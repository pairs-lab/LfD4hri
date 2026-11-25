import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import numpy as np
import torch
from MVD_S.preprocess import temporal_indices, make_views, decode_samples
from MVD_S.inference import load_labels, Recognizer
from MVD_S.checkpoint import read_checkpoint, load_classifier, initialize_for_training
from MVD_S.integration import add_action_arguments, action_recognition, demo_inputs, prepare_demo_outputs
from MVD_S.train import read_annotations, optimizer_groups


class ContractTests(unittest.TestCase):
    def test_temporal_matches_upstream_two_clips(self):
        for n in (1,2,3,16,17,33,128):
            tick=(n-1)/16
            all_indices=[]
            for i in range(16):
                for c in range(2):
                    start=int(np.round(tick*i+tick/2*c))
                    end=int(np.round(tick*i+tick/2*(c+1)))
                    all_indices.append((start+end)//2)
            self.assertEqual(temporal_indices(n),[all_indices[0::2],all_indices[1::2]])

    def test_short_clip_repeats_real_indices(self):
        self.assertEqual(temporal_indices(1),[[0]*16,[0]*16])
        with self.assertRaises(ValueError):temporal_indices(0)

    def test_training_sampling_causal_bounds(self):
        indices=temporal_indices(57,clips=1,rng=np.random.default_rng(0))[0]
        self.assertEqual(indices,sorted(indices))
        self.assertTrue(all(0<=i<57 for i in indices))

    def test_spatial_crop_layout_and_rgb_normalization(self):
        for h,w in ((224,448),(448,224),(224,224)):
            frames=np.zeros((16,h,w,3),np.uint8);frames[...,0]=255
            views,meta=make_views([frames],crops=3)
            self.assertEqual(len(views),3)
            self.assertEqual(tuple(views[0].shape),(3,16,224,224))
            self.assertAlmostEqual(float(views[0][0,0,0,0]),(1-.485)/.229,places=5)
            self.assertAlmostEqual(float(views[0][2,0,0,0]),-.406/.225,places=5)
            self.assertEqual(meta[1]['crop_xy'],[0,112] if h>w else [112,0] if w>h else [0,0])

    def test_strict_label_count_but_duplicate_task_names_allowed(self):
        self.assertEqual(len(load_labels()),174)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.txt';p.write_text('pick\n'*173)
            with self.assertRaises(ValueError):load_labels(p)

    def test_missing_or_pretrain_only_classifier_rejected(self):
        with self.assertRaises(FileNotFoundError):read_checkpoint('/no/such/mvd.pth')
        with self.assertRaisesRegex(ValueError,'fine-tuned'):
            load_classifier(Mock(),{'encoder.foo':torch.zeros(1)})
        with self.assertRaisesRegex(ValueError,'fine-tuned'):
            load_classifier(Mock(),{'head.weight':torch.zeros(400,384)})

    def test_safe_native_namespace_and_ddp_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pth'
            torch.save({'model':{'module.a':torch.ones(2)},'args':argparse.Namespace(data_set='SSV2')},p)
            values,receipt=read_checkpoint(p)
            self.assertEqual(list(values),['a'])
            self.assertEqual(len(receipt['sha256']),64)
            with self.assertRaisesRegex(ValueError,'SHA-256'):read_checkpoint(p,'0'*64)

    def test_pretrain_transfer_cannot_silently_ignore_wrong_backbone(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,3))
        with self.assertRaisesRegex(ValueError,'Unqualified'):
            initialize_for_training(model,{'encoder.unrelated':torch.zeros(1)})

    def test_smoke_checkpoint_not_accepted_as_learned_classifier(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'smoke.pth'
            torch.save({'model':{'head.weight':torch.zeros(174,384)},
                        'mvd_metadata':{'software_smoke_only':True}},p)
            with self.assertRaisesRegex(ValueError,'Software-smoke'):
                Recognizer(p)

    def test_default_parser_is_tsm(self):
        p=argparse.ArgumentParser();add_action_arguments(p)
        self.assertEqual(p.parse_args([]).action_backend,'tsm')
        self.assertEqual(p.parse_args(['--action_backend','mvd_s']).mvd_preset,'paper')

    def test_tsm_dispatch_preserves_class_index_mapping(self):
        api=ModuleType('mmaction2.mmaction.apis')
        api.init_recognizer=Mock(return_value=object())
        scores=torch.zeros(174);scores[17]=1
        api.inference_recognizer=Mock(return_value=SimpleNamespace(pred_score=scores))
        from MVD_S.inference import DEFAULT_LABEL_MAP
        args=SimpleNamespace(config='original.py',action_checkpoint='original.pth',
                             label_map=DEFAULT_LABEL_MAP,device='cpu',reduced_frame_video='original.mp4')
        with patch.dict(sys.modules,{'mmaction2.mmaction.apis':api}):
            self.assertEqual(action_recognition(args),'Pick something up')
        api.init_recognizer.assert_called_once_with('original.py','original.pth',device='cpu')

    def test_mvd_dispatch_does_not_load_tsm(self):
        from MVD_S.inference import DEFAULT_LABEL_MAP
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)/'weights.pth';cp.touch()
            args=SimpleNamespace(action_backend='mvd_s',mvd_checkpoint=cp,label_map=DEFAULT_LABEL_MAP,
                                 device='cpu',reduced_frame_video='video.mp4')
            fake=Mock();fake.predict.return_value={'action':'Touch something','backend':'mvd_s'}
            with patch('MVD_S.integration._mvd',return_value=fake):
                self.assertEqual(action_recognition(args),'Touch something')
                self.assertEqual(args.action_recognition_result['backend'],'mvd_s')

    def test_view_merge_uses_average_softmax(self):
        r=Recognizer.__new__(Recognizer);r.device=torch.device('cpu');r.labels=load_labels()
        r.checkpoint={};r.label_map_sha256='test'
        logits=torch.zeros(2,174);logits[0,0]=4;logits[1,1]=1
        r.model=Mock(return_value=logits)
        with patch('MVD_S.inference.decode_samples',return_value=([],{})), \
             patch('MVD_S.inference.make_views',return_value=([torch.zeros(1),torch.zeros(1)],[{},{}])):
            result=r.predict('unused',batch_size=2)
        np.testing.assert_allclose(result['scores'],logits.softmax(-1).mean(0).numpy(),rtol=1e-6)

    def test_annotations_and_output_layer_lr(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'a video.mp4').touch();p=d/'train.csv';p.write_text('a video.mp4 17\n')
            self.assertEqual(read_annotations(p,d)[0][1],17)
            p.write_text('a video.mp4 174\n')
            with self.assertRaises(ValueError):read_annotations(p,d)
        model=torch.nn.Module();model.head=torch.nn.Linear(2,3)
        groups=optimizer_groups(model,lr=.001,layer_decay=.7,weight_decay=.05)
        self.assertTrue(all(group['lr']==.001 for group in groups))
        self.assertEqual({group['weight_decay'] for group in groups},{0.,.05})

    def test_single_video_and_fresh_run_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'plain_name.mp4';video.touch()
            args=SimpleNamespace(single_video=True,video=video,action_backend='mvd_s',mvd_run_dir=d/'new')
            folder,names=demo_inputs(args)
            self.assertEqual(Path(folder),d)
            self.assertEqual(names,['plain_name.mp4'])
            prepare_demo_outputs(args)
            self.assertEqual(Path(args.image_dir),d/'new/image_dir')
            with self.assertRaises(FileExistsError):prepare_demo_outputs(args)

    def test_native_pretrain_fields_are_explicitly_discarded(self):
        model=torch.nn.Module();model.head=torch.nn.Linear(2,3)
        model.fc_norm=torch.nn.LayerNorm(2);model.norm=torch.nn.Identity()
        model.pos_embed=torch.zeros(1,4,2)
        values={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('head.','fc_norm.'))}
        values.update({'norm.weight':torch.ones(2),'norm.bias':torch.zeros(2),
                       'pos_embed':torch.zeros(1,4,2),'mask_token_img':torch.zeros(1),
                       'decoder_img.proj.weight':torch.zeros(2,2)})
        report=initialize_for_training(model,values)
        self.assertIn('decoder_img.proj.weight',report['discarded_pretrain_fields'])
        self.assertEqual(set(report['missing_initialized']),{'head.weight','head.bias','fc_norm.weight','fc_norm.bias'})

    def test_timm_image_registry_not_overwritten(self):
        import timm
        from timm.models import model_entrypoint
        before=model_entrypoint('vit_small_patch16_224')
        from MVD_S import model
        self.assertIs(before,model_entrypoint('vit_small_patch16_224'))

    def test_all_demo_parsers_accept_backend_without_loading_detector(self):
        import ast
        root=Path(__file__).resolve().parents[2]
        for name in ('demo.py','demo_wo_data.py','demo_wo_key.py','demo_wo_os.py'):
            tree=ast.parse((root/name).read_text())
            parser_function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
            module=ast.Module(body=[parser_function],type_ignores=[])
            namespace={'argparse':argparse,'ROOT_PATH':str(root)}
            exec(compile(module,name,'exec'),namespace)
            with patch.object(sys,'argv',[name,'--action_backend','mvd_s','--mvd_checkpoint','test.pth','--single_video']):
                args=namespace['parse_args']()
            self.assertEqual(args.action_backend,'mvd_s');self.assertTrue(args.single_video)


if __name__=='__main__':unittest.main()
