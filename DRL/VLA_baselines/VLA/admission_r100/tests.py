"""Representation/queue fixtures only; never counted as robot success trials."""
import copy
import json
import signal
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import measured_pose,decode_seer,decode_depth
from .falcon_depth import rgb_parity
from .evaluate import approved,FINAL,DEV,CONF,callbacks
from .guard import checked_stop
from VLA.new_vlas.action_contracts import absolute20_from_canonical,canonical_from_absolute20
from VLA.backup_vlas_r93.contracts import seer_local_from_canonical


class ContractTests(unittest.TestCase):
    def test_previous_command_action_and_polarity_roundtrip(self):
        rng=np.random.default_rng(100)
        for _ in range(256):
            rot=Rotation.random(random_state=rng).as_matrix();columns=rot[:,:2].T.reshape(6)
            position=np.array([.0,-.45,.35])+rng.uniform(-.02,.02,3)
            current=measured_pose(position,columns);previous=current.copy();previous[:3,3]+=rng.uniform(-.01,.01,3)
            wanted=np.r_[rng.uniform(-.02,.02,3),rng.uniform(-.03,.03,3),rng.choice([-1.,1.])]
            local=seer_local_from_canonical(wanted,current,previous)
            decoded,_=decode_seer(np.repeat(local[None],3,axis=0),current,previous)
            np.testing.assert_allclose(decoded[0],wanted,atol=1e-7,rtol=0)
            self.assertEqual(decoded[0,6],wanted[6])

    def test_previous_requested_command_uses_same_shared_clamps(self):
        rot=Rotation.from_euler('xyz',[0,np.pi,0]).as_matrix();columns=rot[:,:2].T.reshape(6)
        current=measured_pose([0,-.45,.70],columns);native=np.zeros((3,7));native[:,2]=1.;native[:,6]=-1
        action,command=decode_seer(native,current,current)
        expected=absolute20_from_canonical(current[:3,3],columns,action[0])
        np.testing.assert_allclose(command[:3,3],expected[:3],atol=1e-7)
        self.assertLessEqual(command[2,3],.72)

    def test_rgb_quantization_strict_budget(self):
        a=np.zeros((384,384,3),np.uint8);b=a.copy();b[0,0,0]=1
        self.assertEqual(rgb_parity(a,b)['different_channel_values'],1)
        b[0,0,0]=2
        with self.assertRaises(ValueError):rgb_parity(a,b)
        b=a.copy();b[0,:20,:]=1
        with self.assertRaises(ValueError):rgb_parity(a,b)

    def test_depth_transport_and_decompression_bomb_rejected(self):
        import base64,zlib
        raw=np.full((384,384),.3,np.float32)
        payload={'front_depth_zlib_f32':base64.b64encode(zlib.compress(raw.tobytes())).decode(),
            'front_intrinsics':np.eye(3).tolist(),'front_extrinsics_OpenCV':np.eye(4).tolist()}
        np.testing.assert_array_equal(decode_depth(payload)['depth_z_m'],raw)
        bad={**payload,'front_depth_zlib_f32':base64.b64encode(zlib.compress(raw.tobytes()*2)).decode()}
        with self.assertRaises(ValueError):decode_depth(bad)

    def test_disjoint_locked_roles(self):
        self.assertEqual((len(FINAL),len(DEV),len(CONF)),(200,10,10))
        self.assertFalse(set(FINAL)&set(DEV));self.assertFalse(set(FINAL)&set(CONF));self.assertFalse(set(DEV)&set(CONF))
        self.assertFalse(set(FINAL)&set(range(20300001,20300201)))

    def test_requires_frozen_native_method_for_final(self):
        method={'family':'seer','R100_frozen_before_final':True,'frozen_for_user_requested_benchmark':True}
        cohort={'role':'fresh_secondary_final800','seeds':FINAL}
        self.assertTrue(approved(method,cohort));self.assertFalse(approved({**method,'family':'Oracle'},cohort))
        self.assertFalse(approved(method,{**cohort,'seeds':FINAL[:-1]}))
        self.assertFalse(approved({**method,'R100_frozen_before_final':False},cohort))

    def test_native_reference_callbacks_restore_after_failure(self):
        import VLA.autorefinement_r97.evaluate as evaluator
        import VLA.autonomy_r83.evaluate as physical
        old=evaluator.approved_final,physical.render
        with self.assertRaises(RuntimeError):
            with callbacks():
                self.assertIs(evaluator.approved_final,approved)
                raise RuntimeError('synthetic')
        self.assertEqual((evaluator.approved_final,physical.render),old)

    def test_guard_never_signals_R99_or_reused_PID(self):
        wrong={'pid':100,'command':['python','-m','VLA.refinement_r99.evaluate']}
        with mock.patch('VLA.admission_r100.guard.alive',return_value=True),mock.patch(
            'VLA.admission_r100.guard.identity',return_value=wrong),mock.patch('os.kill') as kill:
            with self.assertRaises(ValueError):checked_stop(wrong)
            kill.assert_not_called()
        good={'pid':100,'command':['python','-m','VLA.admission_r100.native_gate']}
        with mock.patch('VLA.admission_r100.guard.alive',return_value=True),mock.patch(
            'VLA.admission_r100.guard.identity',return_value=wrong),mock.patch('os.kill') as kill:
            with self.assertRaises(ValueError):checked_stop(good)
            kill.assert_not_called()

    def test_adapter_save_reload_includes_new_branch_and_rejects_missing(self):
        import torch
        from .checkpoint_io import save_adaptation,load_adaptation
        model=torch.nn.Module();model.native=torch.nn.Linear(4,3);model.privileged=torch.nn.Linear(67,3)
        model.frozen=torch.nn.Linear(4,3);model.frozen.requires_grad_(False)
        with tempfile.TemporaryDirectory(prefix='r100_SYNTHETIC_NOT_EXPERIMENT_') as tmp:
            root=Path(tmp);save_adaptation(model,root);other=copy.deepcopy(model)
            with torch.no_grad():other.privileged.weight.zero_()
            load_adaptation(other,root)
            self.assertTrue(torch.equal(other.privileged.weight,model.privileged.weight))
            proof=json.loads((root/'ADAPTATION_SCHEMA.json').read_text())
            self.assertFalse(any(k.startswith('frozen.') for k in proof['keys']))
            other.extra=torch.nn.Linear(2,2)
            with self.assertRaises(ValueError):load_adaptation(other,root)

    def test_locked_native_LR_does_not_reset_after_resume(self):
        from .recipe import RECIPES,learning_rate
        for r in RECIPES.values():
            values=[learning_rate(i,r) for i in range(r['maximum_updates']+1)]
            self.assertGreater(min(values),0);self.assertLessEqual(max(values),r['lr'])
            self.assertEqual(learning_rate(130-2,r),learning_rate(128,r))
            self.assertNotEqual(learning_rate(128,r),learning_rate(0,r))

    def test_immutable_manifest_never_silently_overwritten(self):
        from .workflow import immutable
        with tempfile.TemporaryDirectory(prefix='r100_SYNTHETIC_') as tmp:
            p=Path(tmp)/'frozen.json';immutable(p,{'checkpoint':'a'});immutable(p,{'checkpoint':'a'})
            with self.assertRaises(ValueError):immutable(p,{'checkpoint':'b'})
            self.assertEqual(json.loads(p.read_text()),{'checkpoint':'a'})

    def test_interrupted_training_does_not_reuse_partial_directory(self):
        from .workflow import Owner,WorkerInterrupted
        from .common import atomic_json
        with tempfile.TemporaryDirectory(prefix='r100_SYNTHETIC_') as tmp:
            root=Path(tmp);family='seer';out=root/family;out.mkdir()
            guard=root/'guard';guard.mkdir()
            atomic_json(guard/'seer_phase_guard00.json',{'status':'stopped_owned_worker_to_preserve_resources'})
            owner=Owner.__new__(Owner);owner.family=family;owner.resource=mock.Mock();owner.status=mock.Mock()
            with mock.patch('VLA.admission_r100.workflow.OUT',root),mock.patch('VLA.admission_r100.workflow.guarded') as launch:
                with self.assertRaises(WorkerInterrupted):owner.gpu(family,['--output',out/'partial'],'phase',module='VLA.admission_r100.train')
                launch.assert_not_called()

    def test_training_resumes_last_commit_without_repeating_full_budget(self):
        from .workflow import Owner,WorkerInterrupted
        from .common import atomic_json
        with tempfile.TemporaryDirectory(prefix='r100_SYNTHETIC_') as tmp:
            root=Path(tmp);family='seer';folder=root/family;folder.mkdir()
            parent=root/'parent';commit=folder/'phase_attempt00/committed_130';finish=folder/'phase_attempt01/committed_2050'
            steps={str(parent):2,str(commit):130,str(finish):2050};calls=[]
            owner=Owner.__new__(Owner)
            def gpu(f,args,label,**kwargs):
                if '--op' in args:return
                output=Path(args[args.index('--output')+1]);remaining=args[args.index('--additional-updates')+1]
                calls.append(remaining)
                if len(calls)==1:
                    atomic_json(output/'LATEST_COMMITTED.json',{'checkpoint':str(commit)})
                    raise WorkerInterrupted('synthetic interruption')
                atomic_json(output/'COMPLETED.json',{'checkpoint':str(finish)})
            owner.gpu=gpu
            with mock.patch('VLA.admission_r100.workflow.OUT',root),mock.patch(
                'VLA.admission_r100.workflow.qualified',side_effect=lambda p,f:{'updates':steps[str(p)]}):
                self.assertEqual(owner.fit(family,parent,'phase'),finish)
            self.assertEqual(calls,[2048,1920])

    def test_bad_native_weight_hash_never_becomes_a_completed_checkpoint(self):
        from .train import qualified
        from .common import atomic_json
        with tempfile.TemporaryDirectory(prefix='r100_SYNTHETIC_') as tmp:
            root=Path(tmp);(root/'weights.safetensors').write_bytes(b'not a model')
            atomic_json(root/'TRAINING.json',{'code':{},'family':'seer','completed':True,
                'actual_full_native_model':True,'weights_sha256':'f'*64})
            with self.assertRaises(ValueError):qualified(root,'seer')

    def test_ESM_frozen_encoder_boundary_keeps_mask_and_resolves_index_dtype(self):
        import torch
        from .falcon_native import align_esm_patch_tokens
        mask=torch.tensor([True,False]);features=torch.randn(1,4,3)
        output={'x_norm_patchtokens':features,'masks':mask,'metadata':None}
        actual=align_esm_patch_tokens(output,torch.bfloat16)
        self.assertIs(actual['masks'],mask);self.assertIsNone(actual['metadata'])
        self.assertEqual(actual['x_norm_patchtokens'].dtype,torch.bfloat16)
        self.assertTrue(torch.equal(features,output['x_norm_patchtokens']))
        destination=torch.zeros((1,4,3),dtype=torch.bfloat16)
        destination[torch.tensor([0])]=actual['x_norm_patchtokens']
        self.assertTrue(torch.equal(destination,features.bfloat16()))
        with self.assertRaises(ValueError):align_esm_patch_tokens({},torch.bfloat16)

    def test_measured_FALCON_budget_preserves_actual_primary_headroom(self):
        from .guard import admitted_fraction
        for free in (8000,9000,10200,14000):
            r={'GPU_free_MiB':free,'GPU_total_MiB':15941}
            fraction=admitted_fraction(True,r,['--family','falcon_esm'])
            self.assertLessEqual(fraction,.46)
            self.assertGreaterEqual(free-fraction*r['GPU_total_MiB'],2800-1e-6)
            self.assertLessEqual(admitted_fraction(True,r,['--family','seer']),.37)

    def test_nonreentrant_checkpoint_preserves_adapters_behind_frozen_stem(self):
        import torch,warnings
        from torch.utils.checkpoint import checkpoint
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for reentrant in (True,False):
                native=torch.nn.Parameter(torch.ones(2,2));head=torch.nn.Parameter(torch.ones(1))
                frozen_input=torch.ones(1,2)
                output=checkpoint(lambda x:x@native,frozen_input,use_reentrant=reentrant)
                (output*head).sum().backward()
                self.assertGreater(float(head.grad.abs().sum()),0)
                if reentrant:self.assertIsNone(native.grad)
                else:self.assertGreater(float(native.grad.abs().sum()),0)

    def test_FALCON_admission_rejects_any_dead_native_branch(self):
        from .falcon_native import Runtime
        runtime=Runtime.__new__(Runtime);runtime._training_samples=8
        runtime.branch_gradients={'vision':0.,'language':1.};runtime._gradient_groups={'vision':'vision_LoRA','language':'language_LoRA'}
        runtime.load_proof={}
        with self.assertRaises(ValueError):runtime.verify_gradients()
        runtime.branch_gradients['vision']=2.;runtime.verify_gradients()
        self.assertEqual(runtime.load_proof['actual_native_branch_gradient_checks']['vision_LoRA']['positive_gradient_parameters'],1)

    def test_only_measured_low_memory_admitted_Seer_may_share_primary(self):
        from .workflow import concurrent_fit_admitted
        from .common import atomic_json
        with tempfile.TemporaryDirectory(prefix='r100_SYNTHETIC_') as tmp:
            root=Path(tmp);family=root/'seer';cp=root/'synthetic_native_commit'
            with mock.patch('VLA.admission_r100.workflow.OUT',root):
                self.assertFalse(concurrent_fit_admitted('seer'));self.assertFalse(concurrent_fit_admitted('falcon_esm'))
                atomic_json(family/'NATIVE_ADMISSION_READY.json',{'passed':True,'actual_integration_trials':4})
                atomic_json(family/'TRAINER_CODE_ROUNDTRIP.json',{'passed':True,'effective_batch':16,
                    'checkpoint':str(cp),'fresh_native_reload':{'passed':True}})
                atomic_json(cp/'TRAINING.json',{'actual_full_native_model':True,'peak_GPU_MiB':2563})
                self.assertTrue(concurrent_fit_admitted('seer'))
                atomic_json(cp/'TRAINING.json',{'actual_full_native_model':True,'peak_GPU_MiB':4000})
                self.assertFalse(concurrent_fit_admitted('seer'))
                atomic_json(cp/'TRAINING.json',{'actual_full_native_model':True,'peak_GPU_MiB':2563})
                atomic_json(family/'CONCURRENT_TRAINING_DEFERRED.json',{'synthetic':True})
                self.assertFalse(concurrent_fit_admitted('seer'))

    def test_owned_orphan_recovery_keeps_start_time_and_module_guards(self):
        from .guard import same_owned_process
        expected={'pid':100,'ppid':90,'start_ticks':'12345','command':['python','-m','VLA.admission_r100.evaluate']}
        orphan={**expected,'ppid':1}
        with mock.patch('VLA.admission_r100.guard.alive',side_effect=lambda pid:pid==100),mock.patch(
            'VLA.admission_r100.guard.identity',return_value=orphan),mock.patch('os.kill') as kill:
            self.assertTrue(same_owned_process(expected));checked_stop(expected)
            kill.assert_called_once_with(100,signal.SIGTERM)
        with mock.patch('VLA.admission_r100.guard.alive',return_value=True),mock.patch(
            'VLA.admission_r100.guard.identity',return_value=orphan),mock.patch('os.kill') as kill:
            self.assertFalse(same_owned_process(expected))
            with self.assertRaises(ValueError):checked_stop(expected)
            kill.assert_not_called()
        with mock.patch('VLA.admission_r100.guard.alive',side_effect=lambda pid:pid==100),mock.patch(
            'VLA.admission_r100.guard.identity',return_value={**orphan,'start_ticks':'REUSED'}):
            self.assertFalse(same_owned_process(expected))


if __name__=='__main__':unittest.main()
