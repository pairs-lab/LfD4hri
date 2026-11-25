"""Actual three-view native128 loader and four TRAIN physical reset checks."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,default_collate
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment,reset_source_episode
from VLA.evaluate_smolvla_v9_source import reset_observation_errors,validate_reset_observation
from VLA.smolvla.target_crop import TargetCropDataset,TargetCropSmolVLA,target_view,CAMERA,NATIVE_CAMERA,SCHEMA
from VLA.scene64_conditioning import encode_numeric


def main():
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import PolicyFeature,FeatureType
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from safetensors import safe_open
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','projection','admission','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=TrainPipelineConfig.from_pretrained(str(a.checkpoint));cfg.policy.device='cpu'
    cfg.policy.input_features[NATIVE_CAMERA]=PolicyFeature(type=FeatureType.VISUAL,shape=(3,384,384))
    cfg.rename_map[CAMERA]=NATIVE_CAMERA
    if cfg.dataset.image_transforms.enable or cfg.policy.max_state_dim!=67:raise ValueError('Unaugmented actual67 source required')
    base=make_dataset(cfg);data=TargetCropDataset(base,a.projection,a.admission)
    stats={cfg.rename_map.get(k,k):v for k,v in data.meta.stats.items()}
    with safe_open(a.checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors',framework='np') as saved:
        for key in ('observation.state','action'):
            for field in ('mean','std'):
                # Native normalizers materialize dataset statistics in FP32;
                # LeRobot's JSON metadata may retain double precision. Compare
                # the ACTUAL deployed dtype, not harmless JSON mantissa bits.
                np.testing.assert_array_equal(np.asarray(stats[key][field],np.float32),saved.get_tensor(key+'.'+field))
    proc,_=make_pre_post_processors(cfg.policy,pretrained_path=None,dataset_stats=stats,
        preprocessor_overrides={'device_processor':{'device':'cpu'}})
    ids=np.linspace(0,len(base)-1,14,dtype=int).tolist()
    for ei,row in enumerate(data.corrections):
        start=len(base)+(data.correction_ends[ei-1] if ei else 0)
        ids.extend(start+fi for fi in (0,row['steps']//2,row['steps']-1))
    if len(ids)!=128:raise ValueError('All38 corrections at initial/mid/terminal plus14 originals required')
    receipts=[]
    for bi,batch in enumerate(DataLoader(data,batch_size=2,sampler=ids,num_workers=0)):
        selected=ids[bi*2:bi*2+2];individual=default_collate([data[i] for i in selected])
        for key in ('action','observation.state',*data.meta.camera_keys):torch.testing.assert_close(batch[key],individual[key],rtol=0,atol=0)
        actual=_preprocess_dataset_batch(copy.deepcopy(batch),data.meta.camera_keys,cfg.rename_map,proc)
        prepared,_=SmolVLAPolicy.prepare_images(SimpleNamespace(config=cfg.policy),actual)
        if len(prepared)!=3:raise ValueError('The actual third image was silently discarded')
        for j,index in enumerate(selected):
            sample=data[index];raw=data.raw64[index];views=[]
            for key in ('observation.images.front','observation.images.wrist'):
                tensor=sample[key][-1] if sample[key].ndim==4 else sample[key]
                views.append(np.rint(tensor.permute(1,2,0).numpy()*255).astype(np.uint8))
            masks=data.raw_masks(index);crop,receipt=target_view(*views,*masks);receipts.append({'index':index,**receipt})
            env=SimpleNamespace(normalized_gripper_closure=lambda:float(raw[6]),
                target_mask=lambda camera,target: masks[0 if camera=='front' else 1])
            policy=object.__new__(TargetCropSmolVLA);policy.env=env
            obs=SimpleNamespace(front_rgb=views[0],wrist_rgb=views[1],joint_positions=raw[:6],prompt=sample['task'])
            with patch('VLA.smolvla.target_crop.model_input',return_value={'scene64':raw,'numeric67':encode_numeric(raw)}):
                sensor=policy.build_sensor_batch(obs)
            inference=proc(sensor)
            inferred,_=SmolVLAPolicy.prepare_images(SimpleNamespace(config=cfg.policy),inference)
            for k in range(3):torch.testing.assert_close(prepared[k][j:j+1],inferred[k],rtol=0,atol=0)
            torch.testing.assert_close(actual['observation.state'][j:j+1].reshape(1,67),inference['observation.state'].reshape(1,67),rtol=0,atol=0)
        if bi%8==0:atomic_json(a.output/'progress.json',{'actual_native_batches':bi+1,'planned_batches':64})
    # These are actual PyBullet reset/mask observations, not fake-env native
    # tensor examples above; qualification distinctly reports both scopes.
    source=Path(data.audit['source']);physical=[]
    for t in ('touch','pick','move','pick_place'):
        row=next(r for r in data.parent_rows if r['task']['name']==t and not r.get('paired_scene_source') and not r.get('recovery_start_actions_not_training_labels'))
        env=audit_environment(source,image_size=384,required_success_steps=1,use_semantic_success=False)
        try:
            obs=reset_source_episode(env,source,row)
            with np.load(source/row['file'],allow_pickle=False) as z:
                reference={k:z[k][:1] for k in ('joint_positions','gripper_closure','front_rgb','wrist_rgb')}
                validate_reset_observation(reset_observation_errors(obs,env.normalized_gripper_closure(),reference))
                for role in ('front','wrist'):np.testing.assert_array_equal(env.target_mask(role,'pick'),z[role+'_pick_mask'][0])
                expected,ref=target_view(z['front_rgb'][0],z['wrist_rgb'][0],z['front_pick_mask'][0],z['wrist_pick_mask'][0])
            actual,receipt=target_view(obs.front_rgb,obs.wrist_rgb,env.target_mask('front','pick'),env.target_mask('wrist','pick'))
            if receipt!=ref or int(np.max(np.abs(actual.astype(np.int16)-expected.astype(np.int16))))>1:
                raise ValueError('Actual simulator mask/box/view differs from TRAIN crop')
            Image.fromarray(np.concatenate((obs.front_rgb,obs.wrist_rgb,actual),axis=1)).save(a.output/(t+'_actual_reset_raw_front_wrist_GTcrop.png'))
            physical.append({'task':t,'source_file':row['file'],'actual_current_target_masks_equal':True,**receipt})
        finally:env.close()
    bound={str(path.resolve()):sha256_file(path) for path in (Path(__file__),Path(__file__).with_name('target_crop.py'),
        a.projection/'audit.json',a.admission,a.checkpoint/'scene67_contract.json',a.checkpoint/'model.safetensors')}
    atomic_json(a.output/'audit.json',{'passed':True,'schema':SCHEMA,'samples':128,'actual_native_batches':64,
        'all38_corrective_initial_middle_terminal_included':True,'all_three_train_inference_images_exact':True,
        'original_state67_and_action_normalization_unchanged':True,'physical_TRAIN_resets':physical,
        'crop_receipts':receipts,'bound_files':bound,'source_frames':len(data),'camera_or_robot_pose_changed':False,
        'fake_environment_scope':'native_tensor_transport_only; separate four real PyBullet resets above',
        'long_fit_qualified':False,'actual_CUDA_updates':0,'model_SR_established':False})


if __name__=='__main__':main()
