"""Exercise actual CPU preprocessor + opt-in deployed bounded-state adapter.

Real20 TRAIN observations plus3 explicitly synthetic transport-only velocity/
force extremes. No model allocation, CUDA update, simulator action or SR.
"""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import encode_numeric
from VLA.smolvla.bounded_feedback_r38 import BoundedFeedbackSmolVLA,load_stats,bound,SCHEMA


def run(a):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    if a.output.exists():raise FileExistsError('Keep distinct actual CPU transport proof')
    cfg=PreTrainedConfig.from_pretrained(str(a.checkpoint));cfg.device='cpu'
    processor,_=make_pre_post_processors(cfg,str(a.checkpoint),preprocessor_overrides={'device_processor':{'device':'cpu'}})
    items=json.loads((a.fixture/'requests_with_LOCAL_gold.json').read_text());proof=json.loads((a.fixture/'audit.json').read_text())
    if not proof['passed'] or sha256_file(a.fixture/'requests_with_LOCAL_gold.json')!=proof['request_file_sha256']:raise ValueError('Actual fixture required')
    mean,std=load_stats(a.checkpoint);rows=[]
    cases=[(item,np.asarray(item['payload']['scene64'],np.float32),'actual_original_TRAIN') for item in items]
    for field,value in ((32,2.),(34,-2.),(39,1000.)):
        raw=np.asarray(items[0]['payload']['scene64'],np.float32).copy();raw[field]=value
        cases.append((items[0],raw,'SYNTHETIC_transport_only_field'+str(field)))
    for item,raw,role in cases:
        payload=item['payload'];decode=lambda value:np.asarray(Image.open(BytesIO(base64.b64decode(value))).convert('RGB')).copy()
        obs=SimpleNamespace(front_rgb=decode(payload['front_rgb_png']),wrist_rgb=decode(payload['wrist_rgb_png']),
            joint_positions=raw[:6],prompt=payload['instruction'])
        policy=object.__new__(BoundedFeedbackSmolVLA);policy.env=SimpleNamespace(normalized_gripper_closure=lambda:float(raw[6]))
        policy.bound_mean=mean;policy.bound_std=std;numeric=encode_numeric(raw)
        with patch('VLA.smolvla.scene67_adapter.model_input',return_value={'scene64':raw,'numeric67':numeric}):batch=policy.build_sensor_batch(obs)
        actual=processor(batch);bounded,changed=bound(numeric,mean,std)
        expected=(bounded-mean)/(std+np.float32(1e-8))
        np.testing.assert_allclose(actual['observation.state'].numpy().reshape(-1),expected,rtol=0,atol=1e-6)
        if np.max(np.abs(expected))>10.0001:raise ValueError('Native normalized bound not respected')
        # Real RGB is unchanged; this ablation never edits the camera pixels.
        for key,reference in (('observation.images.camera1',obs.front_rgb),('observation.images.camera2',obs.wrist_rgb)):
            np.testing.assert_array_equal(np.rint(batch[key][0].permute(1,2,0).numpy()*255).astype(np.uint8),reference)
        np.testing.assert_array_equal(policy.last_input['scene64'],raw)
        rows.append({'source_file':item['source_file'],'frame':item['source_frame'],'scope':role,
            'clipped_indices':np.flatnonzero(changed).tolist(),'maximum_native_normalized_abs':float(np.max(np.abs(expected)))})
    atomic_json(a.output,{'passed':True,'schema':SCHEMA,'actual_TRAIN_observations':len(items),'synthetic_transport_only':3,
        'rows':rows,'actual_native_preprocessor_exercised':True,'raw_RGB_and_scene64_retained':True,
        'NN_model_loaded':False,'neural_controls_or_new_task_trials':0,
        'bound_files':{str(p.resolve()):sha256_file(p) for p in (Path(__file__),Path(__file__).parent/'smolvla/bounded_feedback_r38.py',
            a.checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors',a.fixture/'audit.json')}})
    print('passed real20 + explicit synthetic3 native CPU transport; NOT neural qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','fixture','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
