"""Actual native VLA with two unchanged RGBs plus causal sim-GT image."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from VLA.common.provenance import sha256_file
from VLA.new_vlas.action_contracts import pack_native20,native8,normalize_bounds
from VLA.stable_native_r44.conditioning import bounded_append
from .contract import require_visual
from .runtime import create_runtime
from .scene_map import render,task_from_instruction,contract


def main(args):
    from flask import Flask,jsonify,request
    torch.set_num_threads(2);torch.manual_seed(202643)
    cp=args.checkpoint.resolve();cfg=require_visual(cp)
    saved=json.loads((cp/'saved_values.json').read_text())
    for name,key in [('trainable_masters.pt','trainable_master_sha256'),('config.json','config_sha256')]:
        if sha256_file(cp/name)!=saved[key]:raise ValueError('Own checkpoint bytes changed')
    lineage=cfg['r53_lineage']
    if lineage['actual_extra_updates_with_third_GT_image']<1500:
        raise ValueError('Only completed1500-update own3-view fit may serve benchmark actions')
    runtime=create_runtime(cfg['family'],cfg['stats'])
    runtime.load_trainables(cp/'trainable_masters.pt');runtime.to('cuda');runtime.eval()
    metadata={'family':cfg['family'],'checkpoint':str(cp),'actual_optimizer_updates':cfg['actual_optimizer_updates'],
        'privileged_visual_observation':contract(),'numeric_privileged_fields':67,'native_horizon':runtime.horizon,
        'action_source':'actual_native_pretrained_VLA_finetuned_with_two_RGB_plus_causal_GT_diagram',
        'oracle_actions_or_privileged_geometric_action_correction':False,
        'checkpoint_master_sha256':saved['trainable_master_sha256']}
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=8*2**20
    @app.get('/health')
    def health():return jsonify(metadata)
    @app.post('/act')
    def act():
        payload=request.get_json()
        if set(payload)!={'instruction','scene64','ee_position','ee_rotation_6d','front_png','wrist_png'}:
            raise ValueError('Only original causal payload supported')
        raw=np.asarray(payload['scene64'],np.float32)
        xyz,six=np.asarray(payload['ee_position'],np.float32),np.asarray(payload['ee_rotation_6d'],np.float32)
        task=task_from_instruction(payload['instruction'])
        prefix=(pack_native20(xyz,six,np.asarray(raw[6])) if cfg['family']=='xvla' else
            normalize_bounds(native8(xyz,six,raw[27]),cfg['stats']['adapter']['proprio']))
        gt=render(raw,task)
        sample={'instruction':payload['instruction'],'state':bounded_append(prefix,raw,cfg['stats']['scene67']),
            'privileged_map':gt}
        for role in ('front','wrist'):
            sample[role]=np.asarray(Image.open(BytesIO(base64.b64decode(payload[role+'_png']))).convert('RGB'))
            if sample[role].shape!=(384,384,3):raise ValueError('Original camera domain changed')
        actions=runtime.predict(sample)
        if actions.shape!=((30,20) if cfg['family']=='xvla' else (8,7)) or not np.isfinite(actions).all():
            raise ValueError('Actual native output malformed')
        receipt=runtime.r53_native_visual_receipt
        if receipt['actual_views']!=3:raise ValueError('Third visual view not consumed by native neural forward')
        return jsonify(actions=actions.tolist(),actual_neural_model=True,
            input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            privileged_map_RGB_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),native_visual_receipt=receipt,
            action_kind='native_absolute_ee20_closure' if cfg['family']=='xvla' else 'delta_ee7_open01')
    app.run(host='127.0.0.1',port=args.port,threaded=False,use_reloader=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    main(parser.parse_args())
