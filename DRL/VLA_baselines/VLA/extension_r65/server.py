"""Read current simulator observations, return only actual native neural chunks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.scene64_conditioning import encode_numeric, validate_scene, augment_prompt
from VLA.privileged_visual_r53.scene_map import render, task_from_instruction
from .common import SCHEMA, native_contract, HORIZONS, privileged_coverage


def main(a):
    from flask import Flask,jsonify,request
    from VLA.privileged_native_server import decode_image
    cp=a.checkpoint.resolve();c=native_contract(cp)
    if c['model']!=a.model:raise ValueError('Wrong own native checkpoint family')
    if a.model=='vla_jepa':
        import torch
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(13500*2**20/torch.cuda.get_device_properties(0).total_memory)
        from .jepa import Runtime
        runtime=Runtime(c['stats'],cp)
    else:
        from .fast import Runtime
        runtime=Runtime(cp)
    meta={'schema':SCHEMA,'model':a.model,'checkpoint':str(cp),'GT3_valid':True,
        'privileged_coverage':privileged_coverage(a.model),
        'native_horizon':HORIZONS[a.model],'actual_neural_model':True,
        'action_kind':'canonical_delta_ee_7d','no_Oracle_or_geometric_action_repair':True}
    a.output.mkdir(parents=True,exist_ok=False);atomic_json(a.output/'metadata.json',meta)
    app=Flask(__name__)
    @app.get('/health')
    def health():return jsonify(meta)
    @app.post('/act')
    def act():
        p=request.get_json();raw=validate_scene(p['scene64']);instruction=p['instruction'];task=task_from_instruction(instruction)
        gt=render(raw,task);h=HORIZONS[a.model]
        sample={'front':decode_image(p['front_png']),'wrist':decode_image(p['wrist_png']),
            'map':gt,'raw64':raw,'numeric67':encode_numeric(raw),'instruction':instruction,
            'prompt':augment_prompt(instruction,raw),'task':task,'action':np.zeros((h,7),np.float32),
            'action_is_pad':np.zeros(h,bool)}
        digest=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
        seed=int.from_bytes(hashlib.sha256(json.dumps([p['seed'],instruction,p['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
        result=runtime.predict(sample,seed)
        return jsonify(actions=result.tolist(),actual_neural_model=True,input_sha256=digest,
            map_RGB_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),native_views=3,
            actual_native_input_receipt=runtime.last_input_receipt,
            action_kind='canonical_delta_ee_7d')
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=tuple(HORIZONS),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
