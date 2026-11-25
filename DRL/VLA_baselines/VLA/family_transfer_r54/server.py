"""Native neural 5x7 policy server; GT is input, never an action controller."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.scene64_conditioning import encode_numeric,validate_scene
from VLA.privileged_visual_r53.scene_map import render,task_from_instruction
from .common import SCHEMA,checkpoint_contract


def main(a):
    from flask import Flask,jsonify,request
    from VLA.privileged_native_server import decode_image
    a.output.mkdir(parents=True,exist_ok=False);cp=a.checkpoint.resolve()
    if a.model=='smolvla':
        import torch
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(13500*2**20/torch.cuda.get_device_properties(0).total_memory)
        from .smol import Runtime,BaselineRuntime
        runtime=BaselineRuntime(cp) if a.baseline else Runtime(cp,checkpoint_contract(cp)['stats'])
    else:
        if a.baseline:raise ValueError('Do not misrepresent old joint pi0 as a new EEF checkpoint')
        from .runtime_pi0 import Runtime
        runtime=Runtime(cp)
    meta={'schema':SCHEMA,'model':a.model,'checkpoint':str(cp),'native_neural_model':True,
        'input_mode':'preserved_two_RGB_67numeric' if a.baseline else 'two_RGB_plus_GT3_and_native_numeric',
        'GT3_valid':not a.baseline,'native_horizon':5,'action_kind':'canonical_delta_ee_7d',
        'oracle_or_geometric_action_repair':False}
    atomic_json(a.output/'metadata.json',meta);app=Flask(__name__)
    @app.get('/health')
    def health():return jsonify(meta)
    @app.post('/act')
    def act():
        try:
            payload=request.get_json();raw=validate_scene(payload['scene64'])
            instruction=payload['instruction'];task=task_from_instruction(instruction)
            gt=render(raw,task)
            sample={'front':decode_image(payload['front_png']),'wrist':decode_image(payload['wrist_png']),
                'map':gt,'raw64':raw,'numeric67':encode_numeric(raw),'instruction':instruction,
                'task':task,'action':np.zeros((5,7),np.float32),'action_is_pad':np.zeros(5,bool)}
            from VLA.scene64_conditioning import augment_prompt
            sample['prompt']=augment_prompt(instruction,raw)
            digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],instruction,payload['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
            result=runtime.predict(sample,seed)
            return jsonify(actions=result.tolist(),actual_neural_model=True,input_sha256=digest,
                map_RGB_sha256=None if a.baseline else hashlib.sha256(gt.tobytes()).hexdigest(),
                native_views=2 if a.baseline else 3,action_kind='canonical_delta_ee_7d')
        except (ValueError,KeyError) as e:return jsonify(error=str(e)),422
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('smolvla','pi0'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--baseline',action='store_true');main(p.parse_args())
