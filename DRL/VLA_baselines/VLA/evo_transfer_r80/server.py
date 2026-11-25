"""Only native VLA actions. Privileged observations are not a controller."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flask import Flask, jsonify, request
from VLA.common.provenance import sha256_file
from VLA.privileged_native_server import decode_image
from VLA.privileged_visual_r53.scene_map import render, task_from_instruction
from .common import read, verify, PORT
from .data import observed_sample
from .native import Runtime


def main(a):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.82)
    proof=read(a.checkpoint/'TRAINING.json');verify(proof['code'])
    if not proof['completed'] or sha256_file(a.checkpoint/'learned.safetensors')!=proof['learned_sha256']:
        raise ValueError('Actual completed learned checkpoint required')
    if not read(a.reload_proof)['passed']:raise ValueError('Fresh reload qualification required')
    runtime=Runtime(read(a.checkpoint/'stats.json'),a.checkpoint)
    meta={'model':'evo1_r80','checkpoint':str(a.checkpoint.resolve()),
        'checkpoint_sha256':proof['learned_sha256'],'actual_neural_model':True,
        'native_horizon':50,'native_channels':24,'physical_channels':7,
        'views':3,'privileged_numeric_dim':67,'state_dim':91,
        'native_preprocessing':'ImageNet448','action_source':'actual_native_Evo1_flow_head',
        'Oracle_waypoint_or_geometric_action_repair':False}
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=8*2**20
    @app.get('/health')
    def health():return jsonify(meta)
    @app.post('/act')
    def act():
        payload=request.get_json()
        expected={'instruction','scene64','ee_position','ee_rotation_6d','front_png','wrist_png','seed','inference_index'}
        if set(payload)!=expected:raise ValueError('Undeclared policy input')
        gt=render(np.asarray(payload['scene64'],np.float32),task_from_instruction(payload['instruction']))
        sample=observed_sample(payload,decode_image(payload['front_png']),decode_image(payload['wrist_png']),gt)
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],
            payload['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
        result=runtime.predict(sample,seed)
        return jsonify(actions=result.tolist(),actual_neural_model=True,input_sha256=digest,
            native_input_receipt=runtime.last_input,map_RGB_sha256=hashlib.sha256(gt.tobytes()).hexdigest())
    app.run(host='127.0.0.1',port=a.port,threaded=False,use_reloader=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reload-proof',type=Path,required=True);p.add_argument('--port',type=int,default=PORT)
    main(p.parse_args())
