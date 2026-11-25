"""Native stateful VLANeXt server; history advances once per actual observation."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import read,verify
from .native import Runtime
from .data import observation74
from VLA.common.provenance import sha256_file

def main(a):
    import torch
    from flask import Flask,request,jsonify
    from VLA.new_vlas.action_contracts import native8
    from VLA.privileged_native_server import decode_image
    from VLA.privileged_visual_r53.scene_map import render,task_from_instruction
    from VLA.scene64_conditioning import validate_scene
    torch.set_num_threads(2)
    method=read(a.method);verify(method['code']);verify(method['weights_sha256'])
    meta=read(Path(method['checkpoint'])/'TRAINING.json')
    runtime=Runtime(meta['stats'],checkpoint=Path(method['checkpoint'])/'weights.safetensors')
    state={'key':None,'expected_index':0,'history':deque(maxlen=8)}
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=8*2**20
    @app.get('/health')
    def health():
        return jsonify(actual_neural_model=True,method_sha256=sha256_file(a.method),family='vlanext_s',
            native_horizon=8,checkpoint=method['checkpoint'],Oracle_or_geometric_action_residual=False)
    @app.post('/act')
    def act():
        p=request.get_json()
        expected={'instruction','scene64','ee_position','ee_rotation_6d','front_png','wrist_png','seed','inference_index'}
        if set(p)!=expected:
            raise ValueError('Unknown or incomplete observation fields')
        key=(p['seed'],p['instruction']);index=int(p['inference_index'])
        if index==0:
            state.update(key=key,expected_index=0,history=deque(maxlen=8))
        if state['key']!=key or state['expected_index']!=index:
            raise ValueError('History skip/order mismatch; prefix must remain1')
        raw=validate_scene(p['scene64']);task=task_from_instruction(p['instruction'])
        prefix=native8(np.asarray(p['ee_position']),np.asarray(p['ee_rotation_6d']),raw[27])[:6]
        measured=observation74(prefix,raw,meta['stats'])
        if not state['history']:
            state['history'].extend([measured.copy() for _ in range(7)])
        state['history'].append(measured)
        s={'front':decode_image(p['front_png']),'wrist':decode_image(p['wrist_png']),
            'map':render(raw,task),'history74':np.stack(state['history']),'instruction':p['instruction']}
        seed=int.from_bytes(hashlib.sha256(json.dumps([*key,index]).encode()).digest()[:8],'little')&((1<<63)-1)
        actions=runtime.predict(s,seed)
        state['expected_index']+=1
        return jsonify(actions=actions.tolist(),actual_neural_model=True,
            input_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),
            map_RGB_sha256=hashlib.sha256(s['map'].tobytes()).hexdigest(),
            receipt={'family':'vlanext_s','actual_native_views':3,'numeric_dim':74,
                'future_teacher_phase_or_labels':False,'geometric_action_residual_or_Oracle':False,
                'native':runtime.last_input})
    app.run(host='127.0.0.1',port=a.port,threaded=False,use_reloader=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',type=Path,required=True)
    p.add_argument('--port',type=int,default=5984);main(p.parse_args())
