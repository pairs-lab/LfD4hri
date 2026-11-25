import argparse
from collections import deque
from pathlib import Path
import hashlib
import json
import numpy as np
from VLA.common.provenance import sha256_file
from .common import read,verify
from .native import Runtime

def main(a):
    import torch
    from flask import Flask,request,jsonify
    from VLA.new_vlas.action_contracts import native8
    from VLA.replacement_r84.data import observation74
    from VLA.scene64_conditioning import validate_scene
    from VLA.privileged_native_server import decode_image
    from VLA.privileged_visual_r53.scene_map import render,task_from_instruction
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.80)
    method=read(a.method);verify(method['code']);verify(method['weights_sha256'])
    meta=read(Path(method['checkpoint'])/'TRAINING.json')
    r=Runtime(meta['stats'],checkpoint=Path(method['checkpoint'])/'weights.safetensors')
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=8*2**20
    state={'key':None,'expected':0,'values':deque(maxlen=2)}
    @app.get('/health')
    def health():
        return jsonify(actual_neural_model=True,method_sha256=sha256_file(a.method),
            family='dita',native_horizon=10,checkpoint=method['checkpoint'])
    @app.post('/act')
    def act():
        p=request.get_json()
        if set(p)!={'instruction','scene64','ee_position','ee_rotation_6d','front_png','wrist_png','seed','inference_index'}:
            raise ValueError('Incomplete or undeclared observation')
        key=p['seed'],p['instruction'];idx=int(p['inference_index'])
        if idx==0:state.update(key=key,expected=0,values=deque(maxlen=2))
        if state['key']!=key or state['expected']!=idx:raise ValueError('Missing real history timestamp; execute1 required')
        raw=validate_scene(p['scene64']);task=task_from_instruction(p['instruction'])
        prefix=native8(np.asarray(p['ee_position']),np.asarray(p['ee_rotation_6d']),raw[27])[:6]
        observed={'front':decode_image(p['front_png']),'wrist':decode_image(p['wrist_png']),
            'map':render(raw,task),'numeric':observation74(prefix,raw,meta['stats'])}
        if not state['values']:state['values'].append(observed)
        state['values'].append(observed)
        s={'instruction':p['instruction'],'history74':np.stack([v['numeric'] for v in state['values']]),
            'history_images':{name:[v[name] for v in state['values']] for name in ('front','wrist','map')}}
        seed=int.from_bytes(hashlib.sha256(json.dumps([*key,idx]).encode()).digest()[:8],'little')&((1<<63)-1)
        actions=r.predict(s,seed);state['expected']+=1
        return jsonify(actions=actions.tolist(),actual_neural_model=True,
            input_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),
            map_RGB_sha256=hashlib.sha256(observed['map'].tobytes()).hexdigest(),
            receipt={'family':'dita','actual_native_views':3,'numeric_dim':74,
                'future_teacher_phase_or_labels':False,'geometric_action_residual_or_Oracle':False,
                'native':r.last_input})
    app.run(host='127.0.0.1',port=a.port,threaded=False,use_reloader=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',type=Path,required=True)
    p.add_argument('--port',type=int,default=5985);main(p.parse_args())
