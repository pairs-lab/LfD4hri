"""Authentic family neural inference with causal state and strict receipts."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import read,verify
from .native_gate import family_runtime
from .contracts import measured_pose,decode_seer,decode_depth
from VLA.common.provenance import sha256_file


def main(a):
    import torch
    from flask import Flask,request,jsonify
    from VLA.privileged_native_server import decode_image
    from VLA.privileged_visual_r53.scene_map import render,task_from_instruction
    from VLA.scene64_conditioning import validate_scene,encode_numeric
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    method=read(a.method);verify(method['code']);verify(method['weights_sha256'])
    meta=read(Path(method['checkpoint'])/'TRAINING.json');family=method['family']
    runtime=family_runtime(family,meta['stats'],Path(method['checkpoint']))
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=8*1024**2
    state={'key':None,'index':0,'history':deque(maxlen=7),'previous_command':None}
    @app.get('/health')
    def health():return jsonify(actual_neural_model=True,family=family,native_horizon=runtime.horizon,
        checkpoint=method['checkpoint'],method_sha256=sha256_file(a.method),Oracle_or_geometric_action_residual=False)
    @app.post('/act')
    def act():
        p=request.get_json();fields={'instruction','scene64','ee_position','ee_rotation_6d','front_png','wrist_png','seed','inference_index'}
        if family=='falcon_esm':fields|={'front_depth_zlib_f32','front_intrinsics','front_extrinsics_OpenCV'}
        if set(p)!=fields:raise ValueError('Missing/unknown native observation fields')
        index=int(p['inference_index']);key=(p['seed'],p['instruction'])
        if index==0:state.update(key=key,index=0,history=deque(maxlen=7),previous_command=None)
        if state['key']!=key or state['index']!=index:raise ValueError('Native causal history skip/reset mismatch')
        raw=validate_scene(p['scene64']);task=task_from_instruction(p['instruction'])
        front=decode_image(p['front_png']);wrist=decode_image(p['wrist_png'])
        seed=int.from_bytes(hashlib.sha256(json.dumps([*key,index]).encode()).digest()[:8],'little')&((1<<63)-1)
        s={'raw64':raw,'numeric67':encode_numeric(raw),'front':front,'wrist':wrist,
            'map':render(raw,task),'instruction':p['instruction'],'task':task,
            'ee_position':np.asarray(p['ee_position'],np.float32),'ee_rotation_6d':np.asarray(p['ee_rotation_6d'],np.float32)}
        if family=='seer':
            pose=measured_pose(s['ee_position'],s['ee_rotation_6d'])
            native_state=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3]).as_euler('xyz'),float(raw[6]<.5)]
            record=(front,wrist,native_state.astype(np.float32),raw)
            if not state['history']:state['history'].extend([record]*6);state['previous_command']=pose.copy()
            state['history'].append(record)
            s.update(front_history=np.stack([x[0] for x in state['history']]),
                wrist_history=np.stack([x[1] for x in state['history']]),
                state_history=np.stack([x[2] for x in state['history']]),scene64_history=np.stack([x[3] for x in state['history']]))
            native=runtime.predict(s,seed);actions,next_command=decode_seer(native,pose,state['previous_command'])
            state['previous_command']=next_command
        else:
            if family=='falcon_esm':s.update(decode_depth(p))
            actions=runtime.predict(s,seed)
        if actions.shape!=(runtime.horizon,7) or not np.isfinite(actions).all():raise ValueError('Native canonical output chunk malformed')
        state['index']+=1
        return jsonify(actions=actions.tolist(),actual_neural_model=True,
            input_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),
            map_RGB_sha256=hashlib.sha256(s['map'].tobytes()).hexdigest(),
            receipt={'family':family,'actual_native_views':2 if family in ('seer','falcon_esm') else 3,
                'native':runtime.last_input,'future_teacher_phase_or_labels':False,
                'geometric_action_residual_or_Oracle':False,'prefix_executed':1})
    app.run(host='127.0.0.1',port=a.port,threaded=False,use_reloader=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--memory-fraction',type=float,default=.86);main(p.parse_args())
