import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flask import Flask,jsonify,request
from VLA.common.artifact_io import atomic_json
from VLA.intern_r72.common import bindings
from VLA.intern_r72.reload import load
from VLA.intern_r72.data import Inputs,decode
from VLA.intern_r72.native import prediction
from VLA.scene64_conditioning import validate_scene,encode_numeric
from VLA.privileged_native_server import decode_image
from VLA.privileged_visual_r53.scene_map import render,task_from_instruction


def main(a):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(202672)
    cp=a.checkpoint.resolve();model,proof,stats=load(cp);inputs=Inputs(stats)
    a.output.mkdir(parents=True,exist_ok=False)
    meta={'model':'internvla_a15_r72','actual_neural_model':True,'checkpoint':str(cp),
        'checkpoint_sha256':proof['trainables_sha256'],'completed_optimizer_updates':proof['completed_optimizer_updates'],
        'adaptation':proof['adaptation'],'native_horizon':50,'native_action_dim':32,'numeric_dimensions':67,
        'native_views':3,'continuous_state67_and_causal_measured_feedback_text':True,
        'no_Oracle_or_geometric_action_repair':True,'code':bindings()}
    atomic_json(a.output/'metadata.json',meta)
    app=Flask(__name__)
    @app.get('/health')
    def health():return jsonify(meta)
    @app.post('/act')
    def act():
        payload=request.get_json();raw=validate_scene(payload['scene64'])
        gt=render(raw,task_from_instruction(payload['instruction']))
        sample={'instruction':payload['instruction'],'raw64':raw,'numeric67':encode_numeric(raw),
            'front':decode_image(payload['front_png']),'wrist':decode_image(payload['wrist_png']),'map':gt}
        seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],payload['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
        native=prediction(model,inputs(sample,device='cuda'),seed)
        return jsonify({'actions':decode(native,stats['intern_action']).tolist(),'actual_neural_model':True,
            'native_input_receipt':inputs.last_receipt,
            'input_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            'map_RGB_sha256':hashlib.sha256(gt.tobytes()).hexdigest()})
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=5940);main(p.parse_args())
