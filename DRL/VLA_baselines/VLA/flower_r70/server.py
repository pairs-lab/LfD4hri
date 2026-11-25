"""Same frozen R69 native inputs/actions, supports actual trained checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flask import Flask,jsonify,request
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70.common import checkpoint,freeze_test_code,read
from VLA.flower_r69.native import load_foundation,adapt,native_batch,canonical_actions
from VLA.flower_r69.gate import prediction
from VLA.privileged_native_server import decode_image
from VLA.privileged_visual_r53.scene_map import render,task_from_instruction
from VLA.scene64_conditioning import validate_scene,encode_numeric


def main(a):
    cp=a.checkpoint.resolve();proof=checkpoint(cp,allow_rolling=a.allow_rolling)
    binding=freeze_test_code();stats=read(cp/'stats.json')
    torch.set_num_threads(2);torch.manual_seed(202669)
    # Inference alongside the existing trainer is explicitly capped. This
    # process has no optimizer and cannot appropriate another model's memory.
    torch.cuda.set_per_process_memory_fraction(4.0*2**30/torch.cuda.get_device_properties(0).total_memory)
    model,tokenizer,base=load_foundation();adapt(model)
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    saved=torch.load(cp/'trainables.pt',weights_only=True,mmap=True,map_location='cpu')
    if set(params)!=set(saved):raise ValueError('Learned master key set mismatch')
    with torch.no_grad():
        for name,p in params.items():
            value=saved[name]
            if value.dtype!=torch.float32 or value.shape!=p.shape or not torch.isfinite(value).all():
                raise ValueError('Learned master precision/shape/finite differs')
            p.copy_(value)
            if not torch.equal(p.detach().cpu(),value):raise ValueError('Saved master reload not exact')
    del saved
    model.requires_grad_(False)
    a.output.mkdir(parents=True,exist_ok=False)
    meta={'model':'flower_r69','checkpoint':str(cp),'native_horizon':10,'native_views':3,
        'numeric_dimensions':67,'actual_neural_model':True,'actual_new_optimizer_updates':proof['actual_new_optimizer_updates'],
        'development_only':True,'privileged_causal_GT3_state67':True,'no_Oracle_or_geometric_action_repair':True,
        'source_and_plant_binding':binding,'checkpoint_sha256':proof['trainables_sha256'],
        'all_saved_FP32_masters_exact':True}
    atomic_json(a.output/'metadata.json',meta)
    app=Flask(__name__)
    @app.get('/health')
    def health():return jsonify(meta)
    @app.post('/act')
    def act():
        payload=request.get_json();raw=validate_scene(payload['scene64'])
        gt=render(raw,task_from_instruction(payload['instruction']))
        sample={'front':decode_image(payload['front_png']),'wrist':decode_image(payload['wrist_png']),
                'map':gt,'numeric67':encode_numeric(raw),'instruction':payload['instruction']}
        seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],
            payload['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
        native=prediction(model,native_batch(sample,tokenizer,stats),seed)
        result={'actions':canonical_actions(native,stats).tolist(),'native_predictions':native.tolist(),
            'input_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            'map_RGB_sha256':hashlib.sha256(gt.tobytes()).hexdigest(),'actual_neural_model':True,
            'native_views':3,'actual_native_input_receipt':model.last_input_shapes}
        return jsonify(result)
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=5930)
    p.add_argument('--allow-rolling',action='store_true');main(p.parse_args())
