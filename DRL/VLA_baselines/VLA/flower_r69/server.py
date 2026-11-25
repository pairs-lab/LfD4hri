"""Read-only native FLOWER observation→neural-action server for safety gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flask import Flask, jsonify, request
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import DATA
from VLA.flower_r69.native import load_foundation, adapt, native_batch, canonical_actions
from VLA.flower_r69.gate import prediction
from VLA.privileged_native_server import decode_image
from VLA.privileged_visual_r53.scene_map import render, task_from_instruction
from VLA.scene64_conditioning import encode_numeric, validate_scene


def main(a):
    cp = a.checkpoint.resolve()
    proof = json.loads((cp/'GATE2.json').read_text())
    reload_proof = json.loads(a.reload_proof.read_text())
    if not proof['passed'] or not reload_proof['passed'] or not reload_proof['fresh_process']:
        raise ValueError('Own completed actual optimizer and fresh-process reload required')
    for path,digest in proof['code'].items():
        if sha256_file(path) != digest:
            raise ValueError('Bound gate code changed: '+path)
    if sha256_file(cp/'trainables.pt') != proof['saved_trainables_sha256']:
        raise ValueError('Checkpoint file changed')
    stats = json.loads((DATA/'stats.json').read_text())
    audit = json.loads((DATA/'audit.json').read_text())
    if sha256_file(DATA/'stats.json') != audit['stats_sha256']:
        raise ValueError('TRAIN normalizer changed')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.79)
    model,tokenizer,base = load_foundation()
    adapt(model)
    actual = {n:p for n,p in model.named_parameters() if p.requires_grad}
    saved = torch.load(cp/'trainables.pt',map_location='cpu',weights_only=True,mmap=True)
    if set(actual) != set(saved):
        raise ValueError('Native trainable key set mismatch')
    with torch.no_grad():
        for key,p in actual.items():
            value = saved[key]
            if value.shape != p.shape or value.dtype != torch.float32:
                raise ValueError('Native saved master precision/shape mismatch')
            p.copy_(value)
            if not torch.equal(p.detach().cpu(),value):
                raise ValueError('Saved master reload not exact')
    del saved
    a.output.mkdir(parents=True,exist_ok=False)
    meta = {'model':'flower_r69','native_horizon':10,'checkpoint':str(cp),
        'actual_neural_model':True,'native_views':3,'numeric_dimensions':67,
        'schema':'flower_r69_causal_GT3_state67_native_EEF7',
        'no_Oracle_or_geometric_action_repair':True,'benchmark_or_high_SR_established':False,
        'strict_loaded_tensors':base['strict_exact_loaded_tensors']}
    atomic_json(a.output/'metadata.json',meta)
    app=Flask(__name__)
    @app.get('/health')
    def health():
        return jsonify(meta)
    @app.post('/act')
    def act():
        payload=request.get_json()
        raw=validate_scene(payload['scene64']);task=task_from_instruction(payload['instruction'])
        gt=render(raw,task)
        sample={'front':decode_image(payload['front_png']), 'wrist':decode_image(payload['wrist_png']),
            'map':gt, 'numeric67':encode_numeric(raw), 'instruction':payload['instruction']}
        seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],
            payload['inference_index']]).encode()).digest()[:8],'little') & ((1<<63)-1)
        native=prediction(model,native_batch(sample,tokenizer,stats),seed)
        actions=canonical_actions(native,stats)
        receipt={'native_input':model.last_input_shapes,'native_action_space':'EEF7',
            'native_OPEN01_to_canonical_close_sign_applied_once':True,
            'complete_numeric67_consumed':True,'official_FLOWER_rectified_flow':True}
        return jsonify(actions=actions.tolist(),native_predictions=native.tolist(),
            input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            map_RGB_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
            actual_neural_model=True,native_views=3,actual_native_input_receipt=receipt)
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reload-proof',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=5927)
    main(p.parse_args())
