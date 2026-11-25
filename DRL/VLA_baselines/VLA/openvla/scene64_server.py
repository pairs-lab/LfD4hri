"""Fitted native OpenVLA+full64 language; no scripted action corrections."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import PROMPT_SCHEMA,augment_prompt,validate_scene


def main():
    from flask import Flask,jsonify,request
    import torch
    from VLA.privileged_native_server import decode_image
    from VLA.openvla.privileged_loading import PrivilegedOpenVLA,verify_adapter_values
    from VLA.openvla.finetune import validate_checkpoint_manifest
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--unnorm-key',required=True);a=p.parse_args()
    cp=a.checkpoint.resolve();a.output.mkdir(parents=True,exist_ok=False)
    validate_checkpoint_manifest(cp);cfg=json.loads((cp/'train_config.json').read_text())
    if cfg.get('privileged_prompt_schema')!=PROMPT_SCHEMA or not cfg.get('privileged_model_inputs'):
        raise ValueError('Own fitted full64 input checkpoint required')
    torch.set_num_threads(2)
    native=PrivilegedOpenVLA(checkpoint=str(cp),unnorm_key=a.unnorm_key);native._load()
    proof=verify_adapter_values(native.model,cp)
    proof.update(loader_sha256=sha256_file(Path(__file__).with_name('privileged_loading.py')),
        checkpoint_manifest_sha256=sha256_file(cp/'checkpoint_manifest.json'),model_SR_established=False)
    atomic_json(a.output/'adapter_reload.json',proof)
    if not proof['passed']:raise ValueError('Actual FP32 QLoRA reload mismatch; detailed values preserved')
    metadata={'model':'openvla','checkpoint':str(cp),'privileged_schema':PROMPT_SCHEMA,'actual_feedback_fields':64,
        'target_domain_finetuned':True,'adaptation_regime':'full64_native_prompt_QLoRA','oracle_or_ik_controller':False,
        'action_kind':'canonical_delta_ee_7d','all_saved_adapter_values_exact':True}
    atomic_json(a.output/'metadata.json',metadata);app=Flask(__name__)
    @app.get('/metadata')
    def describe():return jsonify(metadata)
    @app.post('/act')
    def act():
        try:
            value=request.get_json()
            if value.get('schema')!=PROMPT_SCHEMA:raise ValueError('Explicit current scene64 required')
            state=validate_scene(value['scene64']);prompt=augment_prompt(value['instruction'],state)
            if prompt!=value['prompt']:raise ValueError('Actual complete language observation mismatch')
            front=decode_image(value['front_rgb_png'])
            action=native.predict(SimpleNamespace(front_rgb=front,prompt=prompt))
            if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Malformed native model output')
            return jsonify(actions=action[None].tolist(),action_kind='canonical_delta_ee_7d',
                action_source='native_neural_model',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        except (ValueError,KeyError) as error:return jsonify(error=str(error)),422
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':main()
