"""Actual native OpenPI EEF actions; no absolute-joint reinterpretation."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import PROMPT_SCHEMA,augment_prompt,validate_scene
from VLA.pi0.eef_scene32_training import build_config,ACTION_SCHEMA


def main():
    import jax
    from flask import Flask,jsonify,request
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from openpi.models import model as model_api
    from openpi.policies import policy_config
    from VLA.privileged_native_server import decode_image
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('pi0','pi0_fast'),required=True)
    for name in ('checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args();cp=a.checkpoint.resolve();a.output.mkdir(parents=True,exist_ok=False)
    if jax.default_backend()!='gpu':raise ValueError('Real native GPU required')
    contract=json.loads((cp/'scene32_checkpoint_contract.json').read_text())
    if contract.get('action_schema')!=ACTION_SCHEMA or contract['model']!=a.model:raise ValueError('Wrong action-space checkpoint')
    for path,digest in contract['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('EEF input/action implementation changed')
    precision=Path(contract['actual_precision_proof'])
    if sha256_file(precision)!=contract['precision_proof_sha256']:raise ValueError('Actual saved precision evidence changed')
    proof=json.loads(precision.read_text());cfg=build_config(a.model,contract['stats'],cp,contract['requested_updates'])
    restore=model_api.restore_params
    def preserving(path,*args,**kwargs):kwargs['dtype']=None;return restore(path,*args,**kwargs)
    with patch.object(model_api,'restore_params',preserving):native=policy_config.create_trained_policy(cfg,str(cp))
    learned=flatten_dict(nnx.state(native._model).filter(cfg.trainable_filter).to_pure_dict())
    for row in proof['trainable_tensors']:
        value=np.asarray(learned[tuple(row['path'])])
        if str(value.dtype)!='float32' or hashlib.sha256(value.tobytes()).hexdigest()!=row['saved_sha256']:raise ValueError('Actual learned EEF parameter reload is lossy')
    atomic_json(a.output/'exact_reload.json',{'passed':True,'every_learned_value_exact':True,'action_schema':ACTION_SCHEMA,'model_SR_established':False})
    metadata={'model':a.model,'checkpoint':str(cp),'privileged_schema':PROMPT_SCHEMA,'actual_feedback_fields':64,
        'target_domain_finetuned':True,'action_schema':ACTION_SCHEMA,'adaptation_regime':'shared_EEF_delta_native_OpenPI_LoRA_with_80_20_recovery',
        'oracle_or_ik_controller':False,'action_kind':'canonical_delta_ee_7d'}
    atomic_json(a.output/'metadata.json',metadata);app=Flask(__name__)
    @app.get('/metadata')
    def info():return jsonify(metadata)
    @app.post('/act')
    def act():
        try:
            payload=request.get_json();raw=validate_scene(payload['scene64']);prompt=augment_prompt(payload['instruction'],raw)
            if payload.get('schema')!=PROMPT_SCHEMA or payload['prompt']!=prompt:raise ValueError('Actual complete current observation differs')
            observed={'image':decode_image(payload['front_rgb_png']),'wrist_image':decode_image(payload['wrist_rgb_png']),
                'joints':((raw[:6]+np.pi)%(2*np.pi)-np.pi).astype(np.float32),'gripper':raw[6:7],
                'pre_action_scene64':raw,'prompt':prompt}
            kwargs={}
            if a.model=='pi0':
                seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],payload['inference_index']]).encode()).digest()[:8],'little')
                kwargs['noise']=np.random.default_rng(seed).standard_normal((5,cfg.model.action_dim),dtype=np.float32)
            result=np.asarray(native.infer(observed,**kwargs)['actions'],np.float32)
            if result.shape!=(cfg.model.action_horizon,7) or not np.isfinite(result).all():raise ValueError('Malformed native EEF command; no fallback')
            result[:,6]=2*np.clip(result[:,6],0,1)-1
            return jsonify(actions=result.tolist(),action_kind='canonical_delta_ee_7d',action_source='native_neural_model',
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        except (ValueError,KeyError) as error:return jsonify(error=str(error)),422
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':main()
