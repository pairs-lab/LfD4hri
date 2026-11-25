"""Actual native OpenPI sampling with32 causal state plus64 language fields."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import PROMPT_SCHEMA,augment_prompt,parse_prompt,validate_scene
from VLA.pi0.scene32_inputs import SCHEMA


def load(checkpoint,model,output):
    from flax import nnx
    from flax.traverse_util import flatten_dict
    from openpi.models import model as model_api
    from openpi.policies import policy_config
    from VLA.pi0.scene32_training import build_config
    contract=json.loads((checkpoint/'scene32_checkpoint_contract.json').read_text())
    proof_path=Path(contract['actual_precision_proof'])
    if contract['model']!=model or contract['state_schema']!=SCHEMA or sha256_file(proof_path)!=contract['precision_proof_sha256']:
        raise ValueError('Own actual native32 saved checkpoint required')
    for file,digest in contract['bound_files'].items():
        if sha256_file(file)!=digest:raise ValueError('Saved native32 input implementation changed')
    proof=json.loads(proof_path.read_text());cfg=build_config(model,contract['stats'],checkpoint,contract['requested_updates'])
    stock=model_api.restore_params
    def preserve(path,*args,**kwargs):
        if Path(path).resolve()!=(checkpoint/'params').resolve():raise ValueError('Unexpected model weight restore')
        kwargs['dtype']=None  # retain learned FP32 and frozen BF16 EXACTLY
        return stock(path,*args,**kwargs)
    with patch.object(model_api,'restore_params',preserve):native=policy_config.create_trained_policy(cfg,str(checkpoint))
    loaded=flatten_dict(nnx.state(native._model).filter(cfg.trainable_filter).to_pure_dict())
    if set(loaded)!={tuple(row['path']) for row in proof['trainable_tensors']}:raise ValueError('Learned tensor set changed on reload')
    for row in proof['trainable_tensors']:
        value=np.asarray(loaded[tuple(row['path'])])
        if str(value.dtype)!='float32' or hashlib.sha256(value.tobytes()).hexdigest()!=row['saved_sha256']:
            raise ValueError('Actual saved learned native32 value changed on reload')
    atomic_json(output/'exact_reload.json',{'passed':True,'every_learned_value_exact':True,
        'trainable_values':proof['trainable_values'],'precision_proof_sha256':sha256_file(proof_path),'model_SR_established':False})
    return native,cfg


def main():
    import jax
    from flask import Flask,jsonify,request
    from VLA.privileged_native_server import decode_image
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('pi0','pi0_fast'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
    if jax.default_backend()!='gpu':raise ValueError('Actual neural GPU required; no CPU stand-in')
    a.output.mkdir(parents=True,exist_ok=False);cp=a.checkpoint.resolve();native,cfg=load(cp,a.model,a.output)
    metadata={'model':a.model,'checkpoint':str(cp),'privileged_schema':PROMPT_SCHEMA,'actual_feedback_fields':64,
        'state_schema':SCHEMA,'actual_numeric_state_fields':32,'target_domain_finetuned':True,
        'adaptation_regime':'native_OpenPI_LoRA_with_simstate32_and_scene64_language',
        'action_kind':'absolute_joint_targets_and_closure','action_horizon':5,'action_dim':7,
        'native_model_action_dim':cfg.model.action_dim,'oracle_or_ik_controller':False,
        'saved_learned_values_reload_exact':True,'unavailable_camera_masked':a.model=='pi0_fast'}
    atomic_json(a.output/'metadata.json',metadata);app=Flask(__name__)
    @app.get('/metadata')
    def info():return jsonify(metadata)
    @app.post('/act')
    def act():
        try:
            payload=request.get_json()
            if payload.get('schema')!=PROMPT_SCHEMA:raise ValueError('Explicit causal64 input required')
            raw=validate_scene(payload['scene64']);prompt=augment_prompt(payload['instruction'],raw)
            if prompt!=payload['prompt'] or np.max(np.abs(parse_prompt(prompt)-raw))>.000051:raise ValueError('Actual language/feedback identity differs')
            front,wrist=decode_image(payload['front_rgb_png']),decode_image(payload['wrist_rgb_png'])
            observed={'image':front,'wrist_image':wrist,'joints':((raw[:6]+np.pi)%(2*np.pi)-np.pi).astype(np.float32),
                'gripper':raw[6:7].copy(),'pre_action_scene64':raw,'prompt':prompt}
            actual=native._input_transform(copy.deepcopy(observed))
            if actual['state'].shape!=(32,) or int(actual['tokenized_prompt_mask'].sum())>=1024:
                raise ValueError('Actual native feedback dropped or prefix truncated')
            digest=hashlib.sha256(json.dumps([payload['seed'],prompt,payload['inference_index']]).encode()).digest()
            kwargs={}
            if a.model=='pi0':kwargs['noise']=np.random.default_rng(int.from_bytes(digest[:8],'little')).standard_normal((5,cfg.model.action_dim),dtype=np.float32)
            result=np.asarray(native.infer(observed,**kwargs)['actions'],np.float32)
            if result.shape!=(5,7) or not np.isfinite(result).all():raise ValueError('Malformed native command; no fallback')
            return jsonify(actions=result.tolist(),action_kind=metadata['action_kind'],action_source='native_neural_model',
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        except (ValueError,KeyError) as error:return jsonify(error=str(error)),422
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':main()
