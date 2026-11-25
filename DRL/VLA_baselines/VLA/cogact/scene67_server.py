"""Actual official CogACT VLM + fitted DiT and observed-state conditioning."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import PROMPT_SCHEMA,augment_prompt,validate_scene,encode_numeric
from VLA.cogact.scene67_learning import StateAdapter,SCHEMA,OFFICIAL
from VLA.cogact.scene67_conditioning import conditioned_call


def main():
    from flask import Flask,jsonify,request
    from VLA.cross_vla120_server import load_cogact
    from VLA.privileged_native_server import decode_image
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cp=a.checkpoint.resolve()
    cfg=json.loads((cp/'scene67_contract.json').read_text())
    if cfg['schema']!=SCHEMA or sha256_file(cp/'learned.pt')!=cfg['learned_sha256']:raise ValueError('Own fitted CogACT checkpoint required')
    for path,digest in cfg['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Fitted CogACT implementation changed')
    model,_=load_cogact(OFFICIAL,a.output/'official_load');learned=torch.load(cp/'learned.pt',map_location='cpu',weights_only=False)
    model.action_model.load_state_dict(learned['action_model'],strict=True);adapter=StateAdapter(cfg['stats']);adapter.load_state_dict(learned['state_adapter'],strict=True)
    for key,value in learned['action_model'].items():
        if not torch.equal(model.action_model.state_dict()[key].cpu(),value):raise ValueError('Actual native DiT learned weight changed on reload')
    for key,value in learned['state_adapter'].items():
        if not torch.equal(adapter.state_dict()[key],value):raise ValueError('Learned observation adapter reload differs')
    del learned
    adapter=adapter.cuda().eval();model.action_model.eval();model.norm_stats={'pybullet_scene67':{'action':cfg['action_stats']}}
    atomic_json(a.output/'exact_reload.json',{'passed':True,'all_saved_DiT_and_state_adapter_values_exact':True,
        'official_VLM_retained':True,'model_SR_established':False})
    metadata={'model':'cogact','checkpoint':str(cp),'privileged_schema':PROMPT_SCHEMA,'actual_feedback_fields':64,
        'target_domain_finetuned':True,'adaptation_regime':SCHEMA,'oracle_or_ik_controller':False,
        'action_kind':'native_delta_ee_and_open_gripper','frozen_native_VLM_required':True}
    atomic_json(a.output/'metadata.json',metadata);app=Flask(__name__)
    @app.get('/metadata')
    def info():return jsonify(metadata)
    @app.post('/act')
    def act():
        try:
            payload=request.get_json();raw=validate_scene(payload['scene64']);prompt=augment_prompt(payload['instruction'],raw)
            if payload.get('schema')!=PROMPT_SCHEMA or payload['prompt']!=prompt:raise ValueError('Actual complete causal input differs')
            image=Image.fromarray(decode_image(payload['front_rgb_png'])).resize((224,224),Image.Resampling.LANCZOS)
            ids=model.vlm.llm_backbone.tokenizer(prompt,truncation=False).input_ids
            if len(ids)+258>=2048:raise ValueError('Native CogACT context would truncate')
            state=torch.from_numpy(encode_numeric(raw)).unsqueeze(0).cuda()
            original=model.action_model.net.forward
            def conditioned(*args,**kwargs):
                # Native CFG contains [conditional, uncondition] branches.
                # Never inject target feedback into the null branch.
                return conditioned_call(original,args,kwargs,adapter(state))
            seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],payload['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode(),patch.object(model.action_model.net,'forward',conditioned):
                torch.manual_seed(seed)
                values,_=model.predict_action(image=image,instruction=prompt,unnorm_key='pybullet_scene67',
                    cfg_scale=1.5,use_ddim=True,num_ddim_steps=10,do_sample=False)
            if values.shape!=(16,7) or not np.isfinite(values).all():raise ValueError('Malformed native DiT commands')
            return jsonify(actions=values.tolist(),action_kind=metadata['action_kind'],action_source='native_neural_model',
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        except (ValueError,KeyError) as error:return jsonify(error=str(error)),422
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':main()
