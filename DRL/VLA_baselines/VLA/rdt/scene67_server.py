"""Full-scene RDT native neural actions; learned input gains, no planner."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import PROMPT_SCHEMA,augment_prompt,validate_scene
from VLA.rdt.server import RDT_ROOT,prepare_sensor_tensors
from VLA.rdt.scene67_inputs import append_scene_conditioning
from VLA.rdt.scene67_native import SCHEMA,load


def main():
    from flask import Flask,jsonify,request
    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    from VLA.rdt.signal_precision import install_fp32_signal_islands
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','output','embeddings','vision'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args();cp=a.checkpoint.resolve();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((cp/'pybullet_rdt_config.json').read_text());contract=cfg['scene67']
    if cfg.get('model_family')!='rdt_scene67' or contract.get('schema')!=SCHEMA:raise ValueError('Own trained full-scene RDT checkpoint required')
    for path,digest in contract['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Fitted scene67 input implementation changed')
    policy,_=load(cp,10,dtype=torch.float32)
    saved=torch.load(cp/'pytorch_model.bin',map_location='cpu',weights_only=True)
    if set(saved)!=set(policy.state_dict()) or any(not torch.equal(v,policy.state_dict()[k].cpu()) for k,v in saved.items()):
        raise ValueError('Actual learned native RDT values changed on reload')
    del saved
    torch.set_num_threads(2);policy=policy.to('cuda').eval();install_fp32_signal_islands(policy)
    vision=SiglipVisionTower(str(a.vision),args=None).to('cuda',dtype=torch.bfloat16).eval()
    embeddings=torch.load(a.embeddings,map_location='cpu',weights_only=True);previous={}
    atomic_json(a.output/'exact_reload.json',{'passed':True,'all_saved_values_exact':True,
        'input_schema':SCHEMA,'weights_sha256':sha256_file(cp/'pytorch_model.bin'),'model_SR_established':False})
    metadata={'model':'rdt','checkpoint':str(cp),'privileged_schema':PROMPT_SCHEMA,'actual_feedback_fields':64,
        'target_domain_finetuned':True,'adaptation_regime':SCHEMA,'oracle_or_ik_controller':False,
        'action_kind':'absolute_joint_targets_and_closure','native_action_DOFs':7,'measured_native_proprio_required':True}
    atomic_json(a.output/'metadata.json',metadata);app=Flask(__name__)
    @app.get('/metadata')
    def info():return jsonify(metadata)
    @app.post('/act')
    def act():
        try:
            payload=request.get_json();raw=validate_scene(payload['scene64']);prompt=augment_prompt(payload['instruction'],raw)
            if payload.get('schema')!=PROMPT_SCHEMA or payload['prompt']!=prompt:raise ValueError('Actual full input identity changed')
            if payload['instruction'] not in embeddings:raise ValueError('Missing frozen task-language embedding')
            native=np.asarray(payload['native_proprio'],np.float32)
            if (native.shape!=(16,) or not np.isfinite(native).all() or not np.allclose(native[:6],raw[:6],rtol=0,atol=1e-6)
                or abs(float(native[6])-(1-float(raw[6])))>1e-6):raise ValueError('Actual independently measured native proprio required')
            observed={**payload,'prompt':payload['instruction'],'step':payload['inference_index'],'proprio':native.tolist()}
            inputs=prepare_sensor_tensors(observed,vision.image_processor,embeddings,target_domain=True,previous_images=previous)
            state=append_scene_conditioning(inputs['states'],raw,contract['stats'])
            seed=int.from_bytes(hashlib.sha256(json.dumps([payload['seed'],payload['instruction'],payload['inference_index']]).encode()).digest()[:8],'little')&((1<<63)-1)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode():
                torch.manual_seed(seed)
                image=vision(inputs['images'][0].to('cuda',dtype=torch.bfloat16)).reshape(1,-1,vision.hidden_size)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    trajectory=policy.predict_action(lang_tokens=inputs['lang_tokens'].to('cuda',dtype=torch.bfloat16),
                        lang_attn_mask=inputs['lang_attn_mask'].to('cuda'),img_tokens=image,state_tokens=state.to('cuda'),
                        action_mask=inputs['state_elem_mask'].unsqueeze(1).to('cuda',dtype=torch.bfloat16),ctrl_freqs=inputs['ctrl_freqs'].to('cuda'))[0]
            values=torch.cat((trajectory[:,:6],1-trajectory[:,10:11].clamp(0,1)),dim=1).float().cpu().numpy()
            if values.shape!=(10,7) or not np.isfinite(values).all():raise ValueError('Malformed neural output')
            return jsonify(actions=values.tolist(),action_kind=metadata['action_kind'],action_source='native_neural_model',
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        except (ValueError,KeyError) as error:return jsonify(error=str(error)),422
    app.run(host='127.0.0.1',port=a.port,threaded=False)


if __name__=='__main__':main()
