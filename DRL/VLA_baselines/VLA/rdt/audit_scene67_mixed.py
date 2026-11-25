"""128 actual mixed native samples, byte-exact sensors and7-DOF labels."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoImageProcessor
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import native_gripper_closure
from VLA.rdt.scene67_native import MixedDataset,SCHEMA,gated_state
from VLA.rdt.training_data import RDTBatchCollator,wrap_joints,validate_language_embeddings
from VLA.rdt.server import prepare_sensor_tensors
from VLA.rdt.simstate_inputs import append_causal_conditioning
from VLA.privileged_native_trial import png


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','projection','recovery-admission','embeddings','vision','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve all actual gates')
    data=MixedDataset(a.source,a.projection,a.recovery_admission,samples_per_episode=4,seed=20261015,action_horizon=10)
    language,_=validate_language_embeddings(a.embeddings,a.embeddings.with_suffix('.manifest.json'),
        a.source/'index.json',data.prompts)
    processor=AutoImageProcessor.from_pretrained(str(a.vision),local_files_only=True)
    collator=RDTBatchCollator(processor,language)
    originals=[]
    for task in ('touch','pick','move','pick_place'):
        originals.extend([i for i,r in enumerate(data.base.index) if r['task']['name']==task][:3])
    corrections=[]
    for task in ('touch','pick','move','pick_place'):
        corrections.extend([data.original_length+i for i,r in enumerate(data.corrective.index) if r['task']['name']==task][:5])
    reports=[]
    for ei in originals+corrections:
        sample=data[ei];batch=collator([sample]);new=ei>=data.original_length
        row=data.index[ei];source=data.correction_data.source if new else a.source
        with np.load(source/row['file'],allow_pickle=False) as z:
            raw=z['audit_pre_action_PRIVILEGED_state64'][sample.step_ids] if new else data.projected[row['file']][sample.step_ids]
            targets=z['next_joint_targets'];closure=native_gripper_closure(z);observed_open=z['gripper_open'][sample.step_ids].copy()
        for bi,frame in enumerate(sample.step_ids):
            images=sample.images[bi];proprio=np.r_[raw[bi,:6],observed_open[bi],np.zeros(9)]
            payload={'prompt':sample.prompts[bi],'step':int(frame),'proprio':proprio.tolist(),
                'front_rgb_png':png(images[2]),'wrist_rgb_png':png(images[3]),
                'previous_front_rgb_png':png(images[0]),'previous_wrist_rgb_png':png(images[1])}
            actual=prepare_sensor_tensors(payload,processor,language,target_domain=True,previous_images={})
            for field in ('images','lang_tokens','lang_attn_mask','state_elem_mask','ctrl_freqs'):
                torch.testing.assert_close(batch[field][bi:bi+1],actual[field],rtol=0,atol=0)
            legacy_stats={k:data.stats[k][:18] for k in ('mean','std')}
            legacy=append_causal_conditioning(actual['states'],raw[bi,:18],legacy_stats)
            warm=gated_state(SimpleNamespace(scene67_gain=torch.zeros(49)),batch['states'][bi:bi+1])
            torch.testing.assert_close(warm,legacy,rtol=0,atol=0)
            future=np.minimum(int(frame)+np.arange(10),row['steps']-1)
            np.testing.assert_array_equal(batch['actions'][bi,:,:6].numpy(),wrap_joints(targets[future]))
            np.testing.assert_array_equal(batch['actions'][bi,:,10].numpy(),1-np.asarray(closure[future],np.float32))
            np.testing.assert_array_equal(np.flatnonzero(batch['state_elem_mask'][bi].numpy()),[0,1,2,3,4,5,10])
        reports.append({'file':row['file'],'corrective':new,'samples':len(sample.step_ids),'step_ids':sample.step_ids.tolist()})
    weights=np.asarray(data.weights())
    if abs(weights[:data.original_length].sum()-.8)>1e-8 or abs(weights[data.original_length:].sum()-.2)>1e-8:
        raise ValueError('Rehearsal/correction sampler mass differs')
    bound=[Path(__file__),Path(__file__).with_name('scene67_native.py'),Path(__file__).with_name('scene67_inputs.py'),
        Path(__file__).resolve().parents[1]/'native_corrective_dataset.py',a.projection/'audit.json',a.recovery_admission]
    atomic_json(a.output,{'passed':True,'schema':SCHEMA,'samples':sum(r['samples'] for r in reports),'actual_batches':len(reports),
        'rows':reports,'original_frames':data.audit['frames'],'correction_frames':data.recovery_contract['frames'],
        'projection_audit_sha256':sha256_file(a.projection/'audit.json'),'recovery_admission_sha256':sha256_file(a.recovery_admission),
        'initial_conditioning_exact_legacy18':True,'native_motor_DOFs':[0,1,2,3,4,5,10],
        'model_weights_loaded':False,'long_fit_qualified':False,'model_SR_established':False,
        'bound_files':{str(f.resolve()):sha256_file(f) for f in bound}})


if __name__=='__main__':main()
