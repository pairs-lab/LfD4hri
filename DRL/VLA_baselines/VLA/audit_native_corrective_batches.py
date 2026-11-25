"""Actual correction-only native batches; never full-fit or SR proof."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.native_corrective_dataset import NativeCorrectiveDataset,rdt_dataset
from VLA.scene64_conditioning import normalization


def audit_rdt(data,output):
    import torch
    from transformers import AutoImageProcessor
    from VLA.rdt.training_data import RDTBatchCollator,wrap_joints,validate_language_embeddings
    from VLA.rdt.scene67_inputs import append_scene_conditioning,EXTRA_SLOTS
    from VLA.rdt.server import prepare_sensor_tensors
    from VLA.privileged_native_trial import png
    root=Path(__file__).resolve().parents[1]
    processor=AutoImageProcessor.from_pretrained(str(root/'VLA/checkpoints/siglip-so400m-patch14-384'),local_files_only=True)
    language_path=root/'VLA/audits/development40_rdt_r14_20261013/qualified_embeddings.pt'
    parent=Path(data.admission['source']);parent_rows=json.loads((parent/'index.json').read_text())
    language,_=validate_language_embeddings(language_path,language_path.with_suffix('.manifest.json'),
        parent/'index.json',[r['task']['prompt'] for r in parent_rows])
    stats,contract=normalization(data.states);native=rdt_dataset(data,stats);collator=RDTBatchCollator(processor,language)
    reports=[]
    for ei,row in enumerate(data.rows):
        # Use the unchanged native loader with a complete within-episode draw
        # to GUARANTEE initial/middle/terminal coverage, then retain3 real rows.
        native.samples_per_episode=row['steps'];full=native[ei]
        ids=sorted({0,row['steps']//2,row['steps']-1});positions=[int(np.flatnonzero(full.step_ids==i)[0]) for i in ids]
        sample=replace(full,images=full.images[positions],states=full.states[positions],actions=full.actions[positions],
            state_mask=full.state_mask[positions],action_valid_lengths=full.action_valid_lengths[positions],
            step_ids=full.step_ids[positions],prompts=tuple(full.prompts[i] for i in positions))
        batch=collator([sample]);start=data.ends[ei-1] if ei else 0
        with np.load(data.source/row['file'],allow_pickle=False) as source_payload:
            actual_native_open=source_payload['gripper_open'].copy()
        for bi,frame in enumerate(ids):
            raw=data[start+frame];state=raw['pre_action_scene64'];proprio=np.r_[state[:6],actual_native_open[frame],np.zeros(9)]
            payload={'prompt':raw['original_prompt'],'proprio':proprio.tolist(),'step':frame,
                'front_rgb_png':png(raw['image']),'wrist_rgb_png':png(raw['wrist_image']),
                'previous_front_rgb_png':png(raw['previous_image']),
                'previous_wrist_rgb_png':png(raw['previous_wrist_image'])}
            inference=prepare_sensor_tensors(payload,processor,language,target_domain=True,previous_images={})
            conditioned=append_scene_conditioning(inference['states'],state,stats)
            for field in ('images','lang_tokens','lang_attn_mask','state_elem_mask','ctrl_freqs'):
                torch.testing.assert_close(batch[field][bi:bi+1],inference[field],rtol=0,atol=0)
            torch.testing.assert_close(batch['states'][bi:bi+1],conditioned,rtol=0,atol=0)
            payload_data=data.episode(ei);future=np.minimum(frame+np.arange(10),row['steps']-1)
            np.testing.assert_array_equal(batch['actions'][bi,:,:6].numpy(),wrap_joints(payload_data['next_joint_targets'][future]))
            np.testing.assert_array_equal(batch['actions'][bi,:,10].numpy(),1-raw_native_closure(payload_data)[future])
            valid=[0,1,2,3,4,5,10]
            np.testing.assert_array_equal(np.flatnonzero(batch['state_elem_mask'][bi].numpy()),valid)
            if torch.any(batch['actions'][bi,:,list(EXTRA_SLOTS)]!=0):raise ValueError('Conditioning became motor labels')
        reports.append({'file':row['file'],'actual_native_batch':len(ids),'step_ids':ids,'current_and_previous_camera_parity':True})
    return {'passed':True,'model':'rdt','samples':sum(r['actual_native_batch'] for r in reports),'actual_batches':len(reports),
        'rows':reports,'native_action_indices':[0,1,2,3,4,5,10],'real_previous_camera_handoff':True,
        'normalization_scope':'correction_only_diagnostic_NOT_full_TRAIN_normalization','normalization_contract':contract}


def raw_native_closure(payload):
    from VLA.common.native_action_fields import native_gripper_closure
    return np.asarray(native_gripper_closure(payload),np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('rdt','pi0','pi0_fast'),required=True)
    for name in ('admission','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve actual failed batch gates')
    a.output.mkdir(parents=True);data=NativeCorrectiveDataset(a.admission)
    if a.model=='rdt':result=audit_rdt(data,a.output)
    else:
        from VLA.pi0.scene64_training import native_statistics
        from VLA.pi0.scene32_inputs import statistics
        from VLA.pi0.scene32_training import audit
        stats=native_statistics(data);stats['state'],contract=statistics(data.states)
        result=audit(data,a.model,stats,a.output)
        result['normalization_scope']='correction_only_diagnostic_NOT_full_TRAIN_normalization'
    result.update(scope='actual_native_corrective_batches_NOT_long_fit_or_model_SR',correction_episodes=38,
        correction_frames=len(data),long_fit_qualified=False,neural_weights_loaded=False,model_SR_established=False,
        admission_sha256=sha256_file(a.admission),implementation_sha256=sha256_file(Path(__file__)),
        dataset_implementation_sha256=sha256_file(Path(__file__).with_name('native_corrective_dataset.py')))
    atomic_json(a.output/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','normalization_contract','normalization')}))


if __name__=='__main__':main()
