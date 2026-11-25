"""Actual recovery trainer entry/lineage/sampler wiring, NO model/CUDA.

Only the heavy native optimizer entry is replaced with an inspection callback.
Dataset, tokenization, lineage validator, CLI forwarding and sampler are real.
This is not an optimizer, weight-reload or neural-control qualification.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def run(root, output):
    from transformers import AutoProcessor
    from VLA.openvla import finetune as native
    from VLA.openvla import recovery_balanced as recipe
    from VLA.scene64_conditioning import PROMPT_SCHEMA
    gate_path=root/'native128_batch2/audit.json'
    gate=json.loads(gate_path.read_text())
    warm=Path(gate['warmstart'])
    cfg=json.loads((warm/'train_config.json').read_text())
    contract=json.loads((Path(cfg['dataset'])/'privileged_dataset.json').read_text())
    admission=next(Path(p) for p in gate['bound_files'] if p.endswith('/native_view_admission.json'))
    args=SimpleNamespace(gate=gate_path,warmstart=warm,updates=2,batch_size=2,
        projection=Path(contract['projection']),admission=admission,dataset=Path(cfg['dataset']),
        output=root/'UNUSED_entry_inspection_not_a_fit',base_model=cfg['base_model'],qualification=None)
    reports={}
    def inspect():
        actual=native.parse_args()
        lineage=native.validate_lineage(actual)
        if actual.batch_size!=2 or lineage['privileged_prompt_schema']!=PROMPT_SCHEMA or lineage['frames']!=117736:
            raise ValueError('Actual native trainer CLI/schema/count wiring differs')
        processor=AutoProcessor.from_pretrained(str(warm),trust_remote_code=True,local_files_only=True)
        dataset=native.PyBulletOpenVLADataset(actual.dataset,processor,None,None,None)
        if len(dataset)!=lineage['frames']:
            raise ValueError('Actual forwarded native dataset differs')
        all_ids=list(native.EpochRandomSampler(dataset,seed=actual.seed,epoch=0,start_index=0))
        resumed=list(native.EpochRandomSampler(dataset,seed=actual.seed,epoch=0,start_index=16))
        if resumed!=all_ids[16:]:
            raise ValueError('Own2-update batch2/accumulation4 resume sample cursor differs')
        sample=dataset[all_ids[0]]
        if int((sample['labels']!=-100).sum())!=8:
            raise ValueError('Actual native supervised action suffix differs')
        reports.update(actual_batch_size=actual.batch_size,actual_raw_frames=len(dataset),
            actual_language_schema=lineage['privileged_prompt_schema'],sample_cursor_exact=True,
            actual_first_sample_index=all_ids[0],sample_within_native_dataset=True,
            required_resume_identity_keys_present=all(k in native.RESUME_INVARIANT_KEYS for k in (
                'recovery_balanced_schema','recovery_input_gate_sha256','warmstart_manifest_sha256','sampling_weights_sha256')))
        if not reports['required_resume_identity_keys_present']:
            raise ValueError('New sampler/recovery/warmstart omitted from exact resume identity')
        return 0
    with patch.object(native,'main',inspect):
        recipe.train(args)
    if args.output.exists():
        raise ValueError('Entry inspection must never create a fit or optimizer')
    atomic_json(output,{'passed':True,'scope':'actual_NATIVE_trainer_entry_lineage_dataset_sampler_preGPU_ONLY',
        **reports,'real_model_loaded':False,'optimizer_updates':0,'neural_robot_controls':0,
        'GPU_qualification_or_SR_established':False,'native_optimizer_entry_replaced_by_inspection_callback':True,
        'input_gate_sha256':sha256_file(gate_path),'recipe_sha256':sha256_file(Path(recipe.__file__)),
        'implementation_sha256':sha256_file(Path(__file__))})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve each actual entry audit')
    run(args.cycle.resolve(),args.output.resolve())
