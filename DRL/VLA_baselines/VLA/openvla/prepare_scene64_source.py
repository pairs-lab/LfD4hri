"""Immutable OpenVLA full64 pilot view; preserves parent NPZ bytes and splits."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.source_training_gate import require_training_source
from VLA.scene64_conditioning import PROMPT_SCHEMA
from VLA.scene64_dataset import Scene64NativeDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('projection','native-preflight','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();data=Scene64NativeDataset(a.projection,1)
    proof=json.loads(a.native_preflight.read_text())
    if (not proof.get('passed') or proof.get('pilot_only') is not False or proof.get('mode')!='language'
        or proof.get('formatted_tokenized_frames')!=len(data)
        or proof.get('projection_audit_sha256')!=sha256_file(a.projection/'audit.json')):
        raise ValueError('FULL actual native64 language/image transform preflight required')
    for path,digest in proof['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Native input preflight stale')
    require_training_source(data.source,model='smolvla',optimizer_updates=1)
    a.output.mkdir(parents=True,exist_ok=False);parts=[];refs=[]
    for ei,row in enumerate(data.rows):
        with np.load(data.source/row['file'],allow_pickle=False) as z:values=z['canonical_actions'].copy()
        values[:,6]=(1-values[:,6])/2;parts.append(values)
        refs.extend({'episode_index':ei,'frame_index':fi} for fi in range(len(values)))
    values=np.concatenate(parts);key=a.output.name
    stats={name:arr.tolist() for name,arr in [('mean',values.mean(0)),('std',values.std(0)),
        ('min',values.min(0)),('max',values.max(0)),('q01',np.quantile(values,.01,axis=0)),('q99',np.quantile(values,.99,axis=0))]}
    stats['mask']=[True]*6+[False]
    atomic_json(a.output/'dataset_statistics.json',{key:{'action':stats,'num_transitions':len(values),'num_trajectories':len(data.rows)}})
    atomic_json(a.output/'samples.json',refs)
    root=Path(__file__).resolve().parents[1]
    dependencies={'native_preflight':a.native_preflight,'causal_projection':a.projection/'audit.json',
        'train_stats':a.projection/'stats.json','prompt_builder':root/'scene64_conditioning.py',
        'dataset_implementation':root/'scene64_dataset.py','input_schema':root/'scene64_schema.py',
        'trainer_implementation':Path(__file__).with_name('train_scene64_prompt.py'),
        'native_original_trainer':Path(__file__).with_name('train_privileged_prompt.py'),
        'native_inference_adapter':Path(__file__).with_name('adapter.py'),
        'source_view_preparation':Path(__file__)}
    bound={k:{'path':str(v.resolve()),'sha256':sha256_file(v)} for k,v in dependencies.items()}
    pre=a.output/'pilot_preconditions.json'
    atomic_json(pre,{'passed':True,'model':'openvla','schema':PROMPT_SCHEMA,'max_optimizer_updates':1000,
        'scope':'bounded_pipeline_pilot','inherited_common_physics_split_and_original_RGB_evidence':True,
        'actual_native_transform_batch_CUDA_reload_controls_still_required':True,
        'stable_success_or_data_sufficiency_certified':False,'dependencies':bound})
    view=a.output/'qualified_source';view.mkdir()
    for row in data.rows:os.link(data.source/row['file'],view/row['file'])
    (view/'index.json').write_bytes((data.source/'index.json').read_bytes())
    manifest=copy.deepcopy(json.loads((data.source/'dataset_manifest.json').read_text()))
    inherited=[{**item,'role':'inherited_common_'+item['role']} for item in manifest['training_qualification']['proofs']]
    manifest.update(candidate_only=False,training_certified=False,qualification_parent=str(data.source),
        qualification_parent_index_sha256=data.audit['source_index_sha256'],qualification_parent_manifest_sha256=data.audit['source_manifest_sha256'],
        purpose='OpenVLA full64 native-language bounded pilot; no stable-success claim',
        training_qualification={'scope':'bounded_pipeline_pilot','allowed_models':['openvla'],'max_optimizer_updates':1000,
            'proofs':[*inherited,{'role':'openvla_full64_native_preconditions','path':str(pre.resolve()),'sha256':sha256_file(pre)}]})
    atomic_json(view/'dataset_manifest.json',manifest);require_training_source(view,model='openvla',optimizer_updates=1000)
    atomic_json(a.output/'conversion_audit.json',{'dataset_key':key,'uses_original_source_arrays_without_reencoding':True})
    atomic_json(a.output/'privileged_dataset.json',{'model':'openvla','schema':PROMPT_SCHEMA,'projection':str(a.projection.resolve()),
        'qualified_source':str(view.resolve()),'original_source':str(data.source),'original_index_sha256':data.audit['source_index_sha256'],
        'frames':len(data),'episodes':len(data.rows),'candidate_only':False,'max_optimizer_updates':1000,'dependencies':bound})


if __name__=='__main__':main()
