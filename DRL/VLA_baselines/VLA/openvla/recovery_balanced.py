"""Native OpenVLA: exact warm start, measured recovery, balanced motions.

All high-level outputs remain the original action-token neural decoder. This
separate experiment never edits a live recipe/source/checkpoint. Offline
sampling strata use TRAIN labels only, never become runtime observations.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_dataset import Scene64NativeDataset
from VLA.native_corrective_dataset import NativeCorrectiveDataset
from VLA.scene64_conditioning import PROMPT_SCHEMA

SCHEMA = 'OpenVLA_full64_native_QLoRA_recovery20_motion_balanced_stationarycap5_v2'


def stratum(command, observed):
    if command[6]<-.5 and observed[30]:return 'release_wrong_held_body'
    if observed[17]:
        if command[6]<-.5:return 'release_current_hold'
        return 'held_transport' if np.linalg.norm(command[:3]) > .001 else 'held_wait'
    if command[6] > .5:
        return 'close_without_current_verified_hold'
    if command[6]<-.5 and (observed[6]>.05 or observed[10]>.5):
        near=np.linalg.norm(observed[11:13])<.08 and -observed[13]-.5*observed[26]<.12
        return 'near_target_reopen' if near else 'reopen_from_hover'
    if np.linalg.norm(command[:2]) > .005 or abs(command[5]) > .005:
        return 'open_xy_or_yaw_alignment'
    return 'open_vertical_reach' if np.linalg.norm(command[:3]) > .001 else 'open_wait'


def cap_stationary(weights,labels,maximum=.05):
    """Preserve task/target mass; do not upweight rare no-op waits to1/3.

    Meaningful OPEN recovery/release labels are separate strata, never capped
    as waits. No labels are deleted or changed. Categories stay offline.
    """
    weights=np.asarray(weights,np.float64).copy();groups=defaultdict(list)
    for i,(task,target,stage) in enumerate(labels):groups[task,target].append(i)
    before=after=0.;rows=[]
    for (task,target),positions in groups.items():
        ids=np.asarray(positions);wait=np.asarray([labels[i][2] in ('open_wait','held_wait') for i in ids])
        total=float(weights[ids].sum());old=float(weights[ids[wait]].sum());before+=old
        new=min(old,maximum*total)
        if old>new:
            moving=total-old
            if moving<=0:raise ValueError('All-wait task/target cannot qualify as a motion training source')
            weights[ids[wait]]*=new/old;weights[ids[~wait]]*=(total-new)/moving
        actual=float(weights[ids[wait]].sum());after+=actual
        if not np.isclose(weights[ids].sum(),total) or actual>maximum*total+1e-12:
            raise ValueError('Stationary cap changed task/target mass')
        rows.append({'task':task,'target':target,'before_wait_mass':old,'after_wait_mass':actual,'task_target_mass':total})
    return weights,{'maximum_stationary_fraction_per_task_target':maximum,
        'before_wait_mass':before,'after_wait_mass':after,'meaningful_reopen_release_NOT_capped':True,'rows':rows}


def sampling_weights(original, correction):
    labels, counts, strata = [], Counter(), defaultdict(set)
    for row, observed in zip(original.rows, np.split(original.states, original.ends[:-1]), strict=True):
        with np.load(original.source/row['file'], allow_pickle=False) as z:
            commands = z['canonical_actions']
        task, target = row['task']['name'], row['task']['pick_name']
        for command, state in zip(commands, observed, strict=True):
            group = task, target, stratum(command, state)
            labels.append(group)
            counts[group] += 1
            strata[task, target].add(group[-1])
    target_counts = Counter(task for task, _ in strata)
    correction_counts = Counter()
    for row in correction.rows:
        correction_counts[row['task']['name']] += row['steps']
    weights = [.8/(4*target_counts[task]*len(strata[task,target])*counts[task,target,stage]) for task,target,stage in labels]
    weights,stationary=cap_stationary(weights,labels)
    weights=weights.tolist()
    for row in correction.rows:
        weights.extend([.2/(4*correction_counts[row['task']['name']])]*row['steps'])
    weights = np.asarray(weights, np.float64)
    if not np.isclose(weights[:len(original)].sum(), .8) or not np.isclose(weights[len(original):].sum(), .2):
        raise ValueError('Exact80/20 original/corrective probability mass differs')
    return weights, {'source_strata': [{'task':t,'target':o,'stratum':s,'frames':n} for (t,o,s),n in sorted(counts.items())],
        'original_mass':float(weights[:len(original)].sum()),'corrective_mass':float(weights[len(original):].sum()),
        'schema':SCHEMA,'runtime_teacher_phase_injected':False,'stationary_mass_guard':stationary}


class Mixed:
    def __init__(self, projection, admission):
        self.original = Scene64NativeDataset(projection, 1)
        self.corrective = NativeCorrectiveDataset(admission, 1)
        self.weights, self.sampling = sampling_weights(self.original, self.corrective)
    def __len__(self):
        return len(self.original)+len(self.corrective)
    def __getitem__(self, index):
        return self.original[index] if index < len(self.original) else self.corrective[index-len(self.original)]


def dataset_for(root, processor, mixed):
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from VLA.openvla.train_privileged_prompt import NativePromptDataset
    # Initialize only the unchanged native normalization/token/image members;
    # do not instantiate a second huge raw dataset while shadowing it.
    dataset = NativePromptDataset.__new__(NativePromptDataset)
    dataset.root, dataset.raw = root, mixed
    dataset.processor, dataset.action_tokenizer = processor, ActionTokenizer(processor.tokenizer)
    dataset.prompt_builder_cls = PurePromptBuilder
    dataset.dataset_key = json.loads((root/'conversion_audit.json').read_text())['dataset_key']
    stats = json.loads((root/'dataset_statistics.json').read_text())[dataset.dataset_key]['action']
    dataset.low, dataset.high, dataset.mask = (np.asarray(stats[k], bool if k=='mask' else np.float32) for k in ('q01','q99','mask'))
    return dataset


def audit(a):
    from transformers import AutoProcessor
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    a.output.mkdir(parents=True, exist_ok=False)
    mixed = Mixed(a.projection, a.admission)
    processor = AutoProcessor.from_pretrained(str(a.warmstart), trust_remote_code=True, local_files_only=True)
    dataset = dataset_for(a.dataset, processor, mixed)
    indices = np.linspace(0, len(mixed.original)-1, 14, dtype=int).tolist()
    for ei, row in enumerate(mixed.corrective.rows):
        offset = len(mixed.original)+(mixed.corrective.ends[ei-1] if ei else 0)
        indices.extend(offset+fi for fi in (0,row['steps']//2,row['steps']-1))
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side='right')
    maximum = count = 0
    for begin in range(0, len(indices), a.batch_size):
        selected = indices[begin:begin+a.batch_size]
        items = [dataset[i] for i in selected]
        batch = collator(items)
        if int((batch['labels'] != -100).sum()) != 8*len(selected):
            raise ValueError('Unchanged native seven action tokens plus EOS required')
        for index, item in zip(selected, items, strict=True):
            raw = mixed[index]
            inference = processor.tokenizer('In: What action should the robot take to '+raw['prompt'].lower()+'?\nOut: ',truncation=False).input_ids
            if item['input_ids'][:-8].tolist() != inference:
                raise ValueError('EXACT training/inference token prefix differs')
            maximum = max(maximum,len(item['input_ids']))
            expected = raw['canonical_actions'][0].copy()
            expected[6] = (1-expected[6])/2
            np.testing.assert_array_equal(expected,raw['language_actions'][0])
            count += 1
    if count != 128:
        raise ValueError('All38 corrections at initial/middle/terminal plus14 original frames required')
    np.save(a.output/'sampling_weights.npy',mixed.weights,allow_pickle=False)
    from VLA.openvla.finetune import validate_checkpoint_manifest
    validate_checkpoint_manifest(a.warmstart)
    from VLA.openvla.privileged_loading import verify_adapter_values
    bound = {str(p.resolve()):sha256_file(p) for p in (Path(__file__),a.projection/'audit.json',a.admission,
        a.dataset/'privileged_dataset.json',a.warmstart/'checkpoint_manifest.json')}
    atomic_json(a.output/'audit.json',{'passed':True,'schema':SCHEMA,'samples':count,'batch_size':a.batch_size,
        'actual_native_batches':int(np.ceil(count/a.batch_size)),'source_frames':len(mixed.original),'corrective_frames':len(mixed.corrective),
        'frames':len(mixed),'sampling':mixed.sampling,'max_native_training_tokens':maximum,
        'exact_all_128_inference_token_prefixes':True,'warmstart':str(a.warmstart.resolve()),
        'weights_path':str((a.output/'sampling_weights.npy').resolve()),'weights_sha256':sha256_file(a.output/'sampling_weights.npy'),
        'bound_files':bound,'new_CUDA_or_neural_SR_claimed':False})


def train(a):
    import peft
    from safetensors.torch import load_file
    from VLA.openvla import finetune as native
    from VLA.openvla import train_privileged_prompt as privileged
    from VLA.openvla.privileged_loading import verify_adapter_values
    # The shared trainer defaults to the historical18-language schema.
    # Its lineage validator and saved config MUST describe the actual64
    # source, exactly as the original train_scene64_prompt wrapper does.
    privileged.PROMPT_SCHEMA = PROMPT_SCHEMA
    gate = json.loads(a.gate.read_text())
    if not gate.get('passed') or gate['schema'] != SCHEMA or gate['batch_size'] != a.batch_size:
        raise ValueError('Own actual native batch/motion/recovery gate required')
    for path,digest in gate['bound_files'].items():
        if sha256_file(path) != digest:
            raise ValueError('Own warmstart/input qualification changed')
    if sha256_file(gate['weights_path']) != gate['weights_sha256']:
        raise ValueError('Actual balanced sampling weights changed')
    if a.updates > 2:
        q = json.loads(a.qualification.read_text()) if a.qualification else {}
        if not q.get('passed') or q.get('schema') != SCHEMA or q.get('batch_size') != a.batch_size or q.get('warmstart_manifest_sha256') != sha256_file(a.warmstart/'checkpoint_manifest.json'):
            raise ValueError('Own CUDA2/saved precision/exact reload8 qualification required')
        for path,digest in q['bound_files'].items():
            if sha256_file(path) != digest:
                raise ValueError('Own actual balanced-fit proof changed')
    mixed = Mixed(a.projection,a.admission)
    weights = torch.from_numpy(np.load(gate['weights_path'],allow_pickle=False))
    np.testing.assert_array_equal(weights.numpy(),mixed.weights)
    get_peft = peft.get_peft_model
    def warm(base,*args,**kwargs):
        model = get_peft(base,*args,**kwargs)
        peft.set_peft_model_state_dict(model,load_file(str(a.warmstart/'adapter_model.safetensors'),device='cpu'))
        evidence = verify_adapter_values(model,a.warmstart)
        if not evidence['passed']:
            raise ValueError('Exact pretrained-adapter warm start failed')
        atomic_json(a.output/'exact_warmstart.json',{**evidence,'parent':str(a.warmstart.resolve()),
            'optimizer_is_new_recipe_NOT_claimed_exact_resume':True})
        return model
    class WeightedEpoch(native.EpochRandomSampler):
        def __iter__(self):
            generator=torch.Generator().manual_seed((self.seed+self.epoch)&((1<<63)-1))
            ids=torch.multinomial(weights,len(self.data_source),replacement=True,generator=generator).tolist()
            yield from ids[self.start_index:]
    old_main=native.main
    def launch_native():
        # privileged.main installed its native source/argument hooks already.
        old_parse,old_lineage=native.parse_args,native.validate_lineage
        def arguments():
            parsed=old_parse();parsed.batch_size=a.batch_size;return parsed
        def lineage(parsed):
            info=old_lineage(parsed)
            return {**info,'frames':len(mixed),'episodes':len(mixed.original.rows)+len(mixed.corrective.rows),
                'original_source_frames':len(mixed.original),'actual_corrective_frames':len(mixed.corrective),
                'recovery_balanced_schema':SCHEMA,'recovery_input_gate_sha256':sha256_file(a.gate),
                'warmstart_manifest_sha256':sha256_file(a.warmstart/'checkpoint_manifest.json'),
                'warmstart_prior_family_optimizer_updates':1000,'sampling_weights_sha256':gate['weights_sha256']}
        native.parse_args,native.validate_lineage=arguments,lineage
        native.EpochRandomSampler=WeightedEpoch
        native.PyBulletOpenVLADataset=lambda root,processor,*args,**kwargs:dataset_for(root,processor,mixed)
        native.RESUME_INVARIANT_KEYS += ('recovery_balanced_schema','recovery_input_gate_sha256','warmstart_manifest_sha256','sampling_weights_sha256')
        return old_main()
    # Keep the stock loss, optimizer, atomic checkpoint and exact own2->1000
    # resume. Only new-recipe initialization/data sampling are replaced.
    forwarded=[sys.argv[0],'--dataset',str(a.dataset),'--output',str(a.output),'--base-model',a.base_model,
        '--steps',str(a.updates),'--grad-accumulation','4','--learning-rate','0.0001','--seed','20261015','--save-freq','250']
    if a.updates>2:
        forwarded += ['--resume-from',str(a.output/'checkpoints/000002'),'--smoke-qualification',str(a.qualification)]
    saved_argv=sys.argv
    try:
        sys.argv=forwarded
        with patch.object(peft,'get_peft_model',warm),patch.object(native,'main',launch_native):
            privileged.main()
    finally:
        sys.argv=saved_argv


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('audit','train'))
    for name in ('dataset','projection','admission','warmstart','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--base-model')
    parser.add_argument('--batch-size',type=int,choices=(1,2),default=2)
    parser.add_argument('--updates',type=int,choices=(2,1000),default=2)
    a=parser.parse_args()
    if a.mode=='audit':audit(a)
    else:train(a)
