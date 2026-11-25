"""Native RDT loss/Adam with full67 feedback and fresh native corrections."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from VLA.common.provenance import sha256_file
from VLA.rdt.scene67_native import MixedDataset,SCHEMA,NEW_SLOTS,load


def main():
    import VLA.rdt.training as trainer
    import VLA.rdt.checkpoint_loading as loader
    p=argparse.ArgumentParser(add_help=False)
    for name in ('projection','recovery-admission','input-gate'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scene67-qualification',type=Path);extra,remainder=p.parse_known_args()
    old=sys.argv;sys.argv=[old[0],*remainder]
    try:a=trainer.parse_args()
    finally:sys.argv=old
    if a.mode!='train' or a.max_episodes is not None:raise ValueError('Full immutable source plus qualified corrections required')
    inputs=json.loads(extra.input_gate.read_text())
    if (not inputs.get('passed') or inputs.get('samples')!=128 or inputs.get('schema')!=SCHEMA
        or inputs['projection_audit_sha256']!=sha256_file(extra.projection/'audit.json')
        or inputs['recovery_admission_sha256']!=sha256_file(extra.recovery_admission)):
        raise ValueError('Actual own mixed native128 batch gate required')
    for path,digest in inputs['bound_files'].items():
        if sha256_file(path)!=digest:raise ValueError('Own actual mixed input proof changed')
    if a.steps>2:
        if extra.scene67_qualification is None:raise ValueError('Own actual mixed67 CUDA/save/reload8 qualification required')
        q=json.loads(extra.scene67_qualification.read_text())
        if not q.get('passed') or q.get('schema')!=SCHEMA or q.get('actual_neural_controls')!=8 or q.get('actual_optimizer_updates')!=2:
            raise ValueError('Own new-model numerical/control gate failed')
        for path,digest in q['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Own numerical/control qualification changed')
    torch.cuda.set_per_process_memory_fraction(8500*2**20/torch.cuda.get_device_properties(0).total_memory)
    datasets=[];native_loader=trainer.DataLoader;target=trainer._target_contract
    def dataset(args):
        d=MixedDataset(args.source,extra.projection,extra.recovery_admission,
            samples_per_episode=args.samples_per_episode,seed=args.seed,action_horizon=args.action_horizon)
        datasets.append(d);return d
    def dataloader(d,*args,**kwargs):
        if isinstance(d,MixedDataset):
            generator=kwargs['generator'];kwargs['shuffle']=False
            kwargs['sampler']=torch.utils.data.WeightedRandomSampler(d.weights(),len(d),replacement=True,generator=generator)
        return native_loader(d,*args,**kwargs)
    def contract(cfg,*,step=None):
        if len(datasets)!=1:raise ValueError('Actual mixed dataset must be instantiated exactly once')
        d=datasets[0]
        cfg.update(privileged_model_inputs=True,privileged_pose_consumed=True,
            policy_inputs=['front_rgb','wrist_rgb','native7_proprio','text','pre_action_simscene67'],
            scene67={'schema':SCHEMA,'stats':d.stats,'recovery':d.recovery_contract,
                'projection':str(extra.projection.resolve()),'projection_audit_sha256':sha256_file(extra.projection/'audit.json'),
                'new_learned_gain_slots':list(NEW_SLOTS),'native_motor_DOFs':[0,1,2,3,4,5,10],
                'normalization_contract':d.normalization_contract,
                'bound_files':{str(p.resolve()):sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('scene67_native.py'),
                    Path(__file__).with_name('scene67_inputs.py'),extra.recovery_admission,extra.input_gate)},
                'oracle_controller_at_deployment':False,'future_or_teacher_action_in_input':False})
        (a.output/'train_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        return {**target(cfg,step=step),'format_version':3,'model_family':'rdt_scene67',
            'privileged_model_inputs':True,'scene67':cfg['scene67']}
    trainer.parse_args=lambda:a;trainer._make_dataset=dataset;trainer.DataLoader=dataloader;trainer._target_contract=contract
    loader.load_checkpoint=load
    # Keep the source/native signal guard. The long-stage proof is produced
    # from THIS model's two-update save and exact deployed reload, not r25.
    return trainer.main()


if __name__=='__main__':raise SystemExit(main())
