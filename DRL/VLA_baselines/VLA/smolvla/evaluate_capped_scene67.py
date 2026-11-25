"""Unchanged fitted67 evaluator with a per-process CUDA memory budget.

Only adds resource accounting and an applied-step journal. The checkpoint,
policy, private RNG, environment, actions and scoring remain the originals.
"""
import argparse
from pathlib import Path
import sys
import torch
from VLA.common.artifact_io import atomic_json


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu-memory-cap-mib',type=int,required=True)
    parser.add_argument('--resource-audit-root',type=Path,required=True)
    a,remaining=parser.parse_known_args()
    if not 2500<=a.gpu_memory_cap_mib<=6000:
        raise ValueError('Explicit bounded inference-only GPU allocation required')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(a.gpu_memory_cap_mib*2**20/total,device=0)
    from VLA import evaluate_scene67_smolvla as original
    policy_class=original.Scene67SmolVLA
    journal=a.resource_audit_root/'applied_step_resource_journal.json'
    class AccountedPolicy(policy_class):
        def __init__(self,env,*args,**kwargs):
            super().__init__(env,*args,**kwargs)
            step=env.step
            def accounted(action):
                result=step(action)
                atomic_json(journal,{'task':env.task.name,'applied_steps_current_task':int(result[0].step),
                    'terminal_current_task':bool(result[2]),'success':bool(result[3]['success']),
                    'safety_violation':result[3].get('safety_violation'),
                    'gpu_allocated_mib':torch.cuda.memory_allocated()/2**20,
                    'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'gpu_memory_cap_mib':a.gpu_memory_cap_mib,
                    'actions_model_and_scoring_changed':False})
                return result
            env.step=accounted
        def reset(self,task):
            super().reset(task)
            atomic_json(journal,{'task':task.name,'applied_steps_current_task':0,'terminal_current_task':False,
                'gpu_memory_cap_mib':a.gpu_memory_cap_mib,'actions_model_and_scoring_changed':False})
    original.Scene67SmolVLA=AccountedPolicy
    old=sys.argv
    try:
        sys.argv=[old[0],*remaining]
        original.main()
    finally:
        sys.argv=old
        atomic_json(a.resource_audit_root/'peak_memory.json',{'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'gpu_memory_cap_mib':a.gpu_memory_cap_mib,'inference_only':True,'model_or_action_changes':False})


if __name__=='__main__':main()
