"""Keep native evaluators unchanged; correctly label selection vs final roles."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .common import read


def main(a):
    role=read(a.cohort)['role'] if a.cohort else 'development_prefix_pilot'
    if role not in ('development_prefix_pilot','development_selection','development_confirmation','untouched_final'):
        raise ValueError('Explicit cohort role required')
    if a.model in ('xvla','vla_adapter'):
        if a.cohort:
            import VLA.evaluate_task_chunks_r52 as runner
        else:
            import VLA.evaluate_native_chunks_r45 as runner
        from VLA.privileged_visual_r53.evaluate import main as evaluate
        args=SimpleNamespace(family=a.model,port=a.port,output=a.output,visual_mode=a.mode,
            execution_steps=a.prefix,seed_count=2,schedule=a.schedule,cohort=a.cohort)
    else:
        import VLA.family_transfer_r54.evaluate as runner
        evaluate=runner.main
        args=SimpleNamespace(model=a.model,port=a.port,output=a.output,prefix=a.prefix,cohort=a.cohort)
    original=runner.atomic_json
    def atomic(path,value):
        if Path(path).name in ('progress.json','plan.json'):
            value={**value,'R55_cohort_role':role,'used_for_DEVELOPMENT_selection':role!='untouched_final',
                'fixed_cohort_never_used_for_optimization':role=='untouched_final',
                'untouched_final_outcome_NEVER_input_to_subsequent_training':role=='untouched_final'}
        original(path,value)
    runner.atomic_json=atomic
    evaluate(args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('xvla','vla_adapter','smolvla','pi0'),required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',type=int);p.add_argument('--cohort',type=Path);p.add_argument('--schedule',type=Path)
    p.add_argument('--mode',choices=('map3','native2','baseline'),default='map3');main(p.parse_args())
