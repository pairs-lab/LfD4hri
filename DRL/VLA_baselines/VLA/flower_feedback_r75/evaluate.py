"""Scoped reuse of R70's exact plant, observations, actions and evaluation.

The only behavioral substitution is the causal queue scheduler. Original source
files are unchanged; injections are local to this new Python process and restored
in finally. Original native query receipts validate every neural action chunk.
"""
import argparse
import json
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70 import evaluate as original
from VLA.flower_r70.common import freeze_test_code
from VLA.flower_feedback_r75.scheduler import FeedbackQueue


CONFIG = {'native_maximum_prefix':10, 'near_unattached_prefix':2, 'near_radius_m':.12,
          'contact_attachment_event_invalidates_unexecuted_suffix':True,
          'new_action_synthesis_or_modification':False,
          'physics_success_cameras_observation_tensors_unchanged':True,
          'new_weights_or_finetune_required_for_this_scheduler':False}


def binding():
    result=freeze_test_code()
    result.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    return result


def main(a):
    if a.prefix != 10: raise ValueError('Ablation reference native prefix is fixed at ten')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    journal_path=a.output.with_name(a.output.name+'_FEEDBACK_EVENTS.jsonl')
    saved=(original.deque, original.model_input, original.freeze_test_code)
    latest=[None]; queue_id=[0]
    with journal_path.open('x') as stream:
        def log(event):
            stream.write(json.dumps(event)+'\n');stream.flush()
        def queue_factory():
            queue_id[0]+=1;number=queue_id[0]
            q=FeedbackQueue(enabled=not a.static_control,
                log=lambda event:log({'episode_queue_id':number,**event}))
            latest[0]=q;return q
        def observe(env, obs):
            state=saved[1](env,obs)
            if latest[0] is not None:latest[0].observe(state['scene64'],obs.step)
            return state
        original.deque, original.model_input, original.freeze_test_code = queue_factory, observe, binding
        try:
            original.main(a)
            atomic_json(a.output/'FEEDBACK_SCHEDULER.json',{'configuration':CONFIG,
                'static_control':a.static_control,'event_journal':str(journal_path),
                'scoped_R70_reuse':True,'binding':binding()})
        finally:
            original.deque, original.model_input, original.freeze_test_code=saved


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=5950)
    p.add_argument('--prefix',type=int,default=10);p.add_argument('--static-control',action='store_true')
    main(p.parse_args())
