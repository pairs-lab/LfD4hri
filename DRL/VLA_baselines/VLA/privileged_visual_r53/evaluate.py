"""Unchanged R45/R52 motor rollout, plus checks of each real neural3-view RPC."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

from .scene_map import render,task_from_instruction,contract


def main(args):
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipts=args.output.with_suffix('.neural_receipts.jsonl').open('x')
    Session=requests.Session
    mode=args.visual_mode
    class AuditedSession(Session):
        def request(self,method,url,**kwargs):
            response=super().request(method,url,**kwargs)
            response.raise_for_status()
            if url.endswith('/health'):
                metadata=response.json()
                if mode=='map3' and metadata.get('privileged_visual_observation')!=contract():
                    raise ValueError('Wrong visual observation model server')
                if mode=='native2' and metadata.get('privileged_visual_observation') is not None:
                    raise ValueError('Expected preserved two-camera baseline')
            if url.endswith('/act'):
                reply,payload=response.json(),kwargs['json']
                if mode=='map3':
                    gt=render(np.asarray(payload['scene64'],np.float32),task_from_instruction(payload['instruction']))
                    digest=hashlib.sha256(gt.tobytes()).hexdigest()
                    if reply.get('privileged_map_RGB_sha256')!=digest or reply.get('native_visual_receipt',{}).get('actual_views')!=3:
                        raise ValueError('Actual third image missing or different from identical causal TRAIN transform')
                receipts.write(json.dumps({'input_sha256':reply['input_sha256'],
                    'privileged_map_RGB_sha256':reply.get('privileged_map_RGB_sha256'),
                    'actual_native_visual_receipt':reply.get('native_visual_receipt'),
                    'action_sha256_float32':hashlib.sha256(np.asarray(reply['actions'],np.float32).tobytes()).hexdigest(),
                    'mode':mode})+'\n');receipts.flush()
            return response
    requests.Session=AuditedSession
    try:
        if args.cohort:
            import VLA.evaluate_task_chunks_r52 as runner
            from VLA.evaluate_frozen_cohort_r52 import main as evaluate
        else:
            import VLA.evaluate_native_chunks_r45 as runner
            evaluate=runner.main
        original=runner.atomic_json
        original_interaction=runner._interaction_audit
        def interaction(env):
            result=original_interaction(env)
            result.update(privileged_policy_input=True,privileged_numeric_fields=67,
                R53_visual_mode=mode)
            return result
        runner._interaction_audit=interaction
        def atomic(path,value):
            if Path(path).name in ('plan.json','progress.json'):
                value={**value,'R53_visual_mode':mode,
                    'privileged_visual_observation':contract() if mode=='map3' else None,
                    'actual_neural_query_receipts':str(args.output.with_suffix('.neural_receipts.jsonl').resolve()),
                    'native_policy_only_action_source_existing_shared_safety_guard_unchanged':True}
            original(path,value)
        runner.atomic_json=atomic
        evaluate(args)
    finally:
        receipts.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('xvla','vla_adapter'),required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--visual-mode',choices=('map3','native2'),default='map3')
    parser.add_argument('--execution-steps',type=int)
    parser.add_argument('--seed-count',type=int,default=2)
    parser.add_argument('--schedule',type=Path)
    parser.add_argument('--cohort',type=Path)
    a=parser.parse_args()
    if bool(a.cohort)!=bool(a.schedule):parser.error('Explicit schedule and cohort must be supplied together')
    main(a)
