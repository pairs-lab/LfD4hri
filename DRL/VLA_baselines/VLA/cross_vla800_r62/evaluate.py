"""Process-local file-I/O shim around unchanged, qualified native evaluators."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import requests
from .common import TASKS,read
from .media import DiscardWriter,discard_snapshot


def main(a):
    method=read(a.method)['method'];folder=a.output.parent
    if a.model in ('smolvla','pi0'):
        import VLA.family_transfer_r54.evaluate as engine
    else:
        import VLA.evaluate_task_chunks_r52 as engine
    raw_atomic=engine.atomic_json;pending={};Session=requests.Session
    original_writer=engine.imageio.get_writer;original_snapshot=engine._save_observation_snapshot
    engine.imageio.get_writer=DiscardWriter;engine._save_observation_snapshot=discard_snapshot
    journal=(folder/'NEURAL_QUERY_RECEIPTS.jsonl').open('x')
    def atomic(path,value):
        path=Path(path)
        if path.name=='trajectory.json':pending[path]=value;return
        if path.name=='progress.json':
            for target,trace in pending.items():raw_atomic(target,trace)
            pending.clear()
        if path.name in ('progress.json','plan.json'):
            value={**value,'R62_measured_one_cohort_800':True,'primary_PUT_requires_release':False,
                'video_files_saved':False,'original_overview_and_mask_call_cadence_preserved':True,
                'no_Oracle_or_geometric_action_repair':True,
                'updated_UTC':datetime.now(timezone.utc).isoformat()}
        raw_atomic(path,value)
    engine.atomic_json=atomic
    class Recorded(Session):
        def request(self,verb,url,**kw):
            start=time.monotonic();response=super().request(verb,url,**kw)
            response.raise_for_status()
            if url.endswith('/act'):
                reply=response.json();payload=kw['json']
                digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                if not reply.get('actual_neural_model') or reply.get('input_sha256')!=digest:
                    raise ValueError('Actual neural/model input receipt absent')
                native=np.asarray(reply['actions'],np.float32)
                shape=(5,7) if a.model in ('smolvla','pi0') else (8,7)
                if native.shape!=shape or not np.isfinite(native).all():raise ValueError('Bad native VLA actions')
                journal.write(json.dumps({'model':a.model,'input_sha256':digest,'actions_float32':native.tolist(),
                    'native_actions_float32_SHA256':hashlib.sha256(native.tobytes()).hexdigest(),
                    'native_views':reply.get('native_views',(reply.get('native_visual_receipt') or {}).get('actual_views',2)),
                    'GT_map_SHA256':reply.get('map_RGB_sha256',reply.get('privileged_map_RGB_sha256')),
                    'wall_latency_s':time.monotonic()-start,'UTC':datetime.now(timezone.utc).isoformat()})+'\n');journal.flush()
            return response
    requests.Session=Recorded
    args=SimpleNamespace(model=a.model,port=a.port,output=a.output,mode=method['mode'],
        prefix=method.get('prefix'),schedule=Path(method['schedule']) if method.get('schedule') else None,
        cohort=a.cohort)
    try:
        from VLA.autocycle_r55.evaluate import main as evaluate
        evaluate(args)
    finally:
        for target,trace in pending.items():raw_atomic(target,trace)
        requests.Session=Session;engine.imageio.get_writer=original_writer
        engine._save_observation_snapshot=original_snapshot;engine.atomic_json=raw_atomic;journal.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('smolvla','pi0','vla_adapter'),required=True)
    p.add_argument('--method',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
