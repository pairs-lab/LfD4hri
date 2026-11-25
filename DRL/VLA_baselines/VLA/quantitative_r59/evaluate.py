"""Use the unchanged R52 motor engine; suppress media and buffer log I/O only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import requests
from VLA.common.artifact_io import atomic_json
from .common import OUT, SCHEDULE, PORT, read


def main():
    import VLA.evaluate_task_chunks_r52 as engine
    # None is false in the original writer branch: no overview render/encode.
    original_writer = engine.imageio.get_writer
    original_snapshot = engine._save_observation_snapshot
    engine.imageio.get_writer = lambda *a, **k: None
    engine._save_observation_snapshot = lambda *a, **k: None
    raw_atomic = engine.atomic_json
    pending = {}
    def atomic(path, value):
        path = Path(path)
        if path.name == 'trajectory.json':
            pending[path] = value
            return
        if path.name == 'progress.json':
            for target, trace in pending.items(): raw_atomic(target,trace)
            pending.clear()
            value = {**value, 'R59_actual_no_video':True,
                'primary_success':'frozen semantic first-hit; PUT release not required',
                'terminal_release_supervisor':False, 'updated_UTC':datetime.now(timezone.utc).isoformat()}
        raw_atomic(path,value)
    engine.atomic_json = atomic
    Session = requests.Session
    latency = (OUT/'query_latency.jsonl').open('x')
    class TimedSession(Session):
        def request(self, method, url, **kwargs):
            started = time.monotonic()
            reply = super().request(method,url,**kwargs)
            if url.endswith('/act'):
                latency.write(json.dumps({'wall_time_s':time.monotonic()-started,
                    'UTC':datetime.now(timezone.utc).isoformat(), 'HTTP_status':reply.status_code,
                    'input_sha256':hashlib.sha256(json.dumps(kwargs['json'],sort_keys=True).encode()).hexdigest()})+'\n')
                latency.flush()
            return reply
    requests.Session = TimedSession
    try:
        from VLA.privileged_visual_r53.evaluate import main as evaluate
        evaluate(SimpleNamespace(family='xvla',port=PORT,output=OUT/'episodes',visual_mode='map3',
            execution_steps=30, seed_count=200, schedule=SCHEDULE, cohort=OUT/'cohort.json'))
    finally:
        for target, trace in pending.items(): raw_atomic(target,trace)
        engine.imageio.get_writer = original_writer
        engine._save_observation_snapshot = original_snapshot
        requests.Session = Session
        latency.close()


if __name__ == '__main__': main()
