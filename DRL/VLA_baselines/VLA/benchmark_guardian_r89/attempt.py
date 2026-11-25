"""One native recovery attempt; guardian owns bounded safe retry policy."""
import argparse
import fcntl
import json
from pathlib import Path
import traceback
from VLA.common.artifact_io import atomic_json
import VLA.dita800_r87.run as original
from .common import bindings,read,utc
from VLA.common.provenance import sha256_file


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.OUT=a.output;original.PORT=a.port
        saved_popen=original.subprocess.Popen;saved_json=original.atomic_json
        def launch(command,*args,**kwargs):
            command=list(command)
            if 'VLA.dita800_r87.evaluate' in command:
                command[command.index('VLA.dita800_r87.evaluate')]='VLA.benchmark_guardian_r89.evaluate'
                command+=['--resume-progress',str(a.resume_progress)]
            return saved_popen(command,*args,**kwargs)
        def write(path,value):
            if Path(path).name=='FROZEN_METHOD.json':
                value={**value,'recovery_only_no_policy_change':True,
                    'resume_snapshot':str(a.resume_progress),'resume_completed_trials':read(a.resume_progress)['completed_trials']}
            saved_json(path,value)
        original.subprocess.Popen=launch;original.atomic_json=write
        owner=original.Owner();owner.bound.update(bindings())
        # Both reused AST transformation and exact map-reference implementation
        # are part of this attempt's immutable code receipt, not just R89.
        from VLA.receipt_recovery_r88 import evaluate,map_reference
        for module in (evaluate,map_reference):
            path=Path(module.__file__);owner.bound[str(path)]=sha256_file(path)
        try:owner.main()
        except BaseException as e:
            atomic_json(a.output/'ERROR.json',{'UTC':utc(),'error':str(e),'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('integration_error_NOT_model_SR',error=str(e));raise
        finally:owner.stop();original.subprocess.Popen=saved_popen;original.atomic_json=saved_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-progress',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    main(p.parse_args())
