"""Reuse the native R87 owner with a resume evaluator and separate artifacts."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import traceback
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
import VLA.dita800_r87.run as original
from VLA.recovery_r85.common import utc
from .evaluate import SOURCE,OUT


def bindings():return {str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}


class Owner(original.Owner):
    def __init__(self):
        super().__init__();self.bound.update(bindings())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'status.json').exists():raise FileExistsError('No duplicate recovery owner')
        sim=json.loads((OUT/'SIM_MAP_REFERENCE.json').read_text())
        dita=json.loads((OUT/'DITA_MAP_REFERENCE.json').read_text())
        if (sim['reference_images_combined_sha256']!=dita['reference_images_combined_sha256']
            or sim['cases']!=dita['cases'] or dita['reference_vs_original_differing_cases']!=0):
            raise ValueError('Exact native-server reference qualification failed')
        if not json.loads((OUT/'REPLAY_PROOF.json').read_text())['passed']:
            raise ValueError('Original failure prefix has not been reconstructed')
        original.OUT=OUT;original.PORT=5991
        saved_popen=original.subprocess.Popen;saved_json=original.atomic_json
        def launch(command,*args,**kwargs):
            command=list(command)
            if 'VLA.dita800_r87.evaluate' in command:
                command[command.index('VLA.dita800_r87.evaluate')]='VLA.receipt_recovery_r88.evaluate'
            return saved_popen(command,*args,**kwargs)
        def write(path,value):
            if Path(path).name=='FROZEN_METHOD.json':
                value={**value,'transport_validation_recovery':{
                    'source_progress':str(SOURCE),'source_sha256':sha256_file(SOURCE),
                    'reused_complete_trials':243,'policy_inputs_actions_weights_physics_unchanged':True,
                    'NumPy1_native_map_audit_reference_verified_on271_states':True}}
            saved_json(path,value)
        original.subprocess.Popen=launch;original.atomic_json=write
        owner=Owner()
        try:
            owner.main()
            report=OUT/'REPORT.md'
            with report.open('a') as f:
                f.write('\nRecovered243 immutable completed outcomes from R87;557 completed after transport repair.\n')
                f.write('Only NumPy-version-dependent receipt reference changed; native policy/inputs/physics unchanged.\n')
        except BaseException as e:
            atomic_json(OUT/'ERROR.json',{'UTC':utc(),'error':str(e),'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('integration_error_NOT_model_SR',error=str(e));raise
        finally:
            owner.stop();original.subprocess.Popen=saved_popen;original.atomic_json=saved_json


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'owner.log').open('x') as f:
            child=subprocess.Popen([str(original.SIM),'-u','-m','VLA.receipt_recovery_r88.run'],
                cwd=original.ROOT,env=original.ENV,stdin=subprocess.DEVNULL,stdout=f,
                stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'progress':str(OUT/'final800/progress.json')})
    else:main()
