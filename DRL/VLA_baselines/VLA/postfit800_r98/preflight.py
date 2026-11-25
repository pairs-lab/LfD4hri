"""Qualify CPU contracts and real plant replay without touching active fit."""
from pathlib import Path
import subprocess

from .common import (ROOT,OUT,SOURCE,CP,CPU,PY,SIM,ENV,read,verify,bindings,
    identity,alive,atomic_json,utc)
from VLA.benchmark_guardian_r89.common import live_children


def main():
    OUT.mkdir(exist_ok=True)
    env={**ENV,'CUDA_VISIBLE_DEVICES':''}
    result=subprocess.run([str(CPU),'-m','unittest','VLA.postfit800_r98.tests',
        'VLA.autorefinement_r97.tests','VLA.recovery_quality_r90.tests',
        'VLA.dita_terminal_recovery_r95.tests'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    (OUT/'CPU_TESTS.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise ValueError('CPU regression/fault-injection preflight failed')
    receipts=[]
    for label,python,original in (('native',PY,True),('evaluator',SIM,False)):
        path=OUT/f'TRANSPORT_{label}.json'
        args=[str(python),'-m','VLA.recovery_quality_r90.transport','--family','vlanext_s','--output',str(path)]
        if original:args.append('--original')
        subprocess.run(args,cwd=ROOT,env=env,check=True,timeout=180)
        receipts.append(read(path))
    if receipts[0]['case_hashes']!=receipts[1]['case_hashes'] or receipts[0]['TRAIN_cases']!=228:
        raise ValueError('Native S/evaluator reference bytes differ')
    subprocess.run([str(SIM),'-m','VLA.postfit800_r98.replay','--output',str(OUT/'ACTUAL_PLANT_REPLAY.json')],
        cwd=ROOT,env=env,check=True,timeout=300)
    state=read(SOURCE/'LIVE.json');parent=identity(state['pid']);optimizer=identity(state['child_pid'])
    if (not alive(state['pid']) or not alive(state['child_pid']) or state['stage']!='recovery_fit9216'
        or parent['command'][-1]!='VLA.dita_terminal_recovery_r95.others'
        or optimizer['ppid']!=parent['pid'] or 'fit' not in optimizer['command']
        or str(CP) not in optimizer['command'] or live_children(parent['pid'])!={optimizer['pid']}):
        raise ValueError('Active fit/scheduler identity changed; do not pause anything')
    atomic_json(OUT/'REQUEST.json',{'UTC':utc(),'legacy_scheduler':parent,'active_optimizer':optimizer,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'latest_user_request':'automatically finish fit, preflight testing then direct actual800,200/action',
        'optimizer_must_continue_unsignaled':True,'direct800_does_not_claim_CONF_pass':True})
    atomic_json(OUT/'QUALIFICATION.json',{'passed':True,'code':bindings(),
        'CPU_regression_and_fault_injection_passed':True,'native_and_evaluator_numpy':[r['numpy'] for r in receipts],
        'TRAIN_map_byte_cases':228,'actual_plant_replay':read(OUT/'ACTUAL_PLANT_REPLAY.json'),
        'GPU_model_or_optimizer_loaded_by_this_preflight':False,
        'original_optimizer_and_training_code_unchanged':True,
        'native_reload_and_actual_integration4_still_required_before800':True})
    print({'passed':True,'output':str(OUT),'actual_training_progress_at_preflight':read(CP/'status.json')['updates']})


if __name__=='__main__':main()
