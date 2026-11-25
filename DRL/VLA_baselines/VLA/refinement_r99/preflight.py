"""CPU tests, transport and plant replay before GPU ownership."""
import subprocess
from .common import ROOT,OUT,CPU,PY,SIM,ENV,read,verify,bindings,atomic_json,utc,FINAL,DEV,CONF
from VLA.autorefinement_r97.common import metric_guard


def main():
    OUT.mkdir(exist_ok=True);metric_guard()
    if not read(OUT/'TRAIN_DATA_AUDIT.json')['passed']:
        raise ValueError('Run actual TRAIN data/RGB audit first')
    if not (OUT/'VISUAL_REVIEW.md').exists():raise ValueError('Main-agent actual image review required')
    env={**ENV,'CUDA_VISIBLE_DEVICES':''}
    modules=['VLA.refinement_r99.tests','VLA.postfit800_r98.tests',
        'VLA.autorefinement_r97.tests','VLA.recovery_quality_r90.tests',
        'VLA.dita_terminal_recovery_r95.tests']
    result=subprocess.run([str(CPU),'-m','unittest',*modules],cwd=ROOT,env=env,
        capture_output=True,text=True,timeout=180)
    (OUT/'CPU_TESTS.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise ValueError('R99 regression/fault-injection tests failed')
    receipts=[]
    for label,python in (('native',PY),('evaluator',SIM)):
        path=OUT/f'TRANSPORT_{label}.json'
        subprocess.run([str(python),'-m','VLA.recovery_quality_r90.transport','--family','vlanext_s',
            '--original','--output',str(path)],cwd=ROOT,env=env,check=True,timeout=180)
        receipts.append(read(path))
    if receipts[0]['case_hashes']!=receipts[1]['case_hashes'] or receipts[0]['TRAIN_cases']!=228:
        raise ValueError('Actual native/evaluator228 TRAIN RGB map cases differ')
    subprocess.run([str(SIM),'-m','VLA.postfit800_r98.replay','--output',str(OUT/'ACTUAL_PLANT_REPLAY.json')],
        cwd=ROOT,env=env,check=True,timeout=300)
    atomic_json(OUT/'REQUEST.json',{'UTC':utc(),'user_authorized_refine_and_new_actual800':True,
        'final_seeds_locked_before_fit':FINAL,'development_seeds':DEV,'confirmation_seeds':CONF,
        'R98_outcomes_used_only_as_prior_development_evidence':True,
        'original_thresholds_and_physics_preserved':True,'no_unrelated_process_cancelled':True})
    atomic_json(OUT/'QUALIFICATION.json',{'passed':True,'UTC':utc(),'code':bindings(),
        'CPU_unit_and_fault_injection_modules':modules,'TRAIN_data_audit':str(OUT/'TRAIN_DATA_AUDIT.json'),
        'native_evaluator_map_byte_cases':228,'actual_plant_replay':read(OUT/'ACTUAL_PLANT_REPLAY.json'),
        'fresh_CUDA_gradient_gate_reload_DEV_CONF_and_actual800_still_required':True})
    print({'passed':True,'status':str(OUT/'status.json')},flush=True)


if __name__=='__main__':main()
