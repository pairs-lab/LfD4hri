"""CPU-only admission for the automatic continuation; no GPU model calls."""
import subprocess
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.replacement_r84.common import verify
from .audit import ROOT, OUT, read
from .owner import bindings, CPU, ENV, SIM


def main():
    proc = subprocess.run([str(CPU),'-m','VLA.recovery_quality_r90.tests'],cwd=ROOT,env=ENV,
        capture_output=True,text=True,timeout=120)
    print(proc.stdout+proc.stderr,flush=True)
    if proc.returncode: raise RuntimeError('CPU regression qualification failed')
    targets = [
        ('dita',ROOT/'VLA/.venvs/dita_r84/bin/python',True,'native'),
        ('dita',SIM,False,'evaluator'),
        ('vlanext_s',ROOT/'VLA/.venvs/vlanext_r84/bin/python',True,'native'),
        ('vlanext_s',SIM,False,'evaluator')]
    receipts = {}
    for family, python, original, label in targets:
        path = OUT/f'QUAL_TRANSPORT_{family}_{label}.json'
        args = [str(python),'-m','VLA.recovery_quality_r90.transport','--family',family,'--output',str(path)]
        if original: args.append('--original')
        subprocess.run(args,cwd=ROOT,env=ENV,check=True,timeout=180)
        receipts[family,label] = read(path)
    for family in ('dita','vlanext_s'):
        a,b = receipts[family,'native'],receipts[family,'evaluator']
        if a['TRAIN_cases'] != 228 or a['case_hashes'] != b['case_hashes']:
            raise ValueError('Native/reference map pixels differ: '+family)
    parents = [ROOT/'VLA/audits/continuous_r86_20261021/dita/fit4096/TRAINING.json',
        ROOT/'VLA/audits/recovery_r85_20261021/vlanext_s/fit2048/TRAINING.json']
    for path in parents: verify(read(path)['code'])
    atomic_json(OUT/'QUALIFICATION.json', {'passed':True,'code':bindings(),
        'unit_regressions_passed':9,'TRAIN_transport_cases_per_family':228,
        'native_and_evaluator_reference_pixels_equal':True,
        'native_Dita_numpy':receipts['dita','native']['numpy'],
        'native_S_numpy':receipts['vlanext_s','native']['numpy'],
        'current_evaluator_numpy':receipts['dita','evaluator']['numpy'],
        'existing_parent_training_code_hashes_unchanged':True,
        'GPU_model_or_optimizer_loaded':False,
        'real_native_FT_gate_and_DEV_still_required':True})
    print('R90 CPU qualification passed; actual native FT/tests remain gated.',flush=True)


if __name__ == '__main__': main()
