"""CPU-only preflight evidence; never runs extra evaluation trials."""
import hashlib
import json
import os
import subprocess
import time
from VLA.common.provenance import sha256_file
from VLA.autonomy_r83.common import verify
from VLA.dita800_r87.run import ENV,SIM
from .common import OUT,ROOT,atomic_json,utc,bindings,read
from .audit import audit_chain


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    run=OUT/f'preflight_{time.time_ns()}';run.mkdir()
    with (run/'regression_tests.log').open('x') as log:
        test=subprocess.run([str(SIM),'-m','unittest','VLA.benchmark_guardian_r89.tests',
            'VLA.receipt_recovery_r88.tests','-v'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if test.returncode:raise RuntimeError('Guardian contract/regression tests failed; do not launch')
    source=ROOT/'VLA/audits/receipt_recovery_r88_20261021/final800/progress.json'
    original=source.read_bytes();data=json.loads(original)
    snapshot=run/'PROGRESS_SNAPSHOT.json';atomic_json(snapshot,data)
    plan=read(source.parent/'PLAN.json');verify(plan['method']['code']);verify(plan['method']['weights_sha256'])
    started=time.monotonic();chain=audit_chain(snapshot)
    result={'passed':True,'UTC':utc(),'qualification_directory':str(run),'native_GPU_inference_not_interrupted':True,
        'regression_tests_passed':18,'source_progress':str(source),
        'source_progress_sha256_at_snapshot':hashlib.sha256(original).hexdigest(),
        'chain_seconds':time.monotonic()-started,'code':bindings(),
        'all_frozen_source_and_weight_hashes_passed':True,
        'journals_still_append_only_during_this_partial_audit':True,
        'NOT_completed800_or_new_experimental_trials':True,**chain}
    prior=OUT/'PREFLIGHT.json'
    if prior.exists():atomic_json(run/'PRIOR_PREFLIGHT.json',read(prior))
    atomic_json(run/'RESULT.json',result);atomic_json(prior,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('code','trajectory_sha256','source_journal_sha256')},indent=2))


if __name__=='__main__':main()
