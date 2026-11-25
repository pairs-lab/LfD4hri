"""Same audited substep replay, explicitly scoped to completed chunk2 forty.

Process-local source/case selection only; original replay helper and actual
VLA/evaluator runtimes remain unchanged. This is diagnosis, not a new SR.
"""
import argparse
import json
from pathlib import Path
from VLA import replay_unattested_success_r41 as native
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    proof_file=native.BASE/'Smol_native2_completion_history40_r41/audit.json'
    proof=json.loads(proof_file.read_text())
    if not proof['passed_trace_integrity'] or proof['trials']!=40:raise ValueError('Complete actual forty required')
    native.SOURCE=native.BASE/'Smol_native_chunk2_development40_r41/progress.json'
    native.CASES=tuple((r['seed'],r['task']) for r in proof['cases']
        if r['predeclared_success'] and r['task']!='touch' and not r['correct_attachment_observed'])
    if len(native.CASES)!=2:raise ValueError('Exact two declared unattested cases required for helper mapping')
    native.main(a.output)
    atomic_json(a.output/'explicit_scope.json',{'source':str(native.SOURCE),'cases':list(native.CASES),
        'process_local_overrides_ONLY':True,'native_replay_helper_source_changed':False,
        'actual40_scores_or_actions_modified':False,'800_data_used':False,
        'wrapper_sha256':sha256_file(Path(__file__)),'proof_sha256':sha256_file(proof_file)})
