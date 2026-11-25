"""Independent final800 outcome -> trajectory -> actual-neural-query chain audit."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from VLA.dita800_r87.audit import final_audit
from VLA.common.provenance import sha256_file
from .common import complete_rows,read


def audit_chain(path):
    """Also usable on a stable snapshot of an unfinished benchmark."""
    data,rows,keys=complete_rows(path)
    journals={};traces=OrderedDict();trajectory_hashes={};checked=0
    for row in rows:
        source=Path(row['trajectory'])
        if source not in traces:
            traces[source]=read(source)
            trajectory_hashes[str(source)]=sha256_file(source)
            if len(traces)>2:traces.popitem(last=False)
        trace=traces[source][row['task']]
        if len(trace)!=row['steps'] or any(bool(s['success']) for s in trace)!=row['success']:
            raise ValueError('Outcome row/actual controller trajectory mismatch')
        journal=source.parent.parent/'NEURAL_QUERIES.jsonl'
        if journal not in journals:
            values={}
            with journal.open() as stream:
                import json
                for line in stream:
                    if not line.endswith('\n'):continue
                    q=json.loads(line);key=(q['seed'],q['task'],q['index'])
                    if key in values:raise ValueError('Duplicate native query in one attempt')
                    if q['receipt']['future_teacher_phase_or_labels'] or q['receipt']['geometric_action_residual_or_Oracle']:
                        raise ValueError('Not the frozen actual-neural executor')
                    values[key]=(q['input_sha256'],np.asarray(q['actions'][0],np.float32))
            journals[journal]=values
        for index,step in enumerate(trace):
            digest,action=journals[journal][row['seed'],row['task'],index]
            if step['step']!=index+1 or step['query_sha256']!=digest:
                raise ValueError('Controller step not linked to the actual native query')
            if not np.array_equal(action,np.asarray(step['action'],np.float32)):
                raise ValueError('Requested model action was replaced')
            if not np.isfinite(np.asarray(step['post_action_scene64'])).all():
                raise ValueError('Nonfinite committed physical state')
            checked+=1
    if checked!=sum(r['steps'] for r in rows):raise ValueError('Unaccounted complete-trial controls')
    return {'independent_outcome_trajectory_neural_query_chain_passed':True,
        'actual_complete_trials_verified':len(rows),
        'actual_neural_controls_verified':checked,'original_failure_evidence_retained':True,
        'trajectory_sha256':trajectory_hashes,'source_journal_sha256':{str(p):sha256_file(p) for p in journals}}


def certify(path):
    summary=final_audit(path);chain=audit_chain(path)
    if chain['actual_neural_controls_verified']!=read(path)['motor_controls']:
        raise ValueError('Final800 contains incomplete/unaccounted control counts')
    return {**summary,**chain}
