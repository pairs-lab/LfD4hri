import argparse
from collections import Counter
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.flower_r70.report import classify
from VLA.intern_r72.common import read


def main(a):
    all_results={};lines=['# InternVLA-A1.5 — actual native policy trials','',
        'Per-action measured development/confirmation, not final800 or five evaluated training seeds.',
        'Same R62 plant/profile/semantic first-hit criteria. PUT does not require release/settle.','']
    for name in ('development40_fit1024','development40_fit2048','confirmation40'):
        path=a.output/name/'progress.json'
        if not path.exists():continue
        p=read(path);counts=Counter()
        for row in p['rows']:counts[classify(row,read(row['trajectory'])[row['task']])]+=1
        all_results[name]={'per_action':p['per_action'],'completed_trials':p['completed_trials'],
            'checkpoint':p['checkpoint'],'observed_outcome_labels':dict(counts)}
        lines+=['## '+name,'','|Action|Successes/trials|Measured SR|','|---|---:|---:|']
        for task,v in p['per_action'].items():lines.append(f"|{task}|{v['successes']}/{v['n']}|{v['SR_percent']:.1f}%|")
        lines+=['','Trace labels: '+str(dict(counts)),'']
    atomic_json(a.output/'MEASURED_PER_ACTION.json',all_results)
    (a.output/'MEASURED_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
