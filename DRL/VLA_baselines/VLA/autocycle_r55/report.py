"""Report-only receipts; this module never controls model selection/fitting."""
from datetime import datetime,timezone
from pathlib import Path
import math
from VLA.common.artifact_io import atomic_json
from .common import MODELS,TASKS,read


def wilson(successes,n):
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def final_result(path):
    p=read(path)
    expected={(seed,t) for seed in range(20269701,20269731) for t in TASKS}
    if p['completed_trials']!=120 or p['planned_trials']!=120:return None
    if len(p['rows'])!=120 or {(r['seed'],r['task']) for r in p['rows']}!=expected:
        raise ValueError('Final row identity/coverage is wrong')
    counts={t:sum(bool(r['success']) for r in p['rows'] if r['task']==t) for t in TASKS}
    if sum(counts.values())!=p['successes']:raise ValueError('Success receipt inconsistent')
    return {'checkpoint':p['checkpoint'],'successes':p['successes'],'trials':120,
        'SR':p['successes']/120,'Wilson95':wilson(p['successes'],120),'per_action':counts,
        'safety_aborts':sum(bool(r.get('safety_violation')) for r in p['rows']),'source':str(path)}


def report(root):
    root=Path(root);state=read(root/'status.json') if (root/'status.json').exists() else {}
    methods=state.get('selected_methods',{});results={}
    frozen=(root/'ALL_METHODS_FROZEN_BEFORE_FINAL.json').exists()
    lines=['# R55 — autonomous four-VLA continuation','',
        'Updated UTC: '+datetime.now(timezone.utc).isoformat(),
        'Stage: '+state.get('stage','not_started'),'',
        'Final outcomes never enter diagnosis, training, sampler or checkpoint selection. All methods freeze before ANY R55 final rollout.',
        'Methods use simulator GT, not vanilla RGB/text-only VLA. Existing common motor/IK/contact-assist/safety interface is retained.',
        'Place is the existing goal-region success criterion, not proof of released/stable placement.','',
        '| Model | Confirmed development readiness | Final successes /120 | SR | 95% Wilson CI |',
        '|---|---|---:|---:|---:|']
    for model in MODELS:
        method=methods.get(model);ready='pending' if method is None else str(method.get('stable_development_target_met',False))
        path=root/(model+'_final120/progress.json')
        result=final_result(path) if frozen and path.exists() else None
        if result is None:
            lines.append(f'| {model} | {ready} | pending | pending | pending |')
            results[model]={'status':'pending_or_failed_NOT_zero_SR'}
        else:
            results[model]=result;lo,hi=result['Wilson95']
            lines.append(f"| {model} | {ready} | {result['successes']}/120 | {100*result['SR']:.1f}% | {100*lo:.1f}–{100*hi:.1f}% |")
    all_complete=all(r.get('trials')==120 for r in results.values())
    lines+=['',f'All four actual final cohorts complete: {all_complete}.',
        'The readiness target is at least24/40 in EACH of two development cohorts, at most4 safety aborts/cohort and at least3/10 for EVERY task in BOTH. It is not a guarantee of final accuracy.',
        'At most two gated continuation rounds/family. A second round needs an actual pooled-development80 gain; no infinite best-test-score search.',
        'Errors: '+str(state.get('errors',{})),
        'Original checkpoints, failures, videos and earlier independent tests remain preserved. No global maximum or intrinsic model ceiling is claimed.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(root/'report_results.json',{'all_four_complete':all_complete,'results':results,
        'report_only_not_input_to_training':True,'all_methods_frozen_before_final':frozen})
