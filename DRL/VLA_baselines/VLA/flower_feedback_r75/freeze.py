"""Evidence-based candidate release, NEVER a performance ceiling claim."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70.common import ROOT,read,verify
from VLA.flower_r71.watch import candidate_wins


BASE=ROOT/'VLA/audits/flower_postfit_r70_20261019'
BALANCED=ROOT/'VLA/audits/target_balance_r74_20261019/flower'


def completed(path):
    if not path.exists(): return None
    d=read(path)
    if d.get('status')!='completed_actual_development' or d.get('completed_trials')!=40:
        return None
    if any(d['per_action'][t]['n']!=10 for t in ('touch','pick','move','pick_place')):
        raise ValueError('Incomplete task coverage cannot support a freeze decision')
    return d


def review(output):
    original=completed(BASE/'final_development40/progress.json')
    if original is None: raise ValueError('Completed baseline40 required')
    options=[{'name':'FLOWER4096_static10','development':original,
              'confirmation_path':BASE/'final_confirmation40/progress.json',
              'evaluation_entry_point':'VLA.flower_r70.evaluate','prefix':10}]
    adaptive=completed(output/'final_development40/progress.json')
    if adaptive is not None and candidate_wins(original,adaptive):
        options.append({'name':'FLOWER4096_feedback_r75','development':adaptive,
                        'confirmation_path':output/'final_confirmation40/progress.json',
                        'evaluation_entry_point':'VLA.flower_feedback_r75.evaluate','prefix':10})
    balanced=completed(BALANCED/'final_development40/progress.json')
    decision=BALANCED/'DEVELOPMENT_SELECTION.json'
    if balanced is not None and decision.exists() and read(decision)['candidate_selected']:
        options.append({'name':'FLOWER4096_plus128_target_balanced_static10','development':balanced,
                        'confirmation_path':BALANCED/'final_confirmation40/progress.json',
                        'evaluation_entry_point':'VLA.flower_r70.evaluate','prefix':10})
    def score(option):
        rows=option['development']['rows']
        return (sum(r['success'] for r in rows),-sum(bool(r.get('safety_violation')) for r in rows))
    # Stable max retains the simpler original on an exact tie. Selection never
    # reads confirmation outcomes or final800; their completion is only a gate.
    selected=max(options,key=score)
    dev=selected['development'];confirmation=completed(selected['confirmation_path'])
    cp=Path(dev['checkpoint']);training=read(cp/'TRAINING.json');verify(training['code'])
    digest=sha256_file(cp/'trainables.pt')
    if digest!=training['trainables_sha256']:raise ValueError('Learned checkpoint changed')
    parity=read(output/'STATIC_PARITY.json')
    if not parity['passed']:raise ValueError('No exact unchanged-runner control parity')
    result={'selection_scope':'development40 only, never confirmation/final800 scores',
        'candidate':selected['name'],'checkpoint':str(cp),'trainables_sha256':digest,
        'stats_sha256':sha256_file(cp/'stats.json'),'native_model':'FLOWER',
        'evaluation_entry_point':selected['evaluation_entry_point'],'prefix':10,
        'development_per_action':dev['per_action'],
        'confirmation_per_action':None if confirmation is None else confirmation['per_action'],
        'confirmation_source':str(selected['confirmation_path']),
        'ready_for_larger_quantitative_evaluation':confirmation is not None,
        'not_a_final800_or_five_seed_result':True,
        'finetune_was_needed_for_selected_variant':selected['name']=='FLOWER4096_plus128_target_balanced_static10',
        'no_new_input_action_or_normalizer_training_required_for_feedback_scheduler':True,
        'performance_ceiling_reached':False,'zero_failure_or_safety_certification_claim':False,
        'PUT_requires_release_and_settle':False,
        'all_candidates_development':{o['name']:o['development']['per_action'] for o in options},
        'known_remaining_gap':'Existing bank does not certify actual drop/wrong-object recovery demonstrations.',
        'checkpoint_not_copied_overwritten_or_deleted':True}
    atomic_json(output/'FREEZE_REVIEW.json',result)
    lines=['# FLOWER R75 — candidate freeze review','',
           'Checkpoint selection uses development outcomes only. Confirmation is reported independently.',
           '',f"Selected: **{result['candidate']}**",f"Checkpoint: `{cp}`",'',
           f"Ready for larger quantitative evaluation: **{confirmation is not None}**",'',
           '| Action | Development | Independent confirmation |','|---|---:|---:|']
    for task in ('touch','pick','move','pick_place'):
        d=dev['per_action'][task];c=None if confirmation is None else confirmation['per_action'][task]
        value='not completed' if c is None else f"{c['successes']}/{c['n']} ({c['SR_percent']:.1f}%)"
        lines.append(f"|{task}|{d['successes']}/{d['n']} ({d['SR_percent']:.1f}%)|{value}|")
    lines+=['','This is a reproducible candidate release, not a claim of a model ceiling, failure-free control,',
            '800-episode stability, or five measured training seeds. PUT remains goal-region first-hit,',
            'not release/settle. Every abort and timeout is included. No threshold, plant, or action was repaired.',
            'The feedback scheduler and target-balanced fine-tune were tested as separate variants;',
            'their effects must not be added together without a new closed-loop test.','']
    (output/'FREEZE_REVIEW.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);review(p.parse_args().output)
