"""Independent, CPU-only continuation review; never edits a live method/job.

Example:
  CUDA_VISIBLE_DEVICES='' python -m VLA.continuation_review_r101 \
    --output VLA/audits/continuation_review_r101_20261025

Completed R99/DEV evidence is re-audited. Active final800 is a status snapshot,
not a certificate or a source of new training labels/checkpoint selection.
"""
import argparse
from collections import Counter
from pathlib import Path
import unittest

import numpy as np

from VLA.admission_r100.common import OUT, TASKS, atomic_json, read, utc, verify
from VLA.common.provenance import sha256_file


def trace_diagnosis(progress):
    """Observable hold/command events, NOT proof of an architectural cause."""
    per = {task: Counter() for task in TASKS}
    cached_path, cached = None, None
    for row in progress['rows']:
        if row['trajectory'] != cached_path:
            cached_path = row['trajectory']
            cached = read(cached_path)
        trace = cached[row['task']]
        if len(trace) != row['steps'] or bool(trace[-1]['success']) != bool(row['success']):
            raise ValueError('Outcome/trace mismatch')
        pre = np.asarray([s['pre_action_scene64'] for s in trace], np.float64)
        post = np.asarray([s['post_action_scene64'] for s in trace], np.float64)
        action = np.asarray([s['action'] for s in trace], np.float64)
        if pre.shape != (len(trace), 64) or post.shape != pre.shape or action.shape != (len(trace), 7):
            raise ValueError('Wrong measured-state/action shape')
        if not all(np.isfinite(a).all() for a in (pre, post, action)):
            raise ValueError('Nonfinite physical trace')
        held_before, held_after = pre[:, 17] > .5, post[:, 17] > .5
        for index, s in enumerate(trace):
            # The diagnostic field records correct ATTACHMENT identity;
            # scene17 additionally requires contact at BOTH pads. Losing
            # bilateral contact is not necessarily dropping the object.
            for values, diagnostic in ((pre[index], s['before']), (post[index], s['after'])):
                correct_attachment = bool(values[31] > .5 and values[30] < .5)
                bilateral = bool(correct_attachment and values[28] > .5 and values[29] > .5)
                if correct_attachment != bool(diagnostic['correct_object_held']):
                    raise ValueError('Independent attachment identity fields disagree')
                if bool(values[17] > .5) != bilateral:
                    raise ValueError('Bilateral hold does not match attachment and per-pad contact')
        c = per[row['task']]
        c['episodes'] += 1
        c['successes'] += int(row['success'])
        c['safety_aborts'] += int(bool(row.get('safety_violation')))
        if row['task'] == 'touch':
            c['failed_touch'] += int(not row['success'])
            continue
        ever, final = bool(np.any(held_before | held_after)), bool(held_after[-1])
        c['ever_correct_hold'] += int(ever)
        c['failed_without_correct_hold'] += int(not row['success'] and not ever)
        c['failed_after_hold_lost_before_end'] += int(not row['success'] and ever and not final)
        c['failed_still_correctly_held_at_end'] += int(not row['success'] and final)
        c['failed_still_correctly_attached_at_end'] += int(
            not row['success'] and post[-1, 31] > .5 and post[-1, 30] < .5)
        c['ever_lost_hold'] += int(np.any(held_before & ~held_after))
        c['open_command_while_held_steps'] += int(np.sum(held_before & (action[:, 6] < 0)))
    for task, c in per.items():
        names = ('failed_touch',) if task == 'touch' else (
            'failed_without_correct_hold', 'failed_after_hold_lost_before_end',
            'failed_still_correctly_held_at_end')
        if c['episodes'] != c['successes'] + sum(c[n] for n in names):
            raise ValueError('Failure categories must form a disjoint partition')
    return {k: dict(v) for k, v in per.items()}


def pending_admission(family):
    from VLA.admission_r100.train import qualified
    from VLA.admission_r100.workflow import foundations
    root = OUT / family
    gate = Path(read(root / 'SELECTED_NATIVE_GATE.json')['checkpoint'])
    meta = qualified(gate, family)
    trainer = read(root / 'TRAINER_CODE_ROUNDTRIP.json')
    qualified(Path(trainer['checkpoint']), family)
    reload = read(gate / 'FRESH_RELOAD.json')
    admission = read(root / 'NATIVE_ADMISSION_READY.json')
    data = read(root / 'DATA_AUDIT.json')
    if not (data['passed'] and trainer['passed'] and reload['passed'] and admission['passed']):
        raise ValueError('Missing pending native admission gate: ' + family)
    if trainer['effective_batch'] != 16 or admission['actual_integration_trials'] != 4:
        raise ValueError('Actual trainer/smoke coverage differs: ' + family)
    foundation_files = foundations(family)
    verify(foundation_files)
    return {'selected_native_gate': str(gate), 'gate_updates': meta['updates'],
        'foundation_files_verified': foundation_files, 'effective_batch': 16,
        'fresh_reload_anchors': len(reload['max_errors']),
        'fresh_reload_max_error': max(reload['max_errors']),
        'train_trajectories': data['trajectories'], 'train_frames': data['frames'],
        'long_target_domain_finetune_completed': (root / 'adapt2048_FINISHED.json').exists(),
        'new_closed_loop_SR_NOT_implied': True}


def main(output):
    if output.exists():
        raise FileExistsError('Preserve the previous review; select a new output directory')
    qualification = read(OUT / 'AUTOMATION_QUALIFICATION.json')
    verify(qualification['code'])
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
        'VLA.admission_r100.tests', 'VLA.postfit800_r98.tests', 'VLA.autorefinement_r97.tests',
        'VLA.continuation_review_r101_tests'))
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    if not tests.wasSuccessful():
        raise ValueError('Continuation contract/queue regression')
    from VLA.refinement_r99.common import OUT as R99, audit_result as audit_r99
    saved = read(R99 / 'CERTIFIED_RESULT.json')
    result = audit_r99(Path(saved['source']))
    for key in ('actual_trials', 'per_action', 'source_sha256', 'checkpoint',
                'actual_neural_controls_verified'):
        if saved[key] != result[key]:
            raise ValueError('Recomputed R99 certificate differs: ' + key)
    print('Re-audited actual R99 final800', flush=True)
    diagnosis = trace_diagnosis(read(result['source']))
    from VLA.admission_r100.evaluate import audit_result as audit_native
    selection = read(OUT / 'seer/DEV_SELECTION.json')
    dev = [audit_native(Path(row['progress'])) for row in selection['candidates']]
    confirm = audit_native(Path(read(OUT / 'seer/confirmation40_COMPLETE.json')['progress']))
    pending = {}
    for family in ('gr00t_n15', 'falcon_esm'):
        print('Verifying authentic pending model bytes: ' + family, flush=True)
        pending[family] = pending_admission(family)
    snapshot = read(OUT / 'WORKFLOW_STATUS.json')
    verify(qualification['code'])
    report = {'UTC': utc(), 'passed': True, 'CPU_tests_passed': tests.testsRun,
        'protected_source_files_unchanged': len(qualification['code']),
        'R99_independent_completed_final_audit': result,
        'R99_observable_trace_diagnosis': diagnosis,
        'Seer_DEV_candidates_reaudited': dev, 'Seer_CONF_reaudited': confirm,
        'pending_native_models': pending, 'active_status_snapshot_NOT_certified': snapshot,
        'causal_model_architecture_failure_NOT_proven': True,
        'final_outcomes_NOT_used_to_change_training_or_checkpoint_selection': True,
        'existing_policy_source_weights_metrics_or_workers_changed': False,
        'no_five_seed_outcomes_invented': True,
        'metric': result['metric']}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'AUDIT.json', report)
    lines = ['# R101: independent continuation review', '', f"UTC: {report['UTC']}", '',
        f"{tests.testsRun} CPU tests PASS; {len(qualification['code'])} protected source hashes unchanged.", '',
        '## VLANeXt-S R99: measured final800', '',
        '|Action|Successes/n|SR|Safety aborts|', '|---|---:|---:|---:|']
    for task, values in result['per_action'].items():
        lines.append(f"|{task}|{values['successes']}/{values['n']}|{values['SR_percent']:.2f}%|{values['safety_aborts']}|")
    lines += ['', 'Independent outcome/trace/query audit verified ' +
        str(result['actual_neural_controls_verified']) + ' actual neural controls.', '',
        '## Observable failure partition (not an architectural explanation)', '',
        '|Action|Failed without correct hold|Failed after hold lost|Failed while still holding|',
        '|---|---:|---:|---:|']
    for task in ('pick', 'move', 'pick_place'):
        c = diagnosis[task]
        lines.append(f"|{task}|{c.get('failed_without_correct_hold',0)}|{c.get('failed_after_hold_lost_before_end',0)}|{c.get('failed_still_correctly_held_at_end',0)}|")
    lines += ['', 'These hold groups use bilateral contact plus correct attachment identity. '
        'Attachment identity alone has a weaker definition in the diagnostic JSON. '
        'Contact loss is NOT, by itself, evidence of an object drop. '
        'Safety aborts overlap these disjoint failure groups. A hold-loss event can occur even in a later successful episode.',
        'The dominant Move/Put failure group is still holding the correct object at termination. This does NOT establish why goal transport failed; '
        'it does not support claiming premature opening is the dominant cause.', '',
        '## Seer: completed development/confirmation only', '',
        '|Checkpoint updates|Role|Touch|Pick|Move|Put|Total|', '|---:|---|---:|---:|---:|---:|---:|']
    for audit, role in [(d, 'DEV selection') for d in dev] + [(confirm, 'independent CONF')]:
        counts = [audit['per_action'][t]['successes'] for t in TASKS]
        updates = read(Path(audit['checkpoint']) / 'TRAINING.json')['updates']
        lines.append('|'+ '|'.join([str(updates), role, *[str(c)+'/10' for c in counts], str(sum(counts))+'/40'])+'|')
    lines += ['', 'Current final800 is still running; no partial-final certificate is issued here.', '',
        '## Pending models', '',
        'GR00T-N1.5 and FALCON-ESM: initialization, optimizer/RNG trainer fixture, foundations, '
        'native processor and four-action integration receipts rechecked. These checks do not imply high success rate. '
        'Long fits remain with the existing durable owner, after the current Seer job; no duplicate GPU owner was launched.', '',
        'The existing bounded sequence is fit → DEV40 → optional continuation → independent CONF40 → frozen800 if ready.', '',
        'Metric: original R62 semantic first-hit; NOT strict1cm; PUT release/settle is not required. '
        'These are not five measured seeds. No existing benchmark, threshold, checkpoint, worker or policy source was modified.']
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print({'passed': True, 'report': str(output / 'REPORT.md')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
