"""Audit existing TRAIN and completed DEV only; never tune on final800."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.audit import cohort_audit

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / 'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4/qualified_native_data'
OUT = ROOT / 'VLA/audits/recovery_quality_r90_20261022'
TASKS = ('touch', 'pick', 'move', 'pick_place')


def read(path):
    return json.loads(Path(path).read_text())


def regimes(raw, action):
    """Audit bins, NOT success rules, expert phases or model inputs."""
    raw, action = np.asarray(raw), np.asarray(action)
    if raw.ndim != 2 or raw.shape[1] != 64 or action.shape != (len(raw), 7):
        raise ValueError('Actual pre-action64 / canonical7 chronology required')
    if not np.isfinite(raw).all() or not np.isfinite(action).all():
        raise ValueError('Nonfinite data')
    xy = np.linalg.norm(raw[:, 11:13], axis=1)
    held = raw[:, 17] > .5
    unheld = raw[:, 31] < .5
    near = (raw[:, 9] <= .14) & unheld
    low = (raw[:, 9] <= .10) & unheld
    close = action[:, 6] > .2
    contact = np.any(raw[:, 28:30] > .5, axis=1)
    previous_hold = np.r_[False, held[:-1]]
    return {
        'all': np.ones(len(raw), bool),
        'unheld_under14cm': near,
        'unheld_under10cm_offset_over2cm': low & (xy > .02),
        'unheld_under14cm_offset_over2cm': near & (xy > .02),
        'unheld_under14cm_descending_over5mm': near & (action[:, 2] < -.005),
        'unheld_under14cm_lateral_over5mm': near & (np.linalg.norm(action[:, :2], axis=1) > .005),
        'unheld_under14cm_raising_over5mm': near & (action[:, 2] > .005),
        'close_without_verified_hold': close & ~held,
        'target_contact_without_verified_hold': contact & ~held,
        'correct_bilateral_hold': held,
        'held_translation_over5mm': held & (np.linalg.norm(action[:, :3], axis=1) > .005),
        'held_close_command': held & close,
        'held_open_command': held & (action[:, 6] < -.2),
        'actual_hold_lost_between_pre_action_frames': previous_hold & ~held,
        'wrong_attached': raw[:, 30] > .5,
    }


def bank_audit(bank=BANK):
    bank = Path(bank)
    proof, rows = read(bank/'audit.json'), read(bank/'index.json')
    if not proof['passed'] or proof.get('future_labels_in_observations') is not False:
        raise ValueError('Qualified causal TRAIN bank required')
    if sha256_file(bank/'index.json') != proof['index_sha256']:
        raise ValueError('TRAIN index changed')
    totals, tasks, sources, coverage = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    grippers, recovery = Counter(), Counter()
    for row in rows:
        path = bank / row['cache']
        if sha256_file(path) != row['cache_sha256']:
            raise ValueError('TRAIN numeric cache changed: ' + str(path))
        with np.load(path, allow_pickle=False) as a:
            raw, action = a['scene64'], a['canonical7']
        if len(raw) != row['frames']:
            raise ValueError('Frame counter mismatch')
        values, counts = np.unique(action[:, 6], return_counts=True)
        grippers.update({str(float(v)): int(c) for v, c in zip(values, counts, strict=True)})
        recovery[str(row.get('recovery_kind'))] += 1
        for name, mask in regimes(raw, action).items():
            count = int(mask.sum())
            totals[name] += count
            tasks[row['task']][name] += count
            sources[row['source_kind']][name] += count
            if count:
                coverage[row['task']+'/'+name].add(row['pick_name'])
    return {
        'scope': 'actual_complete_TRAIN_bank_NOT_neural_SR', 'episodes': len(rows),
        'frames': sum(r['frames'] for r in rows), 'dataset': str(bank.resolve()),
        'index_sha256': sha256_file(bank/'index.json'), 'numeric_cache_hashes_checked': True,
        'totals': dict(totals), 'per_action': dict(tasks), 'per_source_kind': dict(sources),
        'target_coverage_count': {k: len(v) for k, v in coverage.items()},
        'gripper_label_counts': dict(grippers), 'recovery_kind_episodes': dict(recovery),
        'regime_thresholds_are_diagnostic_not_success_or_planner': True,
        'no_new_RGB_or_demonstrations_implied': True,
    }


def development_audit(path):
    path = Path(path)
    progress = read(path)
    if progress['cohort_role'] != 'development_selection':
        raise ValueError('Only DEV may inform adaptation; CONF/final prohibited')
    base = cohort_audit(path, require_complete=True)
    details, totals = [], defaultdict(Counter)
    for row in progress['rows']:
        trace = read(row['trajectory'])[row['task']]
        pre = np.asarray([s['pre_action_scene64'] for s in trace])
        action = np.asarray([s['action'] for s in trace])
        count = {k: int(mask.sum()) for k, mask in regimes(pre, action).items()}
        totals[row['task']].update(count)
        hold = pre[:, 17] > .5
        # Audit labels only. An unheld lateral move can be a correct approach;
        # never call it transport/recovery without a verified hold.
        post = np.asarray([s['post_action_scene64'] for s in trace])
        hold_any = bool(np.any(hold) or np.any(post[:, 17] > .5))
        details.append({
            'seed': row['seed'], 'task': row['task'], 'actual_success': row['success'],
            'actual_safety_violation': row.get('safety_violation'),
            'ever_correct_hold': hold_any, 'regime_frames': count,
            'safety_pre_rollback_diagnostics_retained': any(
                'safety_diagnostics_before_rollback' in s for s in trace),
            'terminal_pre_pad_xyz_m': pre[-1, 7:10].tolist(),
            'terminal_pre_target_relative_xyz_m': pre[-1, 11:14].tolist(),
            'terminal_action': action[-1].tolist(),
        })
    return {'scope': 'completed_DEV_only_NOT_final_adaptation', 'source': str(path.resolve()),
        'source_sha256': sha256_file(path), 'per_action': base['per_action'],
        'regime_frames_by_action': dict(totals), 'episodes': details,
        'neural_SR_is_not_inferred_from_teacher_replay_or_TRAIN_loss': True}


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    result = bank_audit(a.bank)
    atomic_json(a.output/'TRAIN_SUPPORT.json', result)
    stages = [ROOT/'VLA/audits/continuous_r86_20261021'/family/f'dev40_fit{step}/progress.json'
              for family, step in (('dita', 1024), ('dita', 4096), ('vlanext_s', 2048))]
    for path in stages:
        d = development_audit(path)
        atomic_json(a.output/(path.parent.parent.name+'_'+path.parent.name+'_DEV_AUDIT.json'), d)
    atomic_json(a.output/'AUDIT_COMPLETE.json', {
        'passed': True, 'TRAIN_frames': result['frames'], 'completed_DEV_cohorts': len(stages),
        'code_sha256': sha256_file(Path(__file__)), 'final800_traces_inspected_for_adaptation': False,
        'running_training_source_or_dataset_modified': False,
        'task_performance_improvement_claimed': False})
    print(json.dumps({'TRAIN_frames': result['frames'], 'regimes': result['totals']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, default=BANK)
    p.add_argument('--output', type=Path, default=OUT/'support_audit')
    main(p.parse_args())
