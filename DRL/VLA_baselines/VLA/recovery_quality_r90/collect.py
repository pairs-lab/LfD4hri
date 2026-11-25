"""Finite TRAIN-only near-miss pilot; expert actions never enter deployment."""
import argparse
import json
from pathlib import Path
import numpy as np

import VLA.collect_pose_feedback_r52 as base
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .audit import OUT, TASKS, regimes

TARGETS = ('blue block', 'red apple')
KINDS = ('low_offset_open', 'low_offset_closed')


def intervention(task, target, kind, tick):
    if task not in TASKS or target not in TARGETS or kind not in KINDS or not 0 <= tick < 8:
        raise ValueError('Only predeclared TRAIN interventions')
    i = TASKS.index(task)*len(TARGETS)+TARGETS.index(target)
    sx, sy = ((1, 1), (1, -1), (-1, 1), (-1, -1))[i % 4]
    # Lateral miss only in safe air; never command a downward collision.
    xy = (.030*sx, .025*sy) if tick == 0 else (0., 0.)
    return np.array([*xy, 0., 0., 0., 0., 1. if kind.endswith('closed') else -1.], np.float32)


class NearMissStart(base.PerturbedStart):
    def reset(self, task):
        if task != base.TaskSpec.from_dict(self.row['task']):
            raise ValueError('TRAIN task identity changed')
        obs = base.reset_source_episode(self.env, self.source, self.row)
        raw, actions = self.numeric['scene64'], self.numeric['canonical7']
        ids = np.flatnonzero((raw[:, 9] <= .14) & (raw[:, 9] >= .11) & (raw[:, 31] == 0))
        if not len(ids):
            raise ValueError('No original safe low-unheld TRAIN anchor; do not invent one')
        anchor = int(ids[0])
        for i in range(anchor):
            np.testing.assert_allclose(base.scene_state(self.env, obs), raw[i], rtol=0, atol=1e-6)
            obs, _, done, info = self.env.step(actions[i])
            self.prefix.append({'kind': 'original_TRAIN_command', 'action': actions[i].tolist()})
            if done or info.get('safety_violation'):
                raise ValueError('Original TRAIN prefix terminated')
        np.testing.assert_allclose(base.scene_state(self.env, obs), raw[anchor], rtol=0, atol=1e-6)
        for tick in range(8 if self.kind.endswith('closed') else 1):
            action = intervention(task.name, task.pick_name, self.kind, tick)
            obs, _, done, info = self.env.step(action)
            self.prefix.append({'kind': self.kind, 'action': action.tolist()})
            if done or info.get('safety_violation') or self.env.gripper.grasped_object is not None:
                raise ValueError('Near-miss intervention failed safe/unheld gate')
        state = base.scene_state(self.env, obs)
        if not .10 <= state[9] <= .145 or np.linalg.norm(state[11:13]) < .018:
            raise ValueError('Intended measured low offset was not reached')
        if self.kind.endswith('closed') and state[6] < .8:
            raise ValueError('Intended measured empty closure was not reached')
        self.initial_state, self.original_anchor = state.tolist(), anchor
        self.env.step_count = self.env.success_streak = 0
        self.env._last_success = False
        self.last_obs = self.env.observe()
        return self.last_obs


def main(a):
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    proof = json.loads(base.PROJECTION.read_text())
    source = Path(proof['source'])
    if sha256_file(source/'index.json') != proof['source_index_sha256']:
        raise ValueError('TRAIN source index changed')
    manifest = json.loads((source/'dataset_manifest.json').read_text())
    if manifest['split'] != 'train' or manifest['final_test_used']:
        raise ValueError('Only original TRAIN scenes, never DEV/CONF/final scenes')
    rows = json.loads((source/'index.json').read_text())
    native = {r['source']: r for r in json.loads((base.NATIVE/'index.json').read_text())}
    chosen = [next(r for r in rows if r['task']['name'] == task and r['task']['pick_name'] == target
        and not r.get('paired_scene_source') and not r.get('recovery_start_actions_not_training_labels')
        and not r['file'].endswith('_closedhover.npz')) for task in TASKS for target in TARGETS]
    planned = len(chosen)*len(KINDS)
    atomic_json(out/'plan.json', {
        'scope': 'TRAIN_only_near_miss_expert_candidates_NOT_VLA_SR', 'source': str(source),
        'source_sha256': sha256_file(source/'index.json'),
        'script_sha256': sha256_file(Path(__file__)), 'targets': TARGETS, 'kinds': KINDS,
        'candidate_episodes': planned, 'source_files': [r['file'] for r in chosen],
        'source_scene_groups': [r['scene_group'] for r in chosen],
        'interventions_are_real_motor_commands_NOT_supervised_labels': True,
        'expert_actions_only_for_data_never_VLA_deployment': True,
        'selection_uses_DEV_instances_or_final_outcomes': False,
        'candidate_only_NOT_automatically_admitted': True,
        'current_R86_data_stats_weights_source_unchanged': True,
    })
    env = base.audit_environment(source, robot='ur5e', image_size=384, control_hz=10, seed=0,
        required_success_steps=1, success_threshold_scale=1., use_semantic_success=False)
    results = []
    # The existing exact replay helper instantiates this class. Replace only
    # in this separate CPU process; no on-disk or running-job code is changed.
    old = base.PerturbedStart
    base.PerturbedStart = NearMissStart
    try:
        for row in chosen:
            nr = native[str(source/row['file'])]
            with np.load(base.NATIVE/nr['cache'], allow_pickle=False) as loaded:
                numeric = {key: loaded[key] for key in ('scene64', 'canonical7')}
            for kind in KINDS:
                stem = row['task']['name']+'_'+row['task']['pick_name'].replace(' ', '_')+'_'+kind
                result = {'source_file': row['file'], 'kind': kind, 'task': row['task'], 'success': False}
                proxy = NearMissStart(env, source, row, numeric, kind)
                try:
                    arrays, meta = base.collect_episode(proxy, base.TaskSpec.from_dict(row['task']),
                        policy_factory=base.UprightCorrectionTeacher)
                    arrays['audit_pre_action_PRIVILEGED_state64'] = np.asarray(proxy.states, np.float32)
                    path = out/(stem+'.npz')
                    np.savez_compressed(path, **arrays)
                    result.update(meta, file=path.name, payload_sha256=sha256_file(path),
                        perturbation_actions_NOT_labels=proxy.prefix, initial_state64=proxy.initial_state,
                        sensor_review=base.review_images(out, stem, arrays),
                        recovery_regime_frames={k: int(v.sum()) for k, v in regimes(
                            arrays['audit_pre_action_PRIVILEGED_state64'], arrays['canonical_actions']).items()})
                    if meta['success'] and not meta.get('safety_violation'):
                        result['replay'] = base.replay_candidate(env, source, row, numeric, kind, arrays)
                        result['candidate_replay_pass'] = True
                except Exception as error:
                    result['error'] = f'{type(error).__name__}: {error}'
                    result['perturbation_actions_NOT_labels'] = proxy.prefix
                results.append(result)
                atomic_json(out/'index.json', results)
                atomic_json(out/'status.json', {
                    'status': 'collecting_TRAIN_candidates_NOT_neural_testing', 'completed': len(results),
                    'planned': planned, 'actual_expert_successes_NOT_VLA_SR': sum(r['success'] for r in results),
                    'actual_replay_passes': sum(bool(r.get('candidate_replay_pass')) for r in results)})
                print(json.dumps({k: result.get(k) for k in
                    ('kind', 'file', 'success', 'candidate_replay_pass', 'error', 'safety_violation')}), flush=True)
        atomic_json(out/'audit.json', {
            'scope': 'candidate_TRAIN_expert_corrections_NOT_neural_SR',
            'planned': planned, 'all_candidate_attempts_preserved': len(results) == planned,
            'qualified_replay_candidates': sum(bool(r.get('candidate_replay_pass')) for r in results),
            'training_admission_pending': True, 'new_checkpoint_or_VLA_SR_claimed': False})
        atomic_json(out/'status.json', {
            'status': 'pilot_complete_pending_data_admission_NOT_neural_SR', 'completed': len(results),
            'planned': planned, 'actual_expert_successes_NOT_VLA_SR': sum(r['success'] for r in results),
            'actual_replay_passes': sum(bool(r.get('candidate_replay_pass')) for r in results)})
    finally:
        base.PerturbedStart = old
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUT/'near_miss_pilot')
    main(p.parse_args())
