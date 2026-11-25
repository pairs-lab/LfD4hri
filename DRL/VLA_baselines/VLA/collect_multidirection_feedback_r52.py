"""Complete TRAIN-only recovery collection with all four tilt quadrants.

Extends the measured/replayed collector; no deployment controller is added.
Fault signs are fixed by TRAIN task/target indices, never evaluation outcomes.
"""
import argparse
from pathlib import Path

import numpy as np

import VLA.collect_pose_feedback_r52 as base
from VLA.common.provenance import sha256_file

TASKS = ('touch', 'pick', 'move', 'pick_place')
TARGETS = ('blue block', 'red block', 'red apple', 'green pear', 'white egg')
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def fault_action(task, target, kind):
    sx, sy = SIGNS[(TASKS.index(task)*len(TARGETS)+TARGETS.index(target)) % len(SIGNS)]
    if kind == 'wrist_tilt':
        return np.array([.012*sx, .010*sy, 0., .045*sx, .040*sy, .025*sx*sy, -1.], np.float32)
    if kind == 'closed_empty':
        return np.array([.015*sx, .012*sy, 0., 0., 0., 0., 1.], np.float32)
    raise ValueError('Unqualified intervention')


class MultiDirectionStart(base.PerturbedStart):
    def reset(self, task):
        if task != base.TaskSpec.from_dict(self.row['task']):
            raise ValueError('TRAIN task identity changed')
        obs = base.reset_source_episode(self.env, self.source, self.row)
        raw, actions = self.numeric['scene64'], self.numeric['canonical7']
        candidates = np.flatnonzero((raw[:, 9] <= .23) & (raw[:, 9] >= .18) & (raw[:, 31] == 0))
        if not len(candidates):
            raise ValueError('No safe high unheld TRAIN anchor')
        anchor = int(candidates[0])
        for i in range(anchor):
            np.testing.assert_allclose(base.scene_state(self.env, obs), raw[i], rtol=0, atol=1e-6)
            obs, _, done, info = self.env.step(actions[i])
            self.prefix.append({'kind': 'original_TRAIN_command', 'action': actions[i].tolist()})
            if done or info.get('safety_violation'):
                raise ValueError('Original TRAIN prefix terminated')
        np.testing.assert_allclose(base.scene_state(self.env, obs), raw[anchor], rtol=0, atol=1e-6)
        perturb = fault_action(task.name, task.pick_name, self.kind)
        for j in range(1 if self.kind == 'wrist_tilt' else 8):
            action = perturb.copy()
            if j:
                action[:3] = 0
            obs, _, done, info = self.env.step(action)
            self.prefix.append({'kind': self.kind, 'action': action.tolist()})
            if done or info.get('safety_violation') or self.env.gripper.grasped_object is not None:
                raise ValueError('Intervention failed actual safe-air/unheld gate')
        state = base.scene_state(self.env, obs)
        if self.kind == 'closed_empty' and state[6] < .8:
            raise ValueError('Measured empty closure not reached')
        if self.kind == 'wrist_tilt' and base.upright_error(obs.ee_rotation_6d)[1] < .02:
            raise ValueError('Measured tilt not reached')
        self.initial_state, self.original_anchor = state.tolist(), anchor
        self.env.step_count = self.env.success_streak = 0
        self.env._last_success = False
        self.last_obs = self.env.observe()
        return self.last_obs


def main(args):
    if tuple(args.targets.split(',')) != TARGETS:
        raise ValueError('Keep predeclared five-target sweep fixed')
    base.PerturbedStart = MultiDirectionStart
    original = base.atomic_json
    def atomic(path, value):
        if Path(path).name == 'plan.json':
            value = {**value, 'multidirection_extension_sha256': sha256_file(Path(__file__)),
                'fault_quadrants': SIGNS, 'fault_selection': 'fixed_TASK_index_times5_plus_TARGET_index_mod4',
                'both_roll_and_pitch_signs_per_task': True,
                'all_fault_commands_recorded_and_replayed_not_supervised_labels': True}
        original(path, value)
    base.atomic_json = atomic
    base.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', default=','.join(TARGETS))
    main(parser.parse_args())
