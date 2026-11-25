"""TRAIN labels only; do not use this sampler to generate deployed actions."""
from collections import defaultdict
import numpy as np
from VLA.family_transfer_r54.data import Data
from VLA.fit_feedback_r52 import draw as original_draw


MODES = ('pre_close', 'contact_unattached', 'closing_unattached', 'first_correct_hold')
RECIPE = {
    'task_balance': 'uniform four actions',
    'ordinary_sampler_probability': .5,
    'grasp_transition_probability_non_touch': .5,
    'touch': 'unchanged inherited R52 sampler conditioned on touch',
    'transition_modes': MODES,
    'within_mode': 'uniform available target, then episode, then frame',
    'original_bank_preserved': True,
    'adds_wrong_object_or_drop_demonstrations': False,
    'teacher_phase_or_future_action_in_observation': False,
}


def flags(raw, action):
    empty = raw[:, 31] < .5
    correct = (raw[:, 31] > .5) & (raw[:, 30] < .5)
    previous_correct = np.r_[False, correct[:-1]]
    # Actual deployed controller has a +/-0.2 Schmitt trigger, not +/-0.5.
    closing = action[:, 6] > .2
    return {
        'pre_close': empty & closing & (raw[:, 10] < .5),
        'contact_unattached': empty & closing & raw[:, 28:30].any(axis=1),
        'closing_unattached': empty & closing & (raw[:, 6] < .8),
        'first_correct_hold': correct & ~previous_correct,
    }


class GraspData(Data):
    def __init__(self):
        super().__init__(horizon=10)
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for ei, row in enumerate(self.rows):
            n = self.numeric(ei)
            for mode, flag in flags(n['scene64'], n['canonical7']).items():
                frames = np.flatnonzero(flag)
                if len(frames):
                    self.transitions[row['task']][mode][row['pick_name']].append((ei, frames))
        for task in ('pick', 'move', 'pick_place'):
            if any(not self.transitions[task][mode] for mode in MODES):
                raise ValueError('Missing measured TRAIN transition support: ' + task)
        self.last_sampling_mode = None

    def draw(self, rng):
        task = rng.choice(sorted(self.groups))
        if task == 'touch' or rng.random() < .5:
            # Conditioning the old uniform-task sampler preserves all of its
            # source/target/phase probabilities; no test outcome enters here.
            for _ in range(1000):
                ei, fi = original_draw(self, rng)
                if self.rows[ei]['task'] == task:
                    self.last_sampling_mode = 'original_sampler'
                    return ei, fi
            raise RuntimeError('Unexpected uniform-task sampler rejection exhaustion')
        mode = str(rng.choice(MODES))
        choices = self.transitions[task][mode]
        target = rng.choice(sorted(choices))
        ei, frames = choices[target][int(rng.integers(len(choices[target])))]
        self.last_sampling_mode = mode
        return int(ei), int(rng.choice(frames))

    def coverage(self):
        return {task: {mode: {
            'target_names': sorted(targets),
            'episodes': sum(len(v) for v in targets.values()),
            'frames': sum(len(frames) for v in targets.values() for _, frames in v),
        } for mode, targets in modes.items()} for task, modes in self.transitions.items()}
