"""Only discard stale neural suffixes; never synthesize or modify an action."""
from collections import deque
import numpy as np


class FeedbackQueue:
    def __init__(self, *, enabled=True, near_radius=.12, near_prefix=2, log=None):
        if near_radius <= 0 or not 1 <= near_prefix <= 10:
            raise ValueError('Positive radius and native prefix in 1..10 required')
        self.enabled, self.near_radius, self.near_prefix = enabled, near_radius, near_prefix
        self.log = log or (lambda event: None)
        self.pending = deque()
        self.observation = self.query_observation = None
        self.consumed = 0
        self.step = None
        self.query_number = 0

    @staticmethod
    def signature(raw):
        return tuple(raw[i] for i in (17, 28, 29, 30, 31))

    def observe(self, raw, step):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (64,) or not np.isfinite(raw).all():
            raise ValueError('Complete finite current scene64 required')
        self.observation, self.step = raw.copy(), int(step)
        if not self.enabled or not self.pending or self.query_observation is None:
            return
        reason = None
        if self.signature(raw) != self.signature(self.query_observation):
            reason = 'measured_contact_or_attachment_changed'
        elif not raw[31] and np.linalg.norm(raw[11:14]) <= self.near_radius and self.consumed >= self.near_prefix:
            reason = 'near_target_unattached_feedback_budget'
        if reason:
            self.log({'event': 'discard_unexecuted_suffix', 'reason': reason,
                      'observation_step': self.step, 'consumed_neural_actions': self.consumed,
                      'discarded_neural_actions': len(self.pending), 'query_number': self.query_number,
                      'current_contact_attachment': [float(v) for v in self.signature(raw)],
                      'query_contact_attachment': [float(v) for v in self.signature(self.query_observation)],
                      'current_target_distance_m': float(np.linalg.norm(raw[11:14]))})
            self.pending.clear()

    def extend(self, actions):
        if self.pending or self.observation is None:
            raise ValueError('Only an empty, observed queue can accept a fresh neural chunk')
        actions = np.asarray(actions)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError('Exactly ten finite native neural actions required')
        self.pending.extend(action.copy() for action in actions)
        self.query_observation, self.consumed = self.observation.copy(), 0
        self.query_number += 1

    def popleft(self):
        self.consumed += 1
        return self.pending.popleft()

    def __bool__(self):
        return bool(self.pending)

    def __len__(self):
        return len(self.pending)
