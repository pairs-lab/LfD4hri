"""Native binary-gripper postprocessing; no simulator target/action correction."""
import numpy as np


def native_gripper_command(value, *, convention):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Nonfinite native gripper prediction')
    value = np.clip(value, 0., 1.)
    if convention == 'xvla_closed01':
        # Official X-VLA LIBERO binary gripper threshold.
        return 1. if value > .5 else -1.
    if convention == 'adapter_open01':
        # Official Adapter normalize_gripper_action(binarize=True), followed
        # by invert_gripper_action. Exactly0.5 stays0 as in the official code.
        return float(-np.sign(2*value-1))
    raise ValueError('Unknown native gripper convention')
