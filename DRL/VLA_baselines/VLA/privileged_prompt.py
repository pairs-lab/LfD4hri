"""Causal simulation feedback as native VLA language input, never a controller.

An explicitly labelled alternative to state-projector inputs. All eighteen
fields come from the SAME pre-action state18 contract. Metres/radians are
rounded to 1e-4, not normalized using validation/test data. This must be used
identically by a training dataset and its deployed adapter.
"""
from dataclasses import replace
import re

import numpy as np

from VLA.simstate_feedback import SCHEMA, feedback_state

PROMPT_SCHEMA = "simstate18_native_language_feedback_v1"
MARKER = "simulation feedback before action:"
FIELD_SLICES = (("joint radians", 0, 6), ("measured closure", 6, 7),
                ("pad xyz meters", 7, 10), ("requested close", 10, 11),
                ("target center minus pad meters", 11, 14),
                ("goal minus object base meters", 14, 17),
                ("correct bilateral grasp", 17, 18))


def validate_state(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (18,) or not np.isfinite(values).all():
        raise ValueError("Exactly eighteen finite pre-action state values required")
    if not 0 <= values[6] <= 1 or not 0 <= values[10] <= 1 or values[17] not in (0, 1):
        raise ValueError("Closure/intent/grasp feedback does not satisfy the causal schema")
    return values


def augment_prompt(instruction, state):
    instruction = str(instruction).strip()
    if not instruction or MARKER in instruction.lower():
        raise ValueError("A nonempty original instruction, not an already augmented prompt, is required")
    values = validate_state(state)
    fields = []
    for label, begin, end in FIELD_SLICES:
        numbers = [f"{0.0 if abs(float(x)) < 0.00005 else float(x):.4f}" for x in values[begin:end]]
        fields.append(label + "=" + ",".join(numbers))
    # No teacher phase, future observation, success, reward, desired action or
    # IK-generated waypoint is serialized. Labels say precisely which point
    # is measured (object CENTER for approach, object BASE for destination).
    return instruction + "; " + MARKER + " " + "; ".join(fields)


def parse_feedback(prompt):
    """Audit-only inverse: proves all fields survived formatting/transport."""
    if str(prompt).lower().count(MARKER) != 1:
        raise ValueError("Missing/duplicated feedback block")
    block = str(prompt).split(MARKER, 1)[1].strip()
    fields = block.split("; ")
    if len(fields) != len(FIELD_SLICES):
        raise ValueError("Missing feedback field")
    result = []
    for field, (label, begin, end) in zip(fields, FIELD_SLICES, strict=True):
        prefix = label + "="
        if not field.startswith(prefix):
            raise ValueError("Feedback field order/name differs")
        numbers = field[len(prefix):].split(",")
        if len(numbers) != end - begin or any(not re.fullmatch(r"-?\d+\.\d{4}", n) for n in numbers):
            raise ValueError("Malformed feedback numbers")
        result.extend(map(float, numbers))
    return validate_state(result)


class PrivilegedPromptPolicy:
    """Decorate a REAL neural adapter, changing only its observed prompt.

    The adapter's output is returned unchanged. This wrapper cannot create
    actions, retry grasps, invoke IK, change physics or insert an oracle. Raw
    state/prompt are retained for auditing the exact input to the network.
    """
    comparable = False

    def __init__(self, env, native_policy):
        if getattr(native_policy, "name", "") == "oracle":
            raise ValueError("An oracle is not a VLA")
        self.env, self.native_policy = env, native_policy
        self.input_modalities = tuple(getattr(native_policy, "input_modalities", ())) + (PROMPT_SCHEMA,)
        self.last_state = None
        self.last_prompt = None

    @property
    def server_metadata(self):
        return getattr(self.native_policy, "server_metadata", {})

    def reset(self, task):
        self.last_state = self.last_prompt = None
        return self.native_policy.reset(task)

    def predict(self, observation):
        self.last_state = feedback_state(self.env, observation)
        self.last_prompt = augment_prompt(observation.prompt, self.last_state)
        return self.native_policy.predict(replace(observation, prompt=self.last_prompt))

    def close(self):
        return self.native_policy.close()
