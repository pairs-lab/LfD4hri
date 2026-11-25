"""TRAIN-only SimState student prefixes, then supervised expert corrections.

Never an evaluation fallback: four preselected scenes, all failed attempts
retained, no training promotion. Capture18 causal features before each LABEL.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.collect_onpolicy_handoff_pilot import StudentPrefixReset
from VLA.common.provenance import sha256_file
from VLA.simstate_feedback import SCHEMA, feedback_state
from VLA.smolvla.simstate_adapter import SimStateSmolVLA


class CapturedPrefixReset(StudentPrefixReset):
    def reset(self, task):
        observation = super().reset(task)
        self.causal_states = []
        return observation

    def step(self, action):
        # No stepPhysics/teleport or state change: observation is measured at
        # the SAME physical instant as the supervised action's sensor frame.
        self.causal_states.append(feedback_state(self.env, self.env.observe()))
        return super().step(action)


def main():
    import torch
    import VLA.collect_onpolicy_handoff_pilot as collector
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3000*1024**2/total)
    collector.SmolVLAPolicyAdapter = SimStateSmolVLA
    collector.StudentPrefixReset = CapturedPrefixReset
    original = collector.collect_episode
    def collect(wrapper, task, **kwargs):
        arrays, metadata = original(wrapper, task, **kwargs)
        prefix = wrapper if isinstance(wrapper, CapturedPrefixReset) else wrapper.env
        values = np.asarray(prefix.causal_states, dtype=np.float32)
        if values.shape != (len(arrays["canonical_actions"]), 18) or not np.isfinite(values).all():
            raise ValueError("Before-label privileged timeline does not match expert observations")
        arrays["audit_pre_action_PRIVILEGED_state18"] = values
        metadata.update(student_privileged_model_inputs=True, causal_input_schema=SCHEMA,
            causal_input_capture="same physical pre-action instant; teacher phase/action never included")
        return arrays, metadata
    collector.collect_episode = collect
    collector.main()
    # The reused pilot retains its candidate-only/no-promotion scope. Add the
    # actual student interface to every receipt, never relabel teacher SR.
    for name in ("plan.json", "audit.json"):
        path = args.output/name
        data = json.loads(path.read_text())
        data.update(student_privileged_model_inputs=True, student_policy_class="SimStateSmolVLA",
            simstate_schema=SCHEMA, adapter_only_override_NOT_controller_override=True,
            wrapper_sha256=sha256_file(Path(__file__)), teacher_is_DATA_LABELS_only=True)
        path.write_text(json.dumps(data, indent=2)+"\n")


if __name__ == "__main__":
    main()
