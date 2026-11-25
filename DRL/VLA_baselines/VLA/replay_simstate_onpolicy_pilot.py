"""Replay every correction path and verify every captured causal state18.

Reuses the unchanged strict canonical-image/native-action replay; never
reruns student inference or promotes a candidate into training.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.simstate_feedback import feedback_state


def main():
    import VLA.replay_onpolicy_handoff_pilot as replay
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    plan = json.loads((args.candidate/"plan.json").read_text())
    index = json.loads((args.candidate/"index.json").read_text())
    if plan.get("student_privileged_model_inputs") is not True or plan["prefix_controls"] != 20:
        raise ValueError("Exact declared privileged20 prefix required")
    order = [(row, mode) for row in index for mode in ("canonical", "native_joint")]
    calls = 0; checked = []
    original = replay.audit_environment
    def environment(*values, **kwargs):
        nonlocal calls
        row, mode = order[calls]; calls += 1
        env = original(*values, **kwargs)
        with np.load(args.candidate/row["file"], allow_pickle=False) as archive:
            expected = archive["audit_pre_action_PRIVILEGED_state18"].copy()
        if expected.shape != (row["steps"], 18):
            raise ValueError("Invalid causal state timeline")
        record = {"file": row["file"], "mode": mode, "checked_before_label_frames": 0, "maximum_state_error": 0.}
        checked.append(record)
        step = env.step; controls = 0
        def audited_step(action):
            nonlocal controls
            if mode == "canonical" and controls >= 20:
                frame = controls-20
                actual = feedback_state(env, env.observe())
                error = float(np.max(np.abs(actual-expected[frame])))
                record["maximum_state_error"] = max(error, record["maximum_state_error"])
                record["checked_before_label_frames"] += 1
                if error > 1e-6:
                    raise ValueError("Recorded privileged input is not BEFORE-label replay physics")
            controls += 1
            return step(action)
        env.step = audited_step
        return env
    replay.audit_environment = environment
    replay.main()
    path = args.output/"audit.json"; proof = json.loads(path.read_text())
    complete = calls == 8 and all(row["checked_before_label_frames"] == index[i//2]["steps"]
        for i, row in enumerate(checked) if row["mode"] == "canonical")
    proof.update(passed=bool(proof["passed"] and complete), causal_state18_rows=checked,
        captured_before_action_feedback_verified=complete, maximum_causal_state18_error=max(r["maximum_state_error"] for r in checked),
        privileged_inputs_NOT_teacher_phase_or_future_state=True, wrapper_sha256=sha256_file(Path(__file__)),
        training_qualified=False, neural_success_rate_measured=False)
    path.write_text(json.dumps(proof, indent=2)+"\n")
    if not proof["passed"]:
        raise ValueError("Preserved correction replay failure; no source promotion")


if __name__ == "__main__":
    main()
