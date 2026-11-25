"""Admit exact on-policy corrections ONLY for canonical-EEF VLA learners.

This does not reclassify native joint parity failures, change reset roles or
silently admit data into an existing fit. All40 attempts remain preserved;
the two completed student prefixes contain no corrective labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.scene64_conditioning import validate_scene


def admission(candidate, strict, output):
    if output.exists(): raise FileExistsError("Immutable corrective admission")
    rows = json.loads((candidate / "index.json").read_text()); plan = json.loads((candidate / "plan.json").read_text())
    proof = json.loads(strict.read_text())
    checked = {r["file"]: r for r in proof["rows"] if r["mode"] == "canonical"}
    if len(rows) != 40 or [r["source_file"] for r in rows] != plan["selected_files"]:
        raise ValueError("All40 original predeclared TRAIN attempts required")
    allowed = []; preserved = []; frames = 0
    for row in rows:
        if not row.get("file"):
            trace = row["failed_prefix_controls_preserved"]
            if not trace or not trace[-1]["strict_success"] or trace[-1].get("safety_violation"):
                raise ValueError("Unlabeled failed/unsafe prefix needs analysis before this admission")
            preserved.append({"source_file": row["source_file"], "actual_student_completed_controls": len(trace),
                "student_early_strict_success": True, "corrective_labels_created": False})
            continue
        path = candidate / row["file"]
        if not checked[row["file"]]["passed"] or sha256_file(path) != row["payload_sha256"]:
            raise ValueError("Every corrective trajectory must exactly replay")
        if not row["success"] or row.get("safety_violation") or not row["correction_labels_only"]:
            raise ValueError("Unsafe/failed/prefix labels cannot be corrective targets")
        with np.load(path, allow_pickle=False) as data:
            n = row["steps"]; raw = data["audit_pre_action_PRIVILEGED_state64"]
            if raw.shape != (n,64) or data["canonical_actions"].shape != (n,7): raise ValueError("Timeline shape differs")
            for state in raw: validate_scene(state)
            np.testing.assert_allclose(raw[:,:6], data["joint_positions"], rtol=0, atol=1e-6)
            np.testing.assert_allclose(raw[:,6], data["gripper_closure"], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(raw[:,:18], data["audit_pre_action_PRIVILEGED_state18"])
            np.testing.assert_array_equal(raw[0], np.asarray(row["initial_scene64"],np.float32))
            if n > 1:
                np.testing.assert_array_equal(raw[1:,17].astype(bool), data["next_verified_grasp"][:-1])
                np.testing.assert_array_equal(raw[1:,10], data["next_requested_gripper_closure"][:-1])
                np.testing.assert_allclose(data["joint_positions"][1:], data["next_observed_joint_positions"][:-1], rtol=0, atol=1e-6)
            if str(data["prompt"].item()) != row["task"]["prompt"]: raise ValueError("Original requested instruction changed")
            for camera in ("front", "wrist"):
                if data[camera+"_rgb"].shape != (n,384,384,3) or data[camera+"_rgb"].dtype != np.uint8:
                    raise ValueError("Raw model RGB shape/type differs")
            if not bool(data["next_raw_success"][-1]): raise ValueError("Missing original task success")
        allowed.append({"file": row["file"], "frames": n, "payload_sha256": row["payload_sha256"]}); frames += n
    if len(allowed) != 38 or frames != 1479 or len(preserved) != 2: raise ValueError("Correction coverage differs")
    result = {"passed": True, "scope": "canonical_EEF_exact_correction_input_action_admission_NOT_model_SR",
        "allowed_models": ["smolvla", "openvla", "cogact"], "candidate": str(candidate.resolve()),
        "frames": frames, "rows": allowed, "preserved_unlabeled_attempts": preserved,
        "native_joint_training_qualified": False, "validation_development_final_used": False,
        "dependencies": {str(p.resolve()): sha256_file(p) for p in (candidate / "index.json", candidate / "plan.json",
            strict, Path(__file__), Path(__file__).with_name("scene64_conditioning.py"))}}
    atomic_json(output, result); print(json.dumps({k:v for k,v in result.items() if k not in {"rows","dependencies"}}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--strict-replay", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); admission(a.candidate, a.strict_replay, a.output)
