"""Exhaustive tail errors hidden by mean FK action-conversion metrics.

No neural weights, controls or thresholds are changed. FK mismatch can be IK
approximation; actual closed-loop replay is needed to establish its impact.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from VLA.common.environment import ROBOT_SPECS
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint


def error_metrics(actual, expected):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != (7,) or expected.shape != (7,) or not np.isfinite([actual, expected]).all():
        raise ValueError("Requires finite seven-channel actions")
    error = actual-expected
    return {"translation_max_abs_m": float(np.abs(error[:3]).max()),
            "translation_l2_m": float(np.linalg.norm(error[:3])),
            "rotation_max_abs_rad": float(np.abs(error[3:6]).max()),
            "gripper_abs_error": float(abs(error[6]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    state = source/"collection_state.json"
    if state.exists() and not json.loads(state.read_text())["status"].startswith("completed"):
        raise ValueError("Wait for immutable complete collection")
    index = json.loads((source/"index.json").read_text())
    hashes = {str(q): sha256_file(q) for q in (source/"index.json", source/"dataset_manifest.json", Path(__file__))}
    payload = source_payload_fingerprint(source, index)
    spec = ROBOT_SPECS["ur5e"]
    client = p.connect(p.DIRECT)
    rows, count = [], Counter()
    try:
        robot = p.loadURDF(str(spec.urdf), (0, 0, 0), useFixedBase=True, physicsClientId=client)
        def fk(joints):
            for joint, value in zip(spec.arm_joint_ids, joints, strict=True):
                p.resetJointState(robot, joint, float(value), physicsClientId=client)
            link = p.getLinkState(robot, spec.mount_link, computeForwardKinematics=True, physicsClientId=client)
            return np.asarray(link[4]), link[5]
        for item in index:
            with np.load(source/item["file"], allow_pickle=False) as data:
                held = data["next_grasped_object"]
                for frame, (joints, targets, closure, expected) in enumerate(zip(data["joint_positions"],
                        data["next_joint_targets"], native_gripper_closure(data), data["canonical_actions"], strict=True)):
                    current = fk((np.asarray(joints, np.float32)+np.pi)%(2*np.pi)-np.pi)
                    target = fk((np.asarray(targets, np.float32)+np.pi)%(2*np.pi)-np.pi)
                    inverse = p.invertTransform((0, 0, 0), current[1])[1]
                    relative = p.multiplyTransforms((0, 0, 0), inverse, (0, 0, 0), target[1])[1]
                    rotation = (np.asarray(p.getEulerFromQuaternion(relative))+np.pi)%(2*np.pi)-np.pi
                    actual = np.r_[np.clip(target[0]-current[0], -.05, .05),
                                   np.clip(rotation, -.25, .25), 2*float(closure)-1]
                    metrics = error_metrics(actual, expected)
                    count["frames"] += 1
                    count["translation_over_2p5mm_frames"] += metrics["translation_max_abs_m"] > .0025
                    count["rotation_over_3mrad_frames"] += metrics["rotation_max_abs_rad"] > .003
                    count["gripper_nonzero_error_frames"] += metrics["gripper_abs_error"] != 0
                    rows.append({"file": item["file"], "task": item["task"]["name"],
                                 "target": item["task"]["pick_name"], "frame": frame,
                                 "correct_target_held_before_audit_only": bool(frame and held[frame-1]),
                                 "correct_target_held_after_audit_only": bool(held[frame]),
                                 "expected_canonical_audit_only": expected.tolist(),
                                 "reconstructed_native_audit_only": actual.tolist(), **metrics})
    finally:
        p.disconnect(client)
    worst = sorted(rows, key=lambda r:r["translation_max_abs_m"], reverse=True)[:20]
    selected = [max((r for r in rows if r["task"] == task), key=lambda r:r["translation_max_abs_m"])
                for task in ("touch", "pick", "move", "pick_place")]
    output.mkdir(parents=True)
    sheet = Image.new("RGB", (768, 408*4), "black")
    for number, row in enumerate(selected):
        with np.load(source/row["file"], allow_pickle=False) as data:
            for k, cam in enumerate(("front", "wrist")):
                rgb = Image.fromarray(data[cam+"_rgb"][row["frame"]])
                rgb.save(output/(row["task"]+"_worst_"+cam+".png"))
                sheet.paste(rgb.resize((384, 384)), (384*k, 408*number+24))
        ImageDraw.Draw(sheet).text((3, 408*number+4),
            f"{row['file']} t{row['frame']} FK max error {row['translation_max_abs_m']*1000:.2f}mm", fill="white")
    sheet.save(output/"worst_fk_source_observations.png")
    verify_source_payload_fingerprint(payload, source, index)
    if any(sha256_file(Path(q)) != h for q,h in hashes.items()):
        raise RuntimeError("Source/diagnostic changed")
    report = {"scope": "Full-source native FK error tails DIAGNOSTIC, not actual control replay/training permission/learned SR",
              "source": str(source), "hashes": hashes, "source_payload_fingerprint": payload,
              "pybullet_binary": p.__file__, "pybullet_binary_sha256": sha256_file(Path(p.__file__)),
              "counts": dict(count), "translation_max_abs_m": max(r["translation_max_abs_m"] for r in rows),
              "rotation_max_abs_rad": max(r["rotation_max_abs_rad"] for r in rows),
              "worst_translation_rows": worst, "exported_samples": selected,
              "diagnostic_outliers": [r for r in rows if r["translation_max_abs_m"] > .0025 or r["rotation_max_abs_rad"] > .003],
              "thresholds_are_per_frame_diagnostics_not_new_acceptance_gates": True,
              "training_authorized": False, "policy_inputs_changed": False, "final_test_used": False}
    (output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("scope", "counts", "translation_max_abs_m", "rotation_max_abs_rad")}))


if __name__ == "__main__":
    main()
