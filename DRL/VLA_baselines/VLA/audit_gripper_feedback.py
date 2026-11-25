"""Compare real gripper joint feedback for empty/held states; audit labels only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.audit_grasp_recovery import FailedPrefixReset
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def sensor_record(env, label):
    contacts = p.getContactPoints(bodyA=env.gripper.body)
    return {"label_audit_only": label, "commanded_closure": env.normalized_gripper_closure(),
        "actual_motor_joint_angle_rad": float(p.getJointState(env.gripper.body, 1)[0]),
        "pad_COM_distance_m_NOT_aperture": env.gripper.width(),
        "contact_count_all_bodies": len(contacts),
        "normal_contact_force_sum_N": float(sum(c[9] for c in contacts)),
        "actually_holding_requested_object_audit_only": env.gripper.grasped_object == env.pick_id}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads((args.source / "index.json").read_text())
    result = []
    with audit_environment(args.source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.25, use_semantic_success=True) as env:
        for folder in sorted(args.corrections.glob("prefix*")):
            metadata = json.loads((folder / "metadata.json").read_text())
            row = next(r for r in rows if r["file"] == metadata["source_failed_episode"])
            reset_source_episode(env, args.source, row)
            opened = sensor_record(env, "empty_open")
            proxy = FailedPrefixReset(env, args.source, row, metadata["recovery_start_actions_not_training_labels"])
            from VLA.common.tasks import TaskSpec
            proxy.reset(TaskSpec.from_dict(row["task"]))
            empty = sensor_record(env, "empty_closed_after_learned_failure")
            with np.load(folder / "correction.npz") as data:
                for action in data["canonical_actions"]:
                    _, _, done, info = env.step(action)
            held = sensor_record(env, "held_after_teacher_correction")
            if empty["actually_holding_requested_object_audit_only"] or not held["actually_holding_requested_object_audit_only"]:
                raise ValueError("Sensor comparison did not reproduce empty versus held states")
            result.append({"prefix": folder.name, "samples": [opened, empty, held],
                "command_signal_aliases_empty_and_held": empty["commanded_closure"] == held["commanded_closure"],
                "actual_motor_angle_difference_rad": abs(empty["actual_motor_joint_angle_rad"] - held["actual_motor_joint_angle_rad"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "scope": "Measured mechanical feedback, NOT proof of sensor classifier reliability",
        "records": result, "policy_state_changed": False, "target_id_used_only_for_audit_label": True,
        "width_is_pad_COM_separation_not_calibrated_inner_aperture": True, "final_test_used": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
