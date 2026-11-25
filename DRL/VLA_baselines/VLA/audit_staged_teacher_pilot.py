"""ALL saved TRAIN pilot scenes, staged teacher physics only, not VLA SR."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.collect_demos import collect_episode
from VLA.collect_paired_grounding_candidate import FrozenRetargetReset
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.evaluate_smolvla_v9_source import verify_source_audit
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher


def valid_initial_staged_action(action):
    """Safe pre-target vertical hover transition, not blind diagonal descent."""
    action = np.asarray(action)
    return bool(action.shape == (7,) and np.isfinite(action).all()
                and np.array_equal(action[[0, 1, 3, 4, 5]], np.zeros(5))
                and abs(float(action[2])) <= .02000001 and action[6] == -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace-report", "source-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--closed-hover-recovery", action="store_true")
    parser.add_argument("--obstacle-aware-touch", action="store_true",
                        help="Offline-only whole-gripper neighbour-clear touch proposal")
    parser.add_argument("--open-verified-touch", action="store_true",
                        help="Additionally wait for measured OPEN before selecting the finger")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    trace = json.loads(args.trace_report.read_text())
    if trace.get("split_mode") != "train_diagnostic" or trace.get("final_test_used") is not False:
        raise ValueError("TRAIN-only teacher repair, no held-out labels")
    source = Path(trace["source"])
    audit, _manifest = verify_source_audit(source, args.source_audit)
    index = {r["file"]: r for r in json.loads((source/"index.json").read_text())}
    selected = [index[r["file"]] for r in trace["results"]]
    args.output.mkdir(parents=True)
    plan = {"scope": "OFFLINE_staged_teacher_physics_pilot_NOT_VLA_success_rate",
        "source": str(source.resolve()), "source_index_sha256": audit["index_sha256"],
        "selected_files": [r["file"] for r in selected], "selection": "ALL completed TRAIN diagnostic scenes",
        "no_failed_scene_skipped": True, "training_performed": False, "validation_used": False, "final_test_used": False,
        "teacher_source_sha256": sha256_file(Path(__file__).with_name("staged_acquisition_teacher.py")),
        "collector_source_sha256": sha256_file(Path(__file__)),
        "robot_camera_controller_success_changed": False}
    if args.closed_hover_recovery:
        plan.update(start_kind="mid_height_contact_free_closed_empty_v1",
                    startup_actions_are_training_labels=False,
                    near_body_slip_or_wrong_object_recovery_claimed=False)
    teacher = StagedAcquisitionTeacher
    if args.obstacle_aware_touch or args.open_verified_touch:
        from VLA.obstacle_aware_touch_teacher import ObstacleAwareTouchTeacher
        teacher = ObstacleAwareTouchTeacher
        plan["obstacle_aware_touch_teacher_sha256"] = sha256_file(Path(__file__).with_name("obstacle_aware_touch_teacher.py"))
    if args.open_verified_touch:
        from VLA.obstacle_aware_touch_teacher import OpenVerifiedTouchTeacher
        teacher = OpenVerifiedTouchTeacher
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    rows = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        for item in selected:
            proxy = FrozenRetargetReset(env, source, item)
            if args.closed_hover_recovery:
                from VLA.offline_closed_hover_recovery import ClosedHoverRecoveryReset
                proxy = ClosedHoverRecoveryReset(proxy)
            arrays, meta = collect_episode(proxy, TaskSpec.from_dict(item["task"]), policy_factory=teacher)
            meta["original_file"] = item["file"]
            if args.closed_hover_recovery:
                reference = proxy.reset(TaskSpec.from_dict(item["task"]))
                exact = (np.array_equal(arrays["front_rgb"][0], reference.front_rgb)
                    and np.array_equal(arrays["wrist_rgb"][0], reference.wrist_rgb)
                    and np.array_equal(arrays["joint_positions"][0], reference.joint_positions)
                    and float(arrays["gripper_closure"][0]) == env.normalized_gripper_closure())
                meta["closed_hover_start_evidence"] = proxy.startup_evidence
                meta["original_reset_pixels_feedback_exact"] = False
                initial_pass = float(arrays["gripper_closure"][0]) >= .95 and arrays["canonical_actions"][0, 6] == -1
                initial_pass &= not bool(arrays["next_grasped_object"][0])
                initial_pass &= item["task"]["name"] == "touch" or bool(arrays["next_verified_grasp"][-1])
            else:
                with np.load(source/item["file"], allow_pickle=False) as reference:
                    exact = all(np.array_equal(arrays[key][0], reference[key][0]) for key in
                                ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure"))
                meta["original_reset_pixels_feedback_exact"] = exact
                initial_pass = valid_initial_staged_action(arrays["canonical_actions"][0])
            meta["replayed_start_pixels_feedback_exact"] = exact
            meta["first_action_descent_m"] = float(arrays["canonical_actions"][0, 2])
            meta["no_downward_action_at_reset"] = meta["first_action_descent_m"] >= 0
            meta["initial_staged_or_recovery_action_check_passed"] = bool(initial_pass)
            folder = args.output/Path(item["file"]).stem
            folder.mkdir()
            np.savez_compressed(folder/"teacher_candidate.npz", **arrays)
            for frame in sorted(set([0, min(10, len(arrays["front_rgb"])-1), len(arrays["front_rgb"])-1])):
                for view in ("front", "wrist"):
                    Image.fromarray(arrays[view+"_rgb"][frame]).save(folder/f"frame{frame:03d}_{view}.png")
            rows.append(meta)
            (args.output/"progress.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(item["file"], "teacher_success", meta["success"], "steps", meta["steps"], "safety", meta["safety_violation"], flush=True)
            if (not meta["success"] or meta["safety_violation"] or not exact
                    or not meta["initial_staged_or_recovery_action_check_passed"]):
                (args.output/"status.json").write_text(json.dumps({"status": "stopped_preserved_teacher_failure_NOT_training", "failed_file": item["file"]}, indent=2)+"\n")
                raise RuntimeError("Preserve the exact failed teacher case; do not train or skip it")
    verify_source_audit(source, args.source_audit)
    (args.output/"results.json").write_text(json.dumps({**plan, "passed": True, "trials": len(rows),
        "teacher_successes": sum(r["success"] for r in rows), "rows": rows}, indent=2)+"\n")


if __name__ == "__main__":
    main()
