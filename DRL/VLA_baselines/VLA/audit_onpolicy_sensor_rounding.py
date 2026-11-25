"""Quantify strict RGB replay failures using the EXISTING source tolerance.

No source pixels/labels, physics or scorer is edited; no new neural trial.
An independent supplemental proof never overwrites the strict failed report.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.cross_vla120_trial import atomic_json
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import reset_observation_errors, validate_reset_observation
from VLA.simstate_feedback import feedback_state
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


def correction_environment(env, task_name):
    # Use the SAME expert-data termination as collection and strict replay,
    # not the ordinary raw distance condition's earlier first hit.
    return NearSurfaceTouchCollection(env,None,None) if task_name == "touch" else env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve every supplemental replay")
    args.output.mkdir(parents=True)
    plan = json.loads((args.candidate/"plan.json").read_text())
    rows = json.loads((args.candidate/"index.json").read_text())
    strict_path = args.candidate/"full_replay/audit.json"
    strict = json.loads(strict_path.read_text())
    native = [r for r in strict["rows"] if r["mode"] == "native_joint"]
    if len(rows) != 4 or len(native) != 4 or any(not r["passed"] for r in native):
        raise ValueError("All original native paths must already pass")
    source = Path(plan["source"])
    original = {r["file"]:r for r in json.loads((source/"index.json").read_text())}
    reports = []
    for row in rows:
        reference_path = args.candidate/row["file"]
        digest = sha256_file(reference_path)
        with np.load(reference_path,allow_pickle=False) as archive:
            reference = {k:archive[k] for k in archive.files}
        result = {"file":row["file"],"passed":False,"error":None,"source_payload_sha256":digest,
            "frames_checked":0,"causal_state18_max_abs_error":0.,"front_changed_pixel_count":0,
            "wrist_changed_pixel_count":0,"maxima":{},"all_segmentation_masks_exact":True}
        env = audit_environment(source,robot="ur5e",image_size=384,control_hz=10,seed=0,
            required_success_steps=1,use_semantic_success=False)
        correction = correction_environment(env,row["task"]["name"])
        try:
            reset_source_episode(env,source,original[row["source_file"]])
            for action in row["recovery_start_actions_not_training_labels"]:
                _,_,done,info = env.step(np.asarray(action,np.float32))
                if done or info.get("safety_violation"):
                    raise ValueError("Original student prefix no longer replays")
            env.step_count,env.success_streak,env._last_success = 0,0,False
            if env.is_strict_success() or _interaction_audit(env)["held_object"] is not None:
                raise ValueError("Invalid initial physical replay state")
            for frame, action in enumerate(reference["canonical_actions"]):
                obs = env.observe()
                errors = reset_observation_errors(obs,env.normalized_gripper_closure(),
                    {k:reference[k][frame:frame+1] for k in ("joint_positions","gripper_closure","front_rgb","wrist_rgb")})
                gate = validate_reset_observation(errors)
                result["existing_source_tolerance"] = gate["limits"]
                for key,value in errors.items():
                    result["maxima"][key] = max(value,result["maxima"].get(key,0))
                state_error = float(np.max(np.abs(feedback_state(env,obs)-reference["audit_pre_action_PRIVILEGED_state18"][frame])))
                result["causal_state18_max_abs_error"] = max(state_error,result["causal_state18_max_abs_error"])
                if state_error > 1e-6:
                    raise ValueError("Causal state differs: never excuse physics as renderer rounding")
                for camera in ("front","wrist"):
                    actual = getattr(obs,camera+"_rgb"); expected = reference[camera+"_rgb"][frame]
                    result[camera+"_changed_pixel_count"] += int(np.count_nonzero(np.any(actual!=expected,axis=-1)))
                    for role in ("pick","place"):
                        if not np.array_equal(env.target_mask(camera,role),reference[camera+"_"+role+"_mask"][frame]):
                            result["all_segmentation_masks_exact"] = False
                            raise ValueError("Object silhouette/identity changed")
                if (_interaction_audit(env)["held_object"] or "") != str(reference["audit_any_held_object_before"][frame]):
                    raise ValueError("Held identity before label changed")
                _,_,done,info = correction.step(action)
                if info.get("safety_violation"):
                    raise ValueError("Physical safety changed")
                if (_interaction_audit(env)["held_object"] or "") != str(reference["audit_any_held_object_after"][frame]):
                    raise ValueError("Held identity after label changed")
                result["frames_checked"] += 1
                if done:
                    if frame != row["steps"]-1:
                        raise ValueError("Replay terminated at another frame")
                    break
            if not info["strict_raw_success"] or result["frames_checked"] != row["steps"]:
                raise ValueError("Full original correction/strict outcome missing")
            if row["task"]["name"] == "touch" and not correction._last_success:
                raise ValueError("Original near-surface touch quality missing")
            result["passed"] = True
        except Exception as error:
            result["error"] = str(error)
        finally:
            env.close()
        if sha256_file(reference_path) != digest:
            raise ValueError("Candidate source was altered")
        reports.append(result)
        atomic_json(args.output/"progress.json",{"rows":reports})
        print(row["task"]["name"],result["passed"],result["front_changed_pixel_count"],result["wrist_changed_pixel_count"],flush=True)
    proof = {"passed":all(r["passed"] for r in reports),"rows":reports,
        "scope":"supplemental_existing_TinyRenderer_tolerance_and_exact_physics_NOT_training_promotion",
        "strict_failed_report_sha256":sha256_file(strict_path),"new_neural_trials":0,
        "source_images_labels_modified":False,"physics_or_success_criteria_changed":False,
        "training_qualified":False,"tolerance_function_sha256":sha256_file(Path(__file__).with_name("evaluate_smolvla_v9_source.py"))}
    atomic_json(args.output/"audit.json",proof)
    if not proof["passed"]:
        raise ValueError("Supplemental sensor/physics replay failed")


if __name__ == "__main__":
    main()
