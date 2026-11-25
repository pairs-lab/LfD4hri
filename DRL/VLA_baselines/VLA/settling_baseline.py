"""OFFLINE separate-process matched-time object settling reference, no fallback."""
import numpy as np
import pybullet as p

from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment


def pose_difference(actual, reference):
    names = set(actual)
    if names != set(reference) or not names:
        raise ValueError("Baseline must include every physical scene object")
    differences = {}
    for name in sorted(names):
        first, second = actual[name], reference[name]
        a, b = np.asarray(first[1], float), np.asarray(second[1], float)
        if (a.shape != (4,) or b.shape != (4,) or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9
                or not np.isfinite([*first[0], *second[0], *a, *b]).all()):
            raise ValueError("Invalid physical baseline pose")
        cosine = abs(float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))))
        differences[name] = {"translation_m": float(np.linalg.norm(np.asarray(first[0])-second[0])),
                             "rotation_rad": float(2*np.arccos(np.clip(cosine, 0, 1)))}
    return differences


def natural_settling_check(actual, reference):
    differences = pose_difference(actual, reference)
    passed = all(v["translation_m"] <= 1e-5 and v["rotation_rad"] <= 1e-4 for v in differences.values())
    return {"passed": passed, "baseline_kind": "separate_simulator_zero_mount_motion_matched_control_ticks",
        "translation_tolerance_m": 1e-5, "rotation_tolerance_rad": 1e-4,
        "object_differences": differences, "robot_specific_object_displacement": not passed,
        "runtime_safety_or_success_changed": False}


def baseline_object_poses(arguments):
    # MUST execute in a spawned child process: PyBullet calls in this codebase
    # use the process-default client, so nested environments are unsafe.
    source, task_dict, gripper_commands = arguments
    with audit_environment(source, image_size=32, required_success_steps=1,
                           use_semantic_success=False, render_observations=False) as env:
        env.reset(TaskSpec.from_dict(task_dict))
        for command in gripper_commands:
            if command not in (-1, 1):
                raise ValueError("Baseline requires exact original OPEN/CLOSE commands")
            _, _, done, info = env.step(np.asarray([0, 0, 0, 0, 0, 0, command], np.float32))
            if (done or info.get("safety_violation") or env.gripper.grasped_object is not None
                    or p.getContactPoints(bodyA=env.gripper.body)):
                raise RuntimeError("No-motion baseline touched/held/terminated; no reference accepted")
        return {name: p.getBasePositionAndOrientation(body) for name, body in env.object_ids.items()}
