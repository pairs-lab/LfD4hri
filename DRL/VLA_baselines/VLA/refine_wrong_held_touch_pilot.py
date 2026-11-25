"""Repair ONLY the pilot's touch terminal quality on its exact TRAIN reset.

Keep the original failed-quality evidence. Non-touch NPZ bytes are hard-linked
unchanged. New candidate remains unqualified, outside the active training jobs.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.collect_demos import collect_episode
from VLA.collect_wrong_held_recovery_pilot import PhysicalWrongHeldReset
from VLA.common.provenance import sha256_file
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.probe_wrong_object_recovery import WrongHeldOpeningProbe
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


class QualityWrongHeldTouch(NearSurfaceTouchCollection):
    def reset(self, task):
        return self.env.reset(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent, output = args.candidate.resolve(), args.output.resolve()
    if output.exists(): raise FileExistsError("Preserve every refinement attempt")
    plan = json.loads((parent / "plan.json").read_text())
    rows = json.loads((parent / "index.json").read_text())
    if not plan["candidate_only"] or len(rows) != 4 or any(r.get("error") for r in rows):
        raise ValueError("Exact completed four-case TRAIN pilot required")
    fixed = {str(parent / "plan.json"): sha256_file(parent / "plan.json"),
             str(parent / "index.json"): sha256_file(parent / "index.json"),
             str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    fixed.update(plan["source_hashes"])
    for row in rows: fixed[str(parent / row["file"])] = sha256_file(parent / row["file"])
    output.mkdir(parents=True)
    repaired = []
    evidence = None
    for row in rows:
        if row["task"]["name"] != "touch":
            os.link(parent / row["file"], output / row["file"])
            repaired.append(copy.deepcopy(row))
            continue
        env = audit_environment(plan["source"], robot="ur5e", image_size=384, control_hz=10, seed=0,
                                required_success_steps=1, use_semantic_success=False)
        env.dist_threshold, env.obj_threshold = recorded_distance_thresholds(row)
        physical = PhysicalWrongHeldReset(env)
        quality = QualityWrongHeldTouch(physical, None, None)
        try:
            arrays, metadata = collect_episode(quality, TaskSpec.from_dict(row["task"]), policy_factory=WrongHeldOpeningProbe)
            with np.load(parent / row["file"], allow_pickle=False) as old:
                initial = {"joint_rad": float(np.max(np.abs(arrays["joint_positions"][0] - old["joint_positions"][0]))),
                    "closure": abs(float(arrays["gripper_closure"][0] - old["gripper_closure"][0])),
                    **{camera+"_intensity": int(np.max(np.abs(arrays[camera+"_rgb"][0].astype(np.int16) - old[camera+"_rgb"][0].astype(np.int16))))
                       for camera in ("front", "wrist")}}
            gap, force = quality.terminal_geometry()
            evidence = {"strict_expert_success": bool(metadata["success"]), "surface_gap_m": gap,
                "max_normal_force_n": force, "initial_sensor_errors_against_original_pilot": initial,
                "near_surface_NOT_force_contact_claim": True, "qualified_for_training": False}
            if not metadata["success"] or gap is None or gap > .0002 or force > 5 or any(initial.values()):
                raise ValueError("Same-reset near-surface touch refinement failed: " + json.dumps(evidence))
            arrays["audit_any_held_object_before"] = np.asarray(physical.held_before)
            arrays["audit_any_held_object_after"] = np.asarray(physical.held_after)
            np.savez_compressed(output / row["file"], **arrays)
            metadata.update(file=row["file"], source_file=row["source_file"], candidate_only=True,
                recovery_start_kind=row["recovery_start_kind"], recovery_start_evidence=physical.initial_evidence,
                startup_task_not_training_instruction=physical.startup_task.to_dict(),
                recovery_start_actions_not_training_labels=physical.startup_actions,
                correction_checks={**row["correction_checks"], "same_initial_sensors": True,
                    "near_surface_gap_at_most_0point2mm_and_force_at_most5N": True},
                teacher_success_and_feedback_checks=True, near_surface_touch_quality=evidence)
            repaired.append(metadata)
            for camera in ("front", "wrist"):
                Image.fromarray(env.render(camera)).save(output / ("touch_terminal_"+camera+".png"))
        finally:
            env.close()
    if any(sha256_file(Path(path)) != h for path, h in fixed.items()):
        raise ValueError("Original pilot/source evidence changed")
    (output / "index.json").write_text(json.dumps(repaired, indent=2) + "\n")
    (output / "plan.json").write_text(json.dumps({**plan, "parent_candidate": str(parent),
        "source_hashes": fixed, "refinement": "same-reset touch <=0.2mm/<=5N, non-touch payloads unchanged"}, indent=2) + "\n")
    (output / "touch_quality.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
