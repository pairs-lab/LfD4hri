"""Preserve committed scenes, replace failed controls, retain RNG cursor."""
import copy
import argparse
import json
import os
from pathlib import Path
import numpy as np

from VLA.collect_demos import collect_episode
from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.tasks import TaskSpec
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "VLA/datasets/candidates/body_v9_diverse_pilot_60_20261008"
NEW = ROOT / "VLA/datasets/candidates/body_v9_diverse_repaired_60_20261008"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=OLD)
    parser.add_argument("--output", type=Path, default=NEW)
    parser.add_argument("--recollect-tasks", default="", help="Also regenerate these task controls at exactly the same original spawn poses")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    recollect_tasks = set(filter(None, args.recollect_tasks.split(",")))
    if recollect_tasks - {"touch", "pick", "move", "pick_place"}:
        raise ValueError("Unknown task in --recollect-tasks")
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads((source/"index.json").read_text())
    state = json.loads((source/"collection_state.json").read_text())
    output.mkdir(parents=True)
    profile = json.loads((ROOT/"VLA/profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())
    merged, repairs = [], []
    with BodyGraspEnvironment(profile, image_size=384, required_success_steps=10) as env:
        for row in rows:
            if row["success"] and row["task"]["name"] not in recollect_tasks:
                os.link(source/row["file"], output/row["file"])
                merged.append(row)
                continue
            task = TaskSpec.from_dict(row["task"])
            arrays, metadata = collect_episode(env, task, policy_factory=CollisionPlannedTeacher)
            np.savez_compressed(output/row["file"], **arrays)
            metadata.update(file=row["file"], repaired_from=str(source/row["file"]),
                            repair_scope="Same task, objects and spawn poses; teacher controls changed")
            merged.append(metadata)
            repairs.append({"file":row["file"],"success":metadata["success"],"safety":metadata["safety_violation"]})
            print(json.dumps(repairs[-1]),flush=True)
    (output/"index.json").write_text(json.dumps(merged,indent=2)+"\n")
    (output/"collection_state.json").write_text(json.dumps(state,indent=2)+"\n")
    (output/"repair_provenance.json").write_text(json.dumps({"parent":str(source),"committed_episodes":len(rows),
        "recollected_tasks": sorted(recollect_tasks),
        "repairs":repairs,"rng_cursor_preserved":True,"original_modified":False,"final_test_used":False},indent=2)+"\n")
    if not all(r["success"] for r in repairs):
        raise RuntimeError("A hard case still failed; do not resume/promote this candidate")
    if (source/"dataset_manifest.json").is_file():
        manifest = copy.deepcopy(json.loads((source/"dataset_manifest.json").read_text()))
        manifest.update(candidate_only=True, repaired_from=str(source),
                        index_sha256=sha256_file(output/"index.json"),
                        gripper_asset="VLA/assets/body_grasp_v9/robotiq_2f_85",
                        native_joint_action_kind="requested_joint_position_command",
                        environment_runtime_fingerprint=environment_runtime_fingerprint(profile),
                        teacher_source_sha256=sha256_file(ROOT/"VLA/collision_planned_teacher.py"),
                        planner_source_sha256=sha256_file(ROOT/"VLA/collision_grasp_planner.py"))
        (output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
