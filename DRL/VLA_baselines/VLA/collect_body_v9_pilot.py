"""New strict teacher source with correct surfaces, real feedback and action intent."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

from VLA.collect_demos import collect_episode
from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.dataset_replay_context import reset_source_episode

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_grouped_train_20261008"
OUTPUT = ROOT / "VLA/datasets/candidates/body_v9_strict_pilot_20261008"
PROFILE = ROOT / "VLA/profiles/body_contact_v9_wrist65_f80_encoder.json"


class SourceResetProxy:
    def __init__(self, env, row):
        self.env, self.row = env, row
    def __getattr__(self, key):
        return getattr(self.env, key)
    def reset(self, task):
        return reset_source_episode(self.env, SOURCE, self.row)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_index = json.loads((SOURCE / "index.json").read_text())
    selected = []
    for name in ("touch", "pick", "move", "pick_place"):
        regular = [r for r in source_index if r["task"]["name"] == name and not r.get("recovery_start_actions_not_training_labels")]
        selected.extend([next(r for r in regular if r["task"]["pick_name"] == "yellow banana"), regular[0]])
    profile = json.loads(PROFILE.read_text())
    OUTPUT.mkdir(parents=True)
    index = []
    with BodyGraspEnvironment(profile, image_size=384, required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
        for number, row in enumerate(selected):
            proxy = SourceResetProxy(env, row)
            task = TaskSpec.from_dict(row["task"])
            arrays, meta = collect_episode(proxy, task, policy_factory=CollisionPlannedTeacher)
            filename = f"{task.name}_{number:03d}.npz"
            np.savez_compressed(OUTPUT / filename, **arrays)
            meta.update(file=filename, paired_scene_source=row["paired_scene_source"], variant=row["variant"],
                paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
                recovery_start_actions_not_training_labels=[], candidate_only=True,
                native_gripper_action_kind="requested_close_intent",
                gripper_state_contract="encoder_fk_aperture_closure_v1",
                safety_min_clearance_m=env.safety_min_clearance, safety_max_mimic_error_rad=env.safety_max_mimic_error)
            index.append(meta)
            (OUTPUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
            print(f"{filename}: success={meta['success']} verified={bool(arrays['next_verified_grasp'][-1])} steps={meta['steps']}", flush=True)
    canonical = json.loads((SOURCE / "audit.json").read_text())["source"]
    (OUTPUT / "audit.json").write_text(json.dumps({"source": canonical, "scope": "Reset provenance, NOT source qualification", "final_test_used": False}, indent=2) + "\n")
    manifest = {"robot": "ur5e", "robot_urdf": "asset/ur5e/ur5e.urdf", "gripper": "Robotiq 2F-85",
        "image_size": 384, "physics_hz": 240, "control_hz": 10, "required_success_steps": 1,
        "scene_profile": "rl_random_env", "native_joint_action_kind": "requested_joint_position_command",
        "gripper_state_contract": "encoder_fk_aperture_closure_v1", "native_gripper_action_kind": "requested_close_intent",
        "front_camera_role": "policy_workspace", "overview_supplied_to_policy": False,
        "observation_profile": profile, "observation_profile_sha256": profile_hash(profile), "cameras": profile["cameras"],
        "candidate_only": True, "declared_task_counts": dict(Counter(r["task"]["name"] for r in index)),
        "index_sha256": sha256_file(OUTPUT / "index.json"), "final_test_used": False,
        "source_assets_preserved": True, "purpose": "New 8-episode physical/data-contract pilot; NOT sufficient training data"}
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not all(r["success"] for r in index):
        raise RuntimeError("Strict pilot has failed episodes; audit, do not consume for training")


if __name__ == "__main__":
    main()
