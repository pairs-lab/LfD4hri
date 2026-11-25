"""Immutable camera-only derivative via exact stored-action physics replay.

Never changes labels, home, wrist, objects or physics. Rejects any state or
wrist-pixel mismatch; this is dataset generation, NEVER a policy fallback.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from VLA.audit_counterfactual_collection import CanonicalSceneResetEnvironment
from VLA.common.observation_profile import validate_profile, profile_hash
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue only a matching interrupted derivative, preserving completed NPZs")
    parser.add_argument("--native-command-labels", action="store_true",
                        help="Keep feedback as state, export actual requested joint commands as native action labels")
    args = parser.parse_args()
    source = args.source.resolve()
    old_manifest = json.loads((source/"dataset_manifest.json").read_text())
    old_paired = json.loads((source/"audit.json").read_text())
    profile = validate_profile(json.loads(args.profile.read_text()))
    old_profile = old_manifest["observation_profile"]
    for key in ("robot", "home_joints", "wrist_frame"):
        if profile[key] != old_profile[key]:
            raise ValueError("Camera derivative cannot alter "+key)
    if profile["cameras"]["wrist"] != old_profile["cameras"]["wrist"]:
        raise ValueError("Wrist must remain exact for this camera-only derivative")
    for key in ("gripper_controller", "gripper_state_contract", "native_gripper_action_kind",
                "derived_gripper_urdf_sha256", "mimic_coupling", "grasp_candidate_scope"):
        if profile.get(key) != old_profile.get(key):
            raise ValueError("Camera derivative cannot alter "+key)
    if not old_paired["passed"]:
        raise ValueError("Source paired audit failed")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    original = Path(old_paired["source"])
    canonical = {row["file"]: row for row in json.loads((original/"index.json").read_text())}
    old_index = json.loads((source/"index.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    identity = profile_hash(profile)
    maxima = {"state_max_abs_error": 0., "next_joints_max_abs_error": 0., "wrist_pixel_error": 0,
              "wrist_changed_pixel_fraction": 0., "next_distance_max_abs_error": 0.}
    new_index, first_images, differences = [], {}, []
    if args.resume:
        new_index = json.loads((args.output/"index.json").read_text())
        if any(row["observation_profile_sha256"] != identity or Path(row["camera_only_derivative_of"]).parent != source for row in new_index):
            raise ValueError("Cannot resume mixed profile/source derivative")
    completed = {row["file"] for row in new_index}
    with environment_class_for_profile(profile)(profile, image_size=384, seed=20261007, required_success_steps=1) as env:
        for number, item in enumerate(old_index):
            if item["file"] in completed:
                if item["variant"] == 0:
                    with np.load(args.output/item["file"], allow_pickle=False) as saved:
                        first_images[item["paired_scene_source"]] = saved["front_rgb"][0]
                continue
            initial = TaskSpec.from_dict(canonical[item["paired_scene_source"]]["task"])
            task = TaskSpec.from_dict(item["task"])
            wrapper = CanonicalSceneResetEnvironment(env, initial, item["recovery_start_actions_not_training_labels"])
            observation = wrapper.reset(task)
            with np.load(source/item["file"], allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            images, wrist_images, pick_masks, place_masks, robot_masks, command_targets = [], [], [], [], [], []
            if "overview" in profile["cameras"]:
                env.render("overview")
                arrays["overview_reset_robot_mask"] = env.robot_mask("overview")
            done, info = False, None
            for frame, action in enumerate(arrays["canonical_actions"]):
                for key in ("joint_positions", "ee_position", "ee_rotation_6d"):
                    error = float(np.max(np.abs(getattr(observation, key)-arrays[key][frame])))
                    maxima["state_max_abs_error"] = max(maxima["state_max_abs_error"], error)
                    if error > 1e-6:
                        raise RuntimeError(f"Physics/state mismatch {item['file']}:{frame}:{key}:{error}")
                pixel_error = int(np.max(np.abs(observation.wrist_rgb.astype(int)-arrays["wrist_rgb"][frame].astype(int))))
                changed_fraction = float(np.any(observation.wrist_rgb != arrays["wrist_rgb"][frame], axis=-1).mean())
                maxima["wrist_pixel_error"] = max(maxima["wrist_pixel_error"], pixel_error)
                maxima["wrist_changed_pixel_fraction"] = max(maxima["wrist_changed_pixel_fraction"], changed_fraction)
                # An extra audit/video camera can cause a 1-LSB TinyRenderer
                # rounding change at a few pixels, despite EXACT world poses.
                # This was reproduced separately (one channel of one pixel,
                # world/state/distance delta zero). No geometry/mask changes
                # permitted; store the ACTUAL new pixels, never stale copies.
                if pixel_error > 1 or changed_fraction > .001 or not np.array_equal(env.target_mask("wrist"), arrays["wrist_pick_mask"][frame]):
                    raise RuntimeError(f"Unchanged wrist pixels/masks differ: {item['file']}:{frame}")
                if observation.gripper_open != arrays["gripper_open"][frame]:
                    raise RuntimeError("Gripper state changed")
                images.append(observation.front_rgb)
                wrist_images.append(observation.wrist_rgb)
                pick_masks.append(env.target_mask("front"))
                place_masks.append(env.target_mask("front", "place"))
                robot_masks.append(env.robot_mask("front"))
                observation, _, done, info = env.step(action)
                command_targets.append(env.requested_arm_targets.copy())
                feedback = arrays.get("next_observed_joint_positions", arrays["next_joint_targets"])
                error = float(np.max(np.abs(observation.joint_positions-feedback[frame])))
                maxima["next_joints_max_abs_error"] = max(maxima["next_joints_max_abs_error"], error)
                if error > 1e-6 or bool(info["raw_success"]) != bool(arrays["next_raw_success"][frame]):
                    raise RuntimeError(f"Post-action labels differ: {item['file']}:{frame}")
                for field in ("ee_object_distance", "object_goal_distance"):
                    distance_error = abs(float(info[field])-float(arrays["next_"+field][frame]))
                    maxima["next_distance_max_abs_error"] = max(maxima["next_distance_max_abs_error"], distance_error)
                    if distance_error > 1e-6:
                        raise RuntimeError("Object/goal physical trajectory changed")
                if done and frame != len(arrays["canonical_actions"])-1:
                    raise RuntimeError("Replay terminated early")
            if not done or not info["success"]:
                raise RuntimeError("Replay teacher outcome differs")
            arrays.update(front_rgb=np.asarray(images), front_pick_mask=np.asarray(pick_masks),
                          front_place_mask=np.asarray(place_masks), front_robot_mask=np.asarray(robot_masks),
                          wrist_rgb=np.asarray(wrist_images))
            if args.native_command_labels:
                arrays["next_observed_joint_positions"] = arrays.get("next_observed_joint_positions", arrays["next_joint_targets"]).copy()
                arrays["next_joint_targets"] = np.asarray(command_targets, dtype=np.float32)
            np.savez_compressed(args.output/item["file"], **arrays)
            metadata = copy.deepcopy(item)
            metadata["observation_profile_sha256"] = identity
            metadata["camera_only_derivative_of"] = str(source/item["file"])
            if args.native_command_labels:
                metadata["native_joint_action_kind"] = "requested_joint_position_command"
            new_index.append(metadata)
            pair_key = item["paired_scene_source"]
            if item["variant"] == 0:
                first_images[pair_key] = images[0]
            else:
                error = int(np.max(np.abs(first_images[pair_key].astype(int)-images[0].astype(int))))
                if error:
                    raise RuntimeError("Re-rendered paired initial front images differ")
                differences.append(error)
            (args.output/"index.json").write_text(json.dumps(new_index, indent=2)+"\n")
            (args.output/"collection_state.json").write_text(json.dumps({"status": "in_progress",
                "completed_episodes": len(new_index), "planned_episodes": len(old_index),
                "observation_profile_sha256": identity, "production_dataset_changed": False}, indent=2)+"\n")
            print(f"Re-rendered {number+1}/{len(old_index)}: {item['file']}", flush=True)
    report = {**old_paired, "observation_profile": profile, "observation_profile_sha256": identity,
              "camera_only_derivative_of": str(source), "derivative_source_index_sha256": sha256_file(source/"index.json"),
              "derivative_source_manifest_sha256": sha256_file(source/"dataset_manifest.json"),
              "exact_replay_errors": maxima, "new_paired_front_max_pixel_error": max(differences),
              "wrist_rgb_rounding_gate": {"max_channel_lsb": 1, "max_changed_pixel_fraction": .001,
                                          "pick_masks_exact": True, "actual_new_pixels_saved": True},
              "numeric_actions_states_labels_unchanged": not args.native_command_labels,
              "canonical_actions_and_measured_states_unchanged": True,
              "native_joint_action_kind": "requested_joint_position_command" if args.native_command_labels else old_manifest.get("native_joint_action_kind", "measured_next_joint_position")}
    (args.output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"collection_state.json").write_text(json.dumps({"status": "completed_candidate_audit_passed",
        "completed_episodes": len(new_index), "planned_episodes": len(old_index),
        "observation_profile_sha256": identity, "production_dataset_changed": False,
        "not_qualified_for_training_yet": True}, indent=2)+"\n")
    print(json.dumps(maxima), flush=True)


if __name__ == "__main__":
    main()
