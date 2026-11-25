"""Fixed-noise text/image interventions on identical source-only scenes.

This diagnostic measures action conditioning, not closed-loop task success.
No target pose, mask or teacher action is supplied to the deployed model.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.tasks import TaskSpec
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.adapter import SmolVLAPolicyAdapter
from VLA.common.provenance import sha256_file


def pair_chunks(first, second):
    length = min(len(first), len(second))
    if length < 1:
        raise ValueError("Empty action chunk")
    return np.asarray(first[:length]), np.asarray(second[:length])


def verify_explicit_source_audit(data, audit_path):
    """Bind V9 conditioning to real camera/payload and TRAIN-reset proofs."""
    from VLA.audit_training_readiness import source_reset_provenance
    from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    data = Path(data).resolve()
    audit = json.loads(Path(audit_path).read_text())
    if (not audit.get("passed") or audit.get("failures")
            or Path(audit.get("dataset", "")).resolve() != data
            or audit.get("index_sha256") != sha256_file(data / "index.json")
            or audit.get("manifest_sha256") != sha256_file(data / "dataset_manifest.json")):
        raise ValueError("Conditioning source audit is failed, stale or for different data")
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest.get("final_test_used") is not False:
        raise ValueError("Final-test source is not a conditioning diagnostic")
    verify_source_payload_for_profile(audit, data)
    verify_environment_runtime_fingerprint(manifest.get("environment_runtime_fingerprint"), manifest["observation_profile"])
    for row in json.loads((data / "index.json").read_text()):
        provenance = source_reset_provenance(data, row)
        if not provenance:
            raise ValueError("Conditioning data needs explicit canonical TRAIN provenance")
        canonical = json.loads((Path(provenance["canonical_source"]) / "dataset_manifest.json").read_text())
        if canonical.get("split") != "train" or canonical.get("final_test_used"):
            raise ValueError("Conditioning source is not derived exclusively from TRAIN")
    return audit


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--paired-data", required=True, type=Path)
    parser.add_argument("--source-audit", type=Path,
                        help="Explicit exhaustive V9 camera/payload audit for a new paired TRAIN candidate")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layouts-per-task", type=int, default=1)
    parser.add_argument("--at-goal-divergence", action="store_true",
                        help="Use the identical late observation immediately before different goal-directed teacher actions")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.layouts_per_task < 1:
        raise ValueError("layouts-per-task must be positive")
    audit_path = args.source_audit or args.paired_data / "audit.json"
    audit = (verify_explicit_source_audit(args.paired_data, audit_path) if args.source_audit is not None
             else json.loads(audit_path.read_text()))
    if not audit["passed"] or (args.source_audit is None and audit.get("final_test_used") is not False):
        raise RuntimeError("Requires passing training-source-only paired observations")
    index = json.loads((args.paired_data / "index.json").read_text())
    manifest_path = args.paired_data / "dataset_manifest.json"
    if manifest_path.exists():
        from VLA.common.profile_binding import checkpoint_profile
        from VLA.common.observation_profile import profile_hash
        profile = checkpoint_profile(args.checkpoint)
        manifest = json.loads(manifest_path.read_text())
        if (profile_hash(profile) if profile else None) != manifest.get("observation_profile_sha256"):
            raise ValueError("Conditioning data differs from the checkpoint's bound sensor/controller profile")
    identity = verify_fingerprint(args.checkpoint)
    sensor = SimpleNamespace(closure=0.0)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    policy = SmolVLAPolicyAdapter(sensor, checkpoint=str(args.checkpoint), replan_steps=1)
    policy._load()
    device = next(policy.model.parameters()).device

    def predict(task, observation, closure, noise_seed):
        sensor.closure = closure
        policy.reset(task)
        # The standard adapter seeds noise from prompt; that would confound a
        # text intervention. All conditions here instead receive IDENTICAL noise.
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        # Use the SAME deployed preprocessing/postprocessing and official
        # sampling path, retaining the whole chunk instead of hiding all but
        # the first action. Different targets may share a saturated first move.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=policy._fp32_masters):
            batch = policy.preprocess(policy.build_sensor_batch(observation))
            state = batch["observation.state"]
            noise = torch.randn((1, policy.model.config.chunk_size, policy.model.config.max_action_dim),
                                device=state.device, dtype=torch.float32, generator=generator)
            chunk = policy.model.predict_action_chunk(batch, noise=noise)
            chunk = policy.postprocess(chunk)
        result = chunk.squeeze(0).float().cpu().numpy()
        if result.shape != (policy.model.config.chunk_size, 7) or not np.isfinite(result).all():
            raise ValueError("Invalid decoded canonical chunk")
        return result

    rows = []
    for task_name in ("touch", "pick", "move", "pick_place"):
        items = [item for item in index if item["task"]["name"] == task_name]
        groups = {}
        for item in items:
            groups.setdefault(item.get("paired_scene_source", task_name), []).append(item)
        for source_episode, items in list(groups.items())[:args.layouts_per_task]:
            if len(items) != 2:
                raise RuntimeError("Each conditioning scene must have exactly two target variants")
            frame = 0
            if args.at_goal_divergence:
                match = next((row for row in audit.get("rows", [])
                              if row["task"] == task_name and row["source_episode"] == source_episode), None)
                if match is None or match.get("first_different_action_frame") is None:
                    raise ValueError("Goal-divergence frame not qualified by the source probe")
                frame = int(match["first_different_action_frame"])
            rows.extend(audit_pair(policy, predict, task_name, source_episode, items, args.paired_data, frame))
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "paired_data": str(args.paired_data.resolve()), "paired_index_sha256": sha256_file(args.paired_data / "index.json"), "rows": rows,
        "source_audit": str(audit_path.resolve()), "source_audit_sha256": sha256_file(audit_path),
        "fixed_noise_across_all_interventions": True, "privileged_policy_inputs": False, "final_test_used": False,
        "scope": "Source-observation action-conditioning diagnostic only; blank/neutral conditions are interventions, not benchmark trials"}
    if args.source_audit is not None:
        verify_explicit_source_audit(args.paired_data, args.source_audit)
    if verify_fingerprint(args.checkpoint)["content_tree_sha256"] != identity["content_tree_sha256"]:
        raise RuntimeError("Checkpoint changed during conditioning diagnostic")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def audit_pair(policy, predict, task_name, source_episode, items, paired_data, frame=0):
        rows = []
        examples = []
        initial_arrays = []
        for item in sorted(items, key=lambda row: row.get("variant", 0)):
            task = TaskSpec.from_dict(item["task"])
            with np.load(paired_data / item["file"]) as episode:
                observation = VLAObservation(
                    front_rgb=episode["front_rgb"][frame], wrist_rgb=episode["wrist_rgb"][frame],
                    joint_positions=episode["joint_positions"][frame], gripper_width=0.0,
                    gripper_open=float(episode["gripper_open"][frame]),
                    ee_position=episode["ee_position"][frame], ee_rotation_6d=episode["ee_rotation_6d"][frame],
                    prompt=task.prompt, step=frame)
                examples.append((task, observation, float(episode["gripper_closure"][frame]),
                                 episode["canonical_actions"][frame:frame + policy.model.config.chunk_size]))
                initial_arrays.append((observation.front_rgb, observation.wrist_rgb,
                                       observation.joint_positions, float(episode["gripper_closure"][frame])))
        if not all(np.array_equal(a, b) for a, b in zip(*initial_arrays, strict=True)):
            raise ValueError("Text interventions require identical initial RGB and proprioception")
        teacher_a, teacher_b = pair_chunks(examples[0][3], examples[1][3])
        expected_difference = teacher_a[0, :3] - teacher_b[0, :3]
        for seed in (20261007, 20261008, 20261009):
            actions, interventions = [], []
            for task, observation, closure, expected in examples:
                actual_chunk = predict(task, observation, closure, seed)
                actual = actual_chunk[0]
                blank = replace(observation, front_rgb=np.zeros_like(observation.front_rgb),
                                wrist_rgb=np.zeros_like(observation.wrist_rgb))
                blank_action = predict(task, blank, closure, seed)[0]
                neutral = replace(observation, prompt="Perform the task")
                neutral_action = predict(replace(task, prompt=neutral.prompt), neutral, closure, seed)[0]
                actions.append(actual_chunk)
                interventions.append({"prompt": task.prompt, "predicted": actual.tolist(),
                    "predicted_chunk": actual_chunk.tolist(), "teacher_chunk": expected.tolist(),
                    "teacher_first_action": expected[0].tolist(), "blank_images_action": blank_action.tolist(),
                    "neutral_text_action": neutral_action.tolist(),
                    "initial_translation_vector_error_m": float(np.linalg.norm(actual[:3] - expected[0, :3])),
                    "image_intervention_translation_change_m": float(np.linalg.norm(actual[:3] - blank_action[:3])),
                    "neutral_text_translation_change_m": float(np.linalg.norm(actual[:3] - neutral_action[:3]))})
            predicted_difference = actions[0][0, :3] - actions[1][0, :3]
            model_a, model_b = pair_chunks(actions[0][:len(teacher_a)], actions[1][:len(teacher_b)])
            rows.append({"task": task_name, "source_episode": source_episode, "observation_frame": frame, "identical_noise_seed": seed,
                "teacher_chunk_text_translation_difference_rmse_m": float(np.sqrt(np.mean((teacher_a[:, :3] - teacher_b[:, :3])**2))),
                "model_chunk_text_translation_difference_rmse_m": float(np.sqrt(np.mean((model_a[:, :3] - model_b[:, :3])**2))),
                "teacher_text_pair_translation_difference_m": float(np.linalg.norm(expected_difference)),
                "model_text_pair_translation_difference_m": float(np.linalg.norm(predicted_difference)),
                "model_text_difference_projection_onto_teacher": float(np.dot(predicted_difference, expected_difference) /
                    (np.dot(expected_difference, expected_difference) + 1e-12)), "conditions": interventions})
            print(json.dumps({key: value for key, value in rows[-1].items() if key != "conditions"}), flush=True)
        return rows


if __name__ == "__main__":
    main()
