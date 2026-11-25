"""One fixed scene block for the explicit 120/model cross-VLA diagnostic.

This is not a camera-binding override for target-domain checkpoints. Qualified
checkpoints must retain their source profile. Unadapted OFFICIAL base models
can explicitly be probed zero-shot on the common V9 scene; that different
adaptation regime is recorded and is never an accepted post-fit baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from VLA.benchmark import (_action_diagnostics, _interaction_audit,
                           _normalized_semantic_gap, _observation_telemetry,
                           _save_observation_snapshot)
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.policy import load_policy
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.provenance import sha256_file
from VLA.common.recorder import NullVideoRecorder, VideoRecorder
from VLA.common.tasks import randomize_rl_scene, select_tasks

TASKS = ("touch", "pick", "move", "pick_place")


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def task_json(task):
    """TaskSpec contains tuples; compare its actual persisted JSON schema."""
    return json.loads(json.dumps(task.to_dict()))


def validate_regime(model, regime, checkpoint, profile):
    bound = checkpoint_profile(checkpoint)
    if regime == "qualified_v9_postfit":
        if bound is None or profile_hash(bound) != profile_hash(profile):
            raise ValueError("Target-domain checkpoint is not bound to the common V9 profile")
    elif regime == "official_zero_shot_v9":
        if model not in {"pi0", "pi0_fast", "openvla", "cogact"} or bound is not None:
            raise ValueError("Zero-shot diagnostic cannot relabel a target-domain checkpoint")
        if model in {"pi0", "pi0_fast"} and checkpoint != f"gs://openpi-assets/checkpoints/{model}_base":
            raise ValueError("Only the declared official openpi base is eligible for this regime")
        root = Path(checkpoint)
        if (root / "adapter_config.json").exists() or (root / "train_config.json").exists():
            raise ValueError("Fine-tuned adapters are not official zero-shot models")
    else:
        raise ValueError("Explicit adaptation regime required")


class FrontHTTPPolicy:
    """Isolate the OpenVLA runtime; do NOT give it joints/IDs/goal coordinates."""
    input_modalities = ("front_rgb", "text")
    comparable = False

    def __init__(self, port, checkpoint):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.url = f"http://127.0.0.1:{port}"
        response = self.session.get(self.url + "/metadata", timeout=30)
        response.raise_for_status()
        self.server_metadata = response.json()
        if (self.server_metadata.get("model") != "openvla"
                or self.server_metadata.get("checkpoint") != checkpoint
                or self.server_metadata.get("action_contract") != "canonical_delta_ee_7d"):
            raise ValueError("OpenVLA server/checkpoint/action identity differs")

    def reset(self, task):
        pass

    def predict(self, observation):
        from VLA.rdt.adapter import _encode_png
        response = self.session.post(self.url + "/act", json={
            "front_rgb_png": _encode_png(observation.front_rgb),
            "prompt": observation.prompt}, timeout=600)
        response.raise_for_status()
        return np.asarray(response.json()["action"], dtype=np.float32)

    def close(self):
        self.session.close()


def run_trial(plan, model_spec, seed, output, video):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Existing trial retained; no reroll or overwrite")
    output.mkdir(parents=True)
    profile = json.loads(Path(plan["observation_profile_file"]).read_text())
    if profile_hash(profile) != plan["observation_profile_sha256"]:
        raise ValueError("Common profile changed")
    model, checkpoint = model_spec["model"], model_spec["checkpoint"]
    validate_regime(model, model_spec["adaptation_regime"], checkpoint, profile)
    rng = np.random.default_rng(seed)
    tasks = [randomize_rl_scene(task, rng) for task in select_tasks("all")]
    success = plan["success"]
    env = environment_class_for_profile(profile)(observation_profile=profile,
        robot="ur5e", image_size=384, control_hz=10, assist_grasp=True, seed=seed,
        required_success_steps=1, success_threshold_scale=success["reward_xml_threshold_scale"],
        allow_distance_only_success=False, use_semantic_success=True,
        semantic_success_thresholds=success["semantic_thresholds"])
    policy = None
    run = {"model": model, "seed": seed, "adaptation_regime": model_spec["adaptation_regime"],
           "accepted_fair_postfit_baseline": False, "tasks": [], "integration_error": None}
    traces = {}
    try:
        policy = (FrontHTTPPolicy(model_spec["port"], checkpoint) if model == "openvla" else
                  load_policy(model, env, checkpoint=checkpoint, port=model_spec.get("port"),
                      unnorm_key=model_spec.get("unnorm_key"), replan_steps=1, seed=seed))
        atomic_json(output / "manifest.json", {"model": model, "seed": seed,
            "robot": "ur5e", "gripper": "Robotiq 2F-85", "control_hz": 10,
            "replan_steps": 1, "scene_profile": "rl_random_env",
            "tasks": [task.to_dict() for task in tasks], "checkpoint": checkpoint,
            "adaptation_regime": model_spec["adaptation_regime"],
            "model_inputs_consumed": list(policy.input_modalities),
            "privileged_object_pose_given_to_vla": False, "oracle_fallback": False,
            "observation_profile_sha256": profile_hash(profile),
            "success": success, "video_recorded": video, "overview_video_only": True})
        recorder = (VideoRecorder(output / "rollout.mp4", fps=10,
                    overview_render=lambda: env.render("overview")) if video else NullVideoRecorder())
        with recorder:
            for task in tasks:
                obs = env.reset(task)
                if env.is_semantic_success():
                    raise ValueError("Initial task success is an invalid scoring/reset condition; no seed replacement")
                reference = Path(plan["scene_audit_root"]) / f"seed{seed}" / task.name
                _save_observation_snapshot(output, task.name, obs, env)
                from PIL import Image
                for camera, pixels in (("front", obs.front_rgb), ("wrist", obs.wrist_rgb)):
                    expected = np.asarray(Image.open(reference / f"{camera}.png"))
                    if not np.array_equal(expected, pixels):
                        raise ValueError("Reset pixels differ from common CPU scene audit")
                expected_reset = json.loads((reference / "reset.json").read_text())
                if (task_json(task) != expected_reset["task"]
                        or obs.joint_positions.tolist() != expected_reset["joint_positions_rad"]):
                    raise ValueError("Reset task or joints differ from shared scene audit")
                policy.reset(task)
                recorder.add(obs, f"{model} | {task.prompt}", repeat=10)
                trace, latencies, actions = [], [], []
                done, info = False, None
                try:
                    while not done:
                        before, sensor = _interaction_audit(env), _observation_telemetry(obs, env)
                        started = time.perf_counter()
                        predicted = np.asarray(policy.predict(obs), dtype=np.float32)
                        latencies.append(time.perf_counter() - started)
                        if predicted.ndim == 2:
                            predicted = predicted[0]  # fixed receding horizon: ONE executable action
                        if predicted.shape != (7,) or not np.isfinite(predicted).all():
                            raise ValueError("Invalid native action; integration error, not a task failure")
                        actions.append(predicted.copy())
                        obs, reward, done, info = env.step(predicted)
                        trace.append({"step": len(actions), "action": predicted.tolist(),
                            "policy_observation": sensor, "result_observation": _observation_telemetry(obs, env),
                            "interaction_before_audit_only": before,
                            "interaction_after_audit_only": _interaction_audit(env),
                            "reward": float(reward), "success": bool(info["success"]),
                            "raw_success": bool(info["raw_success"]),
                            "strict_raw_success": bool(info["strict_raw_success"]),
                            "semantic_raw_success": bool(info["semantic_raw_success"]),
                            "success_streak": int(info["success_streak"]),
                            "ee_object_distance_m": float(info["ee_object_distance"]),
                            "object_goal_distance_m": float(info["object_goal_distance"]),
                            "pick_lift_m": float(info["pick_lift_m"]),
                            "goal_xy_distance_m": info["goal_xy_distance_m"],
                            "goal_vertical_offset_m": info["goal_vertical_offset_m"],
                            "normalized_semantic_gap": _normalized_semantic_gap(task.name, info, env.semantic_success_thresholds),
                            "safety_violation": info.get("safety_violation"),
                            "safety_rollback_restored": info.get("rollback_restored"),
                            "touching_object": bool(info["touching_object"]),
                            "grasped_object": info["grasped_object"] is not None})
                        recorder.add(obs, f"{model} | {task.prompt} | {len(actions)}")
                        if obs.step % 10 == 0:
                            _save_observation_snapshot(output, task.name, obs, env)
                    _save_observation_snapshot(output, task.name, obs, env)
                    recorder.add(obs, f"{model} | {task.name} | {'SUCCESS' if info['success'] else 'FAIL'}", repeat=10)
                    run["tasks"].append({"name": task.name, "prompt": task.prompt,
                        "success": bool(info["success"]), "initial_semantic_success": False,
                        "error": None, "steps": len(actions),
                        "safety_violation": info.get("safety_violation"),
                        "mean_inference_ms": float(np.mean(latencies) * 1000),
                        "action_diagnostics": _action_diagnostics(actions),
                        "ever_strict_raw_success": any(t["strict_raw_success"] for t in trace),
                        "min_ee_object_distance_m": min(t["ee_object_distance_m"] for t in trace),
                        "max_pick_lift_m": max(t["pick_lift_m"] for t in trace),
                        "final_goal_xy_distance_m": info["goal_xy_distance_m"],
                        "final_goal_vertical_offset_m": info["goal_vertical_offset_m"],
                        "final_grasped_object": info["grasped_object"] is not None})
                finally:
                    traces[task.name] = trace
                    atomic_json(output / "trajectory.json", traces)
                    atomic_json(output / "results.json", run)
                print(f"{model} seed={seed} {task.name}: {run['tasks'][-1]['success']}", flush=True)
    except Exception as error:
        run["integration_error"] = f"{type(error).__name__}: {error}"
        run["traceback"] = traceback.format_exc()
        atomic_json(output / "results.json", run)
        raise
    finally:
        if policy is not None:
            policy.close()
        env.close()
    atomic_json(output / "server_metadata.json", getattr(policy, "server_metadata", {}))
    atomic_json(output / "results.json", run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, digest in plan["frozen_files"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("Frozen cross-model runtime changed")
    spec = next(item for item in plan["models"] if item["model"] == args.model)
    if args.seed not in plan["seeds"]:
        raise ValueError("Undeclared seed")
    run_trial(plan, spec, args.seed, args.output, args.video)


if __name__ == "__main__":
    main()
