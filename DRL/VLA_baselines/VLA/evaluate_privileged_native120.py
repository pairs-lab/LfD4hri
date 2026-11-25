"""Frozen development120 for an explicitly privileged-trained OpenVLA pilot."""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit, _save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.observation_profile import profile_hash
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.recorder import NullVideoRecorder, VideoRecorder
from VLA.common.tasks import randomize_rl_scene, select_tasks
from VLA.cross_vla120_trial import atomic_json
from VLA.privileged_native_trial import PrivilegedNativeHTTPPolicy
from VLA.run_cross_vla120 import SEEDS, TASKS, PROFILE_PROTOCOL


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    profile = checkpoint_profile(args.checkpoint); template = json.loads(PROFILE_PROTOCOL.read_text())
    if profile_hash(profile) != template["observation_profile_sha256"]: raise ValueError("Native pilot sensor/robot profile differs")
    from VLA.openvla.finetune import validate_checkpoint_manifest
    identity = validate_checkpoint_manifest(args.checkpoint)
    rows = []
    def save(status, error=None):
        atomic_json(args.output / "progress.json", {"model": "openvla_privileged_language", "status": status,
            "planned_trials": 120, "trials_per_action": 30, "completed_trials": len(rows),
            "successes": sum(r["success"] for r in rows), "statistics": success_statistics(rows) if rows and len(rows) % 4 == 0 else None,
            "rows": rows, "error": error, "privileged_model_inputs": True,
            "oracle_controller_at_deployment": False, "learned_policy_is_only_action_source": True,
            "optimizer_updates": identity["optimizer_step"], "bounded_pilot_NOT_matched_training_budget": True,
            "final_test_used": False})
    save("running")
    try:
        for seed in SEEDS:
            rng = np.random.default_rng(seed); tasks = [randomize_rl_scene(t, rng) for t in select_tasks("all")]
            folder = args.output / f"test_seed{seed}"; folder.mkdir()
            env = environment_class_for_profile(profile)(observation_profile=profile, robot="ur5e", image_size=384,
                control_hz=10, assist_grasp=True, seed=seed, required_success_steps=1, use_semantic_success=True,
                success_threshold_scale=template["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=template["success"]["semantic_thresholds"])
            policy = None; traces = {}
            try:
                policy = PrivilegedNativeHTTPPolicy(env, "openvla", str(args.checkpoint.resolve()), args.port, seed)
                if not policy.server_metadata["target_domain_finetuned"]: raise ValueError("Do not score official weights as finetuned")
                recorder = VideoRecorder(folder / "rollout.mp4", fps=10, overview_render=lambda: env.render("overview")) if seed == SEEDS[0] else NullVideoRecorder()
                with recorder:
                    for task in tasks:
                        obs = env.reset(task); policy.reset(task); trace = []; done = False
                        if env.is_semantic_success(): raise ValueError("No initial-success reroll")
                        _save_observation_snapshot(folder, task.name, obs, env)
                        while not done:
                            before = _interaction_audit(env); action = policy.predict(obs)
                            obs, _, done, info = env.step(action)
                            trace.append({"step": obs.step, "action": action.tolist(), "native_neural_action": policy.last_native.tolist(),
                                "pre_action_PRIVILEGED_state18": policy.last_state.tolist(), "actual_model_prompt": policy.last_prompt,
                                "interaction_before_audit": before, "interaction_after_audit": _interaction_audit(env),
                                "success": bool(info["success"]), "safety_violation": info.get("safety_violation")})
                            recorder.add(obs, "OpenVLA+privileged language | " + task.prompt)
                            if obs.step % 10 == 0: _save_observation_snapshot(folder, task.name, obs, env)
                        _save_observation_snapshot(folder, task.name, obs, env)
                        traces[task.name] = trace; atomic_json(folder / "trajectory.json", traces)
                        rows.append({"seed": seed, "task": task.name, "success": bool(info["success"]),
                            "steps": len(trace), "safety_violation": info.get("safety_violation"), "trajectory": str(folder / "trajectory.json")})
                        save("running")
            finally:
                if policy is not None: policy.close()
                env.close()
        save("completed_development120_privileged_OpenVLA_pilot")
    except Exception as error:
        save("failed_preserve_all_actual_trials", f"{type(error).__name__}: {error}"); raise


if __name__ == "__main__": main()
