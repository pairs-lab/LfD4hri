"""120 learned privileged-input SmolVLA episodes, NOT an oracle-assisted score."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit, _save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.observation_profile import profile_hash
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.recorder import VideoRecorder, NullVideoRecorder
from VLA.common.tasks import randomize_rl_scene, select_tasks
from VLA.cross_vla120_trial import atomic_json
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.run_cross_vla120 import SEEDS, TASKS, PROFILE_PROTOCOL, ROOT
from VLA.smolvla.simstate_adapter import SimStateSmolVLA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("No hidden reroll/overwrite")
    args.output.mkdir(parents=True)
    profile = checkpoint_profile(str(args.checkpoint))
    template = json.loads(PROFILE_PROTOCOL.read_text())
    if profile_hash(profile) != template["observation_profile_sha256"]:
        raise ValueError("Must retain the same UR5e/Robotiq V9 cameras/physics")
    verify_fingerprint(args.checkpoint, full=True)
    rows = []
    def save(status, error=None):
        atomic_json(args.output / "progress.json", {"model": "smolvla_simstate", "status": status,
            "privileged_model_inputs": True, "learned_policy_is_only_action_source": True,
            "oracle_controller_at_deployment": False, "same_seeds_as_cross_vla120": True,
            "completed_trials": len(rows), "planned_trials": 120, "trials_per_action": 30,
            "successes": sum(r["success"] for r in rows), "rows": rows, "error": error,
            "statistics": success_statistics(rows) if rows and len(rows) % 4 == 0 else None,
            "final_test_used": False, "NOT_plain_unprivileged_VLA": True})
    save("running")
    try:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tasks = [randomize_rl_scene(t, rng) for t in select_tasks("all")]
            env = environment_class_for_profile(profile)(observation_profile=profile,
                robot="ur5e", image_size=384, control_hz=10, assist_grasp=True,
                seed=seed, required_success_steps=1, use_semantic_success=True,
                success_threshold_scale=template["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=template["success"]["semantic_thresholds"])
            policy = SimStateSmolVLA(env, str(args.checkpoint), replan_steps=1, seed=seed)
            folder = ROOT / "VLA/smolvla_simstate" / args.output.parent.name / f"test_seed{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            try:
                recorder = (VideoRecorder(folder / "rollout.mp4", fps=10, overview_render=lambda: env.render("overview"))
                            if seed == SEEDS[0] else NullVideoRecorder())
                traces = {}
                with recorder:
                    for task in tasks:
                        obs = env.reset(task); policy.reset(task)
                        if env.is_semantic_success():
                            raise ValueError("Initial success invalid; no replacement scene")
                        _save_observation_snapshot(folder, task.name, obs, env)
                        trace = []; done = False
                        while not done:
                            before = _interaction_audit(env)
                            from VLA.simstate_feedback import feedback_state
                            input_state = feedback_state(env, obs)
                            action = policy.predict(obs)  # ONLY control producer: fine-tuned VLA
                            if action.shape != (7,) or not np.isfinite(action).all():
                                raise ValueError("Neural integration failed; no teacher replacement")
                            obs, reward, done, info = env.step(action)
                            trace.append({"step": obs.step, "action": action.tolist(),
                                "pre_action_PRIVILEGED_state18": input_state.tolist(),
                                "interaction_before_audit": before,
                                "interaction_after_audit": _interaction_audit(env),
                                "success": bool(info["success"]), "safety_violation": info.get("safety_violation")})
                            recorder.add(obs, f"SmolVLA+SimState | {task.prompt} | {obs.step}")
                            if obs.step % 10 == 0:
                                _save_observation_snapshot(folder, task.name, obs, env)
                        _save_observation_snapshot(folder, task.name, obs, env)
                        traces[task.name] = trace
                        rows.append({"seed": seed, "task": task.name, "success": bool(info["success"]),
                            "safety_violation": info.get("safety_violation"), "steps": len(trace),
                            "trajectory": str(folder / "trajectory.json")})
                        atomic_json(folder / "trajectory.json", traces)
                        atomic_json(folder / "results.json", {"rows": rows[-len(traces):], "privileged_inputs": True})
                        save("running")
            finally:
                policy.close(); env.close()
        if Counter(r["task"] for r in rows) != {t: 30 for t in TASKS}:
            raise ValueError("Incomplete120")
        save("completed120_privileged_input_VLA")
    except Exception as error:
        save("stopped_integration_error_no_reroll", str(error))
        raise


if __name__ == "__main__":
    main()
