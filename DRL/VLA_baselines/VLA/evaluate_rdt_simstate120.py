"""Same locked development120, explicitly RDT+SimState, never plain RDT."""
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
from VLA.rdt.simstate_adapter import SimStateRDTPolicy
from VLA.run_cross_vla120 import SEEDS, TASKS, PROFILE_PROTOCOL, ROOT
from VLA.simstate_feedback import feedback_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5734)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("No hidden cohort reroll")
    args.output.mkdir(parents=True)
    profile = checkpoint_profile(args.checkpoint)
    template = json.loads(PROFILE_PROTOCOL.read_text())
    if profile_hash(profile) != template["observation_profile_sha256"]:
        raise ValueError("Same UR5e/Robotiq/cameras/physics required")
    identity = verify_fingerprint(args.checkpoint, full=True)
    rows = []
    def save(status, error=None):
        atomic_json(args.output/"progress.json", {"model": "rdt_simstate", "status": status,
            "checkpoint_content_sha256": identity["content_tree_sha256"],
            "privileged_model_inputs": True, "oracle_controller_at_deployment": False,
            "learned_policy_is_only_action_source": True, "NOT_plain_unprivileged_VLA": True,
            "same_seeds_as_cross_vla120": True, "final_test_used": False,
            "completed_trials": len(rows), "planned_trials": 120, "trials_per_action": 30,
            "successes": sum(r["success"] for r in rows), "rows": rows, "error": error,
            "statistics": success_statistics(rows) if rows and len(rows)%4 == 0 else None})
    save("running")
    try:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            tasks = [randomize_rl_scene(t, rng) for t in select_tasks("all")]
            env = environment_class_for_profile(profile)(observation_profile=profile,
                robot="ur5e", image_size=384, control_hz=10, assist_grasp=True, seed=seed,
                required_success_steps=1, use_semantic_success=True,
                success_threshold_scale=template["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=template["success"]["semantic_thresholds"])
            policy = SimStateRDTPolicy(env, checkpoint=str(args.checkpoint.resolve()), port=args.port, replan_steps=1, seed=seed)
            folder = ROOT/"VLA/rdt_simstate"/args.output.parent.name/f"test_seed{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            traces = {}
            try:
                policy._verify_server()
                recorder = VideoRecorder(folder/"rollout.mp4", fps=10, overview_render=lambda: env.render("overview")) if seed == SEEDS[0] else NullVideoRecorder()
                with recorder:
                    for task in tasks:
                        obs = env.reset(task); policy.reset(task)
                        if env.is_semantic_success():
                            raise ValueError("No initial-success scene replacement")
                        _save_observation_snapshot(folder, task.name, obs, env)
                        done = False; trace = []
                        while not done:
                            before = _interaction_audit(env); state = feedback_state(env, obs)
                            action = policy.predict(obs)
                            if action.shape != (7,) or not np.isfinite(action).all():
                                raise ValueError("Neural integration failed; no teacher replacement")
                            obs, _, done, info = env.step(action)
                            trace.append({"step": obs.step, "action": action.tolist(),
                                "pre_action_PRIVILEGED_state18": state.tolist(), "interaction_before_audit": before,
                                "interaction_after_audit": _interaction_audit(env), "success": bool(info["success"]),
                                "safety_violation": info.get("safety_violation")})
                            recorder.add(obs, "RDT+SimState | "+task.prompt)
                            if obs.step%10 == 0:
                                _save_observation_snapshot(folder, task.name, obs, env)
                        _save_observation_snapshot(folder, task.name, obs, env)
                        traces[task.name] = trace; atomic_json(folder/"trajectory.json", traces)
                        rows.append({"seed": seed, "task": task.name, "success": bool(info["success"]),
                            "safety_violation": info.get("safety_violation"), "steps": len(trace),
                            "trajectory": str(folder/"trajectory.json")})
                        save("running")
            finally:
                policy.close(); env.close()
        if Counter(r["task"] for r in rows) != {t: 30 for t in TASKS}:
            raise ValueError("Incomplete120")
        save("completed120_privileged_input_VLA")
    except Exception as error:
        save("stopped_integration_error_no_reroll", str(error)); raise


if __name__ == "__main__":
    main()
