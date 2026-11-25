"""Eight native learned controls on four TRAIN scenes: API/physics smoke only."""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit, _save_observation_snapshot
from VLA.common.recorder import VideoRecorder
from VLA.common.tasks import TaskSpec
from VLA.cross_vla120_trial import atomic_json
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.rdt.simstate_adapter import SimStateRDTPolicy
from VLA.simstate_feedback import feedback_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=5734)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("No hidden smoke reroll")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.source/"dataset_manifest.json").read_text())
    if manifest.get("split") != "train" or manifest.get("final_test_used") is not False:
        raise ValueError("Smoke must use TRAIN, not development or final scenes")
    index = json.loads((args.source/"index.json").read_text())
    selected = [next(r for r in index if r["task"]["name"] == task
                    and not r.get("paired_scene_source") and not r.get("recovery_start_actions_not_training_labels"))
                for task in ("touch", "pick", "move", "pick_place")]
    identity = verify_fingerprint(args.checkpoint, full=True)
    env = audit_environment(args.source, robot="ur5e", image_size=384, control_hz=10,
        seed=0, required_success_steps=1, use_semantic_success=True)
    policy = SimStateRDTPolicy(env, checkpoint=str(args.checkpoint.resolve()), port=args.port, replan_steps=1, seed=20261014)
    rows = []
    try:
        policy._verify_server()
        atomic_json(args.output/"server_metadata.json", policy.server_metadata)
        with VideoRecorder(args.output/"rollout.mp4", fps=10, overview_render=lambda: env.render("overview")) as recorder:
            for item in selected:
                task = TaskSpec.from_dict(item["task"])
                obs = reset_source_episode(env, args.source, item); policy.reset(task)
                if env.is_semantic_success():
                    raise ValueError("No initial-success replacement")
                _save_observation_snapshot(args.output, task.name, obs, env)
                trace = []
                for _ in range(2):
                    before = _interaction_audit(env); state = feedback_state(env, obs)
                    action = policy.predict(obs)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("Invalid canonical learned action")
                    obs, _, done, info = env.step(action)
                    trace.append({"action": action.tolist(), "pre_action_privileged_state18": state.tolist(),
                        "before": before, "after": _interaction_audit(env),
                        "safety_violation": info.get("safety_violation"), "success": bool(info["success"])})
                    recorder.add(obs, "RDT+SimState two-update API SMOKE ONLY | "+task.prompt)
                    if done:
                        break  # Preserve original safety/success event; never reroll.
                _save_observation_snapshot(args.output, task.name, obs, env)
                rows.append({"task": task.name, "source_file": item["file"], "trace": trace, "steps": len(trace),
                             "safety_abort": bool(info.get("safety_violation"))})
                atomic_json(args.output/"progress.json", {"status": "running_API_smoke_NOT_SR", "rows": rows})
        atomic_json(args.output/"analysis.json", {"status": "completed_four_task_API_smoke_NOT_SR", "rows": rows,
            "checkpoint_content_sha256": identity["content_tree_sha256"],
            "privileged_model_inputs": True, "oracle_controller_at_deployment": False,
            "all_four_tasks_reached": len(rows) == 4, "all_two_controls_completed": all(r["steps"] == 2 for r in rows),
            "neural_success_claimed": False, "final_or_development_scenes_used": False,
            "long_fit_qualified": False, "actual_controls": sum(r["steps"] for r in rows)})
        atomic_json(args.output/"progress.json", {"status": "completed_four_task_API_smoke_NOT_SR", "rows": rows})
    except Exception as error:
        atomic_json(args.output/"progress.json", {"status": "preserved_integration_error_no_reroll", "rows": rows,
            "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        policy.close(); env.close()


if __name__ == "__main__":
    main()
