"""Verify every trained master survives the actual deployment loader exactly."""

import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.environment import ManipulationEnv
from VLA.common.tasks import randomize_rl_scene, select_tasks
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.adapter import SmolVLAPolicyAdapter


def main():
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    identity = verify_fingerprint(checkpoint)
    contract = json.loads((checkpoint / "fp32_master_contract.json").read_text())
    if not contract.get("passed") or not contract.get("saved_weights_and_adam_moments_fp32"):
        raise RuntimeError("Invalid FP32 master contract")
    env = ManipulationEnv(robot="ur5e", image_size=384, seed=20262001,
                          required_success_steps=1, use_semantic_success=True)
    policy = SmolVLAPolicyAdapter(env, checkpoint=str(checkpoint), seed=20262001)
    errors, dtypes, checked = {}, {}, 0
    try:
        task = randomize_rl_scene(select_tasks("touch")[0], np.random.default_rng(20262001))
        observation = env.reset(task)
        policy.reset(task)
        policy._load()
        parameters = dict(policy.model.named_parameters())
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
            for name in contract["trainable_parameter_names"]:
                actual = parameters[name].detach().cpu()
                expected = saved.get_tensor(name)
                if actual.dtype != torch.float32 or expected.dtype != torch.float32:
                    raise RuntimeError(f"Master dtype mismatch on inference reload: {name}")
                errors[name] = float((actual - expected).abs().max())
                dtypes[str(actual.dtype)] = dtypes.get(str(actual.dtype), 0) + actual.numel()
                checked += actual.numel()
        first = policy.predict(observation)
        policy.reset(task)
        repeated = policy.predict(observation)
        deterministic = bool(np.array_equal(first, repeated))
        finite = bool(np.isfinite(first).all())
        _, _, _, info = env.step(first)
        passed = max(errors.values()) == 0 and deterministic and finite
        report = {
            "passed": passed, "measurement": "precision_and_one_action_integration_not_task_success",
            "checkpoint": str(checkpoint), "checkpoint_content_sha256": identity["content_tree_sha256"],
            "checked_trainable_values": checked, "loaded_master_dtypes": dtypes,
            "max_reload_abs_error": max(errors.values()), "per_tensor_reload_abs_errors": errors,
            "repeated_seed_bitwise_identical": deterministic, "finite_action": finite,
            "canonical_action": first.tolist(), "actual_pybullet_action_executed": True,
            "task": task.prompt, "ee_object_distance_after_step_m": float(info["ee_object_distance"]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "per_tensor_reload_abs_errors"}, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Deployment precision smoke failed")
    finally:
        policy.close()
        env.close()


if __name__ == "__main__":
    main()
