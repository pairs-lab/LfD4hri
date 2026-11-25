"""Four TRAIN scenes, two actual neural controls each; NOT an SR episode test."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import requests

from VLA.benchmark import _interaction_audit, _save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.recorder import VideoRecorder
from VLA.common.tasks import TaskSpec
from VLA.cross_vla120_trial import atomic_json
from VLA.privileged_prompt import PROMPT_SCHEMA, augment_prompt
from VLA.simstate_feedback import feedback_state


def png(pixels):
    stream = BytesIO(); Image.fromarray(pixels).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


class PrivilegedNativeHTTPPolicy:
    """Only native-output convention conversion; never choose a target pose."""
    input_modalities = ("front_rgb", "native_optional_wrist_rgb", "text", PROMPT_SCHEMA)
    comparable = False
    def __init__(self, env, model, checkpoint, port, seed):
        self.env, self.model, self.seed = env, model, seed
        self.url = f"http://127.0.0.1:{port}"
        self.session = requests.Session(); self.session.trust_env = False
        response = self.session.get(self.url + "/metadata", timeout=30); response.raise_for_status()
        self.server_metadata = response.json()
        if (self.server_metadata.get("protocol") != "privileged_native_language_v1"
                or self.server_metadata.get("privileged_schema") != PROMPT_SCHEMA
                or self.server_metadata.get("model") != model
                or self.server_metadata.get("checkpoint") != checkpoint
                or self.server_metadata.get("oracle_or_ik_controller") is not False):
            raise ValueError("Wrong native privileged server")
        self.index = 0

    def reset(self, task): self.index = 0
    def predict(self, obs):
        state = feedback_state(self.env, obs); prompt = augment_prompt(obs.prompt, state)
        response = self.session.post(self.url + "/act", json={"instruction": obs.prompt,
            "state18": state.tolist(), "prompt": prompt, "front_rgb_png": png(obs.front_rgb),
            "wrist_rgb_png": png(obs.wrist_rgb), "seed": self.seed, "inference_index": self.index}, timeout=900)
        response.raise_for_status(); data = response.json()
        if data.get("action_source") != "native_neural_model" or data.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError("Neural action/actual input proof missing")
        native = np.asarray(data["actions"], np.float32)
        if native.ndim != 2 or native.shape[1] != 7 or not len(native) or not np.isfinite(native).all():
            raise ValueError("Malformed neural action")
        self.last_state, self.last_prompt, self.last_native = state, prompt, native[0].copy()
        self.index += 1
        kind = data["action_kind"]
        if kind == "absolute_joint_targets_and_closure":
            return self.env.joint_targets_to_canonical(native[0, :6], native[0, 6])
        action = native[0].copy()
        if kind == "native_delta_ee_and_open_gripper": action[6] = 1 - 2 * action[6]
        elif kind != "canonical_delta_ee_7d": raise ValueError("Unknown native command convention")
        return action
    def close(self): self.session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("pi0", "pi0_fast", "openvla", "cogact"))
    p.add_argument("--checkpoint", required=True); p.add_argument("--port", type=int, required=True)
    p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=Path("VLA/evaluation_protocol_v9_postfit_10_per_action.json"))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text()); profile = json.loads(Path(protocol["observation_profile_file"]).read_text())
    if profile_hash(profile) != protocol["observation_profile_sha256"]: raise ValueError("Camera/robot profile changed")
    rows = json.loads((args.source / "index.json").read_text()); results = []
    env = environment_class_for_profile(profile)(observation_profile=profile, robot="ur5e", image_size=384,
        control_hz=10, assist_grasp=True, seed=20261014, required_success_steps=1, use_semantic_success=True,
        success_threshold_scale=protocol["success"]["reward_xml_threshold_scale"],
        semantic_success_thresholds=protocol["success"]["semantic_thresholds"])
    policy = None
    def save(status, error=None):
        atomic_json(args.output / "analysis.json", {"model": args.model, "status": status,
            "rows": results, "neural_controls_executed": len(results), "planned_neural_controls": 8,
            "learned_policy_is_only_action_source": True, "oracle_fallback": False,
            "kinematic_motor_servo_below_policy": True, "privileged_inputs": True,
            "integration_only_NOT_task_SR": True, "fine_tuned_on_this_privileged_input": bool(policy and policy.server_metadata.get("target_domain_finetuned")),
            "final_test_used": False, "error": error})
    save("running")
    try:
        policy = PrivilegedNativeHTTPPolicy(env, args.model, args.checkpoint, args.port, 20261014)
        with VideoRecorder(args.output / "rollout.mp4", fps=10, overview_render=lambda: env.render("overview")) as video:
            for task in ("touch", "pick", "move", "pick_place"):
                row = next(r for r in rows if r["task"]["name"] == task)
                spec = TaskSpec.from_dict(row["task"]); obs = env.reset(spec); policy.reset(spec)
                if env.is_semantic_success(): raise ValueError("Initial success invalid; no scene reroll")
                _save_observation_snapshot(args.output, task, obs, env)
                for _ in range(2):
                    before = _interaction_audit(env); action = policy.predict(obs)
                    obs, _, done, info = env.step(action)
                    results.append({"task": task, "step": obs.step, "source_task_file": row["file"],
                        "native_neural_action": policy.last_native.tolist(), "canonical_action": action.tolist(),
                        "pre_action_PRIVILEGED_state18": policy.last_state.tolist(), "actual_model_prompt": policy.last_prompt,
                        "interaction_before": before, "interaction_after": _interaction_audit(env),
                        "safety_violation": info.get("safety_violation"), "success_observed": bool(info["success"])})
                    video.add(obs, args.model + "+privileged language | " + spec.prompt)
                    _save_observation_snapshot(args.output, task, obs, env); save("running")
                    if done:
                        raise RuntimeError("Probe terminated early; preserve actual native actions, do not fill missing trials")
        save("completed_eight_native_neural_controls_NOT_SR")
    except Exception as error:
        save("integration_failed_preserved", f"{type(error).__name__}: {error}"); raise
    finally:
        if policy is not None: policy.close()
        env.close()


if __name__ == "__main__": main()
