#!/usr/bin/env python3
"""Serve an official RDT checkpoint for the shared UR5e PyBullet scene."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
import sys
from contextlib import nullcontext

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RDT_ROOT = PROJECT_ROOT / "VLA" / "third_party" / "rdt"
sys.path.insert(0, str(RDT_ROOT))

# Berkeley Autolab UR5 joint convention and this URDF use different zero
# offsets.  Align the deterministic PyBullet home pose to the published
# pretraining mean and invert the same calibration on model outputs.
SIM_HOME = np.array(
    (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0.0),
    dtype=np.float32,
)
BERKELEY_UR5_MEAN = np.array(
    (-3.2655642, -1.3217927, 2.1238138, -2.3727081, -1.5671834, 3.0003631),
    dtype=np.float32,
)
JOINT_OFFSET = BERKELEY_UR5_MEAN - SIM_HOME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="robotics-diffusion-transformer/rdt-170m")
    parser.add_argument("--vision", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--embeddings", type=Path, default=Path("VLA/rdt/prompt_embeddings.pt"))
    parser.add_argument("--port", type=int, default=5600)
    return parser.parse_args()


def decode_image(value: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")


def prepare_sensor_tensors(payload, processor, embeddings, *, target_domain, previous_images=None):
    """Actual server preprocessing on CPU, independently testable against training."""
    from VLA.rdt.training_data import wrap_joints

    prompt = payload["prompt"]
    front, wrist = decode_image(payload["front_rgb_png"]), decode_image(payload["wrist_rgb_png"])
    background = Image.new("RGB", front.size, tuple(int(value * 255) for value in processor.image_mean))
    if "previous_front_rgb_png" in payload and "previous_wrist_rgb_png" in payload:
        previous_front = decode_image(payload["previous_front_rgb_png"])
        previous_wrist = decode_image(payload["previous_wrist_rgb_png"])
    elif int(payload.get("step", 0)) == 0 or not previous_images:
        previous_front, previous_wrist = front, wrist
    else:
        previous_front, previous_wrist = previous_images["front"], previous_images["wrist"]
    if previous_images is not None:
        previous_images.update({"front": front, "wrist": wrist})
    frames = [previous_front, previous_wrist, background, front, wrist, background]
    images = torch.cat([processor.preprocess(frame, return_tensors="pt")["pixel_values"] for frame in frames], dim=0).unsqueeze(0)
    proprio = np.asarray(payload["proprio"], dtype=np.float32)
    if proprio.shape != (16,):
        raise ValueError(f"Expected 16D UR5 proprio, got {proprio.shape}")
    states = torch.zeros((1, 1, 128), dtype=torch.float32)
    mask = torch.zeros((1, 128), dtype=torch.float32)
    joints = wrap_joints(proprio[:6]) if target_domain else proprio[:6] + JOINT_OFFSET
    states[0, 0, :6] = torch.from_numpy(joints)
    states[0, 0, 10] = float(proprio[6])
    mask[0, [0, 1, 2, 3, 4, 5, 10]] = 1
    text = embeddings[prompt]
    if text.ndim == 2:
        text = text.unsqueeze(0)
    return {"images": images, "states": states, "state_elem_mask": mask,
            "lang_tokens": text, "lang_attn_mask": torch.ones(text.shape[:2], dtype=torch.bool),
            "ctrl_freqs": torch.tensor([10 if target_domain else 5], dtype=torch.int64)}


def main() -> None:
    from configs.state_vec import STATE_VEC_IDX_MAPPING
    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    from models.rdt_runner import RDTRunner

    args = parse_args()
    checkpoint_root = Path(args.checkpoint).resolve()
    target_path = checkpoint_root / "pybullet_rdt_config.json"
    target_contract = json.loads(target_path.read_text()) if target_path.is_file() else None
    checkpoint_metadata = {"path": str(checkpoint_root)}
    if target_contract is not None:
        from VLA.fingerprint_checkpoint import verify_fingerprint
        from VLA.rdt.training_data import sha256, wrap_joints

        if (
            target_contract.get("format_version") != 1
            or target_contract.get("model_family") != "rdt"
            or target_contract.get("control_hz") != 10
            or target_contract.get("action_horizon") not in (5, 10, 16, 64)
            or target_contract.get("state_dim") != 128
            or target_contract.get("joint", {}).get("convention") != "wrap_to_minus_pi_pi"
            or target_contract.get("joint", {}).get("offset") != [0.0] * 6
            or target_contract.get("valid_indices") != [0, 1, 2, 3, 4, 5, 10]
            or target_contract.get("scaling") != {"state": "identity_SI", "action": "identity_SI"}
            or target_contract.get("language", {}).get("embeddings_sha256") != sha256(args.embeddings)
            or target_contract.get("language", {}).get("manifest_sha256")
            != sha256(args.embeddings.with_suffix(".manifest.json"))
        ):
            raise RuntimeError("RDT checkpoint/input contract or prompt embeddings mismatch")
        fingerprint = verify_fingerprint(checkpoint_root)
        checkpoint_metadata.update({
            "content_tree_sha256": fingerprint["content_tree_sha256"],
            "contract_sha256": sha256(target_path),
        })
    prompt_embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    if target_contract is not None:
        # The upstream constructor creates the RDT core in BF16 by default;
        # loading FP32 fine-tuned tensors through it irreversibly rounds them
        # BEFORE a later .float(). Use the same lossless loader as training.
        from VLA.rdt.checkpoint_loading import load_checkpoint

        policy, _ = load_checkpoint(checkpoint_root, target_contract["action_horizon"], dtype=torch.float32)
        policy = policy.to("cuda").eval()
        if any(parameter.dtype != torch.float32 for parameter in policy.parameters()):
            raise RuntimeError("RDT target-domain deployment downcast FP32 checkpoint weights")
        from VLA.rdt.signal_precision import PROFILE, install_fp32_signal_islands, signal_dtype
        precision_profile = target_contract.get("signal_precision", "bf16")
        input_signal_dtype = signal_dtype(precision_profile)
        if precision_profile == PROFILE:
            install_fp32_signal_islands(policy)
    else:
        # Preserve the official historical zero-shot path and its numerics.
        policy = RDTRunner.from_pretrained(args.checkpoint).to("cuda", dtype=torch.bfloat16).eval()
        precision_profile, input_signal_dtype = "bf16", torch.bfloat16
    if target_contract is not None and policy.pred_horizon != target_contract["action_horizon"]:
        raise RuntimeError("RDT model weights/config and target action horizon disagree")
    vision = SiglipVisionTower(args.vision, args=None).to("cuda", dtype=torch.bfloat16).eval()
    processor = vision.image_processor
    arm_indices = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)]
    gripper_index = STATE_VEC_IDX_MAPPING["right_gripper_open"]
    # Berkeley's EEF values are expressed in that robot installation's world
    # frame, which is not recoverable from the released checkpoint.  RDT was
    # explicitly pretrained with per-dimension state masks across heterogeneous
    # robots, so advertise only the compatible joint + gripper dimensions
    # instead of feeding numerically valid but semantically false PyBullet EEF
    # coordinates.  The predicted representation remains absolute joint state.
    valid_indices = arm_indices + [gripper_index]
    previous_images: dict[str, Image.Image] = {}
    app = Flask(__name__)

    @app.get("/metadata")
    def metadata():
        return jsonify({
            "protocol": "pybullet_ur5e_robotiq_rdt_v1",
            "model": "rdt", "target_domain": target_contract is not None,
            "control_hz": 10 if target_contract is not None else 5,
            "action_horizon": int(policy.pred_horizon), "checkpoint": checkpoint_metadata,
            "weight_dtype": "float32" if target_contract is not None else "bfloat16",
            "compute": "bfloat16_autocast" if target_contract is not None else "bfloat16",
            "signal_precision": precision_profile,
        })

    @app.post("/act")
    def act():
        payload = request.get_json(force=True)
        prompt = payload["prompt"]
        if prompt not in prompt_embeddings:
            return jsonify({"error": f"No precomputed T5 embedding for {prompt!r}"}), 400
        inputs = prepare_sensor_tensors(payload, processor, prompt_embeddings,
                                        target_domain=target_contract is not None, previous_images=previous_images)
        pixels = inputs["images"][0].to("cuda", dtype=torch.bfloat16)
        # Isolate the diffusion realization from previous requests and bind it
        # to the benchmark seed, prompt and actual observation step.
        seed_payload = json.dumps([int(payload.get("seed", 0)), prompt, int(payload.get("step", 0))])
        noise_seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:8], "little") & ((1 << 63) - 1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.manual_seed(noise_seed)
            image_tokens = vision(pixels).reshape(1, -1, vision.hidden_size)
            state = inputs["states"].to("cuda", dtype=input_signal_dtype)
            mask = inputs["state_elem_mask"].unsqueeze(1).to("cuda", dtype=torch.bfloat16)
            text = inputs["lang_tokens"].to("cuda", dtype=torch.bfloat16)
            precision = torch.autocast("cuda", dtype=torch.bfloat16) if target_contract is not None else nullcontext()
            with precision:
                trajectory = policy.predict_action(
                    lang_tokens=text,
                    lang_attn_mask=inputs["lang_attn_mask"].to("cuda"),
                    img_tokens=image_tokens,
                    state_tokens=state,
                    action_mask=mask,
                    # Official configs/dataset_control_freq.json records the
                    # Berkeley Autolab UR5 source trajectories at 5 Hz.
                    ctrl_freqs=inputs["ctrl_freqs"].to("cuda"),
                )[0]
            sim_joints = trajectory[:, arm_indices].float().cpu().numpy()
            if target_contract is None:
                sim_joints = sim_joints - JOINT_OFFSET
            gripper_open = trajectory[:, gripper_index : gripper_index + 1].float().cpu().numpy()
            gripper_open = np.clip(gripper_open, 0.0, 1.0)
            joint_actions = np.concatenate((sim_joints, gripper_open), axis=1)
        return jsonify({"joint_actions": joint_actions.tolist()})

    # Diffusion scheduler history and fork_rng are process-global mutable
    # state. Serialize requests rather than allowing overlapping denoising.
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
