"""Isolated RDT+SimState server; only learned native joints produce actions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request

# Imports the pinned upstream path setup and exact ordinary image processor.
from VLA.rdt.server import RDT_ROOT
from VLA.common.provenance import sha256_file
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.rdt.simstate_inputs import SCHEMA, EXTRA_SLOTS, prepare_privileged_sensor_tensors


def validate_contract(root, embeddings):
    contract = json.loads((root/"pybullet_rdt_config.json").read_text())
    state = contract.get("simstate", {})
    if (contract.get("format_version") != 2 or contract.get("model_family") != "rdt_simstate"
            or contract.get("privileged_model_inputs") is not True or state.get("schema") != SCHEMA
            or state.get("conditioning_slots_NOT_motor_DOFs") != list(EXTRA_SLOTS)
            or state.get("oracle_controller_at_deployment") is not False
            or state.get("future_state_or_actions_in_input") is not False
            or contract.get("valid_indices") != [0,1,2,3,4,5,10] or contract.get("control_hz") != 10
            or contract.get("state_dim") != 128 or contract.get("action_horizon") != 10
            or contract.get("joint", {}).get("convention") != "wrap_to_minus_pi_pi"
            or contract.get("joint", {}).get("offset") != [0.]*6
            or contract.get("scaling") != {"state": "identity_SI", "action": "identity_SI"}):
        raise ValueError("Wrong privileged model/input/action contract")
    checks = [(Path(state["projection"])/"audit.json", state["projection_audit_sha256"]),
              (Path(__file__).with_name("simstate_inputs.py"), state["input_implementation_sha256"]),
              (Path(__file__).resolve().parents[1]/"simstate_feedback.py", state["feature_builder_sha256"]),
              (embeddings, contract["language"]["embeddings_sha256"]),
              (embeddings.with_suffix(".manifest.json"), contract["language"]["manifest_sha256"])]
    if any(sha256_file(path) != digest for path, digest in checks):
        raise ValueError("Privileged feature/projection/language contract changed")
    manifest = json.loads((root/"vla_checkpoint_manifest.json").read_text())
    if (manifest.get("privileged_model_inputs") is not True
            or manifest["model_weights_sha256"] != sha256_file(root/"pytorch_model.bin")
            or manifest["target_contract_sha256"] != sha256_file(root/"pybullet_rdt_config.json")):
        raise ValueError("Real privileged checkpoint save receipt missing")
    return contract


def main():
    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    from VLA.rdt.checkpoint_loading import load_checkpoint
    from VLA.rdt.signal_precision import PROFILE, install_fp32_signal_islands, signal_dtype
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5734)
    args = parser.parse_args()
    root = args.checkpoint.resolve(); contract = validate_contract(root, args.embeddings.resolve())
    identity = verify_fingerprint(root, full=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6500*1024**2/torch.cuda.get_device_properties(0).total_memory)
    policy, _ = load_checkpoint(root, contract["action_horizon"], dtype=torch.float32)
    policy = policy.to("cuda").eval()
    if any(p.dtype != torch.float32 for p in policy.parameters()):
        raise ValueError("Lossy privileged checkpoint reload")
    precision = contract["signal_precision"]
    if precision == PROFILE:
        install_fp32_signal_islands(policy)
    signal = signal_dtype(precision)
    vision = SiglipVisionTower(str(args.vision), args=None).to("cuda", dtype=torch.bfloat16).eval()
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    previous = {}; app = Flask(__name__)

    @app.get("/metadata")
    def metadata():
        return jsonify(protocol="pybullet_ur5e_robotiq_rdt_v1", model="rdt", target_domain=True,
            control_hz=10, action_horizon=contract["action_horizon"], weight_dtype="float32",
            compute="bfloat16_autocast", signal_precision=precision, simstate_schema=SCHEMA,
            privileged_model_inputs=True, oracle_controller_at_deployment=False,
            feature_builder_sha256=contract["simstate"]["feature_builder_sha256"],
            checkpoint={"path": str(root), "content_tree_sha256": identity["content_tree_sha256"]})

    @app.post("/act")
    def act():
        payload = request.get_json(force=True)
        if payload.get("prompt") not in embeddings:
            return jsonify(error="Missing exact frozen T5 prompt"), 400
        try:
            inputs = prepare_privileged_sensor_tensors(payload, vision.image_processor, embeddings,
                previous_images=previous, stats=contract["simstate"]["train_only_state18_stats"])
        except (ValueError, KeyError) as error:
            return jsonify(error=str(error)), 400
        seed_payload = json.dumps([int(payload.get("seed", 0)), payload["prompt"], int(payload.get("step", 0))])
        seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:8], "little") & ((1<<63)-1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.manual_seed(seed)
            images = vision(inputs["images"][0].to("cuda", dtype=torch.bfloat16)).reshape(1, -1, vision.hidden_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                trajectory = policy.predict_action(lang_tokens=inputs["lang_tokens"].to("cuda", dtype=torch.bfloat16),
                    lang_attn_mask=inputs["lang_attn_mask"].to("cuda"), img_tokens=images,
                    state_tokens=inputs["states"].to("cuda", dtype=signal),
                    action_mask=inputs["state_elem_mask"].unsqueeze(1).to("cuda", dtype=torch.bfloat16),
                    ctrl_freqs=inputs["ctrl_freqs"].to("cuda"))[0]
            joints = trajectory[:, :6].float().cpu().numpy()
            opened = np.clip(trajectory[:, 10:11].float().cpu().numpy(), 0., 1.)
            actions = np.concatenate((joints, opened), axis=1)
        if not np.isfinite(actions).all():
            return jsonify(error="Non-finite learned trajectory; no action fallback"), 500
        return jsonify(joint_actions=actions.tolist())

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
