#!/usr/bin/env python3
"""CogACT-Small server adapted to square PyBullet camera frames.

Upstream ``scripts/deploy.py`` asserts that input images are landscape.  The
shared benchmark intentionally uses square cameras, so this wrapper keeps the
official model/load path but performs a direct 224x224 resize.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="CogACT/CogACT-Small")
    parser.add_argument("--unnorm-key", default="berkeley_autolab_ur5")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    from vla import load_vla

    args = parse_args()
    model = load_vla(
        args.checkpoint,
        load_for_training=False,
        # CogACT-Small publishes a DiT-S action head.  The upstream registry
        # accepts the canonical key ``DiT-S`` (not the display name
        # ``DiT-Small``); using the latter fails with KeyError after the large
        # checkpoint has already been loaded.
        action_model_type="DiT-S",
        future_action_window_size=15,
        action_dim=7,
    )
    model.vlm = model.vlm.to(torch.bfloat16)
    action_stats = model.get_action_stats(args.unnorm_key)
    if model.get_action_dim(args.unnorm_key) != 7:
        raise RuntimeError("CogACT benchmark requires seven-dimensional actions")
    model = model.to("cuda").eval()
    app = Flask(__name__)

    @app.get("/metadata")
    def metadata():
        checkpoint = args.checkpoint
        if Path(checkpoint).is_file():
            checkpoint = str(Path(checkpoint).resolve())
        return jsonify({
            "protocol": "pybullet_ur5e_robotiq_cogact_v1", "model": "cogact",
            "checkpoint": checkpoint, "unnorm_key": args.unnorm_key,
            "action_dim": 7, "action_horizon": 16,
            "gripper_convention": "1=open,0=closed",
            "action_stats_sha256": hashlib.sha256(
                json.dumps(action_stats, sort_keys=True).encode()
            ).hexdigest(),
        })

    @app.post("/api/inference")
    def infer():
        image_file = request.files["images"]
        query_file = request.files["json"]
        query = json.load(query_file.stream)
        if query.get("unnorm_key") != args.unnorm_key:
            return jsonify({"error": "Action normalization mismatch"}), 409
        image = Image.open(BytesIO(image_file.read())).convert("RGB").resize(
            (224, 224), Image.Resampling.LANCZOS
        )
        seed_payload = json.dumps([
            int(query.get("seed", 0)), query["task_description"],
            int(query.get("inference_index", 0)),
        ])
        noise_seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:8], "little") & ((1 << 63) - 1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.manual_seed(noise_seed)
            actions, _ = model.predict_action(
                image=image,
                instruction=query["task_description"],
                unnorm_key=args.unnorm_key,
                do_sample=False,
                cfg_scale=args.cfg_scale,
                use_ddim=True,
                num_ddim_steps=args.ddim_steps,
            )
        # Return the whole action chunk.  The benchmark's --execute-steps flag
        # controls how many elements are consumed before observing again.
        return jsonify(actions.tolist())

    # Diffusion scheduler and fork_rng state are shared within this process.
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
