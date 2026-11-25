"""Load official RDT weights at a task-appropriate action chunk length.

Only the action-position embedding rows outside the requested horizon are
removed. Every retained learned weight is copied exactly; the RDT blocks,
conditioning encoders, sparse state layout and diffusion objective are retained.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_checkpoint(checkpoint: Path, action_horizon: int, *, dtype=torch.float32):
    from models.rdt_runner import RDTRunner

    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text())
    source_horizon = int(config["pred_horizon"])
    if not 1 <= action_horizon <= source_horizon:
        raise ValueError("RDT horizon adaptation only supports retaining a pretrained prefix")
    runner = RDTRunner(
        action_dim=int(config["action_dim"]), pred_horizon=int(action_horizon), config=config,
        lang_token_dim=int(config["lang_token_dim"]), img_token_dim=int(config["img_token_dim"]),
        state_token_dim=int(config["state_token_dim"]),
        max_lang_cond_len=int(config["max_lang_cond_len"]), img_cond_len=int(config["img_cond_len"]),
        lang_pos_embed_config=config.get("lang_pos_embed_config"),
        img_pos_embed_config=config.get("img_pos_embed_config"), dtype=dtype,
    )
    state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    key = "model.x_pos_embed"
    if tuple(state[key].shape[:2]) != (1, source_horizon + 3):
        raise RuntimeError("Unexpected RDT action-position embedding shape")
    state[key] = state[key][:, : action_horizon + 3].contiguous()
    runner.load_state_dict(state, strict=True)
    # Preserve the official flattened Hub config format for later native
    # from_pretrained() inference, including the actual shorter horizon.
    runner._hub_mixin_config = {**config, "pred_horizon": int(action_horizon)}
    return runner, {
        "source_action_horizon": source_horizon,
        "action_horizon": int(action_horizon),
        "adaptation": "retain the learned timestep/frequency/state and first H action position rows",
        "retained_weights_copied_strictly": True,
    }
