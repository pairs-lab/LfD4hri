"""Compare native uncached training vector field with cached inference field."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.tasks import TaskSpec
from VLA.smolvla.adapter import SmolVLAPolicyAdapter
from VLA.fingerprint_checkpoint import verify_fingerprint


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = verify_fingerprint(args.checkpoint)
    sensor = SimpleNamespace(closure=0.)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    adapter = SmolVLAPolicyAdapter(sensor, checkpoint=str(args.checkpoint))
    adapter._load()
    policy, model = adapter.model, adapter.model.model
    if not policy.config.use_cache:
        raise RuntimeError("Requires the actual deployed cache-enabled configuration")
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    index = json.loads((args.source / "index.json").read_text())
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for name in ("touch", "pick", "move", "pick_place"):
            item = next(row for row in index if row["task"]["name"] == name)
            task = TaskSpec.from_dict(item["task"])
            adapter.reset(task)
            with np.load(args.source / item["file"]) as episode:
                sensor.closure = float(episode["gripper_closure"][0])
                obs = VLAObservation(episode["front_rgb"][0], episode["wrist_rgb"][0],
                    episode["joint_positions"][0], 0., 1-sensor.closure, episode["ee_position"][0],
                    episode["ee_rotation_6d"][0], task.prompt, 0)
                processed = adapter.preprocess(adapter.build_sensor_batch(obs))
            images, masks = policy.prepare_images(processed)
            state = policy.prepare_state(processed)
            tokens, language_mask = processed["observation.language.tokens"], processed["observation.language.attention_mask"]
            shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
            noise = torch.randn(shape, device=state.device,
                                generator=torch.Generator(device=state.device).manual_seed(20261007))
            prefix, prefix_pad, prefix_att = model.embed_prefix(images, masks, tokens, language_mask, state)
            _, cache = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(prefix_pad, prefix_att),
                position_ids=torch.cumsum(prefix_pad, dim=1)-1, past_key_values=None,
                inputs_embeds=[prefix, None], use_cache=True)
            for time in (1., .5, .1):
                timestep = torch.full((1,), time, device=state.device)
                values = []
                hook = model.action_out_proj.register_forward_hook(lambda module, inputs, output: values.append(output.detach().float()))
                try:
                    model.forward(images, masks, tokens, language_mask, state,
                                  torch.zeros_like(noise), noise=noise, time=timestep)
                finally:
                    hook.remove()
                if len(values) != 1:
                    raise RuntimeError("Unexpected native action projection calls")
                uncached = values[0]
                cached = model.denoise_step(prefix_pad, cache, noise*time, timestep).float()
                repeat = model.denoise_step(prefix_pad, cache, noise*time, timestep).float()
                error = (uncached-cached)[..., :7]
                rows.append({"task": name, "time": time, "active7_flow_rmse": float(error.square().mean().sqrt()),
                    "active7_flow_max_abs_error": float(error.abs().max()),
                    "repeated_cached_max_abs_error": float((repeat-cached).abs().max()),
                    "finite": bool(torch.isfinite(cached).all() and torch.isfinite(uncached).all())})
                print(json.dumps(rows[-1]), flush=True)
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "source_only": True, "final_test_used": False, "rows": rows,
        "passed_numeric_parity": all(row["finite"] and row["active7_flow_rmse"] < .03 and
                                     row["repeated_cached_max_abs_error"] == 0 for row in rows),
        "numeric_rmse_review_limit": .03,
        "scope": "Normalized velocity-field parity; BF16 kernel/reduction differences possible. Not task success or a performance claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
