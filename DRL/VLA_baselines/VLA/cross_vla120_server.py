"""Isolated OpenVLA and memory-conscious OFFICIAL CogACT inference.

CogACT retains its official weights, architectures, image transform and DiT
sampler. Loading uses mmap/empty initialization and BF16 VLM CPU offloading,
not another model or quantization. This port MUST pass its load/shape/value
checks; a failure is an integration failure, never a scored task failure.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OPENVLA_SNAPSHOT = Path("/home/uet/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f")


def metadata_only_llama_assets(output):
    """Reuse the SAME Llama2 tokenizer/config, never OpenVLA model weights."""
    from transformers import LlamaConfig
    config = json.loads((OPENVLA_SNAPSHOT / "config.json").read_text())
    if config["llm_backbone_id"] != "llama2-7b-pure" or config["hf_llm_id"] != "meta-llama/Llama-2-7b-hf":
        raise ValueError("Cached assets are not the identical official Llama2 backbone")
    llm = LlamaConfig(**config["text_config"])
    if (llm.hidden_size, llm.intermediate_size, llm.num_hidden_layers,
            llm.num_attention_heads, llm.num_key_value_heads) != (4096, 11008, 32, 32, 32):
        raise ValueError("Llama2 architecture differs")
    output.mkdir(parents=True, exist_ok=True)
    llm.save_pretrained(output)
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        source, target = OPENVLA_SNAPSHOT / name, output / name
        if source.is_file() and not target.exists():
            target.symlink_to(source)
    return output


def load_cogact(checkpoint, output, *, meta_only=False):
    import torch
    import timm
    from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
    from accelerate.utils import set_module_tensor_to_device
    sys.path.insert(0, str(ROOT / "VLA/third_party/cogact"))
    from vla import load_vla
    from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS

    checkpoint = Path(checkpoint).resolve()
    if checkpoint != (ROOT / "VLA/checkpoints/cogact-small/checkpoints/CogACT-Small.pt").resolve():
        raise ValueError("Only the pinned official CogACT-Small is eligible")
    if checkpoint.stat().st_size != 30215225166:
        raise ValueError("Incomplete official CogACT checkpoint")
    source_load, source_create, source_assign = torch.load, timm.create_model, torch.nn.Module.load_state_dict
    source_register = torch.nn.Module.register_parameter
    original = source_load(checkpoint, map_location="cpu", mmap=True, weights_only=True)["model"]
    if set(original) != {"action_model", "vision_backbone", "llm_backbone", "projector"}:
        raise ValueError("Unexpected official checkpoint components")
    assets = metadata_only_llama_assets(output / "llama2_config_tokenizer_only")
    registry = {**LLAMA2_MODELS["llama2-7b-pure"], "hf_hub_path": str(assets)}

    def mapped_load(path, *args, **kwargs):
        if Path(path).resolve() != checkpoint:
            raise ValueError("Unexpected auxiliary weight load; no substitute weights permitted")
        return {"model": original}

    def empty_vision(*args, **kwargs):
        kwargs["pretrained"] = False  # all vision weights must come from the official CogACT checkpoint
        return source_create(*args, **kwargs)

    def assigned_state(module, state_dict, strict=True, **kwargs):
        # Accelerate's empty-init registration hook would otherwise convert
        # newly assigned mmap Parameters back to meta. Only the assignment
        # operation uses normal registration; constructor allocation stays meta.
        with patch.object(torch.nn.Module, "register_parameter", source_register):
            return source_assign(module, state_dict, strict=strict, assign=True)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(LLAMA2_MODELS, {"llama2-7b-pure": registry}))
        stack.enter_context(patch.object(torch, "load", mapped_load))
        stack.enter_context(patch.object(timm, "create_model", empty_vision))
        stack.enter_context(patch.object(torch.nn.Module, "load_state_dict", assigned_state))
        stack.enter_context(init_empty_weights())
        model = load_vla(str(checkpoint), load_for_training=False,
                         action_model_type="DiT-S", future_action_window_size=15, action_dim=7)
    prefixes = {"action_model": "action_model.", "vision_backbone": "vlm.vision_backbone.",
                "llm_backbone": "vlm.llm_backbone.", "projector": "vlm.projector."}
    values = model.state_dict()
    expected = {prefixes[group] + name: tensor for group, state in original.items() for name, tensor in state.items()}
    if set(values) != set(expected):
        raise ValueError("Official checkpoint/initialized architecture tensor set differs")
    if any(values[name].shape != tensor.shape or values[name].data_ptr() != tensor.data_ptr()
           for name, tensor in expected.items()):
        raise ValueError("Mmap assignment did not preserve every original tensor")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Uninitialized meta parameters remain")
    count = sum(t.numel() for t in expected.values())
    if count != 7553713991:
        raise ValueError("Official tensor count differs")
    audit = {"checkpoint": str(checkpoint), "all_checkpoint_tensor_shapes_and_storage_assignment_exact": True,
        "tensor_values": count, "tensors": len(expected), "quantization_used": False,
        "another_models_weights_used": False, "llama_assets_source": str(OPENVLA_SNAPSHOT),
        "llama_assets_are_config_tokenizer_ONLY": True, "GPU_load_or_inference_established": False,
        "learned_task_success_established": False}
    (output / "load_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if meta_only:
        return model, audit
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required for this official sampler")
    free, _ = torch.cuda.mem_get_info()
    if free < 14500 * 2**20:
        raise RuntimeError("CogACT startup needs a quiet GPU; no competing fit allowed")
    special = {name: torch.float32 for name, _ in model.named_parameters() if name.startswith("action_model.")}
    mapping = infer_auto_device_map(model, max_memory={0: 12500 * 2**20, "cpu": 4 * 2**30},
        no_split_module_classes=["LlamaDecoderLayer", "Block"], dtype=torch.bfloat16, special_dtypes=special)

    def device_for(name):
        candidates = [(prefix, device) for prefix, device in mapping.items()
                      if prefix == "" or name == prefix or name.startswith(prefix + ".")]
        if not candidates:
            raise ValueError("Tensor lacks declared placement")
        return max(candidates, key=lambda row: len(row[0]))[1]

    # Stream directly from read-only mmap to destination; never allocate a full
    # 30GB FP32 or 15GB BF16 anonymous CPU model during placement.
    for name, parameter in list(model.named_parameters()) + list(model.named_buffers()):
        dtype = torch.bfloat16 if name.startswith("vlm.") and parameter.is_floating_point() else parameter.dtype
        if parameter.is_meta:
            raise ValueError("Missing official runtime buffer")
        set_module_tensor_to_device(model, name, device_for(name), value=parameter, dtype=dtype)
    placed = model.state_dict()
    for name, source in expected.items():
        dtype = torch.bfloat16 if name.startswith("vlm.") else source.dtype
        target = placed[name]
        if target.dtype != dtype:
            raise ValueError("Official BF16-VLM/FP32-action precision differs")
        indices = [0, source.numel() // 2, source.numel() - 1]
        source_probe = source.reshape(-1)[indices].to(dtype)
        target_probe = target.reshape(-1)[indices].detach().cpu()
        if not torch.equal(source_probe, target_probe):
            raise ValueError("Transferred weight probe differs from official precision conversion")
    model = dispatch_model(model, device_map=mapping, main_device="cuda:0").eval()
    model.vlm.requires_grad_(False)
    if model.vlm.device.type != "cuda":
        raise ValueError("Official front image/cognition path lacks GPU execution device")
    audit.update(GPU_load_or_inference_established=False, GPU_load_established=True,
        all_tensor_three_value_transfer_probes_exact=True, device_map={k: str(v) for k, v in mapping.items()},
        memory_profile="bf16_vlm_fp32_action_cpu_offload_no_quantization")
    (output / "load_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return model, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("openvla", "cogact"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=5520)
    parser.add_argument("--unnorm-key", default="berkeley_autolab_ur5")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from flask import Flask, jsonify, request
    from PIL import Image

    output = args.audit_output or ROOT / "VLA/audits/cross_vla120_20261013/cogact/runtime_qualification"
    if args.model == "cogact":
        output.mkdir(parents=True, exist_ok=False)
        model, audit = load_cogact(args.checkpoint, output, meta_only=args.qualify_only)
        if args.qualify_only:
            print(json.dumps(audit), flush=True)
            return
        stats = model.get_action_stats(args.unnorm_key)
        if model.get_action_dim(args.unnorm_key) != 7:
            raise ValueError("Official CogACT normalization is not seven-dimensional")
        metadata = {"protocol": "pybullet_ur5e_robotiq_cogact_v1", "model": "cogact",
            "checkpoint": str(Path(args.checkpoint).resolve()), "unnorm_key": args.unnorm_key,
            "action_dim": 7, "action_horizon": 16, "gripper_convention": "1=open,0=closed",
            "memory_profile": audit["memory_profile"], "target_domain": False,
            "action_stats_sha256": hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()}
    else:
        if args.qualify_only:
            raise ValueError("OpenVLA requires an actual quantized load, not a meta qualification")
        from VLA.openvla.adapter import OpenVLAPolicy
        model = OpenVLAPolicy(checkpoint=args.checkpoint, unnorm_key=args.unnorm_key)
        model._load()
        metadata = {"model": "openvla", "checkpoint": args.checkpoint,
            "unnorm_key": args.unnorm_key, "action_contract": "canonical_delta_ee_7d", "target_domain": False}
    app = Flask(__name__)

    @app.get("/metadata")
    def describe():
        return jsonify(metadata)

    @app.post("/act")
    def act():
        if args.model != "openvla":
            return jsonify({"error": "Wrong model endpoint"}), 409
        payload = request.get_json()
        pixels = np.asarray(Image.open(BytesIO(base64.b64decode(payload["front_rgb_png"]))).convert("RGB"))
        action = model.predict(SimpleNamespace(front_rgb=pixels, prompt=payload["prompt"]))
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/incompatible native prediction")
        return jsonify({"action": action.tolist()})

    @app.post("/api/inference")
    def cogact_act():
        if args.model != "cogact":
            return jsonify({"error": "Wrong model endpoint"}), 409
        query = json.load(request.files["json"].stream)
        if query["unnorm_key"] != args.unnorm_key:
            return jsonify({"error": "Normalization differs"}), 409
        pixels = Image.open(request.files["images"].stream).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        seed_text = json.dumps([int(query.get("seed", 0)), query["task_description"], int(query.get("inference_index", 0))])
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little") & ((1 << 63) - 1)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.manual_seed(seed)
            actions, _ = model.predict_action(image=pixels, instruction=query["task_description"],
                unnorm_key=args.unnorm_key, do_sample=False, cfg_scale=1.5, use_ddim=True, num_ddim_steps=10)
        if not np.isfinite(actions).all() or actions.shape != (16, 7):
            raise ValueError("Malformed official CogACT action chunk")
        return jsonify(actions.tolist())

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
