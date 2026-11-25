"""Official native VLA inference with explicit privileged LANGUAGE feedback.

Initial integration probes are unadapted official weights, not matched
post-fine-tuning results. No IK, world state, oracle or action correction exists
in this process. Kinematics, if needed, remains in the simulation client.
"""
import argparse
import base64
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.privileged_prompt import PROMPT_SCHEMA, augment_prompt, parse_feedback, validate_state

TOKEN_LIMIT = 512


def strict_fast_tokenizer_class():
    from VLA.pi0.fast_tokenizer import UR5FASTTokenizer, _find_subsequence
    class StrictFAST(UR5FASTTokenizer):
        def extract_actions(self, tokens, action_horizon, action_dim):
            values = np.asarray(tokens, np.int64).reshape(-1)
            marker = np.asarray(self._paligemma_tokenizer.encode("Action: "), np.int64)
            delimiter = np.asarray(self._paligemma_tokenizer.encode("|", add_eos=True), np.int64)
            position = _find_subsequence(values, marker)
            if position is None: raise ValueError("Malformed FAST neural output: no Action marker (no zero fallback)")
            begin = position + len(marker); hits = np.flatnonzero(values[begin:] == delimiter[0])
            if not len(hits) or hits[0] == 0: raise ValueError("Malformed FAST action delimiter/pieces (no fallback)")
            pieces = values[begin:begin + int(hits[0])]
            action_tokens = self._act_tokens_to_paligemma_tokens(pieces)
            # UniversalActionProcessor.decode catches malformed coefficient
            # counts internally and fabricates an all-zero chunk. A finite
            # output/shape check AFTER decode cannot detect that fallback.
            from VLA.pi0.fast_decoder_integrity import require_actual_coefficients
            require_actual_coefficients(self._fast_tokenizer,action_tokens,action_horizon,action_dim)
            result = np.asarray(self._fast_tokenizer.decode([action_tokens.tolist()],
                time_horizon=action_horizon, action_dim=action_dim)[0], np.float32)
            if result.shape != (action_horizon, action_dim) or not np.isfinite(result).all():
                raise ValueError("Malformed FAST decoded action, no fabricated zero chunk")
            return result
    return StrictFAST


def openpi_config(model_name, checkpoint):
    from openpi import transforms
    from openpi.models import model as model_api, pi0_config, pi0_fast
    from openpi.training import config

    @dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        model_type: model_api.ModelType
        def __call__(self, data):
            state = np.r_[np.asarray(data["joints"], np.float32).reshape(6),
                          np.asarray(data["gripper"], np.float32).reshape(1)]
            base, wrist = np.asarray(data["base_rgb"]), np.asarray(data["wrist_rgb"])
            if self.model_type == model_api.ModelType.PI0_FAST:
                images = {"base_0_rgb": base, "base_1_rgb": np.zeros_like(base), "wrist_0_rgb": wrist}
                state = np.pad(state, (0, 32 - len(state)))
            else:
                images = {"base_0_rgb": base, "left_wrist_0_rgb": wrist}
            result = {"state": state, "image": images,
                      "image_mask": {name: np.True_ for name in images}, "prompt": data["prompt"]}
            if "actions" in data: result["actions"] = data["actions"]
            return result

    @dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self, data):
            return {"actions": np.asarray(data["actions"])[..., :7]}

    model = (pi0_config.Pi0Config(max_token_len=TOKEN_LIMIT) if model_name == "pi0" else
             pi0_fast.Pi0FASTConfig(max_token_len=TOKEN_LIMIT, fast_model_tokenizer=strict_fast_tokenizer_class()))
    def groups(current):
        return transforms.Group(inputs=[Inputs(current.model_type)], outputs=[Outputs()]).push(
            inputs=[transforms.DeltaActions(transforms.make_bool_mask(6, -1))],
            outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(6, -1))])
    return config.TrainConfig(name=model_name + "_privileged_language_integration",
        model=model, data=config.SimpleDataConfig(
            assets=config.AssetsConfig(assets_dir=checkpoint + "/assets", asset_id="ur5e"),
            data_transforms=groups, base_config=config.DataConfig(prompt_from_task=True)))


def decode_image(encoded):
    from PIL import Image
    pixels = np.asarray(Image.open(BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB"))
    if pixels.shape != (384, 384, 3): raise ValueError("Native shared camera input shape differs")
    return pixels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("pi0", "pi0_fast", "openvla", "cogact"), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unnorm-key", default="berkeley_autolab_ur5")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-memory-cap-mib", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from flask import Flask, jsonify, request
    model_name = args.model
    if model_name in {"pi0", "pi0_fast"}:
        from openpi.policies import policy_config
        from openpi.models.tokenizer import PaligemmaTokenizer
        from VLA.pi0.fast_tokenizer import UR5FASTTokenizer
        config = openpi_config(model_name, args.checkpoint)
        native = policy_config.create_trained_policy(config, args.checkpoint)
        tokenizer = PaligemmaTokenizer(4096) if model_name == "pi0" else UR5FASTTokenizer(4096)
        horizon, dimension = config.model.action_horizon, config.model.action_dim
        action_kind = "absolute_joint_targets_and_closure"
    elif model_name == "openvla":
        import torch
        if args.gpu_memory_cap_mib:
            torch.cuda.set_per_process_memory_fraction(args.gpu_memory_cap_mib * 2**20 / torch.cuda.get_device_properties(0).total_memory)
        from VLA.openvla.adapter import OpenVLAPolicy
        native = OpenVLAPolicy(checkpoint=args.checkpoint, unnorm_key=args.unnorm_key)
        native._load()
        adapter_root = Path(args.checkpoint)
        if (adapter_root / "adapter_config.json").is_file():
            from VLA.openvla.finetune import validate_checkpoint_manifest
            validate_checkpoint_manifest(adapter_root)
            contract = json.loads((adapter_root / "train_config.json").read_text())
            if contract.get("privileged_prompt_schema") != PROMPT_SCHEMA or contract.get("privileged_model_inputs") is not True:
                raise ValueError("Ordinary adapter cannot be relabelled as privileged-trained")
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            saved = load_file(str(adapter_root / "adapter_model.safetensors"), device="cpu")
            loaded = get_peft_model_state_dict(native.model)
            if set(saved) != set(loaded) or any(not torch.equal(v, loaded[k].detach().cpu()) for k, v in saved.items()):
                raise ValueError("Every saved learned adapter value must reload exactly")
            (args.output / "adapter_reload.json").write_text(json.dumps({"passed": True,
                "all_saved_adapter_values_exact": True, "tensors": len(saved),
                "values": sum(t.numel() for t in saved.values())}, indent=2) + "\n")
        tokenizer = native.processor.tokenizer
        horizon, dimension = 1, 7
        action_kind = "canonical_delta_ee_7d"
    else:
        from VLA.cross_vla120_server import load_cogact
        native, _ = load_cogact(args.checkpoint, args.output / "official_load")
        tokenizer = native.vlm.llm_backbone.tokenizer
        horizon, dimension = 16, 7
        action_kind = "native_delta_ee_and_open_gripper"
    adapted = model_name == "openvla" and (Path(args.checkpoint) / "adapter_config.json").is_file()
    metadata = {"model": model_name, "checkpoint": args.checkpoint, "unnorm_key": args.unnorm_key,
        "protocol": "privileged_native_language_v1", "privileged_schema": PROMPT_SCHEMA,
        "target_domain_finetuned": adapted, "adaptation_regime": "bounded_privileged_prompt_QLoRA_pilot" if adapted else "official_weights_privileged_prompt_integration_only",
        "action_kind": action_kind, "action_horizon": horizon, "action_dim": dimension,
        "pi0_token_limit": TOKEN_LIMIT if model_name in {"pi0", "pi0_fast"} else None,
        "oracle_or_ik_controller": False, "same_prompt_builder_for_training_and_inference": True}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    app = Flask(__name__)

    @app.get("/metadata")
    def describe(): return jsonify(metadata)

    @app.post("/act")
    def act():
        payload = request.get_json()
        state = validate_state(payload["state18"])
        prompt = augment_prompt(payload["instruction"], state)
        if payload.get("prompt") != prompt or np.max(np.abs(parse_feedback(prompt) - state)) > .000051:
            return jsonify(error="Actual privileged prompt/feedback identity differs"), 409
        front = decode_image(payload["front_rgb_png"])
        if model_name in {"pi0", "pi0_fast"}:
            from openpi_client import image_tools
            joints = (state[:6] + np.pi) % (2 * np.pi) - np.pi
            if model_name == "pi0":
                _, mask = tokenizer.tokenize(prompt)
            else:
                # The native base model has 32 state slots, not this candidate
                # fine-tuning recipe's seven. Audit that ACTUAL runtime prefix.
                _, mask, _, _ = tokenizer.tokenize(prompt, np.zeros(dimension, np.float32), None)
            if int(mask.sum()) > TOKEN_LIMIT:
                return jsonify(error="Native prefix would truncate; no action executed"), 422
            raw = {"base_rgb": image_tools.convert_to_uint8(image_tools.resize_with_pad(front, 224, 224)),
                "wrist_rgb": image_tools.convert_to_uint8(image_tools.resize_with_pad(decode_image(payload["wrist_rgb_png"]), 224, 224)),
                "joints": joints.astype(np.float32), "gripper": state[6:7].copy(), "prompt": prompt}
            actual_prepared = native._input_transform(dict(raw))
            if int(np.asarray(actual_prepared["tokenized_prompt_mask"]).sum()) >= TOKEN_LIMIT:
                return jsonify(error="Actual normalized native prefix exhausted budget; no action executed"), 422
            kwargs = {}
            if model_name == "pi0":
                prompt_key = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "little")
                rng = np.random.default_rng(np.random.SeedSequence((int(payload["seed"]), prompt_key, int(payload["inference_index"]))))
                kwargs["noise"] = rng.standard_normal((horizon, dimension), dtype=np.float32)
            result = np.asarray(native.infer(raw, **kwargs)["actions"], dtype=np.float32)
        elif model_name == "openvla":
            tokens = tokenizer("In: What action should the robot take to " + prompt.lower() + "?\nOut: ", truncation=False).input_ids
            if len(tokens) + 256 + 9 > min(2048, tokenizer.model_max_length):
                return jsonify(error="Native language would truncate"), 422
            result = native.predict(SimpleNamespace(front_rgb=front, prompt=prompt))[None, :]
        else:
            from PIL import Image
            import torch
            builder = native.vlm.get_prompt_builder()
            builder.add_turn(role="human", message=f"What action should the robot take to {prompt.lower()}?")
            ids = tokenizer(builder.get_prompt(), truncation=False).input_ids
            if len(ids) + 258 > min(2048, tokenizer.model_max_length):
                return jsonify(error="CogACT native language would truncate"), 422
            key = json.dumps([int(payload["seed"]), prompt, int(payload["inference_index"])])
            noise_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") & ((1 << 63) - 1)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                torch.manual_seed(noise_seed)
                result, _ = native.predict_action(image=Image.fromarray(front).resize((224, 224), Image.Resampling.LANCZOS),
                    instruction=prompt, unnorm_key=args.unnorm_key, do_sample=False,
                    cfg_scale=1.5, use_ddim=True, num_ddim_steps=10)
        result = np.asarray(result, dtype=np.float32)
        if result.ndim != 2 or result.shape[1] != 7 or not len(result) or not np.isfinite(result).all():
            return jsonify(error="Malformed neural output; no fallback permitted"), 422
        return jsonify(actions=result.tolist(), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                       action_source="native_neural_model", action_kind=action_kind)
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__": main()
