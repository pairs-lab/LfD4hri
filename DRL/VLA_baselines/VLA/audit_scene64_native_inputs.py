"""Actual tokenizer/image/normalization gates for the expanded VLA inputs.

Pilot receipts NEVER qualify a full fit. Full corpus must be rerun after its
physical extraction. Real native transform calls, not hand-made tensor shapes.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from VLA.common.provenance import sha256_file
from VLA.common.artifact_io import atomic_json
from VLA.scene64_dataset import Scene64NativeDataset
from VLA.scene64_conditioning import PROMPT_SCHEMA, NUMERIC_SCHEMA, normalization, parse_prompt, encode_numeric


def selected(data):
    ids = []
    for ei, row in enumerate(data.rows):
        if not data.pilot_only and sum(r["task"]["name"] == row["task"]["name"] for r in data.rows[:ei]) >= 4: continue
        first = data.ends[ei-1] if ei else 0; count = row["steps"]
        ids.extend(first+i for i in sorted({0, count//2, count-1}))
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection", type=Path, required=True); p.add_argument("--pilot", action="store_true")
    p.add_argument("--mode", choices=("language", "openpi", "numeric"), required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Keep failed preflights")
    args.output.mkdir(parents=True)
    data = Scene64NativeDataset(args.projection, pilot=args.pilot)
    indices = selected(data); report = {}; visuals = []
    if args.mode == "language":
        from transformers import AutoProcessor
        from VLA.audit_privileged_native_inputs import SNAPSHOT
        processor = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
        tokenizer = processor.tokenizer; maximum = 0; rounding = 0.
        for begin in range(0, len(data), 128):
            # Do not decompress128 image episodes just to audit language.
            from bisect import bisect_right
            from VLA.scene64_conditioning import augment_prompt
            prompts = [augment_prompt(data.rows[bisect_right(data.ends, i)]["task"]["prompt"], data.states[i])
                for i in range(begin, min(begin+128, len(data)))]
            texts = ["In: What action should the robot take to " + text.lower() + "?\nOut: " for text in prompts]
            tokens = tokenizer(texts, truncation=False).input_ids
            for offset, (prompt, ids) in enumerate(zip(prompts, tokens, strict=True)):
                maximum = max(maximum, len(ids))
                rounding = max(rounding, float(np.max(np.abs(parse_prompt(prompt)-data.states[begin+offset]))))
                if len(ids)+265 > min(2048, tokenizer.model_max_length): raise ValueError("Expanded language would truncate")
        for i in indices:
            raw = data[i]; tensor = processor.image_processor.apply_transform(Image.fromarray(raw["image"]))
            if tuple(tensor.shape) != (6, 224, 224) or not tensor.isfinite().all(): raise ValueError("Native image transform failed")
            path = args.output / f"front_{i}.png"; Image.fromarray(raw["image"]).save(path); visuals.append(str(path))
        report = {"models": ["openvla", "cogact"], "formatted_tokenized_frames": len(data),
            "max_instruction_tokens": maximum, "rounding_max_abs_error": rounding,
            "actual_OpenVLA_image_samples": len(indices), "CogACT_neural_cognition_certified": False}
    elif args.mode == "openpi":
        from openpi.models.tokenizer import PaligemmaTokenizer
        from VLA.pi0.fast_tokenizer import UR5FASTTokenizer
        # Measure before selecting a native budget; never hide truncation.
        continuous, fast = PaligemmaTokenizer(4096), UR5FASTTokenizer(4096)
        max_pi = max_fast = 0; rows = []
        for i in indices:
            raw = data[i]; _, mask = continuous.tokenize(raw["prompt"])
            state = np.r_[raw["joints"], raw["gripper"]]; action = raw["actions"].copy(); action[:, :6] -= state[:6]
            train, tm, _, loss = fast.tokenize(raw["prompt"], state, action)
            infer, im, _, _ = fast.tokenize(raw["prompt"], state, None)
            prefix = int(im.sum()); nt = int(tm.sum())
            if not np.array_equal(train[:prefix], infer[:prefix]) or np.any(loss[:prefix]) or not np.any(loss):
                raise ValueError("Actual FAST prefix/loss mask differs")
            max_pi = max(max_pi, int(mask.sum())); max_fast = max(max_fast, nt)
            if max(max_pi, max_fast) >= 1024: raise ValueError("Expanded native1024 budget exhausted")
            rows.append({"index": i, "pi0_tokens": int(mask.sum()), "FAST_prefix_tokens": prefix, "FAST_total_tokens": nt})
        report = {"models": ["pi0", "pi0_fast"], "native_samples": len(rows), "rows": rows,
            "max_pi0_tokens": max_pi, "max_FAST_tokens": max_fast, "required_native_budget": 1024,
            "full_FAST_codec_physics_certified": False}
    else:
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.types import PolicyFeature, FeatureType
        from lerobot.policies import make_pre_post_processors
        from VLA.rdt.scene67_inputs import append_scene_conditioning, EXTRA_SLOTS
        cfg = SmolVLAConfig(device="cpu", max_state_dim=67)
        cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(67,))
        stats, contract = normalization(data.states)
        tensors = {k: np.asarray(v, np.int64 if k=="count" else np.float32) for k,v in stats.items()}
        proc, _ = make_pre_post_processors(cfg, pretrained_path=None, dataset_stats={"observation.state": tensors},
            preprocessor_overrides={"device_processor": {"device": "cpu"}})
        maximum = 0.; rows = []
        # A real batched fetch/collate path must carry67, never the old32.
        for begin in range(0, min(len(data),128), 2):
            samples = data.__getitems__([begin, begin+1])
            batch = {"observation.state": torch.from_numpy(np.stack([s["numeric_scene67"] for s in samples])),
                "task": [s["original_prompt"] for s in samples]}
            padded = SmolVLAPolicy.prepare_state(SimpleNamespace(config=cfg), proc(batch))
            expected = (batch["observation.state"]-torch.as_tensor(tensors["mean"]))/(torch.as_tensor(tensors["std"])+1e-8)
            if padded.shape != (2,67) or not torch.isfinite(padded).all(): raise ValueError("Native67 was truncated/padded incorrectly")
            error = float((padded-expected).abs().max()); maximum = max(maximum, error)
            if error > 1e-5: raise ValueError("Native67 normalization differs")
            raw = np.stack([s["pre_action_scene64"] for s in samples]); native = torch.zeros(2,1,128)
            native[:,0,:6] = torch.from_numpy((raw[:,:6]+np.pi)%(2*np.pi)-np.pi)
            native[:,0,10] = torch.from_numpy(1-raw[:,6])
            appended = append_scene_conditioning(native, raw, stats)
            torch.testing.assert_close(appended[:,0,list(EXTRA_SLOTS)], expected[:,7:], rtol=0, atol=1e-5)
            rows.append({"indices": [begin, begin+1], "actual_shape": list(padded.shape), "error": error})
        report = {"models": ["smolvla", "rdt"], "native_batches": len(rows), "samples": 2*len(rows),
            "normalization_max_error": maximum, "numeric_names_count": 67,
            "RDT_observation_slots": list(EXTRA_SLOTS), "RDT_action_DOFs_unchanged": [0,1,2,3,4,5,10],
            "normalization_contract": contract}
    report.update(passed=True, pilot_only=args.pilot, training_qualified=False, mode=args.mode,
        projection_audit_sha256=sha256_file(args.projection / "audit.json"), prompt_schema=PROMPT_SCHEMA,
        numeric_schema=NUMERIC_SCHEMA, source_frames=len(data), neural_weights_loaded=False,
        source_images_and_actions_unchanged=True, new_model_SR=None,
        source_front_visuals=visuals, bound_files={str(path.resolve()): sha256_file(path) for path in (
            Path(__file__), Path(__file__).with_name("scene64_conditioning.py"), Path(__file__).with_name("scene64_dataset.py"),
            Path(__file__).with_name("scene64_schema.py"), Path(__file__).parent / "common/artifact_io.py")})
    atomic_json(args.output / "audit.json", report)
    print(json.dumps({k: v for k,v in report.items() if k not in {"rows", "normalization_contract"}}), flush=True)


if __name__ == "__main__": main()
