"""Actual native tokenizers, causal corpus and source views for other VLAs.

CPU preflight, NOT learned success or full fine-tuning qualification. There is
no robot controller here. The GPU queue may consume only a passing receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.privileged_native_dataset import PrivilegedNativeDataset
from VLA.privileged_prompt import PROMPT_SCHEMA, augment_prompt, parse_feedback

SNAPSHOT = Path("/home/uet/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f")


def selected_indices(dataset):
    selected = []
    for task in ("touch", "pick", "move", "pick_place"):
        episodes = [i for i, r in enumerate(dataset.rows) if r["task"]["name"] == task][:4]
        for ei in episodes:
            start = 0 if ei == 0 else dataset.ends[ei - 1]
            length = dataset.ends[ei] - start
            for offset in sorted(set([0, length // 2, length - 1])):
                selected.append(start + offset)
    return selected


def language_audit(args, dataset):
    from PIL import Image
    from transformers import AutoProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
    from prismatic.vla.action_tokenizer import ActionTokenizer
    tokenizer = processor.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer)
    limit = min(int(tokenizer.model_max_length), 2048)
    maximum = 0; quantization = 0.0; frames = 0
    # Every causal sidecar row is formatted AND tokenized. This is stronger
    # than a few reset samples; it is still not a closed-loop success test.
    with (args.output / "causal_prompts.jsonl").open("x") as stream:
        for ei, row in enumerate(dataset.rows):
            start = 0 if ei == 0 else dataset.ends[ei - 1]
            prompts = [augment_prompt(row["task"]["prompt"], s) for s in dataset.states[start:dataset.ends[ei]]]
            texts = ["In: What action should the robot take to " + p.lower() + "?\nOut: " for p in prompts]
            tokenized = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False).input_ids
            for fi, (prompt, ids) in enumerate(zip(prompts, tokenized, strict=True)):
                values = dataset.states[start + fi]
                quantization = max(quantization, float(np.max(np.abs(parse_feedback(prompt) - values))))
                maximum = max(maximum, len(ids))
                if len(ids) + 9 + 256 > limit:
                    raise ValueError("Feedback/action/image tokens would be truncated")
                stream.write(json.dumps({"index": start + fi, "file": row["file"], "frame": fi, "prompt": prompt}, separators=(",", ":")) + "\n")
                frames += 1
            if (ei + 1) % 100 == 0:
                print(f"native language full corpus {ei+1}/{len(dataset.rows)} episodes", flush=True)
    samples = []
    for index in selected_indices(dataset):
        raw = dataset[index]
        human = f"What action should the robot take to {raw['prompt'].lower()}?"
        inference = tokenizer("In: " + human + "\nOut: ", add_special_tokens=True).input_ids
        builder = PurePromptBuilder("openvla")
        builder.add_turn("human", human)
        builder.add_turn("gpt", action_tokenizer(raw["language_actions"][0]))
        training = tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids
        # The native training builder trims trailing whitespace before action
        # pieces. Audit the actual common prefix, retaining the infer marker.
        common = 0
        for a, b in zip(inference, training):
            if a != b: break
            common += 1
        if common < len(inference) - 2:
            raise ValueError("Native training/inference instruction tokens differ")
        transformed = processor.image_processor.apply_transform(Image.fromarray(raw["image"]))
        if tuple(transformed.shape) != (6, 224, 224) or not bool(transformed.isfinite().all()):
            raise ValueError("OpenVLA native image transform failed")
        # CogACT uses the same pinned Llama2-7b-pure instruction builder and
        # tokenizer, but adds [space, EOS] after the instruction. Explicitly
        # distinguish this from an assertion of identical cognition tensors.
        cogact_ids = tokenizer("In: " + human + "\nOut:", add_special_tokens=True, truncation=False).input_ids + [29871, 2]
        if len(cogact_ids) + 256 > limit:
            raise ValueError("CogACT feedback would truncate")
        image_path = args.output / f"input_{index}_front.png"
        Image.fromarray(raw["image"]).save(image_path)
        samples.append({"index": index, "task": dataset.rows[raw["episode_index"]]["task"]["name"],
            "instruction_common_prefix_tokens": common, "inference_tokens": len(inference),
            "training_tokens": len(training), "cogact_instruction_tokens": len(cogact_ids),
            "native_image_shape": list(transformed.shape), "source_front": str(image_path),
            "source_front_sha256": sha256_file(image_path), "causal_prompt": raw["prompt"]})
    return {"models": ["openvla", "cogact"], "all_frames_formatted_and_tokenized": frames,
        "maximum_instruction_tokens": maximum, "native_context_limit": limit,
        "reserved_action_tokens": 9, "reserved_image_tokens": 256,
        "feedback_quantization_max_error": quantization,
        "openvla_actual_image_transform_checked": True,
        "cogact_backbone_is_pinned_llama2_7b_pure": True,
        "cogact_actual_vision_or_neural_tensor_parity_certified": False,
        "samples": samples, "corpus_sha256": sha256_file(args.output / "causal_prompts.jsonl")}


def openpi_audit(args, dataset):
    from openpi.models.tokenizer import PaligemmaTokenizer
    from VLA.pi0.fast_tokenizer import UR5FASTTokenizer
    pi0 = PaligemmaTokenizer(max_len=512)
    fast = UR5FASTTokenizer(max_len=512)
    samples = []; max_pi0 = max_fast = 0
    for index in selected_indices(dataset):
        raw = dataset[index]
        # The token budget check uses a deliberately long tokenizer so no
        # hidden truncation can disappear from the receipt.
        tokens, mask = pi0.tokenize(raw["prompt"])
        if int(mask.sum()) >= 512: raise ValueError("pi0 prompt budget exhausted")
        state = np.r_[raw["joints"], raw["gripper"]].astype(np.float32)
        actions = raw["actions"].copy(); actions[:, :6] -= state[:6]
        # Length measurement for this short seven-DOF target contract. Actual
        # TRAIN statistics/codec physics replay still precede its long fit.
        ft, fm, ar, loss = fast.tokenize(raw["prompt"], state, actions)
        it, im, _, _ = fast.tokenize(raw["prompt"], state, None)
        length = int(fm.sum()); prefix = int(im.sum())
        if length >= 512 or not np.array_equal(ft[:prefix], it[:prefix]):
            raise ValueError("FAST truncation or training/inference prefix mismatch")
        if not np.any(loss) or np.any(loss[:prefix]):
            raise ValueError("FAST teacher action leaked into conditioning loss mask")
        max_pi0 = max(max_pi0, int(mask.sum())); max_fast = max(max_fast, length)
        samples.append({"index": index, "pi0_instruction_tokens": int(mask.sum()),
            "fast_instruction_state_tokens": prefix, "fast_with_native_action_tokens": length,
            "training_inference_prefix_exact": True})
    return {"models": ["pi0", "pi0_fast"], "selected_native_samples": len(samples),
        "maximum_pi0_prompt_tokens": max_pi0, "maximum_fast_prompt_state_action_tokens": max_fast,
        "required_new_token_limit": 512, "old_pi0_48_token_limit_is_not_acceptable": max_pi0 > 48,
        "old_fast_180_token_limit_is_not_acceptable": max_fast > 180,
        "full_corpus_token_bound_or_fast_physics_certified": False, "samples": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("language", "openpi"), required=True)
    p.add_argument("--projection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = PrivilegedNativeDataset(args.projection)
    report = language_audit(args, dataset) if args.mode == "language" else openpi_audit(args, dataset)
    report.update(passed=True, schema=PROMPT_SCHEMA, projection=str(args.projection.resolve()),
        projection_audit_sha256=sha256_file(args.projection / "audit.json"),
        source=str(dataset.source), source_index_sha256=dataset.audit["source_index_sha256"],
        source_manifest_sha256=dataset.audit["source_manifest_sha256"],
        episodes=len(dataset.rows), frames=len(dataset), neural_weights_loaded=False,
        oracle_or_ik_actions_generated=False, future_feedback_used=False,
        data_sufficiency_or_learned_success_established=False,
        prompt_builder_sha256=sha256_file(Path(__file__).with_name("privileged_prompt.py")),
        dataset_implementation_sha256=sha256_file(Path(__file__).with_name("privileged_native_dataset.py")))
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
