"""Compare actual training loader/processor with deployed sensor preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.provenance import sha256_file
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.adapter import SmolVLAPolicyAdapter
from VLA.smolvla.audit_checkpoint_predictions import validate_saved_normalization


def main() -> None:
    import torch
    from torch.utils.data import default_collate
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes_per_task < 1:
        raise ValueError("episodes-per-task must be positive")
    checkpoint, source = args.checkpoint.resolve(), args.source.resolve()
    audit = json.loads(args.audit_report.read_text())
    if not audit.get("passed") or audit.get("failures"):
        raise RuntimeError("Failed source audit")
    identity = verify_fingerprint(checkpoint)
    normalization = validate_saved_normalization(checkpoint, source, audit, identity)
    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    if cfg.dataset.image_transforms.enable:
        raise RuntimeError("This exact equality audit requires the recorded no-augmentation training profile")
    cfg.policy.device = "cpu"
    dataset = make_dataset(cfg)
    preprocess, _ = make_pre_post_processors(
        cfg.policy, checkpoint, preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    sensor = SimpleNamespace(closure=0.0)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    adapter = SmolVLAPolicyAdapter(sensor, checkpoint=str(checkpoint))
    # These official methods only access config: avoid loading neural weights/GPU.
    model_view = SimpleNamespace(config=cfg.policy)
    index = json.loads((source / "index.json").read_text())
    starts = np.cumsum([0] + [int(item["steps"]) for item in index[:-1]])
    selected = []
    for name in ("touch", "pick", "move", "pick_place"):
        numbers = [i for i, item in enumerate(index) if item["task"]["name"] == name]
        selected.extend(numbers[int(j)] for j in np.linspace(0, len(numbers) - 1, min(len(numbers), args.episodes_per_task), dtype=int))
    maxima = {"prepared_image_max_abs_error": 0., "prepared_state_max_abs_error": 0.,
              "language_token_mismatches": 0, "language_mask_mismatches": 0, "image_mask_mismatches": 0}
    rows = []
    for number in selected:
        item = index[number]
        with np.load(source / item["file"], allow_pickle=False) as episode:
            frames = np.unique(np.rint(np.linspace(0, item["steps"] - 1, 4)).astype(int))
            for frame in frames:
                train = _preprocess_dataset_batch(
                    default_collate([dataset[int(starts[number]) + int(frame)]]),
                    dataset.meta.camera_keys, cfg.rename_map, preprocess,
                )
                sensor.closure = float(episode["gripper_closure"][frame])
                observation = VLAObservation(
                    episode["front_rgb"][frame], episode["wrist_rgb"][frame],
                    episode["joint_positions"][frame], .085 * (1 - sensor.closure),
                    float(episode["gripper_open"][frame]), episode["ee_position"][frame],
                    episode["ee_rotation_6d"][frame], str(episode["prompt"].item()), int(frame),
                )
                infer = preprocess(adapter.build_sensor_batch(observation))
                train_images, train_masks = SmolVLAPolicy.prepare_images(model_view, train)
                infer_images, infer_masks = SmolVLAPolicy.prepare_images(model_view, infer)
                if len(train_images) != 2 or len(infer_images) != 2:
                    raise RuntimeError("Expected exactly two populated camera slots")
                image_error = max(float((a - b).abs().max()) for a, b in zip(train_images, infer_images, strict=True))
                state_error = float((SmolVLAPolicy.prepare_state(model_view, train) - SmolVLAPolicy.prepare_state(model_view, infer)).abs().max())
                maxima["prepared_image_max_abs_error"] = max(maxima["prepared_image_max_abs_error"], image_error)
                maxima["prepared_state_max_abs_error"] = max(maxima["prepared_state_max_abs_error"], state_error)
                for key, counter in (("observation.language.tokens", "language_token_mismatches"),
                                     ("observation.language.attention_mask", "language_mask_mismatches")):
                    if train[key].shape != infer[key].shape:
                        raise RuntimeError(f"Different model language shape: {key}")
                    maxima[counter] += int((train[key] != infer[key]).sum())
                maxima["image_mask_mismatches"] += sum(int((a != b).sum()) for a, b in zip(train_masks, infer_masks, strict=True))
                rows.append({"episode": item["file"], "frame": int(frame), "image_error": image_error, "state_error": state_error})
        print(f"{item['file']}: checked {len(rows)} model observations", flush=True)
    passed = all(value == 0 for value in maxima.values())
    report = {
        "passed": passed, "checkpoint": str(checkpoint), "source": str(source),
        "checkpoint_content_sha256": identity["content_tree_sha256"],
        "source_index_sha256": sha256_file(source / "index.json"),
        "adapter_sha256": sha256_file(Path(__file__).with_name("adapter.py")),
        "samples": len(rows), "saved_normalization": normalization, "errors": maxima,
        "methods": ["actual make_dataset and _preprocess_dataset_batch", "deployed build_sensor_batch",
                    "checkpoint processors", "official prepare_images and prepare_state"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("Train/infer model-input mismatch; see saved report")


if __name__ == "__main__":
    main()
