"""Actual LeRobot input/state18/normalizer/padding audit without GPU weights."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.smolvla.simstate_dataset import SimStateDataset


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.types import PolicyFeature, FeatureType
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior input audit")
    base = LeRobotDataset(repo_id="local/simstate18_original_qualified", root=args.dataset)
    data = SimStateDataset(base, args.projection)
    config = SmolVLAConfig.from_pretrained(str(args.base_model))
    if config.max_state_dim < 18:
        raise ValueError("Architecture lacks native state18 capacity")
    config.device = "cpu"
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(18,))
    mapping = {"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
    stats = {mapping.get(k, k): v for k, v in data.meta.stats.items()}
    processor, _ = make_pre_post_processors(config, pretrained_path=None, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    maximum, rows = 0., []
    for index in np.linspace(0, len(data)-1, 128, dtype=int):
        sample = data[int(index)]
        batch = {mapping.get(k, k): (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k,v in sample.items()}
        batch["task"] = [sample["task"]]
        processed = processor(copy.deepcopy(batch))
        state = SmolVLAPolicy.prepare_state(SimpleNamespace(config=config), processed)
        if state.shape != (1, config.max_state_dim) or not torch.isfinite(state).all():
            raise ValueError("Official preprocessing/padding of state18 failed")
        if torch.any(state[:,18:] != 0):
            raise ValueError("New state features wrongly occupy action/unused padding slots")
        raw = sample["observation.state"]
        mean, std = stats["observation.state"]["mean"], stats["observation.state"]["std"]
        expected = (raw - torch.as_tensor(mean)) / (torch.as_tensor(std) + 1e-8)
        error = float((state[0,:18] - expected).abs().max())
        maximum = max(maximum,error)
        if error > 1e-5:
            raise ValueError("Actual processor normalization differs from TRAIN state18 statistics")
        rows.append({"index": int(index), "state_error": error, "finite": True})
    atomic_json(args.output, {"passed": True, "samples": len(rows), "rows": rows,
        "state18_normalization_max_abs_error": maximum, "native_state_padding_dim": config.max_state_dim,
        "projection_audit_sha256": sha256_file(args.projection / "audit.json"),
        "stats_sha256": sha256_file(args.projection / "stats.json"),
        "original_images_and_actions_untouched": True, "neural_task_success_established": False})


if __name__ == "__main__":
    main()
