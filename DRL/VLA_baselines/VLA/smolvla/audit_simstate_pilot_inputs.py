"""CPU early warning for REAL pilot frames; never qualifies pilot for fitting.

Uses the actual original LeRobot rows and official preprocessing, without
relaxing the full-projection training gate or loading neural/GPU weights.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.simstate_feedback import FEATURE_NAMES, SCHEMA


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.configs.train import TrainPipelineConfig
    import draccus
    from VLA.run_simstate_repair_cycle import BASE, training_command

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostic evidence")
    pilot = json.loads((args.pilot / "audit.json").read_text())
    if not pilot["passed"] or not pilot["pilot_only"] or pilot["schema"] != SCHEMA:
        raise ValueError("This early diagnostic accepts only the real causal pilot")
    source = Path(pilot["source"])
    source_rows = json.loads((source / "index.json").read_text())
    offsets, total = {}, 0
    for row in source_rows:
        offsets[row["file"]] = total
        total += int(row["steps"])
    base = LeRobotDataset(repo_id="local/simstate18_original_qualified", root=args.dataset)
    if len(base) != total:
        raise ValueError("Original full LeRobot and source ordering/counts differ")

    meta = copy.deepcopy(base.meta)
    meta.info.features["observation.state"] = {"dtype": "float32", "shape": [18], "names": FEATURE_NAMES}
    original = copy.deepcopy(base.meta.features["observation.state"])
    if tuple(original["shape"]) != (7,) or tuple(meta.features["observation.state"]["shape"]) != (18,):
        raise ValueError("Dataset metadata overlay mutates base or loses state18")
    mapping = {"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
    # Parse the EXACT pending smoke CLI. Factory only infers dataset input
    # schema when input_features is empty; detect a stale pretrained state7.
    command = training_command(BASE, args.pilot, args.output.parent / "never_execute_pilot_fit", 2, 2)
    parsed = draccus.parse(config_class=TrainPipelineConfig, args=list(map(str, command[3:])))
    inferred = {mapping.get(k, k): v for k, v in dataset_to_policy_features(meta.features).items()}
    if parsed.policy.input_features:
        declared = parsed.policy.input_features["observation.state"].shape
        if tuple(declared) != (18,):
            raise ValueError(f"Pending actual CLI leaves stale state shape {declared}")
    else:
        parsed.policy.input_features = {k: v for k, v in inferred.items() if v.type != FeatureType.ACTION}
    if tuple(parsed.policy.input_features["observation.state"].shape) != (18,):
        raise ValueError("Pending factory schema will not use state18")

    config = SmolVLAConfig.from_pretrained(str(args.base_model))
    config.device = "cpu"
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(18,))
    state_stats = json.loads((args.pilot / "stats.json").read_text())
    stats = {mapping.get(k, k): v for k, v in base.meta.stats.items()}
    stats["observation.state"] = {k: np.asarray(v, dtype=np.int64 if k == "count" else np.float32) for k, v in state_stats.items()}
    processor, _ = make_pre_post_processors(config, pretrained_path=None, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": "cpu"}})
    proof_rows, maximum = [], 0.
    for proof in pilot["rows"]:
        payload = args.pilot / "states" / proof["file"]
        if sha256_file(payload) != proof["state_payload_sha256"]:
            raise ValueError("Real pilot state payload changed")
        with np.load(payload, allow_pickle=False) as archive:
            states = archive["state"]
        for frame in sorted({0, len(states)//2, len(states)-1}):
            index = offsets[proof["file"]] + frame
            sample = base[index]
            raw = torch.from_numpy(states[frame].copy())
            if int(sample["index"]) != index or not torch.allclose(sample["observation.state"].reshape(-1), raw[:7], atol=1e-6, rtol=0):
                raise ValueError("Actual loader/source frame/state7 mismatch")
            batch = {mapping.get(k,k): (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k,v in sample.items()}
            batch["observation.state"] = raw.unsqueeze(0)
            batch["task"] = [sample["task"]]
            result = processor(copy.deepcopy(batch))
            padded = SmolVLAPolicy.prepare_state(SimpleNamespace(config=config), result)
            mean = torch.as_tensor(stats["observation.state"]["mean"])
            std = torch.as_tensor(stats["observation.state"]["std"])
            expected = (raw - mean) / (std + 1e-8)
            error = float((padded[0,:18] - expected).abs().max())
            if padded.shape != (1, config.max_state_dim) or not torch.isfinite(padded).all() or torch.any(padded[:,18:] != 0) or error > 1e-5:
                raise ValueError("Actual pilot official normalization/state padding failed")
            if any(not torch.equal(batch[mapping[k]], result[mapping[k]]) for k in mapping):
                raise ValueError("Identity RGB processing unexpectedly modifies original model input")
            maximum = max(maximum, error)
            proof_rows.append({"source_file": proof["file"], "local_frame": frame, "global_index": index,
                "task": sample["task"], "normalized_state_max_abs_error": error})
    atomic_json(args.output, {"passed": True, "pilot_only": True, "qualified_for_training": False,
        "neural_success_established": False, "no_GPU_model_loaded": True,
        "actual_original_loader_frames": len(proof_rows), "rows": proof_rows,
        "pending_smoke_CLI_inferred_state_shape": [18], "state_padding_dim": config.max_state_dim,
        "maximum_normalization_error": maximum, "base_metadata_unmodified": base.meta.features["observation.state"] == original,
        "pilot_audit_sha256": sha256_file(args.pilot / "audit.json"),
        "full_TRAIN_VAL_and_loader_qualification_still_required": True})


if __name__ == "__main__":
    main()
