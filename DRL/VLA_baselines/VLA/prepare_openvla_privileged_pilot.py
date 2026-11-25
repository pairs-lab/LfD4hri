"""Immutable OpenVLA-only 1k-update source view; inherit, never overwrite, data."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.common.source_training_gate import require_training_source
from VLA.privileged_native_dataset import PrivilegedNativeDataset
from VLA.privileged_prompt import PROMPT_SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("projection", "native-preflight", "visual-review", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    dataset = PrivilegedNativeDataset(args.projection, 1)
    require_training_source(dataset.source, model="smolvla", optimizer_updates=1)
    preflight = json.loads(args.native_preflight.read_text()); review = json.loads(args.visual_review.read_text())
    if (not preflight.get("passed") or preflight.get("all_frames_formatted_and_tokenized") != len(dataset)
            or preflight.get("source_index_sha256") != dataset.audit["source_index_sha256"]
            or review.get("actual_native_contact_sheet_reviewed") is not True
            or review.get("preflight_sha256") != sha256_file(args.native_preflight)
            or review.get("neural_success_certified") is not False):
        raise ValueError("Missing native input/source/actual image review proof")
    for image in review.get("images", []):
        if sha256_file(image["path"]) != image["sha256"]: raise ValueError("Reviewed image changed")
    actions = []; references = []
    for ei, row in enumerate(dataset.rows):
        with np.load(dataset.source / row["file"], allow_pickle=False) as archive:
            canonical = archive["canonical_actions"].astype(np.float32, copy=True)
        canonical[:, 6] = (1 - canonical[:, 6]) / 2
        actions.append(canonical)
        references.extend({"episode_index": ei, "frame_index": fi} for fi in range(len(canonical)))
    actions = np.concatenate(actions)
    if len(actions) != len(dataset): raise ValueError("Native action frame count differs")
    key = args.output.name
    action_stats = {"mean": actions.mean(0).tolist(), "std": actions.std(0).tolist(),
        "min": actions.min(0).tolist(), "max": actions.max(0).tolist(),
        "q01": np.quantile(actions, .01, axis=0).tolist(), "q99": np.quantile(actions, .99, axis=0).tolist(),
        "mask": [True] * 6 + [False]}
    (args.output / "dataset_statistics.json").write_text(json.dumps({key: {"action": action_stats,
        "num_transitions": len(actions), "num_trajectories": len(dataset.rows)}}, indent=2) + "\n")
    (args.output / "samples.json").write_text(json.dumps(references, separators=(",", ":")) + "\n")
    dependencies = {"native_preflight": args.native_preflight, "actual_visual_review": args.visual_review,
        "causal_projection": args.projection / "audit.json", "train_stats": args.projection / "stats.json",
        "prompt_builder": Path(__file__).with_name("privileged_prompt.py"),
        "dataset_implementation": Path(__file__).with_name("privileged_native_dataset.py"),
        "native_original_trainer": Path(__file__).with_name("openvla") / "finetune.py",
        "native_inference_adapter": Path(__file__).with_name("openvla") / "adapter.py",
        "trainer_implementation": Path(__file__).with_name("openvla") / "train_privileged_prompt.py"}
    proof = {"passed": True, "model": "openvla", "scope": "bounded_pipeline_pilot", "max_optimizer_updates": 1000,
        "full_original_source_physics_qualification_inherited": True, "native_tokenizers_and_rgb_transforms_checked": True,
        "actual_cuda_update_and_reload_still_required_before_continuation": True,
        "data_sufficiency_or_success_rate_certified": False,
        "dependencies": {k: {"path": str(v.resolve()), "sha256": sha256_file(v)} for k, v in dependencies.items()}}
    proof_path = args.output / "pilot_preconditions.json"; proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    view = args.output / "qualified_source"; view.mkdir()
    for row in dataset.rows: os.link(dataset.source / row["file"], view / row["file"])
    (view / "index.json").write_bytes((dataset.source / "index.json").read_bytes())
    manifest = copy.deepcopy(json.loads((dataset.source / "dataset_manifest.json").read_text()))
    manifest.update(qualification_parent=str(dataset.source), qualification_parent_index_sha256=dataset.audit["source_index_sha256"],
        qualification_parent_manifest_sha256=dataset.audit["source_manifest_sha256"], candidate_only=False,
        training_certified=False, purpose="OpenVLA privileged-language bounded 1k pilot NOT stable-success certification")
    # These are inherited PHYSICS/SPLIT evidence, not the parent's SmolVLA
    # optimizer-budget authorization. Preserve the exact proof bytes while
    # assigning their explicit narrower evidentiary role to this new view.
    inherited = [{**item, "role": "inherited_common_" + item["role"]}
                 for item in manifest["training_qualification"]["proofs"]]
    manifest["training_qualification"] = {"scope": "bounded_pipeline_pilot", "allowed_models": ["openvla"],
        "max_optimizer_updates": 1000, "proofs": [*inherited, {"role": "openvla_privileged_native_preconditions",
            "path": str(proof_path.resolve()), "sha256": sha256_file(proof_path)}]}
    (view / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    require_training_source(view, model="openvla", optimizer_updates=1000)
    (args.output / "conversion_audit.json").write_text(json.dumps({"dataset_key": key,
        "uses_original_source_arrays_without_reencoding": True, "not_legacy_conversion": True}) + "\n")
    (args.output / "privileged_dataset.json").write_text(json.dumps({"schema": PROMPT_SCHEMA,
        "model": "openvla", "projection": str(args.projection.resolve()), "qualified_source": str(view.resolve()),
        "original_source": str(dataset.source), "original_index_sha256": dataset.audit["source_index_sha256"],
        "frames": len(dataset), "episodes": len(dataset.rows), "candidate_only": False,
        "max_optimizer_updates": 1000, "dependencies": proof["dependencies"]}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(dataset), "episodes": len(dataset.rows),
        "max_optimizer_updates": 1000, "neural_training_executed": False}), flush=True)


if __name__ == "__main__": main()
