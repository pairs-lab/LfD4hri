"""Continue SHA-bound local RDT weights AND Adam state after owner loss.

The old checkpoint did not save the DataLoader cursor/generator. This is
explicitly a deterministic restarted-shuffle continuation, NOT bit-exact
restart. Model, objective, optimizer recipe, source and total update budget
remain unchanged. Never reads pickles from an untrusted external checkpoint.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.common.source_training_gate import require_training_source
from VLA.cross_vla120_trial import atomic_json


def verify_saved_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "vla_checkpoint_manifest.json").read_text())
    for name, field in (("config.json", "model_config_sha256"), ("pytorch_model.bin", "model_weights_sha256"),
                        ("training_state.pt", "training_state_sha256"), ("pybullet_rdt_config.json", "target_contract_sha256")):
        if sha256_file(checkpoint / name) != manifest[field]:
            raise ValueError("Interrupted local RDT checkpoint changed: " + name)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state["step"] != manifest["step"] or state["step"] >= manifest["steps"]:
        raise ValueError("Not a partial checkpoint")
    moments = state["optimizer"]["state"].values()
    if not state["optimizer"]["state"] or any(float(item["step"]) != state["step"] or
            any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in (item["exp_avg"], item["exp_avg_sq"])) for item in moments):
        raise ValueError("Saved Adam update counts/FP32 moments differ")
    return manifest, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    cycle = json.loads((args.cycle / "status.json").read_text())
    if Path(f"/proc/{cycle['pid']}").exists():
        raise ValueError("Previous cycle still has an owner; never run duplicate optimization")
    if any(sha256_file(Path(p)) != digest for p, digest in cycle["frozen_files"].items()):
        raise ValueError("Interrupted recipe/source changed")
    manifest, saved = verify_saved_checkpoint(args.checkpoint)
    require_training_source(manifest["lineage"]["source"], model="rdt", optimizer_updates=manifest["steps"])
    proof = {"passed": True, "checkpoint": str(args.checkpoint.resolve()), "saved_optimizer_updates": saved["step"],
        "planned_total_optimizer_updates": manifest["steps"], "remaining_updates": manifest["steps"]-saved["step"],
        "restores_weights_and_Adam": True, "restores_stochastic_RNG": True,
        "exact_DataLoader_cursor_restored": False, "shuffle_restart_explicit": True,
        "source_model_controller_scorer_unchanged": True, "checkpoint_manifest_sha256": sha256_file(args.checkpoint/"vla_checkpoint_manifest.json"),
        "neural_success_established": False}
    if args.audit_output.exists():
        raise FileExistsError("Preserve previous continuation audit")
    atomic_json(args.audit_output, proof)
    if args.inspect_only:
        print(json.dumps(proof, indent=2)); return

    import VLA.rdt.training as trainer
    from VLA.rdt import checkpoint_loading
    from VLA.rdt.signal_qualification import require_signal_qualification
    old_command = json.loads((args.cycle / "signal_qualification.json").read_text())
    # Read the actual recipe directly from the saved config; original source
    # qualification and original smoke recipe are checked before narrowing K.
    tokens = ["train", "--source", manifest["lineage"]["source"], "--audit-report", manifest["lineage"]["audit_report"],
        "--action-contract-report", manifest["lineage"]["action_contract_report"],
        "--language-embeddings", manifest["language_embeddings"], "--language-manifest", manifest["language_manifest"],
        "--base-checkpoint", manifest["base_checkpoint"], "--vision-checkpoint", manifest["vision_checkpoint"],
        "--signal-qualification", str(args.cycle/"signal_qualification.json"), "--output", str(args.output), "--num-workers", "0", "--save-interval", "250"]
    for key in ("steps", "action_horizon", "episode_batch_size", "samples_per_episode", "gradient_accumulation",
                "learning_rate", "weight_decay", "optimizer", "seed", "signal_precision"):
        tokens += ["--"+key.replace("_", "-"), str(manifest[key])]
    original_argv = sys.argv
    sys.argv = ["rdt_continuation", *tokens]
    try:
        recipe = trainer.parse_args()
    finally:
        sys.argv = original_argv
    require_signal_qualification(recipe)
    lineage = trainer.validate_lineage(recipe.source, recipe.audit_report, recipe.action_contract_report)
    # Same override as official local trainer.main(): source validation has
    # the sparse format's default horizon64; this qualified recipe uses10.
    # This is not a relaxation: the saved/smoke contract must still match10.
    lineage["action_horizon"] = recipe.action_horizon
    if lineage != manifest["lineage"]:
        raise ValueError("RDT current lineage differs from saved checkpoint")
    original_load, original_optimizer, original_save = checkpoint_loading.load_checkpoint, trainer._make_optimizer, trainer._save_checkpoint
    base_steps, total_steps = saved["step"], recipe.steps
    def load(_base, horizon, *, dtype=torch.float32):
        model, _ = original_load(args.checkpoint, horizon, dtype=dtype)
        return model, manifest["horizon_adaptation"]
    def optimizer(config, model):
        value = original_optimizer(config, model)
        value.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        return value
    def save(model, optimizer, output, step, epoch, configuration):
        configuration = {**configuration, "steps": total_steps, "continued_from_checkpoint": str(args.checkpoint.resolve()),
            "continued_from_saved_updates": base_steps, "new_updates_requested": total_steps-base_steps,
            "exact_DataLoader_cursor_restored": False, "shuffle_restart_explicit": True,
            "continued_checkpoint_manifest_sha256": proof["checkpoint_manifest_sha256"]}
        return original_save(model, optimizer, output, base_steps+step, epoch, configuration)
    checkpoint_loading.load_checkpoint, trainer._make_optimizer, trainer._save_checkpoint = load, optimizer, save
    recipe.steps = total_steps-base_steps
    result = trainer._train(recipe, lineage)
    summary = json.loads((args.output/"training_summary.json").read_text())
    atomic_json(args.output/"continuation_summary.json", {**proof, "completed_new_updates": summary["steps"],
        "cumulative_optimizer_updates": base_steps+summary["steps"], "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    raise SystemExit(result)


if __name__ == "__main__":
    main()
