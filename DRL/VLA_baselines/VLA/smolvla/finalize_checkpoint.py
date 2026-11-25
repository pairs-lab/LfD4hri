"""Finalize a completed intermediate save for lossless, audited development inference.

The caller must wait until the trainer has advanced beyond this save. We also
check every saved trainable/moment dtype and optimizer update counter. Nothing
in an already-fingerprinted checkpoint is overwritten.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors import safe_open

from VLA.common.provenance import sha256_file
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.training import validate_checkpoint_schema, validate_lineage


ROOT = Path(__file__).resolve().parents[2]


def validate_sampling_contract(source: Path, recorded: dict) -> dict:
    """Reconstruct the recorded sampler, including its target-balance mode."""
    from VLA.smolvla.balanced_sampling import sampling_contract

    if not isinstance(recorded, dict) or "initial_frame_probability" not in recorded:
        raise RuntimeError("Sampling contract is missing its initial-frame probability")
    # Historical task/episode-only runs may omit this field. Never interpret
    # arbitrary truthy strings as a target-balanced recipe.
    target_balanced = recorded.get("target_balanced", False)
    if type(target_balanced) is not bool:
        raise RuntimeError("Sampling target_balanced contract must be boolean")
    expected = sampling_contract(source, recorded["initial_frame_probability"], target_balanced,
                                 recorded.get("grounding_initial_probability", 0.))
    if recorded != expected:
        raise RuntimeError("Sampling contract differs from the audited source")
    return expected


def finalize(checkpoint: Path, source: Path, audit_root: Path, *, source_audit=None,
             roundtrip_report=None, stats_report=None, preprocess_report=None) -> dict:
    checkpoint = checkpoint.resolve()
    if checkpoint.name != "pretrained_model" or checkpoint.parent.parent.name != "checkpoints":
        raise ValueError("Expected a native checkpoints/<microstep>/pretrained_model directory")
    contract_path = checkpoint / "fp32_master_contract.json"
    manifest_path = checkpoint / "vla_checkpoint_manifest.json"
    if (checkpoint / "benchmark_fingerprint.json").exists():
        if not contract_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Already-fingerprinted checkpoint lacks the required contracts")
        verify_fingerprint(checkpoint)
        return json.loads(manifest_path.read_text())
    microsteps = int(checkpoint.parent.name)
    run = checkpoint.parents[2]
    cfg = json.loads((checkpoint / "train_config.json").read_text())
    policy_cfg = json.loads((checkpoint / "config.json").read_text())
    dataset = Path(cfg["dataset"]["root"])
    info = json.loads((dataset / "meta/info.json").read_text())
    lineage = validate_lineage(SimpleNamespace(
        source=source, dataset_root=dataset, action_horizon=policy_cfg["chunk_size"],
        audit_report=source_audit or audit_root / "audit.json",
        roundtrip_report=roundtrip_report or audit_root / "lerobot_roundtrip.json",
        stats_report=stats_report or audit_root / "smolvla_exact_numeric_stats.json",
        preprocess_report=preprocess_report or audit_root / "smolvla_exact_worst_preprocess.json",
    ))
    schema = validate_checkpoint_schema(run, microsteps, info, policy_cfg["chunk_size"],
                                        str(policy_cfg["pretrained_path"]), not policy_cfg["train_expert_only"])
    master = json.loads((run / "master_dtype_audit.json").read_text())
    if (not master.get("passed") or master.get("promoted_after_loading") != 0
            or set(master["loaded_trainable_dtype_counts"]) != {"torch.float32"}):
        raise RuntimeError("Training master-dtype audit failed")
    state = checkpoint.parent / "training_state"
    metadata = json.loads((state / "training_step.json").read_text())
    accumulation = int(metadata["grad_accum_steps"])
    if microsteps != int(metadata["step"]) or microsteps % accumulation:
        raise RuntimeError("Incomplete accumulation group / wrong checkpoint step")
    optimizer_updates = microsteps // accumulation
    if json.loads((state / "scheduler_state.json").read_text())["last_epoch"] != microsteps:
        raise RuntimeError("Scheduler accounting differs from the saved microstep")
    with safe_open(checkpoint / "model.safetensors", framework="np") as saved:
        for name in master["trainable_parameter_names"]:
            if saved.get_slice(name).get_dtype() != "F32":
                raise RuntimeError("Saved master is not FP32: " + name)
    counters = set()
    with safe_open(state / "optimizer_state.safetensors", framework="np") as saved:
        for name in saved.keys():
            if "exp_avg" in name and saved.get_slice(name).get_dtype() != "F32":
                raise RuntimeError("Saved Adam moment is not FP32")
            if name.endswith("/step"):
                counters.add(float(saved.get_tensor(name)))
    if counters != {float(optimizer_updates)}:
        raise RuntimeError(f"Adam counters differ from {optimizer_updates}: {counters}")
    contract = {**master, "saved_weights_and_adam_moments_fp32": True,
                "constructor_sha256": sha256_file(Path(__file__).with_name("precision.py")),
                "trainer_sha256": sha256_file(Path(__file__).with_name("train_fp32.py"))}
    manifest = {
        **lineage, **schema, "mode": "completed_intermediate_save_for_development_only",
        "steps": microsteps, "step_unit": "microbatches", "optimizer_updates": optimizer_updates,
        "gradient_accumulation": accumulation, "batch_size": metadata["batch_size"],
        "effective_batch_size": metadata["batch_size"] * accumulation,
        "master_dtype": "float32", "compute_dtype": "bfloat16_autocast",
        "base_model": policy_cfg["pretrained_path"], "train_vlm": not policy_cfg["train_expert_only"],
        "final_test_used": False,
    }
    sampling_path = run / "sampling_contract.json"
    if sampling_path.exists():
        recorded = json.loads(sampling_path.read_text())
        validate_sampling_contract(source, recorded)
        manifest["sampling_contract"] = recorded
        manifest["sampling_contract_sha256"] = sha256_file(sampling_path)
    for path, data in ((contract_path, contract), (manifest_path, manifest)):
        if path.exists():
            if json.loads(path.read_text()) != data:
                raise RuntimeError("Existing checkpoint contract differs; refusing overwrite: " + str(path))
        else:
            path.write_text(json.dumps(data, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    for name in ("source-audit", "roundtrip-report", "stats-report", "preprocess-report"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.checkpoint, args.source, args.audit_root, source_audit=args.source_audit,
        roundtrip_report=args.roundtrip_report, stats_report=args.stats_report, preprocess_report=args.preprocess_report), indent=2))


if __name__ == "__main__":
    main()
