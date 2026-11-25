"""Postfit evidence -> explicit phase repair -> smoke ->5k refine ->same40.

CPU preparation can run while RDT trains; optional explicit CUDA memory cap
permits sharing with an already optimizing RDT job after headroom checks. Dataset
payload, normalization, robot, sensor and scorer remain unchanged. Fresh
immutable experiment, with completed failures retained and no silent retry.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.provenance import sha256_file
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.phase_balanced_sampling import load_phase_recipe
from VLA.smolvla.audit_checkpoint_predictions import validate_holdout

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous-cycle", "preparation", "output", "model-output", "wait-cycle"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--gpu-share-memory-limit-mib", type=int,
                        help="Explicit6000..7000MiB process-local training allocator cap; permits sharing only while RDT is optimizing")
    args = parser.parse_args()
    if args.gpu_share_memory_limit_mib is not None and not 6000 <= args.gpu_share_memory_limit_mib <= 7000:
        raise ValueError("Sharing requires a bounded6000..7000MiB training allocator cap")
    previous, prep, output, model = map(lambda p: p.resolve(), (args.previous_cycle, args.preparation, args.output, args.model_output))
    if output.exists() or model.exists(): raise FileExistsError("Fresh recipe/model evidence required")
    before = json.loads((previous / "comparison.json").read_text())
    completed = json.loads((previous / "status.json").read_text())
    grounding = json.loads((previous / "postfit_causal_grounding/results.json").read_text())
    if completed["status"] != "completed40_fit_test_cycle_remaining_errors_require_review" or before["trials_each"] != 40:
        raise ValueError("Exact completed40 baseline required")
    if not grounding["sensor_identity_passed"] or not grounding["full_text_not_truncated"]:
        raise ValueError("Neural text/sensor grounding audit failed integration")
    cp = Path(completed["model_output"]) / "train/checkpoints/160000/pretrained_model"
    identity = verify_fingerprint(cp)
    if identity["content_tree_sha256"] != grounding["checkpoint_content_sha256"]:
        raise ValueError("Baseline weights differ from audited model")
    source, validation = prep / "qualified_train", prep / "validation_candidate"
    from VLA.common.source_training_gate import require_training_source
    require_training_source(source, model="smolvla", optimizer_updates=5000)
    validate_holdout(source, validation)
    weights, recipe = load_phase_recipe(source, .1)
    if recipe["observed_phase_mass"]["xy_choice"]["new_probability_mass"] <= recipe["observed_phase_mass"]["xy_choice"]["original_probability_mass"]:
        raise ValueError("Proposed recipe does not address observed localization exposure")
    files = [previous / "comparison.json", previous / "postfit_causal_grounding/results.json", source / "index.json", source / "dataset_manifest.json",
        Path(__file__), ROOT / "VLA/smolvla/training.py", ROOT / "VLA/smolvla/phase_balanced_sampling.py", ROOT / "VLA/smolvla/train_phase_balanced.py",
        ROOT / "VLA/smolvla/train_fp32.py", ROOT / "VLA/run_diagnostic40.py", ROOT / "VLA/analyze_diagnostic40.py"]
    fixed = {str(p): sha256_file(p) for p in files}
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    (output / "phase_recipe.json").write_text(json.dumps(recipe, indent=2)+"\n")
    status = {"scope": "audited_SmolVLA_phase_refinement_then_same40_DEVELOPMENT", "pid": os.getpid(), "status": "running", "error": None,
        "baseline_successes": before["postfit_successes"], "baseline_trials": 40, "updates": 5000,
        "warmstart_optimizer_updates_already_used": 10000, "cumulative_family_updates_after_this_refinement": 15000,
        "dataset_episodes": recipe["episodes"], "dataset_frames": len(weights), "dataset_changed": False,
        "normalization_changed": False, "final_test_used": False, "oracle_fallback": False, "high_success_guaranteed": False,
        "gpu_share_memory_limit_mib": args.gpu_share_memory_limit_mib,
        "warmstart_content_sha256": identity["content_tree_sha256"], "frozen_files": fixed, "history": []}
    def save(stage):
        status.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat())
        (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in fixed.items()): raise ValueError("Bound recipe/source changed")
    environment = {**os.environ, "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    if args.gpu_share_memory_limit_mib is not None:
        environment["VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB"] = str(args.gpu_share_memory_limit_mib)
    def run(stage, module, tokens, report=None, updates=None):
        verify(); save(stage)
        command = [str(ROOT / "VLA/third_party/lerobot/.venv/bin/python"), "-m" if module else "-c"]
        command += ([module] if module else []) + list(map(str, tokens))
        with (output / "logs" / (stage+".log")).open("x") as stream:
            done = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=86400)
        status["history"].append({"stage": stage, "command": command, "exit_code": done.returncode})
        if done.returncode: raise RuntimeError(stage+" failed; preserve log and diagnose")
        if report:
            data = json.loads(Path(report).read_text())
            if data.get("passed") is not True: raise ValueError(stage+" report failed")
            if updates is not None and data["optimizer_steps"] != [float(updates)]: raise ValueError("Wrong actual optimizer count")
        save(stage+"_completed")
    common = ["--source", source, "--dataset-root", prep / "lerobot_dataset", "--audit-report", prep / "qualified_train_audit/audit.json",
        "--roundtrip-report", prep / "fit_audit/pilot_prep/roundtrip.json", "--stats-report", prep / "fit_audit/pilot_prep/stats.json",
        "--preprocess-report", prep / "fit_audit/pilot_prep/smolvla_preprocess.json", "--base-model", cp,
        "--train-vlm", "--train-visual-connector", "--batch-size", 2, "--learning-rate", .000025,
        "--initial-frame-sampling-probability", .1, "--target-balanced-sampling", "--phase-balanced-sampling", "--action-horizon", 5]
    try:
        run("validate", "VLA.smolvla.training", ["validate", *common, "--steps", 5000, "--gradient-accumulation", 16,
            "--warmup-updates", 250, "--decay-updates", 5000])
        deadline = time.monotonic()+48*3600
        while True:
            verify()
            queued = json.loads(args.wait_cycle.read_text())
            if queued["status"] == "completed_fit_test_remaining_errors_require_review": break
            if (args.gpu_share_memory_limit_mib is not None and queued["status"] == "running" and queued["stage"] == "fresh_train"):
                status["shared_with_active_RDT_cycle"] = str(args.wait_cycle.resolve())
                break
            if queued["status"].startswith("stopped") or time.monotonic() > deadline:
                raise RuntimeError("Declared RDT cycle stopped or finite GPU queue wait expired")
            save("CPU_ready_waiting_complete_RDT_GPU_fit_test_cycle"); time.sleep(20)
        minimum = (args.gpu_share_memory_limit_mib+512)*2**20 if args.gpu_share_memory_limit_mib is not None else 10*2**30
        run("cuda_preflight", None, [f"import torch; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); assert free >= {minimum}, f'Insufficient GPU headroom: {{free}}'; print(torch.cuda.get_device_name(0),free,total)"])
        for name, updates, accumulation, warmup in (("smoke", 2, 8, 0), ("train", 5000, 16, 250)):
            target = model / name
            run(name, "VLA.smolvla.training", ["smoke" if name == "smoke" else "train", *common, "--steps", updates,
                "--gradient-accumulation", accumulation, "--warmup-updates", warmup, "--decay-updates", 1000 if name == "smoke" else 5000,
                "--save-freq", 1000, "--output", target])
            saved = target / "checkpoints" / f"{updates*accumulation:06d}" / "pretrained_model"
            proof = output / (name+"_precision.json")
            run(name+"_precision", "VLA.smolvla.audit_training_precision", ["--checkpoint", saved, "--output", proof], proof, updates)
            run(name+"_fingerprint", "VLA.fingerprint_checkpoint", [saved])
            proof = output / (name+"_input_parity.json")
            run(name+"_input_parity", "VLA.smolvla.audit_train_infer_inputs", ["--checkpoint", saved, "--source", source,
                "--audit-report", prep / "qualified_train_audit/audit.json", "--episodes-per-task", 2, "--output", proof], proof)
        cp_new = model / "train/checkpoints/080000/pretrained_model"
        run("development40", "VLA.run_diagnostic40", ["--checkpoint", cp_new, "--output", output / "development40", "--run-tag", model.parent.name+"_postfit40"])
        run("error_analysis", "VLA.analyze_diagnostic40", ["--progress", output / "development40/progress.json", "--output", output / "analysis"])
        after = json.loads((output / "analysis/analysis.json").read_text())
        (output / "comparison.json").write_text(json.dumps({"baseline_successes": before["postfit_successes"],
            "postfit_successes": after["successes"], "trials_each": 40, "per_action": after["statistics"]["by_task"],
            "failure_signatures": after["failure_signatures_overlap"], "stable_success_established": False,
            "next_step": "Review real postfit video/trace; collect and qualify actual TRAIN on-policy corrections if grounding/hold recovery remains poor. Never silently repeat unchanged fitting."}, indent=2)+"\n")
        status["status"] = "completed_phase_repair_fit40_remaining_errors_require_review"; save("done")
    except Exception as exc:
        status.update(status="stopped_error", error=str(exc)); save(status.get("stage", "initialization")); raise


if __name__ == "__main__":
    main()
