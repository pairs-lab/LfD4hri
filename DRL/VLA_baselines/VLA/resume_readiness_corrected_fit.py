"""Reuse completed physics honestly, prepare CPU data, then await200 trials.

Preserves the original stopped workflow and failed readiness. No altered
success rules, scene filtering, oracle fallback, or fabricated image approval.
New optimization cannot start before full baseline error analysis completes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_source_environment_runtime
from VLA.prepare_v9_candidate_fit import require_visual_receipt
from VLA.run_smolvla_v9_pilot import build_plan, parse_args, validate_inputs, checkpoint_path

ROOT = Path(__file__).resolve().parents[1]


def wait_report(path, status, save, verify, deadline):
    while not path.exists():
        verify()
        save(status)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Finite wait expired: {path}")
        time.sleep(20)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stopped-workflow", "corrected-readiness", "diagnostic-progress", "output", "model-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    original, output = args.stopped_workflow.resolve(), args.output.resolve()
    old = json.loads((original / "workflow_status.json").read_text())
    if old["status"] != "stopped_no_silent_retry" or not old["error"].startswith("readiness failed"):
        raise ValueError("Only the explicitly diagnosed readiness stop is covered")
    if (output / "status.json").exists() or args.model_output.exists():
        raise FileExistsError("Do not overwrite a previous continuation/model")
    source = original / "combined_candidate"
    source_audit = original / "combined_source_audit/audit.json"
    replay = original / "combined_full_physics_replay.json"
    native = original / "native_smol512"
    receipt = output / "visual_review_receipt.json"
    readiness = args.corrected_readiness.resolve()
    diagnostic = args.diagnostic_progress.resolve()
    require_visual_receipt(receipt, source, native, source_audit)
    checked = json.loads(replay.read_text())
    rows = json.loads((source / "index.json").read_text())
    if (not checked["passed"] or checked["failures"] or not checked["all_episodes"]
            or checked["mode"] != "both" or checked["native_success_mode"] != "strict"
            or len(checked["rows"]) != 2 * len(rows)
            or any(not r["success"] or r.get("initial_strict_success") for r in checked["rows"])
            or checked["source_index_sha256"] != sha256_file(source / "index.json")
            or checked["source_manifest_sha256"] != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Completed unchanged full physics is required; no bypass")
    verify_source_environment_runtime(checked, source)
    frozen = {str(p): sha256_file(p) for p in (
        Path(__file__).resolve(), ROOT / "VLA/audit_training_readiness.py",
        ROOT / "VLA/analyze_diagnostic50.py", original / "workflow_status.json",
        original / "readiness/readiness.json", replay, source_audit, native / "audit.json",
        source / "index.json", source / "dataset_manifest.json", receipt,
        diagnostic.parent / "plan.json")}
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    history = []
    plan = {"scope": "200_trial_development_analysis_then_new_qualified_finetune",
        "stopped_workflow_preserved": str(original), "completed_full_physics_reused": str(replay),
        "same_robot_cameras_controller_success": True, "episodes_removed": 0,
        "original_source": str(source), "corrected_readiness": str(readiness),
        "diagnostic_progress": str(diagnostic), "model_output": str(args.model_output.resolve()),
        "updates": old["updates"], "batch_size": old["batch_size"], "learning_rate": old["learning_rate"],
        "initial_frame_sampling_probability": old["initial_frame_sampling_probability"],
        "paired_grounding_probability": old["paired_grounding_probability"],
        "GPU_training_waits_for_all200_baseline_trials": True, "frozen_files": frozen,
        "data_sufficiency_established": False, "final_test_used": False}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    def save(status, error=None):
        (output / "status.json").write_text(json.dumps({**plan, "status": status,
            "history": history, "pid": os.getpid(), "error": error,
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in frozen.items()):
            raise ValueError("Immutable preparation/review/diagnostic contract changed")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    def command(stage, command, report=None, pass_key="passed"):
        verify()
        save(stage)
        with (logs / (stage + ".log")).open("x") as stream:
            done = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        history.append({"stage": stage, "command": command, "exit_code": done.returncode})
        if done.returncode:
            raise RuntimeError(f"{stage} failed; preserve evidence and diagnose")
        if report is not None and not json.loads(Path(report).read_text()).get(pass_key):
            raise ValueError(f"{stage} report did not pass")
        save(stage + "_completed")
    def module(stage, name, tokens, **kwargs):
        command(stage, [sys.executable, "-m", name, *map(str, tokens)], **kwargs)
    def audit(stage, dataset):
        path = output / stage / "audit.json"
        counts = Counter(r["task"]["name"] for r in json.loads((dataset / "index.json").read_text()))
        module(stage, "VLA.audit_finetune_data", ["--dataset", dataset, "--output", path.parent,
            "--expected-task-counts", json.dumps(counts), "--allow-stricter-frozen-reward-contract",
            "--recovery-reset-observability", "dual_view"], report=path)
        return path
    try:
        report = wait_report(readiness, "waiting_corrected_readiness", save, verify, time.monotonic() + 7200)
        if (not report["source_integrity_passed"] or report["failures"] or report["group_overlap_count"]
                or report["combined"]["episodes"] != len(rows)
                or len(report["lineage"]) != 1
                or Path(report["lineage"][0]["source"]).resolve() != source):
            raise ValueError("Corrected readiness did not qualify the exact unchanged source")
        frozen[str(readiness)] = sha256_file(readiness)
        train, validation = output / "train_candidate", output / "validation_candidate"
        for split, path in (("train", train), ("validation", validation)):
            module("materialize_" + split, "VLA.materialize_training_split", [
                "--readiness", readiness, "--split", split, "--output", path])
        train_audit, validation_audit = audit("train_candidate_audit", train), audit("validation_audit", validation)
        qualified = output / "qualified_train"
        module("qualify", "VLA.qualify_bounded_refinement", ["--source", train, "--source-audit", train_audit,
            "--native-replay-report", replay, "--output", qualified, "--max-updates", old["updates"],
            "--visual-review-receipt", receipt, "--native-camera-dir", native,
            "--combined-source", source, "--combined-source-audit", source_audit])
        qualified_audit = audit("qualified_train_audit", qualified)
        module("sampler_phase_mass", "VLA.audit_sampling_phase_mass", ["--source", qualified,
            "--output", output / "sampler_phase_mass", "--initial-probability", old["initial_frame_sampling_probability"]])
        sampling = json.loads((output / "sampler_phase_mass/audit.json").read_text())
        if any(sampling["by_task"][t]["phase_metrics"]["close_acquires_correct_object"]["frames"] == 0
               for t in ("pick", "move", "pick_place")):
            raise ValueError("Actual qualified training data lacks acquisition labels")
        tokens = ["--source", qualified, "--validation", validation, "--source-audit", qualified_audit,
            "--validation-audit", validation_audit, "--dataset-root", output / "lerobot_dataset",
            "--audit-root", output / "fit_audit", "--output", args.model_output,
            "--updates", old["updates"], "--experiment-scope", "bounded_refinement_experiment",
            "--batch-size", old["batch_size"], "--learning-rate", old["learning_rate"],
            "--initial-frame-sampling-probability", old["initial_frame_sampling_probability"]]
        if old["paired_grounding_probability"]:
            tokens += ["--paired-grounding-probability", old["paired_grounding_probability"]]
        pilot = parse_args(list(map(str, tokens)))
        validate_inputs(pilot)
        for step in build_plan(pilot):
            if step.gpu:
                break
            command("cpu_" + step.name, step.command, report=step.gate_report)
        deadline = time.monotonic() + 86400
        while True:
            verify()
            progress = json.loads(diagnostic.read_text())
            if progress["status"].startswith("stopped"):
                raise ValueError("Baseline200 stopped: diagnose integration, not silently start training")
            if progress["status"] == "completed200_development_trials":
                break
            save("prepared_CPU_waiting_complete200_baseline")
            if time.monotonic() >= deadline:
                raise TimeoutError("Baseline200 did not finish in finite wait")
            time.sleep(20)
        module("complete200_error_analysis", "VLA.analyze_diagnostic50", ["--progress", diagnostic,
            "--output", output / "baseline200_analysis", "--require-complete"])
        module("finetune_and_source_diagnostics", "VLA.run_smolvla_v9_pilot", [*tokens, "--prepared", "--run"])
        cp = checkpoint_path(args.model_output / "train", old["updates"], pilot.accumulation)
        module("postfit200_development", "VLA.run_diagnostic50", ["--checkpoint", cp,
            "--output", output / "postfit200_development", "--run-tag", "diagnostic50_r13_postfit_20261013"])
        module("postfit200_error_analysis", "VLA.analyze_diagnostic50", [
            "--progress", output / "postfit200_development/progress.json",
            "--output", output / "postfit200_analysis", "--require-complete"])
        save("completed_finetune_and_both200_DEVELOPMENT_evaluations")
    except Exception as exc:
        save("stopped_no_silent_retry", str(exc))
        raise


if __name__ == "__main__":
    main()
