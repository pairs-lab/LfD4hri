"""Finite predeclared1000/5000-update grounding probes; no training changes.

Waits for fully saved intermediate checkpoints AND evidence that training has
advanced past the save before adding an immutable inference manifest. Uses
only TRAIN-counterfactual observations and identical baseline noise seeds.
Does not stop/restart the optimizer, select weights by validation or run an
oracle fallback. Per-probe GPU memory is bounded separately from the trainer.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit-audit", "model-output", "source", "source-audit", "paired-data", "paired-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    frozen = {str(p.resolve()): sha256_file(p) for p in [args.source/"index.json", args.source/"dataset_manifest.json",
        args.source_audit, args.paired_data/"index.json", args.paired_data/"dataset_manifest.json", args.paired_audit, Path(__file__)]}
    plan = {"scope": "predeclared_TRAIN_only_grounding_followup_NOT_success_rate", "updates": [1000, 5000],
        "noise_seeds": [20261010, 20261011, 20261012], "source": str(args.source.resolve()),
        "paired_data": str(args.paired_data.resolve()), "model_output": str(args.model_output.resolve()),
        "frozen_inputs": frozen, "selection_uses_neural_outcomes": False, "training_modified": False,
        "validation_used": False, "final_test_used": False, "wait_limit_seconds": 14400,
        "gpu_probe_memory_fraction": .34, "gpu_free_preflight_gib": 6}
    (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    history = []
    def save(state, stage, error=None):
        (output/"status.json").write_text(json.dumps({**plan, "status": state, "stage": stage, "error": error,
            "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), "history": history}, indent=2)+"\n")
    def unchanged():
        if any(sha256_file(path) != digest for path, digest in frozen.items()):
            raise RuntimeError("Predeclared source/proofs/code changed")
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(ROOT/"VLA/checkpoints/huggingface"), "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    python = str(ROOT/"VLA/third_party/lerobot/.venv/bin/python")
    deadline = time.monotonic()+14400
    try:
        for updates in plan["updates"]:
            microsteps = updates*16
            checkpoint = args.model_output/"train/checkpoints"/f"{microsteps:06d}"/"pretrained_model"
            save("waiting_for_complete_saved_checkpoint", f"wait_{updates}")
            while True:
                unchanged()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Finite4h checkpoint wait expired; optimizer is not modified")
                driver = json.loads((args.fit_audit/"pilot_driver_status.json").read_text())
                if driver["status"] in {"failed", "stopped_error"}:
                    raise RuntimeError("Primary fit failed; no follow-up on partial checkpoint")
                complete = all(p.exists() for p in [checkpoint/"model.safetensors", checkpoint/"config.json",
                    checkpoint.parent/"training_state/optimizer_state.safetensors", checkpoint.parent/"training_state/training_step.json"])
                if updates == 5000:
                    advanced = (checkpoint/"fp32_master_contract.json").exists() and (checkpoint/"benchmark_fingerprint.json").exists()
                else:
                    log = args.fit_audit/"pilot_logs/train.log"
                    values = re.findall(r"(\d+)/80000", log.read_text(errors="replace")) if log.exists() else []
                    advanced = bool(values) and max(map(int, values)) > microsteps+64
                if complete and advanced:
                    break
                time.sleep(10)
            probe = output/f"update{updates:06d}"
            probe.mkdir()
            prep = args.fit_audit/"pilot_prep"
            commands = [
                ("finalize", [python, "-m", "VLA.smolvla.finalize_checkpoint", "--checkpoint", str(checkpoint),
                    "--source", str(args.source), "--audit-root", str(args.fit_audit), "--source-audit", str(args.source_audit),
                    "--roundtrip-report", str(prep/"roundtrip.json"), "--stats-report", str(prep/"stats.json"),
                    "--preprocess-report", str(prep/"smolvla_preprocess.json")]),
                # The primary driver necessarily fingerprints the final save
                # before this follow-up is allowed to use it. Verify/reuse that
                # immutable report; do not fail merely because it exists.
                ("fingerprint", [python, "-m", "VLA.fingerprint_checkpoint", str(checkpoint), "--verify-existing"]),
                ("grounding", [python, "-c",
                    "import torch,runpy,sys; assert torch.cuda.is_available(); free,total=torch.cuda.mem_get_info(); "
                    "assert free>=6*2**30, f'Insufficient probe headroom: {free}'; "
                    "torch.cuda.set_per_process_memory_fraction(.34); "
                    "sys.argv=['audit_target_grounding',*sys.argv[1:]]; runpy.run_module('VLA.smolvla.audit_target_grounding',run_name='__main__')",
                    "--checkpoint", str(checkpoint), "--paired-data", str(args.paired_data), "--source-audit", str(args.paired_audit),
                    "--output", str(probe/"grounding"), "--noise-seeds", *map(str, plan["noise_seeds"])])]
            for name, command in commands:
                unchanged(); save("running", f"{updates}_{name}")
                with (probe/(name+".log")).open("x") as stream:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                history.append({"updates": updates, "stage": name, "exit_code": result.returncode, "command": command})
                if result.returncode:
                    raise RuntimeError(f"{updates}_{name} failed; no optimizer interruption or silent retry")
            save("completed_probe", str(updates))
        save("completed_TRAIN_grounding_diagnostics_only", "done")
    except BaseException as exc:
        save("stopped_followup_ONLY_primary_fit_unchanged", "failure", str(exc))
        raise


if __name__ == "__main__":
    main()
