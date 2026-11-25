"""Serial native GPU setup checks after the running SmolVLA adaptation trial.

These are optimizer/integration tests, NOT model performance evaluations.
Never open final seeds or start a long fine-tune on an unvalidated model.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "VLA/audits/finetune_v8_20261008"
SOURCE = ROOT / "VLA/datasets/candidates/wrist_v8_grouped_train_20261008"


def jobs():
    yield "rdt", [str(ROOT / "VLA/.venvs/rdt/bin/python"), "-m", "VLA.rdt.training", "smoke",
        "--source", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"),
        "--action-contract-report", str(AUDIT / "rdt_action_contract_native.json"),
        "--language-embeddings", str(ROOT / "VLA/rdt/prompt_embeddings_v8_full_grammar_bf16.pt"),
        "--action-horizon", "10", "--samples-per-episode", "4",
        "--smoke-report", str(AUDIT / "rdt_gpu_smoke.json")]
    yield "pi0_fast", [str(ROOT / "VLA/third_party/openpi/.venv/bin/python"), "-m", "VLA.pi0.training", "smoke",
        "--model", "pi0_fast", "--repo-id", "openpi_wrist_v8_grouped_train_20261008",
        "--source-dataset", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"),
        "--roundtrip-report", str(AUDIT / "openpi_roundtrip.json"),
        "--action-adapter-report", str(AUDIT / "pi0_action_adapter_train.json"),
        "--closed-loop-adapter-report", str(AUDIT / "pi0_closed_loop_semantic_train.json"),
        "--action-horizon", "5", "--fast-codec-scale", "20",
        "--fast-codec-report", str(AUDIT / "fast_h5_scale20_closed_loop_replay.json"),
        "--fast-tokenizer-path", "/home/uet/.cache/huggingface/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4",
        "--steps", "1000", "--batch-size", "1", "--gradient-accumulation", "4",
        "--exp-name", "cam_v8_fast_h5_native_gpu_smoke", "--smoke-report", str(AUDIT / "pi0_fast_gpu_smoke.json")]
    yield "openvla", [str(ROOT / "VLA/.venvs/openvla/bin/python"), "-m", "VLA.openvla.finetune",
        "--source", str(SOURCE), "--audit-report", str(AUDIT / "train_source/audit.json"),
        "--dataset", str(ROOT / "VLA/datasets/openvla_wrist_v8_grouped_train_20261008"),
        "--roundtrip-report", str(AUDIT / "openvla_roundtrip.json"),
        "--preprocess-report", str(AUDIT / "openvla_exact_worst_preprocess.json"),
        "--batch-size", "1", "--grad-accumulation", "8", "--steps", "2", "--save-freq", "2",
        "--output", str(ROOT / "VLA/openvla/finetune/cam_v8_qlora_native_gpu_smoke")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps([{ "stage": name, "command": command} for name, command in jobs()], indent=2))
        return
    status_path = AUDIT / "other_models_gpu_smoke_queue_status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    dependency = AUDIT / "cam_v8_grouped_vlm_targetbalance_warm_u1000_driver_status.json"
    history = []
    def write(status, stage):
        status_path.write_text(json.dumps({"status": status, "next_stage": stage, "pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(), "history": history,
            "scope": "Native GPU setup tests, not learned success rates", "final_test_used": False}, indent=2) + "\n")
    while True:
        state = json.loads(dependency.read_text()) if dependency.exists() else {}
        if state.get("status") == "completed_diagnostic_only":
            break
        if str(state.get("status", "")).startswith("failed"):
            write("blocked_by_failed_upstream_job", "dependency")
            raise RuntimeError("Inspect the failed active trial before scheduling another GPU user")
        write("waiting_for_running_smolvla_trial", "dependency")
        time.sleep(10)
    environment = os.environ.copy()
    environment.pop("JAX_PLATFORMS", None)
    environment.update({"HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "HF_HUB_OFFLINE": "1",
        "OPENPI_DATA_HOME": str(ROOT / "VLA/cache/openpi"), "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    for name, command in jobs():
        while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            write("waiting_for_gpu", name)
            time.sleep(5)
        log = ROOT / "VLA/logs" / ("finetune_v8_queued_" + name + "_gpu_smoke.log")
        if log.exists():
            raise FileExistsError(log)
        write("running", name)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        history.append({"model": name, "exit_code": result.returncode, "log": str(log)})
        if result.returncode:
            write("failed", name)
            raise RuntimeError(f"{name} setup failed; do not count it as model task failure")
    write("completed_setup_tests_only", "done")


if __name__ == "__main__":
    main()
