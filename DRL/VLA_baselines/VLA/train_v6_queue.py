#!/usr/bin/env python3
"""Run audited heavyweight VLA fine-tunes serially after current GPU jobs.

Every job has its own immutable experiment directory and log. Failed jobs stop
the queue rather than silently starting an unvalidated replacement model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v6_balanced_600_oneshot"
AUDIT = ROOT / "VLA/audits/camera_v6_balanced_600_oneshot"
OPENPI_PY = ROOT / "VLA/third_party/openpi/.venv/bin/python"
FAST_TOKENIZER = Path("/home/uet/.cache/huggingface/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4")


def active_gpu_compute() -> list[str]:
    result = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"
    ], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def jobs():
    simulator_python = ROOT / "VLA/.venvs/openvla/bin/python"
    def evaluation(model, checkpoint, tag):
        return [str(simulator_python), "-m", "VLA.evaluate_v6_model", "--model", model,
                "--checkpoint", str(checkpoint), "--run-tag", tag, "--development-only"]

    # The two lightweight target-domain fine-tunes are already active. Their
    # final outputs are evaluated once the GPU is quiet, before loading a
    # heavyweight foundation model that occupies nearly the entire device.
    yield "smolvla_evaluate", evaluation(
        "smolvla", ROOT / "VLA/smolvla/finetune/cam_v6_visual_connector_fp32_warmstart1000_bs2_accum8_5000updates/checkpoints/040000/pretrained_model",
        "v6_visual_connector_fp32_warmstart1000_final5000updates",
    )
    yield "rdt_evaluate", evaluation(
        "rdt", ROOT / "VLA/rdt/finetune/cam_v6_h10_fp32_master_warmstart2000_20000/checkpoints/step-020000",
        "v6_h10_warmstart2000_final20000updates",
    )
    common = [
        "--repo-id", "openpi_lerobot_cam_v6_balanced_600_oneshot",
        "--source-dataset", str(SOURCE), "--audit-report", str(AUDIT / "audit.json"),
        "--roundtrip-report", str(AUDIT / "openpi_roundtrip.json"),
        "--action-adapter-report", str(AUDIT / "pi0_action_adapter.json"),
        "--closed-loop-adapter-report", str(AUDIT / "pi0_closed_loop_replay.json"),
        "--steps", "30000", "--batch-size", "1", "--gradient-accumulation", "4",
        "--save-interval", "5000",
    ]
    openpi_jobs = (
        ("pi0", "cam_v6_h10_bs1_accum4_30000updates", ["--action-horizon", "10"]),
        ("pi0_fast", "cam_v6_h5_scale20_bs1_accum4_30000updates", [
            "--action-horizon", "5", "--fast-codec-scale", "20",
            "--fast-tokenizer-path", str(FAST_TOKENIZER),
            "--fast-codec-report", str(AUDIT / "pi0_fast_codec_h5_scale20_replay_full.json"),
        ]),
    )
    # Validate each large model's real GPU update before committing to any
    # long training run. Otherwise FAST/OpenVLA setup errors remain hidden
    # behind tens of thousands of pi0 updates.
    for model, tag, extra in openpi_jobs:
        prefix = [str(OPENPI_PY), "-m", "VLA.pi0.training"]
        arguments = ["--model", model, "--exp-name", tag, *common, *extra]
        yield model + "_smoke", [*prefix, "smoke", *arguments, "--smoke-report", str(AUDIT / (model + "_v6_gpu_smoke.json"))]
    openvla_prefix = [
        str(ROOT / "VLA/.venvs/openvla/bin/python"), "-m", "VLA.openvla.finetune",
        "--source", str(SOURCE), "--audit-report", str(AUDIT / "audit.json"),
        "--dataset", str(ROOT / "VLA/datasets/openvla_cam_v6_balanced_600_oneshot"),
        "--roundtrip-report", str(AUDIT / "openvla_roundtrip.json"),
        "--preprocess-report", str(AUDIT / "openvla_exact_worst_preprocess.json"),
        "--batch-size", "1", "--grad-accumulation", "8",
    ]
    yield "openvla_smoke", [*openvla_prefix, "--steps", "2", "--save-freq", "2",
        "--output", str(ROOT / "VLA/openvla/finetune/cam_v6_qlora_gpu_smoke_bs1_accum8")]
    for model, tag, extra in openpi_jobs:
        prefix = [str(OPENPI_PY), "-m", "VLA.pi0.training"]
        arguments = ["--model", model, "--exp-name", tag, *common, *extra]
        yield model + "_train", [*prefix, "train", *arguments]
        yield model + "_evaluate", evaluation(
            model, ROOT / "VLA/pi0/finetune" / (model + "_pybullet_ur5e_robotiq") / tag / "119999",
            "v6_" + tag + "_final30000updates",
        )
    yield "openvla_train", [*openvla_prefix,
        "--output", str(ROOT / "VLA/openvla/finetune/cam_v6_qlora_bs1_accum8_10000updates"),
        "--steps", "10000", "--save-freq", "1000",
    ]
    yield "openvla_evaluate", evaluation(
        "openvla", ROOT / "VLA/openvla/finetune/cam_v6_qlora_bs1_accum8_10000updates/checkpoints/010000",
        "v6_qlora_final10000updates",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute; otherwise print the concrete queue")
    parser.add_argument("--plan-output", type=Path, help="Save the concrete command plan without starting GPU jobs")
    args = parser.parse_args()
    queue = list(jobs())
    if not args.run:
        plan = [{"job": name, "command": command} for name, command in queue]
        if args.plan_output:
            if args.plan_output.exists():
                raise FileExistsError(args.plan_output)
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            args.plan_output.write_text(json.dumps(plan, indent=2) + "\n")
            print(json.dumps({"plan": str(args.plan_output.resolve()), "jobs": [name for name, _ in queue]}))
        else:
            print(json.dumps(plan, indent=2))
        return 0
    status_path = ROOT / "VLA/logs/v6_training_queue_status.json"
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "HF_HUB_OFFLINE": "1",
        "OPENPI_DATA_HOME": str(ROOT / "VLA/cache/openpi"),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    })
    environment.pop("JAX_PLATFORMS", None)
    history = []
    for name, command in queue:
        while True:
            active = active_gpu_compute()
            if not active:
                break
            status_path.write_text(json.dumps({
                "status": "waiting_for_gpu", "next_job": name, "active_gpu_processes": active,
                "completed_jobs": history, "pid": os.getpid(),
            }, indent=2) + "\n")
            time.sleep(10)
        log = ROOT / "VLA/logs" / ("v6_queue_" + name + ".log")
        if log.exists():
            raise FileExistsError(f"Refusing to overwrite queue log: {log}")
        record = {"job": name, "command": command, "log": str(log),
                  "started_at": datetime.now(timezone.utc).isoformat()}
        status_path.write_text(json.dumps({"status": "running", **record, "completed_jobs": history,
                                          "pid": os.getpid()}, indent=2) + "\n")
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record["exit_code"] = result.returncode
        history.append(record)
        if result.returncode:
            status_path.write_text(json.dumps({"status": "failed", "completed_jobs": history}, indent=2) + "\n")
            return result.returncode
    status_path.write_text(json.dumps({"status": "training_finished", "completed_jobs": history}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
