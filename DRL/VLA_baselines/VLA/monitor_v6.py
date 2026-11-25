"""Continuously audit active training and run serialized development diagnostics.

Only development seeds are opened here. Final testing remains the responsibility
of the quiet-GPU queue. Low task success is recorded honestly, never converted
into an integration error or hidden by changing the success predicate.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "VLA/logs"
RUNS = {
    "rdt": ROOT / "VLA/rdt/finetune/cam_v6_h10_fp32_master_warmstart2000_20000",
    "smolvla": ROOT / "VLA/smolvla/finetune/cam_v6_fp32_master_warmstart10000_bs4_accum4_10000updates",
}
TRAIN_LOGS = {"rdt": LOGS / "rdt_v6_h10_train_20000.log",
              "smolvla": LOGS / "smolvla_v6_fp32_master_train_10000updates.log"}
SOURCE = ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v6_balanced_600_oneshot"
AUDIT = ROOT / "VLA/audits/camera_v6_balanced_600_oneshot"
STOP = False
SMOL_ACCUMULATION = 4
SMOL_UPDATES = 10000
SMOL_TAG = "smolvla_fp32"


def now():
    return datetime.now(timezone.utc).isoformat()


def tail(path, size=100_000):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode("utf-8", errors="replace")


def training_progress(model):
    path = TRAIN_LOGS[model]
    value = tail(path)
    if model == "smolvla":
        steps = re.findall(rf"(\d+)/{SMOL_UPDATES * SMOL_ACCUMULATION}", value)
        step = int(steps[-1]) if steps else 0
    else:
        steps = re.findall(r'"step":\s*(\d+)', value)
        step = int(steps[-1]) if steps else 0
    return {"microstep" if model == "smolvla" else "optimizer_update": step,
            "latest_saved_or_logged_step": step,
            "optimizer_updates": step // SMOL_ACCUMULATION if model == "smolvla" else step,
            "last_log_age_seconds": None if not path.exists() else round(time.time() - path.stat().st_mtime, 1),
            "training_returned": (RUNS[model] / "training_lineage.json").exists() if model == "smolvla" else step >= 20000,
            "numerical_or_traceback_warning": bool(re.search(
                r'Traceback \(most recent|"?(?:loss|grad_norm|grdn)"?\s*:\s*(?:nan|[-+]?inf(?:inity)?)\b', value, re.IGNORECASE))}


def read_queue():
    path = LOGS / "v6_training_queue_status.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "unavailable"}


def resources():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    free_gpu, used_gpu = [int(value.strip()) for value in result.stdout.strip().split(",")]
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available_ram = int(memory["MemAvailable"].split()[0]) / 1024
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True)
    active_trainers = []
    for value in processes.stdout.splitlines():
        try:
            pid = int(value.strip())
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (ValueError, OSError):
            continue
        if (("VLA.rdt.training" in command and str(RUNS["rdt"]) in command.replace("--output ", ""))
                or ("VLA.smolvla.train_fp32" in command and str(RUNS["smolvla"]) in command)):
            active_trainers.append(pid)
        # RDT was launched with a relative output path in the original command.
        elif "VLA.rdt.training" in command and str(RUNS["rdt"].relative_to(ROOT)) in command:
            active_trainers.append(pid)
    return {"gpu_free_mib": free_gpu, "gpu_used_mib": used_gpu, "ram_available_mib": round(available_ram),
            "active_scoped_trainer_pids": active_trainers}


def jobs():
    return [
        {"id": f"rdt_h10_u{step:06d}", "model": "rdt", "step": step,
         "checkpoint": str(RUNS["rdt"] / "checkpoints" / f"step-{step:06d}")}
        for step in (10000, 15000)
    ] + [
        {"id": f"{SMOL_TAG}_u{step:06d}", "model": "smolvla", "step": step * SMOL_ACCUMULATION,
         "checkpoint": str(RUNS["smolvla"] / "checkpoints" / f"{step * SMOL_ACCUMULATION:06d}" / "pretrained_model")}
        for step in (1000, 3000, 6000) if step < SMOL_UPDATES
    ]


class Monitor:
    def __init__(self, interval):
        self.interval = interval
        self.status_path = LOGS / "continuous_monitor_v6_status.json"
        self.events_path = LOGS / "continuous_monitor_v6_events.jsonl"
        self.last_health_event = 0.
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {"jobs": {}}
        for job in self.status["jobs"].values():
            if job.get("status") == "running":
                job.update({"status": "attention_required", "error": "Previous monitor interrupted; preserve partial artifacts for review"})
        self.status.update({"pid": os.getpid(), "started_at": now(), "interval_seconds": interval,
                            "development_seeds": [20262001, 20262002], "replan_steps": [1, 2],
                            "final_test_used": False, "automatic_visual_judgment": False})
        self.environment = os.environ.copy()
        self.environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
                                 "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"), "HF_HUB_OFFLINE": "1"})

    def event(self, event, **values):
        with self.events_path.open("a") as stream:
            stream.write(json.dumps({"at": now(), "event": event, **values}) + "\n")

    def health(self):
        self.status.update({"updated_at": now(), "training": {model: training_progress(model) for model in RUNS},
                            "resources": resources(), "queue": read_queue()})
        self.status["warnings"] = [
            model + ": numerical/traceback warning" for model, value in self.status["training"].items()
            if value["numerical_or_traceback_warning"]
        ] + [
            model + ": training log has not advanced for 15 minutes" for model, value in self.status["training"].items()
            if value["last_log_age_seconds"] is not None and value["last_log_age_seconds"] > 900 and not value["training_returned"]
        ] + [job_id + ": " + value.get("error", "Diagnostic requires review")
             for job_id, value in self.status["jobs"].items() if value.get("status") == "attention_required"]
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.status, indent=2) + "\n")
        temporary.replace(self.status_path)
        if time.monotonic() - self.last_health_event >= self.interval:
            self.event("health_checked", training=self.status["training"], resources=self.status["resources"],
                       queue_status=self.status["queue"].get("status"), warnings=self.status["warnings"])
            self.last_health_event = time.monotonic()

    def child(self, command, log):
        if log.exists():
            raise FileExistsError(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if STOP:
                        raise InterruptedError("Monitor stop requested")
                    try:
                        process.wait(timeout=self.interval)
                    except subprocess.TimeoutExpired:
                        self.health()
                if process.returncode:
                    raise RuntimeError(f"Diagnostic subprocess exited {process.returncode}; inspect {log}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    def diagnose(self, job):
        from VLA.run_evaluation_protocol import validate_protocol, DEFAULT_PROTOCOL, DEFAULT_ORACLE_AUDIT

        protocol, protocol_hash = validate_protocol(DEFAULT_PROTOCOL, DEFAULT_ORACLE_AUDIT)
        if any(seed not in protocol["development_seeds"] for seed in self.status["development_seeds"]):
            raise RuntimeError("Monitor seeds must belong to the locked development split")
        directory = ROOT / "VLA/audits/continuous_v6" / job["id"]
        if directory.exists():
            raise FileExistsError("Refusing duplicate/partial diagnostic: " + str(directory))
        directory.mkdir(parents=True)
        log_root = LOGS / "continuous_v6" / job["id"]
        log_root.mkdir(parents=True, exist_ok=True)
        checkpoint = Path(job["checkpoint"])
        model = job["model"]
        model_python = ROOT / ("VLA/third_party/lerobot/.venv/bin/python" if model == "smolvla" else "VLA/.venvs/rdt/bin/python")
        if model == "smolvla":
            self.child([str(model_python), "-m", "VLA.smolvla.finalize_checkpoint", "--checkpoint", str(checkpoint),
                        "--source", str(SOURCE), "--audit-root", str(AUDIT)], log_root / "finalize.log")
        if not (checkpoint / "benchmark_fingerprint.json").is_file():
            self.child([sys.executable, "-m", "VLA.fingerprint_checkpoint", str(checkpoint)], log_root / "fingerprint.log")
        server = None
        server_stream = None
        port = 5612
        rows, videos = [], []
        try:
            if model == "rdt":
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        raise RuntimeError("Dedicated monitor inference port is occupied")
                except (ConnectionRefusedError, TimeoutError):
                    pass
                server_stream = (log_root / "server.log").open("w")
                server = subprocess.Popen([str(model_python), "-m", "VLA.rdt.server", "--checkpoint", str(checkpoint),
                                           "--embeddings", str(ROOT / "VLA/rdt/prompt_embeddings_v6.pt"),
                                           "--vision", str(ROOT / "VLA/checkpoints/siglip-so400m-patch14-384"), "--port", str(port)],
                                          cwd=ROOT, env=self.environment, stdout=server_stream, stderr=subprocess.STDOUT)
                deadline, last_health = time.monotonic() + 1200, 0.
                while True:
                    if STOP or server.poll() is not None:
                        raise RuntimeError("Diagnostic server stopped before readiness")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Diagnostic server readiness timeout")
                        if time.monotonic() - last_health >= self.interval:
                            self.health()
                            last_health = time.monotonic()
                        time.sleep(1)
            simulator_python = str(model_python) if model == "smolvla" else sys.executable
            for replan in self.status["replan_steps"]:
                for seed in self.status["development_seeds"]:
                    test_id = f"monitor_v6_{job['id']}_seed{seed}_rp{replan}"
                    run = ROOT / "VLA" / model / test_id
                    if (run / "results.json").exists():
                        raise FileExistsError(run)
                    command = [simulator_python, "-m", "VLA.benchmark", "--model", model, "--robot", protocol["robot"],
                               "--tasks", ",".join(protocol["tasks"]), "--test-id", test_id, "--seed", str(seed),
                               "--scene-profile", protocol["scene_profile"], "--control-hz", str(protocol["control_hz"]),
                               "--replan-steps", str(replan), "--required-success-steps", str(protocol["success"]["required_consecutive_steps"]),
                               "--success-mode", "semantic", "--success-threshold-scale", str(protocol["success"]["reward_xml_threshold_scale"]),
                               "--checkpoint", str(checkpoint)]
                    for name, value in protocol["success"]["semantic_thresholds"].items():
                        command += ["--semantic-" + name.replace("_", "-"), str(value)]
                    if model == "rdt":
                        command += ["--port", str(port)]
                    record = replan == 1 and seed == self.status["development_seeds"][0]
                    if record:
                        command += ["--audit-observation-every", "20"]
                    else:
                        command += ["--no-video"]
                    self.child(command, log_root / f"seed{seed}_rp{replan}.log")
                    result = json.loads((run / "results.json").read_text())
                    if len(result["tasks"]) != 4 or any(task["error"] is not None for task in result["tasks"]):
                        raise RuntimeError("Integration error is not a task failure: " + str(run))
                    for task in result["tasks"]:
                        rows.append({"task": task["name"], "seed": seed, "replan_steps": replan,
                                     "success": task["success"], "result": str(run / "results.json")})
                    # Keep completed evidence even if a later diagnostic fails.
                    (directory / "partial_results.json").write_text(json.dumps(rows, indent=2) + "\n")
                    if record:
                        self.child([sys.executable, "-m", "VLA.audit_video", str(run)], log_root / "video_audit.log")
                        videos.append(str(run / "video_contact_sheet.png"))
            offline = [str(model_python) if model == "smolvla" else sys.executable,
                       "-m", f"VLA.{model}.audit_checkpoint_predictions", "--checkpoint", str(checkpoint),
                       "--source", str(SOURCE), "--audit-report", str(AUDIT / "audit.json"),
                       "--episodes-per-task", "6", "--output", str(directory / "offline_imitation.json")]
            if model == "rdt":
                offline += ["--port", str(port)]
            self.child(offline, log_root / "offline_imitation.log")
            from VLA.common.evaluation_statistics import success_statistics

            report = {"model": model, "checkpoint": str(checkpoint), "protocol_sha256": protocol_hash,
                      "final_test_used": False, "integration_passed": True, "rows": rows,
                      "by_replan": [{"replan_steps": replan, "trials": len(selected),
                                     "successes": sum(row["success"] for row in selected),
                                     "success_rate": sum(row["success"] for row in selected) / len(selected),
                                     "uncertainty": success_statistics(selected)}
                                    for replan in self.status["replan_steps"]
                                    for selected in [[row for row in rows if row["replan_steps"] == replan]]],
                      "video_contact_sheets": videos, "requires_manual_visual_review": True,
                      "latency_not_reportable": "Development inference shares GPU with training",
                      "model_stability_established": False}
            (directory / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if server_stream is not None:
                server_stream.close()

    def run(self):
        self.event("monitor_started", pid=os.getpid())
        while not STOP:
            self.status["status"] = "monitoring_training"
            self.health()
            queue = self.status["queue"]
            if queue.get("status") == "training_finished":
                self.status["status"] = "queue_completed_pending_result_review"
                self.health()
                return
            light_training = any(not value["training_returned"] for value in self.status["training"].values())
            capacity = self.status["resources"]
            if (light_training and capacity["active_scoped_trainer_pids"] and queue.get("status") == "waiting_for_gpu"
                    and capacity["gpu_free_mib"] >= 3000 and capacity["ram_available_mib"] >= 2500):
                for job in jobs():
                    if job["id"] in self.status["jobs"]:
                        continue
                    checkpoint = Path(job["checkpoint"])
                    progress = self.status["training"][job["model"]]["latest_saved_or_logged_step"]
                    marker = checkpoint / ("vla_checkpoint_manifest.json" if job["model"] == "rdt" else "train_config.json")
                    # Advance beyond a save before opening it; a directory can
                    # exist while the writer is still serializing weights.
                    if progress <= job["step"] or not marker.is_file():
                        continue
                    self.status["jobs"][job["id"]] = {"status": "running", **job, "started_at": now()}
                    self.status["status"] = "development_diagnostic_running"
                    self.health()
                    try:
                        report = self.diagnose(job)
                        self.status["jobs"][job["id"]].update({"status": "completed", "finished_at": now(),
                                                                "requires_visual_review": True, "by_replan": report["by_replan"]})
                        self.event("diagnostic_completed", job=job["id"], by_replan=report["by_replan"])
                    except Exception as error:
                        self.status["jobs"][job["id"]].update({"status": "attention_required", "error": str(error), "finished_at": now()})
                        self.event("diagnostic_error", job=job["id"], error=str(error))
                    self.health()
                    break
            self.status["status"] = "health_monitor_only_while_final_queue_runs" if not light_training else "waiting_for_next_ready_checkpoint"
            self.health()
            time.sleep(self.interval)
        self.status["status"] = "stopped_by_request"
        self.health()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--smolvla-run", type=Path)
    parser.add_argument("--smolvla-log", type=Path)
    parser.add_argument("--smolvla-accumulation", type=int, default=4)
    parser.add_argument("--smolvla-updates", type=int, default=10000)
    parser.add_argument("--smolvla-tag", default="smolvla_fp32")
    args = parser.parse_args()
    global SMOL_ACCUMULATION, SMOL_UPDATES, SMOL_TAG
    if args.smolvla_accumulation < 1 or args.smolvla_updates < 1 or not re.fullmatch(r"[A-Za-z0-9_]+", args.smolvla_tag):
        raise ValueError("Invalid SmolVLA profile")
    SMOL_ACCUMULATION, SMOL_UPDATES, SMOL_TAG = args.smolvla_accumulation, args.smolvla_updates, args.smolvla_tag
    if args.smolvla_run is not None:
        RUNS["smolvla"] = args.smolvla_run.resolve()
    if args.smolvla_log is not None:
        TRAIN_LOGS["smolvla"] = args.smolvla_log.resolve()
    if args.interval < 10:
        raise ValueError("Monitoring interval must be at least 10 seconds")
    if not args.run:
        print(json.dumps(jobs(), indent=2))
        return
    def stop(_signum, _frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "continuous_monitor_v6.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Monitor(args.interval).run()


if __name__ == "__main__":
    main()
