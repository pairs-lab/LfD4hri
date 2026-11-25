"""Reload monitor improvements only between complete diagnostic jobs.

Never interrupts a benchmark, inference server, or trainer. An exclusive lock
is checked before replacing the driver so two evaluators cannot run together.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--after-job", required=True)
    args = parser.parse_args()
    state_path = ROOT / "VLA/logs/continuous_monitor_v6_status.json"
    while True:
        state = json.loads(state_path.read_text())
        if state.get("pid") != args.pid:
            raise RuntimeError("Driver ownership changed; no process was signalled")
        if state.get("jobs", {}).get(args.after_job, {}).get("status") in ("completed", "attention_required"):
            running = any(job.get("status") == "running" for job in state["jobs"].values())
            child_path = Path(f"/proc/{args.pid}/task/{args.pid}/children")
            children = child_path.read_text().strip() if child_path.exists() else ""
            if not running and not children:
                command = Path(f"/proc/{args.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                if "-m VLA.monitor_v6 --run" not in command:
                    raise RuntimeError("Unexpected process command; no process was signalled")
                os.kill(args.pid, signal.SIGTERM)
                print("Stopped only idle monitor driver at diagnostic boundary:", args.pid, flush=True)
                break
        time.sleep(2)
    deadline = time.monotonic() + 120
    with (ROOT / "VLA/logs/continuous_monitor_v6.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Previous driver did not release its lock; no replacement launched")
                time.sleep(1)
    print("Restarting updated monitor; completed checkpoint results retained", flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "VLA.monitor_v6", "--run", "--interval", "60"])


if __name__ == "__main__":
    main()
