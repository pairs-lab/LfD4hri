"""Finite CPU-only live evidence and post120 analysis watcher; never trains."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from VLA.analyze_simstate120 import analyze
from VLA.cross_vla120_trial import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous monitor output")
    deadline = time.monotonic() + 72*3600
    try:
        while time.monotonic() < deadline:
            cycle = json.loads((args.cycle / "status.json").read_text())
            projections = {}
            for split in ("train", "validation"):
                path = args.projections / split / "progress.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    projections[split] = {k: data.get(k) for k in ("status", "completed_episodes", "planned_episodes", "frames", "held_frames")}
            progress_path = args.cycle / "development120/progress.json"
            evaluation = json.loads(progress_path.read_text()) if progress_path.exists() else None
            summary = {"monitor_pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "cycle_stage": cycle["stage"], "cycle_status": cycle["status"], "cycle_error": cycle.get("error"),
                "causal_projections": projections, "privileged_input_variant_NOT_plain_VLA": True,
                "model_trials": evaluation["completed_trials"] if evaluation else 0,
                "learned_SR120": evaluation["successes"]/120 if evaluation and evaluation["status"] == "completed120_privileged_input_VLA" else None,
                "no_metric_inferred_from_teacher_replays_or_loss": True}
            atomic_json(args.output, summary)
            if evaluation and evaluation["status"] == "completed120_privileged_input_VLA":
                analysis = args.cycle / "post120_failure_audit"
                if not analysis.exists():
                    analyze(progress_path, analysis)
                return
            if cycle["status"] == "failed" or (evaluation and evaluation["status"].startswith("stopped")):
                raise RuntimeError("Cycle failed; monitor preserves failure, never calls it successful")
            if not Path(f"/proc/{cycle['pid']}").exists():
                raise RuntimeError("Cycle owner disappeared before neural qualification")
            time.sleep(30)
        raise TimeoutError("Finite72-hour evidence watch expired")
    except Exception as error:
        current = json.loads(args.output.read_text()) if args.output.exists() else {}
        atomic_json(args.output, {**current, "monitor_status": "stopped_with_preserved_error", "monitor_error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
