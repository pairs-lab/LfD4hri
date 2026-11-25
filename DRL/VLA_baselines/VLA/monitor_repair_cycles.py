"""Read-only job monitor plus one atomic user-facing status snapshot.

Never rewrites job/source/results or reruns failures. Completed task scores
remain separate from pending fits, smoke integration and expert corrections.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time


def read_json(path):
    path = Path(path)
    if not path.exists(): return None
    for _ in range(3):
        try: return json.loads(path.read_text())
        except json.JSONDecodeError: time.sleep(.05)
    return {"status": "snapshot_temporarily_incomplete"}


def tail(path, size=16384):
    if not path.exists(): return ""
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-size))
        return stream.read().decode(errors="replace")


def model_snapshot(root, model):
    state = read_json(root / "status.json") or {}
    result = {"status_file": str((root / "status.json").resolve()), "status": state.get("status"),
        "stage": state.get("stage"), "error": state.get("error"), "intended_optimizer_updates": state.get("updates"),
        "completed_new_development_success_rate": None}
    # A stale JSON snapshot is not a live trainer. Do not signal any PID here.
    # Missing identity is unverified; positive PID must have a live non-zombie
    # owner. Log counters remain diagnostic, never adopted saved updates.
    pid = state.get("pid")
    result["recorded_status"] = state.get("status")
    result["owner_alive"] = None
    if type(pid) is int and pid > 0:
        process = Path(f"/proc/{pid}/stat")
        try:
            alive = process.exists() and process.read_text().rsplit(")",1)[1].split()[0] != "Z"
        except (FileNotFoundError, IndexError):
            alive = False
        result["owner_alive"] = alive
        if state.get("status") == "running" and not alive:
            result.update(status="interrupted_owner_missing", error="Recorded running owner no longer exists; preserve checkpoints and audit before recovery")
    if model == "rdt":
        path = Path(state.get("model_output", "/missing")) / "train/train_metrics.jsonl"
        lines = tail(path).splitlines()
        for line in reversed(lines):
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            result.update(recorded_optimizer_updates=item["step"], last_training_loss=item["loss"])
            break
        progress = root / "development40/progress.json"
    else:
        values = re.findall(r"(\d+)/80000",tail(root / "logs/train.log"))
        if values:
            micro = int(values[-1])
            result.update(completed_microsteps_in_live_log=micro, approximate_optimizer_updates_from_log=micro//16,
                log_counter_NOT_saved_Adam_counter=True)
        progress = root / "development40/progress.json"
    result["development40_progress_file"] = str(progress.resolve())
    evaluation = read_json(progress)
    if evaluation:
        result["development_status"] = evaluation.get("status")
        result["completed_development_trials"] = evaluation.get("completed_trials",0)
        result["development_counts"] = evaluation.get("counts")
    comparison = read_json(root / "comparison.json")
    if comparison:
        result["completed_new_development_success_rate"] = comparison["postfit_successes"] / comparison["trials_each"]
        result["comparison_file"] = str((root / "comparison.json").resolve())
    if model == "rdt" and evaluation and evaluation.get("completed_trials") == 40:
        result["completed_development_counts"] = evaluation.get("counts")
        analysis = read_json(root / "development40/analysis.json")
        if analysis and analysis.get("complete40") is True:
            result["completed_new_development_success_rate"] = analysis["successes"] / 40
            result["completed_development_statistics"] = analysis["statistics"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smolvla", type=Path, required=True)
    parser.add_argument("--rdt", type=Path, required=True)
    parser.add_argument("--baseline-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os
    deadline = time.monotonic()+48*3600
    args.output.parent.mkdir(parents=True,exist_ok=True)
    while True:
        snapshots = {"smolvla":model_snapshot(args.smolvla,"smolvla"), "rdt":model_snapshot(args.rdt,"rdt")}
        baseline = read_json(args.baseline_comparison)
        states = [s.get("status") or "" for s in snapshots.values()]
        ended = all(s.startswith(("completed","stopped","failed")) for s in states)
        snapshot = {"scope":"Live status only; learned scores NOT inferred from optimizer/expert audits",
            "monitor_pid":os.getpid(),
            "updated_at_utc":datetime.now(timezone.utc).isoformat(), "models":snapshots,
            "last_completed_SmolVLA_R14_successes": baseline.get("postfit_successes") if baseline else None,
            "last_completed_SmolVLA_R14_trials":baseline.get("trials_each") if baseline else None,
            "final_test_used":False, "oracle_fallback":False, "stable_high_success_established":False,
            "monitor_status":"both_jobs_terminal_review_required" if ended else "monitoring_active_jobs"}
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot,indent=2)+"\n")
        temporary.replace(args.output)
        if ended or time.monotonic() > deadline: break
        time.sleep(30)


if __name__ == "__main__":
    main()
