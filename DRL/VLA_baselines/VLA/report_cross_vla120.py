"""Refresh a readable SR table from REAL queue evidence; pending is never0%."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from VLA.run_cross_vla120 import TASKS


def render(progress):
    lines = ["# Cross-VLA:120 episodes per model /30 per action", "",
        "Live summary of actual recorded neural trials. Pending/incomplete rates are not0%.", "",
        "Same UR5e/Robotiq V9 scenes,30 seeds, controller, success condition; no oracle fallback.",
        "Adaptation regimes are NOT matched: official-zero-shot rows are NOT fair post-fit paper baselines.", "",
        f"Updated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Queue status: {progress['status']}; recorded neural episodes:{progress['completed_neural_trials']}/720.", "",
        "| Model | Regime | Touch | Pick | Move | Place | SR120 | Status |",
        "|---|---|---|---|---|---|---|---|"]
    for model, row in progress["models"].items():
        cells = []
        for task in TASKS:
            part = row["by_task"][task]
            cells.append(f"{part['successes']}/{part['completed']} (done{part['completed']}/30)"
                         if part["completed"] else "— (done0/30)")
        rate = row["success_rate_for_completed120"]
        sr = f"{100 * rate:.2f}% ({row['successes']}/120)" if rate is not None else f"— (done{row['completed_trials']}/120)"
        lines.append(f"| {model} | {row['adaptation_regime']} | {' | '.join(cells)} | {sr} | {row['status']} |")
    lines += ["", "Safety aborts remain scored task failures. Load/transport/integration failures leave the declared cohort incomplete.", ""]
    for model, row in progress["models"].items():
        if row.get("error"):
            lines += [f"{model}: integration error, preserved without reroll: `{row['error']}`", ""]
        if row["success_rate_for_completed120"] is not None:
            ci = row["wilson_95_interval"]
            lines += [f"{model}:120 trials Wilson95% interval [{100 * ci[0]:.2f}%, {100 * ci[1]:.2f}%].", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=72)
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 72:
        raise ValueError("Finite report wait required")
    deadline = time.monotonic() + args.wait_hours * 3600
    target = args.root / "REPORT.md"
    while True:
        progress = json.loads((args.root / "progress.json").read_text())
        temporary = target.with_suffix(".md.tmp")
        temporary.write_text(render(progress))
        temporary.replace(target)
        if (progress["status"] in {"completed720", "finished_with_explicit_incomplete_models",
                                   "stopped_global_preflight_error_no_reroll"}
                or not Path(f"/proc/{progress['pid']}").exists()
                or time.monotonic() >= deadline):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
