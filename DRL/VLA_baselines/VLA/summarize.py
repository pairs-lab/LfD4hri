#!/usr/bin/env python3
"""Build a compact, reproducible summary from benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ("oracle", "openvla", "pi0", "pi0_fast", "cogact", "rdt", "smolvla")
TASKS = ("touch", "pick", "move", "pick_place")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    rows: list[dict] = []
    for model in MODELS:
        test_dir = ROOT / model / "test_1"
        results = read_json(test_dir / "results.json")
        readiness = read_json(test_dir / "readiness.json")
        if results is not None:
            by_task = {item["name"]: item for item in results.get("tasks", [])}
            errors = [item["error"] for item in by_task.values() if item.get("error")]
            timed_tasks = [
                item
                for item in by_task.values()
                if item.get("mean_inference_ms") is not None and item.get("steps", 0) > 0
            ]
            timed_steps = sum(item["steps"] for item in timed_tasks)
            mean_inference_ms = (
                sum(item["mean_inference_ms"] * item["steps"] for item in timed_tasks)
                / timed_steps
                if timed_steps
                else None
            )
            row = {
                "model": model,
                "status": (
                    "runtime_error"
                    if errors
                    else ("evaluated" if results.get("comparable", True) else "physics_check")
                ),
                "comparable": bool(results.get("comparable", True)),
                "successes": sum(bool(item.get("success")) for item in by_task.values()),
                "trials": len(by_task),
                "step_weighted_mean_inference_ms": mean_inference_ms,
                "tasks": {
                    task: (
                        None
                        if task not in by_task
                        else {
                            "success": bool(by_task[task].get("success")),
                            "steps": by_task[task].get("steps"),
                            "mean_inference_ms": by_task[task].get("mean_inference_ms"),
                            "error": by_task[task].get("error"),
                        }
                    )
                    for task in TASKS
                },
                "reason": "; ".join(errors) if errors else None,
                "artifacts": {
                    "results": str(test_dir / "results.json"),
                    "video": str(test_dir / "rollout.mp4"),
                },
            }
        elif readiness is not None:
            row = {
                "model": model,
                "status": readiness.get("status", "blocked"),
                "comparable": True,
                "successes": None,
                "trials": 0,
                "step_weighted_mean_inference_ms": None,
                "tasks": {task: None for task in TASKS},
                "reason": readiness.get("reason"),
                "artifacts": {"readiness": str(test_dir / "readiness.json")},
            }
        else:
            row = {
                "model": model,
                "status": "not_run",
                "comparable": model != "oracle",
                "successes": None,
                "trials": 0,
                "step_weighted_mean_inference_ms": None,
                "tasks": {task: None for task in TASKS},
                "reason": None,
                "artifacts": {},
            }
        rows.append(row)

    payload = {
        "protocol": "zero-shot integration on deterministic test_1",
        "warning": (
            "Zero-shot scores are not a fair capability comparison. Use identical target-domain "
            "demonstrations and held-out seeds before comparing with retrained TD3."
        ),
        "rows": rows,
    }
    (ROOT / "results_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Zero-shot integration status",
        "",
        "> These runs validate integration, not a fair final capability comparison. All learned",
        "> policies require the same target-domain training split before comparison with retrained TD3.",
        "",
        "| Model | Status | Touch | Pick | Move | Pick/place | Success | Mean inference/step |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        task_cells = []
        for task in TASKS:
            item = row["tasks"][task]
            task_cells.append("—" if item is None else ("✓" if item["success"] else "✗"))
        score = "—" if row["successes"] is None else f'{row["successes"]}/{row["trials"]}'
        latency = row["step_weighted_mean_inference_ms"]
        latency_cell = "—" if latency is None else f"{latency:.1f} ms"
        lines.append(
            f'| {row["model"]} | {row["status"]} | '
            + " | ".join(task_cells)
            + f" | {score} | {latency_cell} |"
        )
    lines.extend(
        (
            "",
            "Latency is the step-weighted mean measured on this machine; it includes model warm-up.",
            "pi0-FAST produced malformed-length action-token chunks on this out-of-domain scene;",
            "those steps use the official tokenizer's zero-action fallback (see `VLA/pi0_fast/test_1/diagnostics.json`).",
            "",
            "Machine-readable details: `VLA/results_summary.json`.",
            "",
        )
    )
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
