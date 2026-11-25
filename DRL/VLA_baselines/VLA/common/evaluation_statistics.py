"""Uncertainty estimates for the locked four-task evaluation protocol."""

from __future__ import annotations

import math
import numpy as np


def wilson_interval(successes: int, trials: int) -> list[float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval needs a valid nonempty binomial sample")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def success_statistics(rows: list[dict]) -> dict:
    """Report per-task intervals and retain correlation within a scene seed."""

    if not rows:
        raise ValueError("No evaluation rows")
    successes = sum(bool(row["success"]) for row in rows)
    by_task = {}
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        count = sum(bool(row["success"]) for row in task_rows)
        by_task[task] = {
            "trials": len(task_rows), "successes": count,
            "success_rate": count / len(task_rows),
            "wilson_95_interval": wilson_interval(count, len(task_rows)),
        }
    seeds = sorted({row["seed"] for row in rows})
    seed_rates = []
    expected_tasks = set(by_task)
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if {row["task"] for row in seed_rows} != expected_tasks or len(seed_rows) != len(expected_tasks):
            raise ValueError("Each seed must contain one complete set of tasks")
        seed_rates.append(np.mean([bool(row["success"]) for row in seed_rows]))
    rng = np.random.default_rng(20261007)
    samples = np.asarray(seed_rates)[rng.integers(0, len(seeds), (10000, len(seeds)))].mean(axis=1)
    return {
        "wilson_95_interval": wilson_interval(successes, len(rows)),
        "seed_block_bootstrap_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
        "seed_blocks": len(seeds), "bootstrap_replicates": 10000,
        "by_task": by_task,
    }
