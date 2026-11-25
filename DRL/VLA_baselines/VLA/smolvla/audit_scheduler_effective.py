"""Audit completed SmolVLA scheduler state without importing the model or Torch.

LeRobot's stock CosineDecayWithWarmupSchedulerConfig rescales BOTH warmup and
decay when configured decay exceeds the declared training microsteps. Native
config/lineage files preserve the requested values, not the effective values.
Passing this audit means the saved state matches that effective schedule; it
does not mean the requested update-unit schedule actually ran, nor establish
model performance. For an explicit 500-update run with 50-update warmup, use
decay=500 (with the same accumulation scaling) to avoid stock auto-rescaling.
Never change an active or fingerprinted experiment to repair its metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEDULER_TYPE = "cosine_decay_with_warmup"
PROJECT_ROOT = Path(__file__).absolute().parents[2]
STOCK_IMPLEMENTATION = PROJECT_ROOT / "VLA/third_party/lerobot/src/lerobot/optim/schedulers.py"


def integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return value


def effective_schedule(training_microsteps: int, accumulation: int,
                       warmup_microsteps: int, decay_microsteps: int) -> dict:
    """Mirror the stock scheduler's int-truncating automatic rescaling."""
    total = integer(training_microsteps, "training_microsteps", 1)
    accumulation = integer(accumulation, "accumulation", 1)
    warmup = integer(warmup_microsteps, "warmup_microsteps")
    decay = integer(decay_microsteps, "decay_microsteps", 1)
    if total % accumulation:
        raise ValueError("A completed run must end on an accumulation boundary")
    if warmup >= decay:
        raise ValueError("Configured warmup must end before configured decay")
    scale = min(1.0, total / decay)
    effective_warmup = int(warmup * scale) if total < decay else warmup
    effective_decay = min(total, decay)
    def units(warm, dec):
        return {"warmup_microsteps": warm, "decay_microsteps": dec,
                "warmup_optimizer_updates": warm / accumulation,
                "decay_optimizer_updates": dec / accumulation}
    return {"training_microsteps": total, "optimizer_updates": total // accumulation,
            "gradient_accumulation": accumulation, "stock_autoscale_applied": total < decay,
            "stock_autoscale_factor": scale,
            "configured_contract": units(warmup, decay),
            "effective_contract": units(effective_warmup, effective_decay),
            "configured_contract_is_effective": warmup == effective_warmup and decay == effective_decay}


def stock_lr_multiplier(current_microstep: int, warmup: int, decay: int,
                        peak_lr: float, decay_lr: float) -> float:
    """Exact stock lambda: cosine uses global step, not step-minus-warmup."""
    step = integer(current_microstep, "current_microstep")
    warmup = integer(warmup, "warmup")
    decay = integer(decay, "decay", 1)
    peak = finite(peak_lr, "peak_lr", 1e-300)
    floor = finite(decay_lr, "decay_lr")
    if floor > peak:
        raise ValueError("Decay LR cannot exceed peak LR")
    if step < warmup:
        if step <= 0:
            return 1.0 / (warmup + 1)
        fraction = 1.0 - step / warmup
        return (1.0 / (warmup + 1) - 1.0) * fraction + 1.0
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(step, decay) / decay))
    alpha = floor / peak
    return (1.0 - alpha) * cosine + alpha


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_checkpoint(checkpoint: Path) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "pretrained_model":
        raise ValueError("Expected a completed native pretrained_model checkpoint")
    paths = {"policy_config": checkpoint / "config.json",
             "train_config": checkpoint / "train_config.json",
             "scheduler_state": checkpoint.parent / "training_state/scheduler_state.json",
             "optimizer_param_groups": checkpoint.parent / "training_state/optimizer_param_groups.json"}
    data = {name: json.loads(path.read_text()) for name, path in paths.items()}
    policy, cfg, state, groups = (data[name] for name in paths)
    schedule_cfg = cfg["scheduler"]
    if schedule_cfg.get("type") != SCHEDULER_TYPE:
        raise ValueError("Only the stock cosine_decay_with_warmup scheduler is supported")
    if policy.get("type") != "smolvla":
        raise ValueError("This audit only supports SmolVLA checkpoints")
    total = integer(cfg["steps"], "train_config.steps", 1)
    accumulation = integer(cfg["accelerator"]["gradient_accumulation"]["steps"], "accumulation", 1)
    warmup = integer(schedule_cfg["num_warmup_steps"], "scheduler.num_warmup_steps")
    decay = integer(schedule_cfg["num_decay_steps"], "scheduler.num_decay_steps", 1)
    peak = finite(schedule_cfg["peak_lr"], "scheduler.peak_lr", 1e-300)
    floor = finite(schedule_cfg["decay_lr"], "scheduler.decay_lr")
    derived = effective_schedule(total, accumulation, warmup, decay)
    effective = derived["effective_contract"]
    last_epoch = integer(state["last_epoch"], "scheduler.last_epoch")
    failures = []
    if last_epoch != total:
        failures.append(f"Saved scheduler.last_epoch={last_epoch} != declared completed microsteps={total}")
    if not checkpoint.parent.name.isdigit() or int(checkpoint.parent.name) != total:
        failures.append("Checkpoint folder does not equal the declared completed microstep")
    expected_fields = {"scheduler_warmup_steps": warmup, "scheduler_decay_steps": decay,
                       "scheduler_decay_lr": floor, "optimizer_lr": peak}
    for label, observed in (("config.json", policy), ("train_config.policy", cfg["policy"])):
        for field, expected in expected_fields.items():
            if observed.get(field) != expected:
                failures.append(f"{label}.{field} differs from configured scheduler")
    expected_factor = stock_lr_multiplier(last_epoch, effective["warmup_microsteps"],
                                         effective["decay_microsteps"], peak, floor)
    if not isinstance(groups, list) or not groups:
        raise ValueError("Saved optimizer must contain at least one parameter group")
    base_lrs, last_lrs = state["base_lrs"], state["_last_lr"]
    if len(base_lrs) != len(groups) or len(last_lrs) != len(groups):
        raise ValueError("Scheduler and optimizer parameter-group counts disagree")
    lr_rows = []
    for number, (base, saved, group) in enumerate(zip(base_lrs, last_lrs, groups, strict=True)):
        base = finite(base, "scheduler.base_lr", 1e-300)
        saved = finite(saved, "scheduler._last_lr")
        optimizer_lr = finite(group["lr"], "optimizer.lr")
        initial_lr = finite(group["initial_lr"], "optimizer.initial_lr", 1e-300)
        expected_lr = base * expected_factor
        if not close(initial_lr, base):
            failures.append(f"Group {number} optimizer.initial_lr differs from scheduler.base_lrs")
        if not close(saved, expected_lr):
            failures.append(f"Group {number} saved scheduler LR differs from the stock formula")
        if not close(optimizer_lr, expected_lr):
            failures.append(f"Group {number} optimizer LR differs from the stock formula")
        lr_rows.append({"group": number, "base_lr": base, "expected_saved_lr": expected_lr,
                        "saved_scheduler_lr": saved, "saved_optimizer_lr": optimizer_lr,
                        "scheduler_max_abs_error": abs(saved - expected_lr),
                        "optimizer_max_abs_error": abs(optimizer_lr - expected_lr)})
    report = {"passed": not failures, "failures": failures, "checkpoint": str(checkpoint),
              "measurement": "saved_effective_scheduler_state_not_model_performance",
              **derived, "scheduler_type": SCHEDULER_TYPE,
              "scheduler_last_epoch": last_epoch, "scheduler_peak_lr": peak,
              "scheduler_decay_lr": floor, "parameter_groups": lr_rows,
              "configs_store_requested_values_only": True,
              "saved_lr_is_next_update_lr_not_previous_adam_update_lr": True,
              "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
              "next_recipe_example": {"optimizer_updates": 500, "warmup_updates": 50,
                  "decay_updates": 500, "explanation": "Set decay equal to run length to avoid stock autoscale; do not alter an active experiment"}}
    if STOCK_IMPLEMENTATION.is_file():
        report["stock_implementation"] = {"path": str(STOCK_IMPLEMENTATION), "sha256": sha256(STOCK_IMPLEMENTATION)}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a previous effective-schedule report")
    report = audit_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
