"""Saved-weight evidence that real smoke updates learn the NEW state columns."""
import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open

from VLA.cross_vla120_trial import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Immutable numeric update proof")
    name = "model.state_proj.weight"
    with safe_open(args.base / "model.safetensors", framework="np") as original, safe_open(args.checkpoint / "model.safetensors", framework="np") as current:
        rows = {}
        for label, first, last in [("pad_and_command",7,11), ("target_relative",11,14),
                                   ("goal_relative",14,17), ("correct_grasp_feedback",17,18)]:
            before = original.get_slice(name)[:64,first:last]
            after = current.get_slice(name)[:64,first:last]
            delta = np.abs(after - before)
            rows[label] = {"changed_values": int(np.count_nonzero(delta)), "max_abs_weight_change": float(delta.max())}
    passed = all(row["changed_values"] > 0 and np.isfinite(row["max_abs_weight_change"]) for row in rows.values())
    atomic_json(args.output, {"passed": passed, "rows": rows, "state18_new_input_columns_receive_real_updates": passed,
                              "model_success_established": False})
    if not passed: raise ValueError("New information did not update its input projection; do not launch long fit")


if __name__ == "__main__":
    main()
