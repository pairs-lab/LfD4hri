"""Expand qualified recovery protocol over the existing frozen geometry pool.

CPU/PyBullet collection only. Failure preserves candidate evidence and never
promotes partial/failed data into any active model's training source.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from config import PLACE_TARGETS
from VLA.audit_training_readiness import physical_scene_key
from VLA.common.provenance import sha256_file
from VLA.common.training_selection import select_probe_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v6_balanced_600_oneshot"
OUTPUT = ROOT / "VLA/datasets/candidates/wrist_v8_closed_empty_visible_64_allvocab_20261008"
AUDIT = ROOT / "VLA/audits/finetune_v8_closed_empty_full_20261008"
# Exact existing profile, not another camera/home recipe.
PROFILE = ROOT / "VLA/profiles/wrist_v8_h65_f80_workspace_scene_grasp.json"


def main():
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("Immutable recovery collection already exists")
    frozen = [ROOT / ("VLA/datasets/candidates/wrist_v8_grouped_" + name + "_20261008")
              for name in ("train", "validation")]
    groups = [{physical_scene_key(r["task"]) for r in json.loads((p / "index.json").read_text())}
              for p in frozen]
    if groups[0] & groups[1]:
        raise ValueError("Existing frozen split overlaps")
    rows = json.loads((SOURCE / "index.json").read_text())
    selected = [r for name in ("touch", "pick", "move", "pick_place")
                for r in select_probe_rows([r for r in rows if r["task"]["name"] == name],
                                          64, False, tuple(PLACE_TARGETS))]
    if any(physical_scene_key(r["task"]) not in groups[0] | groups[1] for r in selected):
        raise ValueError("Requested recovery scene is outside frozen geometry pool")
    counts = {name: 2 * count for name, count in Counter(r["task"]["name"] for r in selected).items()}
    AUDIT.mkdir(parents=True)
    (AUDIT / "frozen_geometry_plan.json").write_text(json.dumps({
        "planned_episodes": sum(counts.values()), "task_counts": counts,
        "source_index_sha256": sha256_file(SOURCE / "index.json"),
        "frozen_split_index_sha256": {p.name: sha256_file(p / "index.json") for p in frozen},
        "unknown_groups": 0, "geometry_reassignment": False, "final_test_used": False,
        "requested_files": [r["file"] for r in selected]}, indent=2) + "\n")
    history = []

    def status(state, stage):
        (AUDIT / "driver_status.json").write_text(json.dumps({"status": state, "stage": stage,
            "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(),
            "history": history, "final_test_used": False, "active_training_source_changed": False,
            "planned_episodes": sum(counts.values())}, indent=2) + "\n")

    commands = [
        ("collect", ["-m", "VLA.audit_counterfactual_collection", "--source", str(SOURCE),
            "--output", str(OUTPUT), "--observation-profile", str(PROFILE), "--layouts-per-task", "64",
            "--include-all-destinations", "--closed-empty-workspace-start"]),
        ("manifest", ["-m", "VLA.finalize_camera_candidate", "--candidate", str(OUTPUT),
            "--camera-report", str(ROOT / "VLA/audits/wrist_v8_scene_grasp_camera_grid_20261008/audit.json")]),
        ("source_audit", ["-m", "VLA.audit_finetune_data", "--dataset", str(OUTPUT),
            "--output", str(AUDIT / "source"), "--expected-task-counts", json.dumps(counts)]),
    ]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
    try:
        for stage, command in commands:
            log = ROOT / "VLA/logs" / ("v8_closed_empty_full_" + stage + ".log")
            status("running", stage)
            with log.open("x") as stream:
                result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT)
            history.append({"stage": stage, "exit_code": result.returncode, "log": str(log)})
            if result.returncode:
                raise RuntimeError(f"Recovery expansion failed at {stage}; retain evidence, do not weaken gate")
        if not json.loads((AUDIT / "source/audit.json").read_text()).get("passed"):
            raise RuntimeError("Exhaustive recovery source audit did not pass")
        status("source_audited_pending_visual_review_and_frozen_split_extension", "done")
    except Exception as exc:
        history.append({"error": repr(exc)})
        status("failed", stage if "stage" in locals() else "setup")
        raise


if __name__ == "__main__":
    main()
