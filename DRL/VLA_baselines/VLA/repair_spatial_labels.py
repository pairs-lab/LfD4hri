#!/usr/bin/env python3
"""Create an immutable VLA dataset with mislabeled named regions regenerated.

Only episodes whose stored destination disagrees with the validated VLA
language map are simulated again.  Unaffected NPZ files are hard-linked
byte-for-byte into a new dataset directory; source metadata and hashes are
retained in the new manifest for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from config import PICK_TARGETS
from VLA.collect_demos import collect_episode
from VLA.common.environment import ManipulationEnv
from VLA.common.spatial import FIXED_DESTINATION
from VLA.common.tasks import TaskSpec, randomize_rl_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_positions(task: TaskSpec) -> list[np.ndarray]:
    positions = [np.asarray(task.pick_position, dtype=np.float64)]
    if task.place_name is not None and task.place_name not in FIXED_DESTINATION:
        assert task.place_position is not None
        positions.append(np.asarray(task.place_position, dtype=np.float64))
    positions.extend(np.asarray(item.position, dtype=np.float64) for item in task.scene_objects)
    return positions


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_index_path = source / "index.json"
    source_manifest_path = source / "dataset_manifest.json"
    source_index_hash = sha256(source_index_path)
    source_manifest_hash = sha256(source_manifest_path)
    audit = json.loads(args.source_audit.read_text())
    if (
        not audit.get("passed")
        or audit.get("failures")
        or Path(audit.get("dataset", "")).resolve() != source
        or audit.get("index_sha256") != source_index_hash
        or audit.get("manifest_sha256") != source_manifest_hash
    ):
        raise RuntimeError("Source dataset audit is failed, stale, or belongs to another dataset")

    index = json.loads(source_index_path.read_text())
    manifest = json.loads(source_manifest_path.read_text())
    output.mkdir(parents=True)
    repaired: list[str] = []
    unchanged: list[str] = []
    with ManipulationEnv(
        robot="ur5e",
        image_size=int(manifest["image_size"]),
        seed=int(manifest["seed"]),
    ) as env:
        for number, item in enumerate(index, start=1):
            task_dict = dict(item["task"])
            place_name = task_dict.get("place_name")
            expected = FIXED_DESTINATION.get(place_name)
            recorded = task_dict.get("place_position")
            needs_repair = expected is not None and tuple(recorded) != tuple(expected)
            destination = output / item["file"]
            updated = dict(item)
            if needs_repair:
                task_dict["place_position"] = list(expected)
                task = TaskSpec.from_dict(task_dict)
                goal = np.asarray(expected, dtype=np.float64)
                if any(
                    np.linalg.norm(position[:2] - goal[:2]) <= 0.13
                    for position in physical_positions(task)
                ):
                    # The old scene was conditioned around the wrong bottom-right
                    # coordinate. Re-sample the complete random_env scene around
                    # the corrected goal while preserving task and target names.
                    seed_material = f"{source_index_hash}:{item['file']}:spatial-v2"
                    repair_seed = int.from_bytes(
                        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
                        byteorder="little",
                    )
                    if task.pick_name not in PICK_TARGETS:
                        raise RuntimeError(f"Unknown pick target in {item['file']}")
                    task = randomize_rl_scene(
                        task,
                        np.random.default_rng(repair_seed),
                        forced_pick_name=task.pick_name,
                        forced_place_name=task.place_name,
                    )
                arrays, regenerated = collect_episode(env, task)
                if not regenerated["success"]:
                    raise RuntimeError(f"Regenerated oracle failed: {item['file']}")
                np.savez_compressed(destination, **arrays)
                updated = {**regenerated, "file": item["file"]}
                repaired.append(item["file"])
            else:
                os.link(source / item["file"], destination)
                unchanged.append(item["file"])
            index[number - 1] = updated
            print(
                f"{number}/{len(index)} {item['file']}: "
                f"{'regenerated' if needs_repair else 'unchanged'}",
                flush=True,
            )

    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (output / "dataset_manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "fixed_destination_mapping": FIXED_DESTINATION,
                "spatial_language_contract": (
                    "VLA/common/spatial.py; named region and coordinate are one-to-one"
                ),
                "spatial_label_repair": {
                    "source": str(source),
                    "source_index_sha256": source_index_hash,
                    "source_manifest_sha256": source_manifest_hash,
                    "source_audit": str(args.source_audit.resolve()),
                    "regenerated_files": repaired,
                    "hardlinked_unchanged_files": len(unchanged),
                },
            },
            indent=2,
        )
        + "\n"
    )
    (output / "collection_state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_episodes": len(index),
                "regenerated_files": repaired,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"episodes": len(index), "repaired": repaired}, indent=2))


if __name__ == "__main__":
    main()
