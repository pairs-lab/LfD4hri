#!/usr/bin/env python3
"""Make an encoder-only index from Google's official sharded T5-XXL index.

RDT only instantiates ``T5EncoderModel``.  The official ``sharded`` revision of
``google/t5-v1_1-xxl`` places the shared embedding and all encoder tensors in
the first two shards, so downloading the three decoder-only shards is wasteful.
This script preserves the downloaded full index next to the filtered one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()
    index_path = args.checkpoint_dir / "pytorch_model.bin.index.json"
    full_path = args.checkpoint_dir / "pytorch_model.bin.index.full.json"
    if not full_path.exists():
        shutil.copy2(index_path, full_path)
    full = json.loads(full_path.read_text())
    weight_map = {
        key: value
        for key, value in full["weight_map"].items()
        if key == "shared.weight" or key.startswith("encoder.")
    }
    required = sorted(set(weight_map.values()))
    expected = ["pytorch_model-00001-of-00005.bin", "pytorch_model-00002-of-00005.bin"]
    if required != expected:
        raise RuntimeError(f"Unexpected official encoder shard layout: {required}")
    missing = [name for name in required if not (args.checkpoint_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing encoder shards: {missing}")
    filtered = {"metadata": full.get("metadata", {}), "weight_map": weight_map}
    index_path.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")
    print(f"Encoder-only T5 index: {len(weight_map)} tensors in {required}")


if __name__ == "__main__":
    main()
