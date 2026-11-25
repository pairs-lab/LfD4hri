"""Audited state18 sidecar over the unchanged canonical LeRobot dataset."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.common.source_training_gate import require_training_source
from VLA.simstate_feedback import FEATURE_NAMES, SCHEMA


def load_projection(root, *, require_full_train=True):
    root = Path(root).resolve()
    audit = json.loads((root / "audit.json").read_text())
    if (not audit.get("passed") or audit.get("schema") != SCHEMA or audit.get("pilot_only")
            or audit.get("future_state_in_model_input") is not False
            or audit.get("teacher_actions_in_model_input") is not False
            or audit.get("images_or_action_labels_modified") is not False):
        raise ValueError("Full causal projection qualification required")
    if require_full_train and (audit.get("split") != "train" or not audit.get("qualified_for_training")):
        raise ValueError("Validation/pilot projection cannot be training input")
    source = Path(audit["source"])
    if (sha256_file(source / "index.json") != audit["source_index_sha256"]
            or sha256_file(source / "dataset_manifest.json") != audit["source_manifest_sha256"]
            or sha256_file(root / "stats.json") != audit["stats_sha256"]):
        raise ValueError("Projected source/statistics changed")
    rows = json.loads((source / "index.json").read_text())
    if [r["file"] for r in rows] != audit["selected_files"] or len(rows) != len(audit["rows"]):
        raise ValueError("Projection frame/episode ordering differs")
    pieces = []
    for original, proof in zip(rows, audit["rows"], strict=True):
        state_file = root / "states" / original["file"]
        if (proof["file"] != original["file"]
                or sha256_file(state_file) != proof["state_payload_sha256"]
                or sha256_file(source / original["file"]) != proof["source_payload_sha256"]):
            raise ValueError("Original labels or state18 payload changed")
        with np.load(state_file, allow_pickle=False) as archive:
            values = archive["state"]
            if values.shape != (proof["frames"], 18) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Invalid projected state payload")
            pieces.append(values)
    values = np.concatenate(pieces)
    if len(values) != audit["frames"]:
        raise ValueError("Incomplete declared frames")
    return values, audit, json.loads((root / "stats.json").read_text())


class SimStateDataset(torch.utils.data.Dataset):
    def __init__(self, base, projection):
        self.base = base
        self.states, self.audit, stats = load_projection(projection)
        if len(base) != len(self.states):
            raise ValueError("LeRobot and full projection frame counts differ")
        self.meta = deepcopy(base.meta)
        self.meta.info.features["observation.state"] = {"dtype": "float32", "shape": [18], "names": FEATURE_NAMES}
        self.meta.stats["observation.state"] = {key: np.asarray(value, dtype=np.float32 if key != "count" else np.int64)
                                              for key, value in stats.items()}

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def features(self):
        return self.meta.features

    def __getitem__(self, index):
        sample = self.base[index]
        global_index = int(sample["index"])
        if global_index != index:
            raise ValueError("No hidden episode subset/remapping of state18 indices permitted")
        state = self.states[index]
        if not torch.allclose(sample["observation.state"].reshape(-1), torch.from_numpy(state[:7]), atol=1e-6, rtol=0):
            raise ValueError("Actual loader original state7 differs from its projected frame")
        original_shape = sample["observation.state"].shape
        return {**sample, "observation.state": torch.from_numpy(state.copy()).reshape(*original_shape[:-1], 18)}

    def __getitems__(self, indices):
        # PyTorch DataLoader prefers this optional batched method. Letting
        # __getattr__ delegate it to LeRobot silently bypasses our state18
        # overlay, returning original state7 for every training batch.
        # Go through the checked overlay for EVERY requested frame instead.
        return [self[int(index)] for index in indices]
