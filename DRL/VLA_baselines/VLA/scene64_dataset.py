"""Lossless shared source view for native scene64/67 VLA inputs.

Incomplete extraction and pilot data can be inspected ONLY when explicitly
requested. Neither permits a long fit. Each model still needs its own
batched-transform/CUDA/saved-precision/reload/robot qualification.
"""
from bisect import bisect_right
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import copy_requested_gripper_field, native_gripper_closure
from VLA.scene64_conditioning import augment_prompt, encode_numeric
from VLA.scene64_schema import SCHEMA, FEATURE_NAMES


def nearest_joint_targets(requested, measured):
    requested, measured = np.asarray(requested,np.float64), np.asarray(measured,np.float64)
    if requested.shape[-1:]!=(6,) or measured.shape!=(6,) or not np.isfinite(np.r_[requested.reshape(-1),measured]).all():
        raise ValueError("Finite six-joint native command and current observation required")
    current=(measured+np.pi)%(2*np.pi)-np.pi
    # OpenPI's DeltaActions then sees a short physical delta, NOT a spurious
    # +/-2pi jump caused by independently wrapping input and future labels.
    delta=(requested-measured+np.pi)%(2*np.pi)-np.pi
    return (current+delta).astype(np.float32)


def load_scene_projection(root, *, pilot=False):
    root = Path(root).resolve(); proof = json.loads((root / "audit.json").read_text())
    if (not proof.get("passed") or proof.get("schema") != SCHEMA or proof.get("feature_names") != FEATURE_NAMES
        or proof.get("teacher_action_or_phase_in_input") is not False or proof.get("future_state_in_input") is not False
        or proof.get("images_or_action_labels_modified") is not False
        or bool(proof.get("pilot_only")) != pilot or proof.get("split") != "train"):
        raise ValueError("Complete matching before-action scene64 replay proof required")
    source = Path(proof["source"])
    if (sha256_file(source / "index.json") != proof["source_index_sha256"]
        or sha256_file(source / "dataset_manifest.json") != proof["source_manifest_sha256"]
        or sha256_file(root / "stats.json") != proof["stats_sha256"]
        or sha256_file(Path(__file__).with_name("simstate_scene64.py")) != proof["feature_builder_sha256"]):
        raise ValueError("Scene/source/statistics implementation changed")
    originals = json.loads((source / "index.json").read_text())
    by_file = {row["file"]: row for row in originals}
    if not pilot and [r["file"] for r in originals] != proof["selected_files"]:
        raise ValueError("Full source coverage/order differs")
    if len(proof["rows"]) != len(proof["selected_files"]): raise ValueError("Missing episode proof")
    rows = []; pieces = []
    for name, item in zip(proof["selected_files"], proof["rows"], strict=True):
        row = by_file[name]; path = root / "states" / name
        if (item["file"] != name or item["frames"] != row["steps"]
            or sha256_file(path) != item["state_payload_sha256"]
            or sha256_file(source / name) != item["source_payload_sha256"]): raise ValueError("Scene/source payload changed")
        with np.load(path, allow_pickle=False) as data: values = data["state"].copy()
        if values.shape != (row["steps"], 64) or values.dtype != np.float32: raise ValueError("Scene timeline differs")
        # Exercising every row catches attachment, padding and rotation defects,
        # not only shape/finite errors in an arbitrary representative frame.
        for value in values: encode_numeric(value)
        pieces.append(values); rows.append(row)
    values = np.concatenate(pieces)
    if len(values) != proof["frames"]: raise ValueError("Incomplete frame extraction")
    return values, proof, rows


class Scene64NativeDataset:
    def __init__(self, projection, action_horizon=10, *, pilot=False):
        self.states, self.audit, self.rows = load_scene_projection(projection, pilot=pilot)
        self.source = Path(self.audit["source"]); self.horizon = int(action_horizon)
        self.ends = np.cumsum([r["steps"] for r in self.rows]).tolist()
        self.pilot_only = pilot
        if self.horizon < 1 or self.ends[-1] != len(self.states): raise ValueError("Invalid native horizon/count")

    def __len__(self): return len(self.states)

    @lru_cache(maxsize=1)
    def episode(self, index):
        with np.load(self.source / self.rows[index]["file"], allow_pickle=False) as archive:
            result = {key: archive[key] for key in ("front_rgb", "wrist_rgb", "joint_positions",
                "gripper_closure", "next_joint_targets", "next_gripper_closure", "canonical_actions", "prompt")}
            copy_requested_gripper_field(archive, result)
        return result

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self): raise IndexError(index)
        ei = bisect_right(self.ends, index); fi = index - (self.ends[ei-1] if ei else 0)
        data = self.episode(ei); raw = self.states[index]
        ordinary = np.r_[data["joint_positions"][fi], data["gripper_closure"][fi]].astype(np.float32)
        if not np.allclose(ordinary, raw[:7], atol=1e-6, rtol=0): raise ValueError("Wrong source/pre-action frame")
        ids = np.minimum(fi + np.arange(self.horizon), len(data["canonical_actions"]) - 1)
        action = np.c_[data["next_joint_targets"][ids], native_gripper_closure(data)[ids]].astype(np.float32)
        action[:, :6] = nearest_joint_targets(data["next_joint_targets"][ids],ordinary[:6])
        canonical = data["canonical_actions"][ids].astype(np.float32, copy=True)
        opened = canonical.copy(); opened[:, 6] = (1-canonical[:, 6])/2
        instruction = str(data["prompt"].item())
        return {"index": index, "episode_index": ei, "frame_index": fi, "image": data["front_rgb"][fi].copy(),
            "wrist_image": data["wrist_rgb"][fi].copy(), "joints": ((ordinary[:6]+np.pi) % (2*np.pi)-np.pi).astype(np.float32),
            "gripper": ordinary[6:].copy(), "actions": action, "canonical_actions": canonical,
            "language_actions": opened, "original_prompt": instruction, "prompt": augment_prompt(instruction, raw),
            "pre_action_scene64": raw.copy(), "numeric_scene67": encode_numeric(raw)}

    def __getitems__(self, indices): return [self[int(index)] for index in indices]
