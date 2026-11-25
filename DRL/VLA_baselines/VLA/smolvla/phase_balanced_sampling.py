"""Explicit training-only phase ablation on unchanged, physics-proved labels.

Keep20% original target/episode-balanced draws;80% balances manipulation
phases within each task/target. No segmentation, pose or hold flag is a model
input. No target-coordinate servo, changed action, scorer or rollout filtering.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.smolvla.balanced_sampling import BalancedInitialSampler, frame_weights

PROFILE = "target_phase_balanced_v1"
PHASE_MASS = {"xy_choice": .40, "descent": .20, "acquisition": .10, "closed_unheld_open": .10,
              "held_transport": .15, "other": .05}
BASE_MIX = .20


def label_phases(data):
    actions = np.asarray(data["canonical_actions"])
    closure = np.asarray(data["gripper_closure"])
    held_after = np.asarray(data["next_grasped_object"])
    verified = np.asarray(data["next_verified_grasp"])
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 2:
        raise ValueError("Invalid actual labels")
    n = len(actions)
    if (closure.shape != (n,) or held_after.shape != (n,) or verified.shape != (n,)
            or held_after.dtype != bool or verified.dtype != bool or not np.isfinite(actions).all()
            or not np.isfinite(closure).all() or np.any((closure < 0) | (closure > 1))):
        raise ValueError("Invalid measured feedback arrays")
    # Exact qualified reset contracts separately prove starts NOT held.
    held_before = np.r_[False, held_after[:-1]]
    phase = np.full(n, "other", dtype="U20")
    phase[held_before] = "held_transport"
    phase[~held_before & (actions[:, 2] < -.003)] = "descent"
    phase[~held_before & (np.linalg.norm(actions[:, :2], axis=1) > .003)] = "xy_choice"
    phase[(actions[:, 6] > .2) & ~held_before & verified] = "acquisition"
    phase[(closure >= .95) & ~held_before & (actions[:, 6] < -.2)] = "closed_unheld_open"
    return phase


def mix_phase_weights(index, labels, initial_probability=.1):
    if len(index) != len(labels):
        raise ValueError("Label/index counts differ")
    base = frame_weights(index, initial_probability, target_balanced=True, grounding_probability=0.)
    groups = defaultdict(lambda: defaultdict(list))
    offset = 0
    for item, phases in zip(index, labels, strict=True):
        if len(phases) != item["steps"] or set(phases) - set(PHASE_MASS):
            raise ValueError("Actual phase labels differ from index")
        key = (item["task"]["name"], item["task"]["pick_name"])
        for phase in np.unique(phases):
            groups[key][str(phase)].extend((offset + np.flatnonzero(phases == phase)).tolist())
        offset += len(phases)
    types = Counter(task for task, _ in groups)
    if set(types) != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("All four tasks required")
    contrast = np.zeros_like(base)
    for (task, _), phases in groups.items():
        mass = 1. / (4*types[task])
        available = sum(PHASE_MASS[p] for p in phases)
        for phase, frames in phases.items():
            contrast[frames] = mass * PHASE_MASS[phase] / available / len(frames)
    weights = BASE_MIX * base + (1-BASE_MIX) * contrast
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1, atol=1e-12):
        raise ValueError("Phase probability mass invalid")
    return weights


def load_phase_recipe(source, initial_probability=.1):
    source = Path(source)
    paths = [source / "index.json", source / "dataset_manifest.json", Path(__file__)]
    hashes = {str(p.resolve()): sha256_file(p) for p in paths}
    index = json.loads((source / "index.json").read_text())
    labels = []
    for item in index:
        if item.get("recovery_start_kind") not in {None, "mid_height_contact_free_closed_empty_v1", "closed_empty_near_object_missed_close"}:
            raise ValueError("Phase recipe does not support held/unknown reset roles")
        with np.load(source / item["file"], allow_pickle=False) as data:
            labels.append(label_phases(data))
    weights = mix_phase_weights(index, labels, initial_probability)
    base = frame_weights(index, initial_probability, target_balanced=True)
    phases = np.concatenate(labels)
    actual = {phase: {"frames": int(np.sum(phases == phase)), "original_probability_mass": float(base[phases == phase].sum()),
                       "new_probability_mass": float(weights[phases == phase].sum())} for phase in PHASE_MASS}
    if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("Source/recipe changed while loading")
    contract = {"profile": PROFILE, "source_hashes": hashes, "frames": len(weights), "episodes": len(index),
        "base_mixture": BASE_MIX, "phase_mixture": 1-BASE_MIX, "requested_phase_mass": PHASE_MASS,
        "observed_phase_mass": actual, "phase_labels_form_disjoint_partition": True, "sampling_with_replacement": True,
        "target_balanced": True, "initial_frame_probability_in_base_component": initial_probability,
        "policy_inputs_changed": False, "source_labels_changed": False, "held_start_assumption": "separately qualified NOT-held reset only",
        "grasp_feedback_used_for_offline_sampling_only": True, "performance_improvement_established": False}
    return weights, contract


class PhaseBalancedSampler(BalancedInitialSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import os
        weights, _ = load_phase_recipe(os.environ["VLA_SMOLVLA_SAMPLING_SOURCE"], float(os.environ["VLA_SMOLVLA_INITIAL_FRAME_PROB"]))
        self.weights = torch.from_numpy(weights)
