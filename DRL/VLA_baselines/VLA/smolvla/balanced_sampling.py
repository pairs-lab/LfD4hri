"""Deterministic task/episode-balanced sampling with explicit initial-state mass.

This is a fine-tuning-recipe ablation, not a dataset or benchmark modification.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def grounding_initial_weights(index):
    """Equal task/target mass on genuine non-recovery text pairs at t=0.

    This is supervised sampling metadata, never an observation to the policy.
    Source audit must separately establish identical pixels/feedback per pair.
    """
    groups = {}
    starts = np.r_[0, np.cumsum([int(row["steps"]) for row in index])[:-1]]
    for number, row in enumerate(index):
        if row.get("grounding_text_intervention") != "target":
            continue
        if row.get("recovery_start_actions_not_training_labels"):
            raise ValueError("Initial grounding mass must not silently select recovery starts")
        key = (row["task"]["name"], row.get("paired_collection_source"),
               row.get("canonical_source"), row.get("paired_scene_source"))
        if not all(key):
            raise ValueError("Grounding sampling needs explicit paired source provenance")
        groups.setdefault(key, []).append((number, row))
    tasks = Counter(key[0] for key in groups)
    if set(tasks) != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("Grounding sampling requires target pairs for every primitive")
    result = np.zeros(sum(int(row["steps"]) for row in index), dtype=np.float64)
    target_counts = Counter((key[0], row["task"]["pick_name"])
                            for key, pair in groups.items() for _, row in pair)
    target_types = Counter(task for task, _ in target_counts)
    for key, pair in groups.items():
        if (len(pair) != 2 or {row.get("variant") for _, row in pair} != {0, 1}
                or len({row["task"]["pick_name"] for _, row in pair}) != 2
                or len({row["task"]["prompt"] for _, row in pair}) != 2):
            raise ValueError("Incomplete or non-target-counterfactual grounding pair")
        for number, row in pair:
            # Alternate targets may occur in several pairs. Do not let this
            # introduce object-class imbalance into the target-balanced base.
            result[starts[number]] = 1 / (4 * target_types[key[0]] * target_counts[(key[0], row["task"]["pick_name"])])
    return result, len(groups)


def frame_weights(index, initial_probability, target_balanced=False, grounding_probability=0.):
    if not 0 < initial_probability < 1:
        raise ValueError("initial_probability must lie strictly between zero and one")
    counts = Counter(item["task"]["name"] for item in index)
    if set(counts) != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("Balanced sampling requires all four task types")
    arrays = []
    target_counts = Counter((item["task"]["name"], item["task"]["pick_name"]) for item in index) if target_balanced else {}
    target_types = Counter(task for task, _ in target_counts)
    for item in index:
        length = int(item["steps"])
        if length < 2:
            raise ValueError("Initial-frame sampling requires at least two frames per episode")
        task = item["task"]["name"]
        mass = (1 / (len(counts) * target_types[task] * target_counts[(task, item["task"]["pick_name"])])
                if target_balanced else 1 / (len(counts) * counts[task]))
        weights = np.full(length, mass * (1 - initial_probability) / (length - 1), dtype=np.float64)
        weights[0] = mass * initial_probability
        arrays.append(weights)
    result = np.concatenate(arrays)
    if not np.isclose(result.sum(), 1, atol=1e-12):
        raise RuntimeError("Sampling probabilities do not sum to one")
    if not 0 <= grounding_probability < 1:
        raise ValueError("grounding_probability must lie in [0, 1)")
    if grounding_probability:
        grounding, _ = grounding_initial_weights(index)
        result = (1 - grounding_probability) * result + grounding_probability * grounding
        if not np.isclose(result.sum(), 1, atol=1e-12):
            raise RuntimeError("Grounding mixture does not sum to one")
    return result


class BalancedInitialSampler(Sampler):
    def __init__(self, dataset_from_indices, dataset_to_indices, *, source_index,
                 initial_probability=.30, episode_indices_to_use=None,
                 target_balanced=False,
                 grounding_probability=0.,
                 drop_n_first_frames=0, drop_n_last_frames=0, shuffle=True,
                 seed=0, absolute_to_relative_idx=None):
        if drop_n_first_frames or drop_n_last_frames or not shuffle or absolute_to_relative_idx is not None:
            raise ValueError("This ablation requires the complete unfiltered map-style dataset")
        if episode_indices_to_use is not None and list(episode_indices_to_use) != list(range(len(source_index))):
            raise ValueError("Episode subset needs its own validated sampling contract")
        lengths = np.asarray([item["steps"] for item in source_index], dtype=np.int64)
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        ends = np.cumsum(lengths)
        if not np.array_equal(starts, dataset_from_indices) or not np.array_equal(ends, dataset_to_indices):
            raise RuntimeError("LeRobot episode offsets differ from audited source index")
        self.weights = torch.from_numpy(frame_weights(source_index, initial_probability, target_balanced, grounding_probability))
        self.seed, self._epoch, self._start_index = int(seed), 0, 0

    def __len__(self):
        return len(self.weights)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Negative sampler epoch")
        self._epoch = int(epoch)

    def state_dict(self):
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state):
        epoch, start = int(state["epoch"]), int(state["start_index"])
        if epoch < 0 or not 0 <= start <= len(self):
            raise ValueError("Invalid sampling restart cursor")
        self._epoch, self._start_index = epoch, start

    def __iter__(self):
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        generator = torch.Generator().manual_seed(epoch_seed)
        selected = torch.multinomial(self.weights, len(self), replacement=True, generator=generator)
        return iter(selected[start:].tolist())


def sampling_contract(source, probability, target_balanced=False, grounding_probability=0.):
    source = Path(source)
    index_path = source / "index.json"
    index = json.loads(index_path.read_text())
    weights = frame_weights(index, probability, target_balanced, grounding_probability)
    mass, initial_mass, target_mass, offset = Counter(), Counter(), Counter(), 0
    for item in index:
        name, length = item["task"]["name"], item["steps"]
        mass[name] += float(weights[offset:offset + length].sum())
        initial_mass[name] += float(weights[offset])
        if target_balanced:
            target_mass[name + ":" + item["task"]["pick_name"]] += float(weights[offset:offset + length].sum())
        offset += length
    contract = {"profile": "task_target_episode_balanced_initial_frames" if target_balanced else "task_episode_balanced_initial_frames", "initial_frame_probability": probability,
            "task_probability_mass": dict(mass), "task_initial_probability_mass": dict(initial_mass),
            "target_balanced": target_balanced, "target_probability_mass": dict(target_mass),
            "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "frames": len(weights), "sampling_with_replacement": True,
            "dataset_images_labels_stats_changed": False,
            "performance_improvement": "NOT ESTABLISHED; requires GPU smoke and source/dev tests"}
    if grounding_probability:
        _, pair_count = grounding_initial_weights(index)
        contract.update(profile="task_target_balanced_with_counterfactual_initial_mass",
            grounding_initial_probability=grounding_probability, grounding_pairs=pair_count,
            total_initial_probability=grounding_probability + (1-grounding_probability)*probability,
            mixture_rule="base draws plus task/requested-target/episode-balanced paired t=0 draws",
            paired_branches_forced_into_same_minibatch=False,
            independent_scene_count_increased_by_sampling=False)
    return contract
