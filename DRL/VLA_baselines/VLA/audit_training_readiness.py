"""Audit raw/paired/recovery source quality and freeze a scene-disjoint split.

This does NOT qualify a small pilot as sufficient training data. Validation
here means held-out training-source layouts, not benchmark/final-test seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from config import PICK_TARGETS, PLACE_TARGETS
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import verify_source_payload_for_profile
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.observation_profile import profile_hash, validate_profile

TASKS = ("touch", "pick", "move", "pick_place")
CANONICAL_RESET = "canonical_spawn_order_then_teacher_retarget"
CLOSED_HOVER = "mid_height_contact_free_closed_empty_v1"


def reset_example_role(item, reset, canonical_cache, data):
    """A canonical reset is not necessarily a text-counterfactual pair.

    Only the explicit, verified single-instruction closed-hover collection may
    be classified separately. Near-object paired recoveries still require two
    texts, identical initial sensors and distinct action labels downstream.
    Metadata assertions alone cannot exempt an episode from the pair check.
    """
    if not reset:
        if item.get("recovery_start_kind") == CLOSED_HOVER:
            raise ValueError("Closed-hover recovery lacks canonical reset provenance")
        return "raw"
    if item.get("recovery_start_kind") != CLOSED_HOVER:
        return "counterfactual_pair"
    original = canonical_cache[reset["canonical_source"]][1][reset["paired_scene_source"]]
    if item.get("variant") != 0 or item["task"] != original["task"]:
        raise ValueError("Standalone closed-hover recovery changed canonical instruction/variant")
    startup = np.asarray(item.get("recovery_start_actions_not_training_labels", []), dtype=float)
    if (startup.ndim != 2 or startup.shape[1] != 7 or not 1 <= len(startup) <= 50
            or not np.isfinite(startup).all() or np.any(startup[:, [0, 1, 3, 4, 5]] != 0)
            or startup[-1, 6] != 1):
        raise ValueError("Closed-hover startup controls are missing or inconsistent")
    evidence = item.get("recovery_start_evidence", {})
    checks = item.get("closed_hover_recovery_checks", {})
    expected = ("initial_measured_closed_empty", "first_corrective_label_open",
                "first_post_action_not_false_grasp", "terminal_verified_grasp")
    if any(checks.get(key) is not True for key in expected):
        raise ValueError("Missing/failed closed-hover collection checks")
    if (evidence.get("kind") != CLOSED_HOVER or evidence.get("grasped_object", "missing") is not None
            or evidence.get("contact_count") != 0
            or any(evidence.get(key) is not False for key in
                   ("object_poses_teleported", "startup_controls_are_training_labels",
                    "near_body_or_natural_failure_recovery_claimed"))):
        raise ValueError("Closed-hover startup is not explicitly contact-free and unassisted")
    values = np.asarray([evidence.get(key, np.nan) for key in
                         ("actual_closure", "surface_clearance_m", "object_clearance_m")], dtype=float)
    closure = float(data["gripper_closure"][0])
    if (not np.isfinite(values).all() or not .95 <= values[0] <= 1 or values[1:].min() < .003
            or not np.isfinite(closure) or closure < .95 or abs(values[0] - closure) > 1e-6
            or float(data["canonical_actions"][0, 6]) != -1
            or bool(data["next_grasped_object"][0])
            or (item["task"]["name"] != "touch" and not bool(data["next_verified_grasp"][-1]))):
        raise ValueError("Closed-hover measured feedback/OPEN label disagrees with recovery evidence")
    return "closed_hover_recovery"


def source_reset_provenance(source, item, canonical_cache=None):
    """Resolve optional canonical reset metadata without misclassifying RAW rows.

    ``paired_source_dataset`` is the dataset containing ``paired_scene_source``;
    ``paired_collection_source`` keeps the identity of the original pair batch.
    All three paths survive materialization so mixed canonical parents need no
    invented, dataset-wide ``audit.json.source``.
    """
    source = Path(source).resolve()
    filename = item.get("paired_scene_source")
    strategy = item.get("paired_reset_strategy")
    startup = item.get("recovery_start_actions_not_training_labels", [])
    if not filename:
        if strategy == CANONICAL_RESET or startup or item.get("canonical_source") or item.get("paired_source_dataset"):
            raise ValueError("Recovery/canonical reset is missing paired_scene_source")
        return {}
    if "variant" not in item:
        raise ValueError("Explicit paired scene is missing its variant")
    if strategy not in (None, CANONICAL_RESET):
        raise ValueError(f"Unsupported paired reset strategy: {strategy}")
    explicit = item.get("canonical_source")
    paired_dataset = item.get("paired_source_dataset")
    if explicit and paired_dataset and Path(explicit).resolve() != Path(paired_dataset).resolve():
        raise ValueError("Conflicting per-row canonical reset datasets")
    canonical = explicit or paired_dataset
    if canonical is None:
        path = source / "audit.json"
        metadata = json.loads(path.read_text()) if path.is_file() else {}
        canonical = metadata.get("source")
    if not isinstance(canonical, str) or not canonical:
        raise ValueError("Paired/recovery source needs per-row canonical_source or legacy audit.json.source")
    canonical = Path(canonical).resolve()
    cache = {} if canonical_cache is None else canonical_cache
    key = str(canonical)
    if key not in cache:
        path = canonical / "index.json"
        original = json.loads(path.read_text())
        if len({row["file"] for row in original}) != len(original):
            raise ValueError("Canonical source index contains duplicate filenames")
        cache[key] = (sha256_file(path), {row["file"]: row for row in original})
    digest, original = cache[key]
    if item.get("canonical_source_index_sha256", digest) != digest:
        raise ValueError("Canonical reset index changed after provenance binding")
    if filename not in original:
        raise ValueError(f"Canonical reset episode missing: {canonical}/{filename}")
    if physical_scene_key(original[filename]["task"]) != physical_scene_key(item["task"]):
        raise ValueError("Paired/recovery provenance changed physical scene")
    return {"canonical_source": str(canonical), "paired_source_dataset": str(canonical),
            "canonical_source_index_sha256": digest, "paired_scene_source": filename,
            "paired_reset_strategy": CANONICAL_RESET,
            "paired_collection_source": str(Path(item.get("paired_collection_source", source)).resolve())}


def verify_readiness_source(source, manifest, audit):
    """Preserve legacy audits, requiring payload/code/profile binding for V9."""
    verify_source_payload_for_profile(audit, source)
    profile = manifest.get("observation_profile")
    if profile is not None:
        profile = validate_profile(profile)
        if profile_hash(profile) != manifest.get("observation_profile_sha256"):
            raise ValueError("Source sensor/home/controller profile hash is invalid")
    v9 = (profile or {}).get("gripper_controller") == "body_contact_v9"
    if v9 or manifest.get("environment_runtime_fingerprint") is not None:
        verify_environment_runtime_fingerprint(manifest.get("environment_runtime_fingerprint"), profile)
    if v9:
        if (profile.get("grasp_candidate_scope") != "all_dynamic_scene_objects"
                or manifest.get("gripper_state_contract") != "encoder_fk_aperture_closure_v1"
                or profile.get("gripper_state_contract") != "encoder_fk_aperture_closure_v1"
                or manifest.get("native_gripper_action_kind") != "requested_close_intent"):
            raise ValueError("V9 needs task-independent grasp scope and separate calibrated feedback/requested-action contracts")
    return v9


def physical_scene_key(task):
    # Include the physical destination, not virtual named-region coordinates.
    from VLA.common.spatial import FIXED_DESTINATION
    objects = [(task["pick_name"], task["pick_position"], False)]
    if task.get("place_name") and task["place_name"] not in FIXED_DESTINATION:
        objects.append((task["place_name"], task["place_position"],
                        task["place_name"].endswith(" bowl")))
    objects.extend((x["name"], x["position"], x["fixed"]) for x in task["scene_objects"])
    if len({x[0] for x in objects}) != len(objects):
        raise ValueError("Duplicate physical object in scene")
    return hashlib.sha256(json.dumps(sorted(objects), separators=(",", ":")).encode()).hexdigest()


def freeze_split(rows, validation_fraction=.2, seed=20261008, preserve_training_vocabulary=False, minimum_train_repetitions=1):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be strictly between zero and one")
    if type(minimum_train_repetitions) is not int or minimum_train_repetitions < 1:
        raise ValueError("Minimum training repetitions must be a positive integer")
    if minimum_train_repetitions > 1 and not preserve_training_vocabulary:
        raise ValueError("Repeated target coverage requires the explicit vocabulary constraint")
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_group"]].append(row)
    # A scene shared by primitives must remain together too. Stratify by its
    # task-set signature; deterministic hash ranking does not inspect results.
    strata = defaultdict(list)
    for group, members in groups.items():
        signature = tuple(sorted({r["task"] for r in members}))
        strata[signature].append(group)
    validation = set()
    def features(row):
        values = {(row["task"], "pick", row["pick_name"])}
        if row.get("place_name"):
            values.add((row["task"], "destination", row["place_name"]))
        return values
    group_features = ({group: set().union(*(features(r) for r in members))
                       for group, members in groups.items()} if preserve_training_vocabulary else {})
    remaining = Counter(feature for values in group_features.values() for feature in values)
    for signature, keys in sorted(strata.items()):
        if len(keys) < 2:
            raise ValueError(f"Cannot hold out a disjoint scene in stratum {signature}")
        keys.sort(key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
        count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
        if not preserve_training_vocabulary:
            validation.update(keys[:count])
            continue
        chosen = []
        for key in keys:
            # Coverage is determined from task labels BEFORE model evaluation,
            # not by successes/losses or rerolling seeds. Keep entire groups.
            if any(remaining[value] <= minimum_train_repetitions for value in group_features[key]):
                continue
            chosen.append(key)
            remaining.subtract(group_features[key])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Insufficient independent scenes for covered training vocabulary and validation quota: {signature}")
        validation.update(chosen)
    return {name: [r for r in rows if (r["scene_group"] in validation) == is_validation]
            for name, is_validation in (("train", False), ("validation", True))}


def describe(rows):
    tasks = {}
    for name in TASKS:
        selected = [r for r in rows if r["task"] == name]
        picks = Counter(r["pick_name"] for r in selected)
        destinations = Counter(r["place_name"] for r in selected if r["place_name"])
        tasks[name] = {
            "episodes": len(selected), "frames": sum(r["frames"] for r in selected),
            "physical_layouts": len({r["scene_group"] for r in selected}),
            "pick_counts": dict(picks), "missing_pick_names": sorted(set(PICK_TARGETS) - picks.keys()),
            "destination_counts": dict(destinations),
            "missing_destination_names": sorted(set(PLACE_TARGETS) - destinations.keys())
                if name in ("move", "pick_place") else [],
            "unique_pick_destination_pairs": len({(r["pick_name"], r["place_name"]) for r in selected}),
        }
    frames = sum(r["frames"] for r in rows)
    return {"episodes": len(rows), "frames": frames, "rgb_images": frames * 2,
            "physical_layouts": len({r["scene_group"] for r in rows}), "tasks": tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--source-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve-training-vocabulary", action="store_true",
                        help="Deterministic pre-training coverage constraint; never use outcomes to choose held-out scenes")
    parser.add_argument("--minimum-train-repetitions", type=int, default=1,
                        help="Minimum independent groups per target label; default preserves old split assignments")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.source) != len(args.source_audit):
        raise ValueError("Each source requires its own source audit")
    rows, failures, lineage = [], [], []
    profile_ids, action_kinds = set(), set()
    grasp_scopes = set()
    synchronization_error = gripper_synchronization_error = command_feedback_gap = 0.
    constant_motion_dimensions = set(range(6))
    paired = defaultdict(list)
    origins = defaultdict(set)
    canonical_cache = {}
    state_contracts = set()
    for source_arg, audit_path in zip(args.source, args.source_audit, strict=True):
        source = source_arg.resolve()
        index_path, manifest_path = source / "index.json", source / "dataset_manifest.json"
        index = json.loads(index_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (not audit.get("passed") or audit.get("failures")
                or Path(audit.get("dataset", "")).resolve() != source
                or audit.get("index_sha256") != sha256_file(index_path)
                or audit.get("manifest_sha256") != sha256_file(manifest_path)):
            raise ValueError(f"Stale/failed source audit: {source}")
        body_v9 = verify_readiness_source(source, manifest, audit)
        profile_ids.add(manifest.get("observation_profile_sha256"))
        action_kinds.add(manifest.get("native_joint_action_kind"))
        grasp_scopes.add(manifest.get("observation_profile", {}).get("grasp_candidate_scope", "selected_target_legacy"))
        state_contracts.add(manifest.get("gripper_state_contract", "commanded_closure"))
        lineage.append({"source": str(source), "source_index_sha256": sha256_file(index_path),
                        "source_manifest_sha256": sha256_file(manifest_path),
                        "source_audit": str(audit_path.resolve()), "source_audit_sha256": sha256_file(audit_path),
                        "source_payload_fingerprint": audit.get("source_payload_fingerprint")})
        for number, item in enumerate(index):
            task, filename = item["task"], item["file"]
            path = source / filename
            if path.resolve().parent != source:
                raise ValueError("Episode path escapes source")
            group = physical_scene_key(task)
            reset = source_reset_provenance(source, item, canonical_cache)
            reset_role = "invalid"
            if reset:
                origins[(reset["canonical_source"], reset["paired_scene_source"])].add(group)
            with np.load(path, allow_pickle=False) as data:
                action, q, command = (data[k] for k in ("canonical_actions", "joint_positions", "next_joint_targets"))
                feedback = data["next_observed_joint_positions"]
                length = len(action)
                if (not length or length != item["steps"] or action.shape != (length, 7)
                        or q.shape != (length, 6) or command.shape != (length, 6) or feedback.shape != (length, 6)):
                    failures.append(f"Shape mismatch: {path}")
                    continue
                numeric = [action, q, command, feedback, data["gripper_closure"], data["next_gripper_closure"]]
                if not all(np.isfinite(a).all() for a in numeric):
                    failures.append(f"Nonfinite: {path}")
                if str(data["prompt"].item()) != task["prompt"]:
                    failures.append(f"Text mismatch: {path}")
                if not item["success"] or not bool(data["next_raw_success"][-1]):
                    failures.append(f"Missing teacher success: {path}")
                if item.get("safety_violation"):
                    failures.append(f"Rejected safety transition: {path}")
                if body_v9:
                    required = ("next_requested_gripper_closure", "next_verified_grasp", "next_bilateral_pad_contact")
                    if any(key not in data or data[key].shape != (length,) for key in required):
                        failures.append(f"Missing or malformed V9 feedback/action/quality fields: {path}")
                    else:
                        requested = data["next_requested_gripper_closure"]
                        if not np.isfinite(requested).all() or np.any((requested < 0) | (requested > 1)):
                            failures.append(f"Invalid V9 requested gripper labels: {path}")
                        if task["name"] != "touch" and not bool(data["next_verified_grasp"][-1]):
                            failures.append(f"Unverified terminal physical grasp: {path}")
                    if item.get("gripper_state_contract", manifest["gripper_state_contract"]) != manifest["gripper_state_contract"]:
                        failures.append(f"Mixed gripper feedback contract: {path}")
                if length > 1:
                    synchronization_error = max(synchronization_error, float(np.max(np.abs(feedback[:-1] - q[1:]))))
                    gripper_synchronization_error = max(gripper_synchronization_error,
                        float(np.max(np.abs(data["next_gripper_closure"][:-1] - data["gripper_closure"][1:]))))
                command_feedback_gap = max(command_feedback_gap, float(np.max(np.abs(command - feedback))))
                constant_motion_dimensions &= set(np.flatnonzero(np.max(np.abs(action[:, :6]), axis=0) < 1e-7).tolist())
                initial_hash = hashlib.sha256(q[0].tobytes())
                for key in ("front_rgb", "wrist_rgb"):
                    image = data[key]
                    if image.dtype != np.uint8 or image.shape != (length, 384, 384, 3):
                        failures.append(f"Invalid RGB format: {path}:{key}")
                    initial_hash.update(image[0].tobytes())
                try:
                    reset_role = reset_example_role(item, reset, canonical_cache, data)
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append(f"Invalid reset-example role: {path}: {exc}")
                if reset_role == "counterfactual_pair":
                    startup = json.dumps(item.get("recovery_start_actions_not_training_labels", []), separators=(",", ":"))
                    pair_key = (reset["paired_collection_source"], reset["canonical_source"],
                                reset["paired_scene_source"], task["name"], startup)
                    paired[pair_key].append((initial_hash.hexdigest(), task["prompt"], action.copy(), item["variant"]))
            rows.append({"source": str(source), "file": filename, "sha256": sha256_file(path),
                         "scene_group": group, "task": task["name"], "frames": length,
                         "pick_name": task["pick_name"], "place_name": task.get("place_name"),
                         "prompt": task["prompt"], "variant": item.get("variant"),
                         "recovery_start_steps": len(item.get("recovery_start_actions_not_training_labels", [])),
                         "reset_example_role": reset_role,
                         **reset})
            if number % 40 == 0:
                print(f"{source.name}: {number}/{len(index)}", flush=True)
    if len(profile_ids) != 1 or None in profile_ids:
        failures.append("Mixed or absent sensor/home profiles")
    if action_kinds != {"requested_joint_position_command"}:
        failures.append("Native actions must be requested commands, not feedback")
    if synchronization_error != 0:
        failures.append("Post-action feedback does not match next observation")
    if gripper_synchronization_error != 0:
        failures.append("Post-action measured gripper feedback does not match next observation")
    if len(state_contracts) != 1:
        failures.append("Mixed gripper state conventions")
    if any(len(groups) != 1 for groups in origins.values()):
        failures.append("Paired/recovery provenance changed physical scene")
    for key, values in paired.items():
        if (len(values) != 2 or {x[3] for x in values} != {0, 1}
                or len({x[0] for x in values}) != 1 or len({x[1] for x in values}) != 2):
            failures.append(f"Counterfactual initial observation/text pair invalid: {key}")
        elif values[0][2].shape == values[1][2].shape and np.array_equal(values[0][2], values[1][2]):
            failures.append(f"Counterfactual labels identical despite changed target: {key}")
    split = freeze_split(rows, preserve_training_vocabulary=args.preserve_training_vocabulary,
                         minimum_train_repetitions=args.minimum_train_repetitions)
    groups = {name: {r["scene_group"] for r in members} for name, members in split.items()}
    overlap = groups["train"] & groups["validation"]
    if overlap:
        failures.append("Scene leakage across train/validation")
    report = {"source_integrity_passed": not failures, "failures": failures,
              "ready_for_long_finetune": False, "data_sufficiency": "NOT ESTABLISHED",
              "observation_profile_sha256": sorted(profile_ids), "lineage": lineage,
              "combined": describe(rows), "splits": {name: describe(members) for name, members in split.items()},
              "group_overlap_count": len(overlap), "split_seed": 20261008,
              "preserve_training_vocabulary": args.preserve_training_vocabulary,
              "minimum_train_repetitions": args.minimum_train_repetitions,
              "split_strategy": "hashed_groups_with_train_vocabulary_constraint" if args.preserve_training_vocabulary else "hashed_scene_groups",
              "feedback_to_next_observation_max_abs_error": synchronization_error,
              "gripper_feedback_to_next_observation_max_abs_error": gripper_synchronization_error,
              "requested_command_vs_feedback_max_gap_rad": command_feedback_gap,
              "constant_canonical_motion_dimensions": sorted(constant_motion_dimensions),
              "counterfactual_pairs_checked": len(paired),
              "raw_unpaired_episodes_checked": sum(not r.get("paired_scene_source") for r in rows),
              "paired_episodes_checked": sum(r["reset_example_role"] == "counterfactual_pair" for r in rows),
              "canonical_reset_episodes_checked": sum(bool(r.get("paired_scene_source")) for r in rows),
              "standalone_closed_hover_recoveries_checked": sum(r["reset_example_role"] == "closed_hover_recovery" for r in rows),
              "gripper_state_contracts": sorted(state_contracts),
              "grasp_candidate_scopes": sorted(grasp_scopes),
              "task_independent_grasp_candidates": grasp_scopes == {"all_dynamic_scene_objects"},
              "blocking_checks": ["Native conversion/loader/parity and fresh train-only normalization for this profile",
                                  "Grasp assistance must use task-independent dynamic scene candidates",
                                  "Missing destination coverage; more independent object/destination/layout demonstrations",
                                  ("Encoder/FK aperture feedback verified; contact/retry coverage and learned recovery still require validation"
                                   if state_contracts == {"encoder_fk_aperture_closure_v1"}
                                   else "Measured gripper-aperture convention not yet calibrated (state is commanded closure)"),
                                  "Small train-set fit followed by disjoint validation and online development rollouts"],
              "caveats": ["Source quality gates are not learned-policy success evidence",
                          "Paired and recovery copies do not multiply independent layout count",
                          "Validation is a training-source diagnostic, not a final benchmark score",
                          "Constant rotation labels limit recovery from orientation errors"],
              "payloads": rows}
    args.output.mkdir(parents=True)
    (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, members in split.items():
        (args.output / f"{name}_episodes.json").write_text(json.dumps(members, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("payloads", "lineage", "combined", "splits")}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
