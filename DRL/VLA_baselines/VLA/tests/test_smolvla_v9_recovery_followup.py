import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from VLA.run_smolvla_v9_recovery_followup import (
    HARNESS, RECOVERY_COUNTS, TASKS, assert_frozen_inputs, inherit_predeclared_selection, main, parse_args,
    prepare_plan, run_followup, select_recovery_enriched, selection_metadata,
    subgroup_summary, validate_primary_completion, wait_for_primary,
)

MODULE = "VLA.run_smolvla_v9_recovery_followup"
CONTRACT = {"train_payload_tree_sha256": "train-tree", "validation_payload_tree_sha256": "val-tree",
            "observation_profile_sha256": "camera-profile"}


def fixture_rows(split):
    rows = []
    base = 10000 if split == "heldout" else 0
    for number, task in enumerate(TASKS):
        for scene in range(3):
            rows.append({"file": f"{task}_normal_{scene}.npz", "success": True,
                         "task": {"name": task, "pick_name": "red apple", "pick_position": [base + number * 100 + scene, 0, 0],
                                  "place_name": None, "place_position": None, "scene_objects": []}})
        for variant in range(RECOVERY_COUNTS[split][task]):
            objects = [("red apple", [base + 1000 + number * 100, 0, 0]),
                       ("green pear", [base + 1000 + number * 100, 1, 0])]
            target, other = objects[variant], objects[1 - variant]
            rows.append({"file": f"{task}_recovery_{variant}.npz", "success": True,
                         "recovery_start_actions_not_training_labels": [[0, 0, 0, 0, 0, 0, 1]],
                         "task": {"name": task, "pick_name": target[0], "pick_position": target[1],
                                  "place_name": None, "place_position": None,
                                  "scene_objects": [{"name": other[0], "position": other[1], "fixed": False}]}})
    return rows


class FakeTime:
    def __init__(self):
        self.now, self.sleeps = 0, []

    def clock(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class SmolVLARecoveryFollowupTest(unittest.TestCase):
    def arguments(self, root, extra=()):
        return parse_args(["--source", str(root / "train"), "--validation", str(root / "val"),
                           "--checkpoint", str(root / "primary/train/checkpoints/008000/pretrained_model"),
                           "--driver-status", str(root / "primary-audit/pilot_driver_status.json"),
                           "--output", str(root / "followup"), *extra])

    def fixtures(self, root):
        for split, folder in (("train", "train"), ("heldout", "val")):
            source = root / folder
            source.mkdir()
            (source / "index.json").write_text(json.dumps(fixture_rows(split)))
            (source / "dataset_manifest.json").write_text(json.dumps({"split": folder}))
            (source / "audit.json").write_text(json.dumps({"passed": True}))

    def prepared(self, root, extra=()):
        self.fixtures(root)
        args = self.arguments(root, extra)
        with patch(MODULE + ".verify_source_inputs", return_value=CONTRACT):
            plan = prepare_plan(args)
        return args, plan

    def primary_status(self, args, plan):
        harness_hash = next(row["sha256"] for row in plan["frozen_files"] if Path(row["path"]) == HARNESS)
        return {"status": "completed_diagnostic_only", "source": str(args.source), "validation": str(args.validation),
                "harness_sha256": harness_hash,
                "stages": [{"name": name, "command": ["python", "-m", "harness", "--checkpoint", str(args.checkpoint)]}
                           for name in ("train_closedloop", "heldout_closedloop")]}

    def test_includes_all_six_train_and_two_val_recovery_rows(self):
        for split in ("train", "heldout"):
            source = fixture_rows(split)
            selected = select_recovery_enriched(source, split)
            expected_recovery = {row["file"] for row in source if row.get("recovery_start_actions_not_training_labels")}
            actual_recovery = {row["file"] for row in selected if row.get("recovery_start_actions_not_training_labels")}
            self.assertEqual(expected_recovery, actual_recovery)
            self.assertEqual(len(selected), 8)
            self.assertTrue(all(sum(row["task"]["name"] == task for row in selected) == 2 for task in TASKS))
            meta = selection_metadata(selected)
            self.assertEqual(meta["recovery_trials"], 6 if split == "train" else 2)
            self.assertEqual(meta["physical_layouts"], 5 if split == "train" else 7)
            self.assertEqual(meta["recovery_physical_layouts"], 3 if split == "train" else 1)
            self.assertNotIn("recovery_start_actions_not_training_labels", json.dumps(meta))

    def test_retry_preserves_original_selection_and_rebinds_only_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, plan = self.prepared(root)
            previous = copy.deepcopy(plan)
            driver = str(Path(HARNESS).with_name("run_smolvla_v9_recovery_followup.py"))
            next(row for row in previous["frozen_files"] if row["path"] == driver)["sha256"] = "old-driver"
            path = root / "previous.json"
            path.write_text(json.dumps(previous))
            inherited = copy.deepcopy(plan)
            inherit_predeclared_selection(inherited, path)
            self.assertTrue(inherited["previous_declaration"]["driver_rebound"])
            self.assertEqual(inherited["selections"], previous["selections"])
            with patch(MODULE + ".verify_source_inputs", return_value=CONTRACT):
                assert_frozen_inputs(args, inherited)
            path.write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                assert_frozen_inputs(args, inherited)

    def test_retry_rejects_changed_selection_harness_source_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _args, plan = self.prepared(root)
            previous = root / "previous.json"
            previous.write_text(json.dumps(plan))
            for kind in ("selection", "harness", "source", "seed"):
                changed = copy.deepcopy(plan)
                if kind == "selection":
                    changed["selections"]["train"]["episode_files"].reverse()
                elif kind == "harness":
                    next(row for row in changed["frozen_files"] if Path(row["path"]) == HARNESS)["sha256"] = "wrong"
                elif kind == "source":
                    changed["source_contract"]["train_payload_tree_sha256"] = "wrong"
                else:
                    command = changed["stages"][0]["command"]
                    command[command.index("--seed") + 1] = "1"
                with self.assertRaises(ValueError):
                    inherit_predeclared_selection(changed, previous)

    def test_ordinary_fills_use_default_spread_and_no_success_outcomes(self):
        rows = fixture_rows("train")
        selected = select_recovery_enriched(rows, "train")
        self.assertEqual([row["file"] for row in selected if row["task"]["name"] == "move"],
                         ["move_normal_0.npz", "move_normal_2.npz"])
        altered = copy.deepcopy(rows)
        for number, row in enumerate(altered):
            row.update(success=bool(number % 2), neural_success=bool((number + 1) % 2))
        self.assertEqual([row["file"] for row in selected],
                         [row["file"] for row in select_recovery_enriched(altered, "train")])

    def test_recovery_count_mismatch_or_duplicate_names_fail_closed(self):
        rows = fixture_rows("train")
        rows[0]["recovery_start_actions_not_training_labels"] = [[0, 0, 0, 0, 0, 0, 1]]
        with self.assertRaisesRegex(ValueError, "metadata counts"):
            select_recovery_enriched(rows, "train")
        duplicate = fixture_rows("train")
        duplicate.append(copy.deepcopy(duplicate[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_recovery_enriched(duplicate, "train")

    def test_virtualenv_python_symlink_is_preserved_in_both_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base-python"
            base.touch()
            executable = root / "venv/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(base)
            args, plan = self.prepared(root, ["--model-python", str(executable)])
            self.assertEqual(args.model_python, executable.absolute())
            self.assertNotEqual(args.model_python, executable.resolve())
            self.assertTrue(all(stage["command"][0] == str(executable) for stage in plan["stages"]))

    def test_frozen_commands_keep_train_normalization_and_exact_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.prepared(Path(directory))
            self.assertEqual(args.wait_timeout, 7200)
            for stage in plan["stages"]:
                command = stage["command"]
                self.assertEqual(command[command.index("--training-source") + 1], str(args.source))
                self.assertEqual(command[command.index("--training-audit-report") + 1], str(args.source_audit))
                self.assertEqual(command[command.index("--episode-files") + 1:], plan["selections"][stage["split"]]["episode_files"])
                self.assertEqual(command[command.index("--episodes-per-task") + 1], "2")
            self.assertEqual([stage["split"] for stage in plan["stages"]], ["train", "heldout"])

    def test_dry_plan_creates_no_files_or_processes_or_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _ = self.prepared(root)
            before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch(MODULE + ".parse_args", return_value=args), \
                    patch(MODULE + ".verify_source_inputs", return_value=CONTRACT), \
                    patch(MODULE + ".subprocess.run") as execute, patch(MODULE + ".wait_for_primary") as wait, \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            execute.assert_not_called()
            wait.assert_not_called()
            self.assertEqual(before, {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()})
            self.assertFalse(args.output.exists())

    def test_timeout_argument_is_finite_positive_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("0", "-1", "nan", "inf", "86401"):
                with self.assertRaises(ValueError):
                    self.arguments(Path(directory), ["--wait-timeout", value])

    def test_wait_handles_transient_json_but_never_sleeps_over_fifteen_seconds(self):
        timer = FakeTime()
        responses = iter([{"status": "running"}, json.JSONDecodeError("partial", "", 0),
                          {"status": "stage_completed"}, {"status": "completed_diagnostic_only"}])
        def reader():
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return result
        result = wait_for_primary("unused", 60, clock=timer.clock, sleep=timer.sleep, reader=reader)
        self.assertEqual(result["status"], "completed_diagnostic_only")
        self.assertEqual(timer.sleeps, [15, 15, 15])

    def test_primary_failure_and_unknown_status_do_not_retry(self):
        for primary, exception in (({"status": "failed", "error": "bad smoke"}, RuntimeError),
                                   ({"status": "completed"}, ValueError)):
            timer = FakeTime()
            with self.assertRaises(exception):
                wait_for_primary("unused", 60, clock=timer.clock, sleep=timer.sleep, reader=lambda: primary)
            self.assertEqual(timer.sleeps, [])

    def test_wait_timeout_is_exactly_bounded(self):
        timer = FakeTime()
        with self.assertRaisesRegex(TimeoutError, "within 2 seconds"):
            wait_for_primary("unused", 2, clock=timer.clock, sleep=timer.sleep, reader=lambda: {"status": "running"})
        self.assertEqual(timer.now, 2)
        self.assertEqual(timer.sleeps, [2])

    def test_run_freezes_names_before_wait_and_records_failure_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.prepared(Path(directory))
            def failed_wait(*_args):
                frozen = json.loads((args.output / "followup_plan.json").read_text())
                self.assertEqual(frozen["selections"], plan["selections"])
                self.assertEqual(json.loads((args.output / "followup_status.json").read_text())["status"], "waiting_primary")
                raise TimeoutError("primary wait timed out")
            with patch(MODULE + ".wait_for_primary", side_effect=failed_wait), patch(MODULE + ".subprocess.run") as execute:
                with self.assertRaises(TimeoutError):
                    run_followup(args, plan)
            execute.assert_not_called()
            self.assertEqual(json.loads((args.output / "followup_status.json").read_text())["status"], "failed")

    def test_completion_must_bind_exact_source_checkpoint_and_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.prepared(Path(directory))
            valid = self.primary_status(args, plan)
            validate_primary_completion(args, plan, valid)
            for field in ("source", "validation", "harness_sha256", "status"):
                changed = copy.deepcopy(valid)
                changed[field] = "/wrong"
                with self.assertRaises(ValueError):
                    validate_primary_completion(args, plan, changed)
            changed = copy.deepcopy(valid)
            changed["stages"][0]["command"][-1] = "/wrong-checkpoint"
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_primary_completion(args, plan, changed)

    def test_changed_predeclaration_during_wait_prevents_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.prepared(Path(directory))
            def completed_wait(*_args):
                (args.output / "followup_plan.json").write_text("changed declaration")
                return self.primary_status(args, plan)
            with patch(MODULE + ".wait_for_primary", side_effect=completed_wait), patch(MODULE + ".subprocess.run") as execute:
                with self.assertRaisesRegex(RuntimeError, "plan changed"):
                    run_followup(args, plan)
            execute.assert_not_called()
            failed = json.loads((args.output / "followup_status.json").read_text())
            self.assertEqual(failed["status"], "failed")

    def test_changed_source_proof_is_rejected_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.prepared(Path(directory))
            args.source_audit.write_text("changed proof")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                assert_frozen_inputs(args, plan)
            args.output.mkdir()
            marker = args.output / "user-result"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                run_followup(args, plan)
            self.assertEqual(marker.read_text(), "preserve")

    def test_subgroups_are_predeclared_and_same_layout_variants_not_inflated(self):
        selected = selection_metadata(select_recovery_enriched(fixture_rows("train"), "train"))
        results = [{"file": row["file"], "task": row["task"], "success": row["group"] == "recovery",
                    "safety_abort": False, "correct_target_grasp_seen": True, "wrong_target_grasp_seen": False}
                   for row in selected["episodes"]]
        summary = subgroup_summary({"results": results}, selected)
        self.assertEqual((summary["recovery"]["trials"], summary["recovery"]["physical_layouts"]), (6, 3))
        self.assertEqual(summary["ordinary"]["trials"], 2)
        with self.assertRaisesRegex(ValueError, "predeclared"):
            subgroup_summary({"results": results[:-1]}, selected)

    def test_only_after_primary_completion_are_two_harnesses_run_sequentially(self):
        with tempfile.TemporaryDirectory() as directory:
            args, plan = self.prepared(Path(directory))
            order = []
            def completed_wait(*_args):
                order.append("primary_done")
                return self.primary_status(args, plan)
            def runner(command, **_kwargs):
                split = "train" if command[command.index("--split-mode") + 1] == "train_diagnostic" else "heldout"
                self.assertIn("primary_done", order)
                order.append(split)
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                selected = plan["selections"][split]
                report = {"scope": "source_diagnostic_not_benchmark", "split_mode": command[command.index("--split-mode") + 1],
                          "source": command[command.index("--source") + 1], "training_normalization_source": str(args.source),
                          "checkpoint": str(args.checkpoint), "trials": 8, "privileged_information_in_controls": False,
                          "oracle_fallback": False, "final_test_used": False,
                          "results": [{"file": row["file"], "task": row["task"], "success": False} for row in selected["episodes"]]}
                (output / "results.json").write_text(json.dumps(report))
                return SimpleNamespace(returncode=0)
            with patch(MODULE + ".verify_source_inputs", return_value=CONTRACT), \
                    patch(MODULE + ".wait_for_primary", side_effect=completed_wait), \
                    patch("VLA.fingerprint_checkpoint.verify_fingerprint", return_value={"content_tree_sha256": "fixed-weights"}), \
                    patch(MODULE + ".subprocess.run", side_effect=runner):
                run_followup(args, plan)
            self.assertEqual(order, ["primary_done", "train", "heldout"])
            self.assertEqual(json.loads((args.output / "followup_status.json").read_text())["status"], "completed_diagnostic_only")
            self.assertEqual(json.loads((args.output / "results.json").read_text())["splits"]["train"]["subgroups"]["recovery"]["trials"], 6)


if __name__ == "__main__":
    unittest.main()
