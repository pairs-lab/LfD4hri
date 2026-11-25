import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from VLA.evaluate_smolvla_v9_source import select_episodes, safety_abort_info
from VLA.run_smolvla_v9_pilot import build_plan, checkpoint_path, main, parse_args


class SmolVLAV9PilotTest(unittest.TestCase):
    def test_separate_anti_memorization_recipe_reaches_both_smoke_and_main(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), ["--initial-frame-sampling-probability", "0.1",
                "--learning-rate", "0.00005", "--batch-size", "2"])
            stages = {stage.name: stage for stage in build_plan(args)}
            for name in ("smoke", "train"):
                cmd = stages[name].command
                for flag, expected in (("--initial-frame-sampling-probability", "0.1"),
                                       ("--learning-rate", "5e-05"), ("--batch-size", "2")):
                    self.assertEqual(cmd[cmd.index(flag)+1], expected)
            with self.assertRaises(ValueError):
                self.args(Path(directory), ["--initial-frame-sampling-probability", "1"])

    def args(self, root, extra=()):
        return parse_args(["--source", str(root / "train_source"), "--validation", str(root / "val_source"),
                           "--dataset-root", str(root / "converted"), "--audit-root", str(root / "audits"),
                           "--output", str(root / "trial"), *extra])

    def test_defaults_and_exact_microstep_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            self.assertEqual((args.updates, args.action_horizon, args.accumulation), (500, 5, 16))
            self.assertEqual(checkpoint_path(args.output / "train", 500, 16).parent.name, "008000")
            self.assertEqual(checkpoint_path(args.output / "smoke", 2, 8).parent.name, "000016")

    def test_train_is_fresh_base_and_validation_never_optimizes_or_normalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            stages = {stage.name: stage for stage in build_plan(args)}
            for key in ("smoke", "train"):
                command = stages[key].command
                self.assertEqual(command[command.index("--base-model") + 1], "lerobot/smolvla_base")
                self.assertIn("--train-vlm", command)
                self.assertIn("--train-visual-connector", command)
                self.assertIn("--target-balanced-sampling", command)
                self.assertNotIn(str(args.validation), command)
            self.assertEqual(stages["smoke"].command[stages["smoke"].command.index("--steps") + 1], "2")
            self.assertEqual(stages["smoke"].command[stages["smoke"].command.index("--warmup-updates") + 1], "0")
            heldout = stages["heldout_imitation"].command
            self.assertEqual(heldout[heldout.index("--normalization-source") + 1], str(args.source))
            for name in ("heldout_closedloop", "train_closedloop"):
                command = stages[name].command
                self.assertEqual(command[command.index("--training-source") + 1], str(args.source))
                self.assertNotIn("VLA.evaluate_tiny_fit", command)

    def test_prepared_mode_does_not_rewrite_converted_data(self):
        with tempfile.TemporaryDirectory() as directory:
            names = {stage.name for stage in build_plan(self.args(Path(directory), ["--prepared"]))}
            self.assertFalse({"convert", "exact_stats", "roundtrip", "preprocess"} & names)
            self.assertIn("validate", names)

    def test_actual_simulation_interpreter_checks_resets_before_cuda_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory), ["--sim-python", "/tmp/deployment-python"])
            stages = build_plan(args)
            names = [stage.name for stage in stages]
            for split in ("train", "heldout"):
                name = split + "_reset_sensor_parity"
                stage = stages[names.index(name)]
                self.assertEqual(stage.command[0], str(args.sim_python))
                self.assertFalse(stage.gpu)
                self.assertTrue(stage.gate_report.endswith(name + ".json"))
                self.assertLess(names.index(name), names.index("cuda_preflight"))
                self.assertLess(names.index(name), names.index("smoke"))

    def test_virtualenv_executable_symlinks_are_not_resolved_to_base_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "base-python"
            executable.touch()
            virtualenv_python = root / "venv/bin/python"
            virtualenv_python.parent.mkdir(parents=True)
            virtualenv_python.symlink_to(executable)
            args = self.args(root, ["--model-python", str(virtualenv_python)])
            self.assertEqual(args.model_python, virtualenv_python.absolute())
            self.assertEqual(args.sim_python, virtualenv_python.absolute())
            self.assertNotEqual(args.model_python, virtualenv_python.resolve())
            self.assertTrue(all(stage.command[0] == str(virtualenv_python) for stage in build_plan(args)))

    def test_fresh_run_can_reuse_separate_immutable_cpu_proof_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep = root / "earlier-audit/pilot_prep"
            args = self.args(root, ["--prepared", "--prep-root", str(prep)])
            self.assertEqual(args.prep_root, prep)
            stages = {stage.name: stage for stage in build_plan(args)}
            self.assertIn(str(prep / "stats.json"), stages["validate"].command)
            self.assertNotIn(str(prep), stages["train_reset_sensor_parity"].command)

    def test_dry_plan_has_no_filesystem_or_execution_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            with patch("VLA.run_smolvla_v9_pilot.parse_args", return_value=args), \
                    patch("VLA.run_smolvla_v9_pilot.subprocess.run") as execute, contextlib.redirect_stdout(io.StringIO()):
                main()
            execute.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_dry_plan_refuses_beyond_existing_qualification_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, ["--updates", "501"])
            args.source.mkdir()
            (args.source / "dataset_manifest.json").write_text(json.dumps({"training_qualification": {"max_optimizer_updates": 500}}))
            with patch("VLA.run_smolvla_v9_pilot.parse_args", return_value=args), self.assertRaisesRegex(ValueError, "qualification"):
                main()

    def test_subset_selection_is_bounded_reproducible_and_layout_distinct(self):
        rows = [{"file": f"{task}_{scene}.npz", "task": {"name": task, "pick_name": "red apple", "pick_position": [scene, 0, 0],
                 "place_name": "blue bowl", "place_position": [0, 4, 0], "scene_objects": []}}
                for task in ("touch", "pick", "move", "pick_place") for scene in range(4)]
        selected = select_episodes(rows)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected, select_episodes(rows))
        for task in ("touch", "pick", "move", "pick_place"):
            self.assertEqual([row["task"]["pick_position"][0] for row in selected if row["task"]["name"] == task], [0, 3])
        with self.assertRaises(ValueError):
            select_episodes(rows, filenames=[selected[0]["file"]] * 8)

    def test_actual_v9_safety_violation_cannot_be_hidden(self):
        self.assertEqual(safety_abort_info({"safety_violation": "mimic joint mismatch"}), (True, "mimic joint mismatch"))
        self.assertEqual(safety_abort_info({"safety_abort": True, "safety_abort_reason": "floor"}), (True, "floor"))
        self.assertEqual(safety_abort_info({"success": True}), (False, None))

    def test_refinement_budget_is_explicit_and_keeps_intermediate_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                self.args(root, ["--updates", "5000"])
            args = self.args(root, ["--updates", "5000", "--experiment-scope", "bounded_refinement_experiment"])
            stages = {stage.name: stage for stage in build_plan(args)}
            train = stages["train"].command
            self.assertEqual(train[train.index("--save-freq")+1], "1000")
            self.assertEqual(train[train.index("--warmup-updates")+1], "500")
            self.assertEqual(train[train.index("--decay-updates")+1], "5000")
            self.assertEqual(checkpoint_path(args.output/"train", 5000, 16).parent.name, "080000")


if __name__ == "__main__":
    unittest.main()
