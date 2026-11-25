from pathlib import Path
import json
import tempfile
import unittest

from VLA.run_rdt_v9_refinement import build_stages, parse_args


class RDTDriverTests(unittest.TestCase):
    def test_source_counts_are_read_not_hard_coded_to_the_old556_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = {"touch": 527, "pick": 527, "move": 527, "pick_place": 526}
            (root / "index.json").write_text(json.dumps([
                {"task": {"name": task}} for task, count in counts.items() for _ in range(count)]))
            args = self.args()
            args.source = root
            stages = {row["name"]: row for row in build_stages(args)}
            command = stages["source_audit"]["command"]
            self.assertEqual(json.loads(command[command.index("--expected-task-counts") + 1]), counts)
            self.assertEqual(command[command.index("--recovery-reset-observability") + 1], "dual_view")
            (root / "index.json").write_text('[{"task":{"name":"pick"}}]')
            with self.assertRaises(ValueError): build_stages(args)

    def args(self, extra=()):
        return parse_args(["--source", "/tmp/rdt-train", "--validation", "/tmp/rdt-validation", "--validation-audit",
            "/tmp/rdt-validation-audit.json", "--cache", "/tmp/cache.pt", "--audit-root", "/tmp/rdt-audit",
            "--output", "/tmp/rdt-run", *extra])

    def test_fresh_base_train_and_separate_two_update_smoke(self):
        stages = {stage["name"]: stage for stage in build_stages(self.args())}
        for name, updates in (("smoke", "2"), ("train", "5000")):
            command = stages[name]["command"]
            self.assertEqual(command[command.index("--steps") + 1], updates)
            self.assertTrue(command[command.index("--base-checkpoint") + 1].endswith("checkpoints/rdt-170m"))
            self.assertNotIn("/tmp/rdt-validation", command)
            self.assertEqual(command[command.index("--optimizer") + 1], "adamw")
            self.assertNotIn("--max-episodes", command)
        self.assertNotEqual(stages["train"]["command"][-1], stages["smoke"]["command"][-1])

    def test_simulator_parity_uses_actual_deployment_binary_before_gpu(self):
        stages = build_stages(self.args())
        names = [row["name"] for row in stages]
        for name in ("train_deployment_reset_parity", "heldout_deployment_reset_parity"):
            command = stages[names.index(name)]["command"]
            self.assertIn("third_party/lerobot/.venv/bin/python", command[0])
            self.assertLess(names.index(name), names.index("gpu_headroom"))
        self.assertLess(names.index("smoke_master_adam"), names.index("train"))
        self.assertLess(names.index("smoke_reload_precision"), names.index("train"))

    def test_budget_cannot_expand(self):
        for value in ("0", "5001", "-1"):
            with self.assertRaises(ValueError):
                self.args(["--updates", value])

    def test_explicit_fp32_signal_recipe_reaches_smoke_and_fit(self):
        profile = "fp32_state_action_readout_latent_v1"
        args = self.args(["--signal-precision", profile, "--development40",
                          "--wait-for-cycle-status", "/tmp/smolvla-cycle-status.json"])
        self.assertTrue(args.development40)
        stages = {row["name"]: row for row in build_stages(args)}
        for name in ("input_smoke", "smoke", "train"):
            command = stages[name]["command"]
            self.assertEqual(command[command.index("--signal-precision") + 1], profile)
        self.assertEqual(self.args().signal_precision, "bf16")

    def test_default_recipe_keeps_native_rdt_history_and_short_replan_contract(self):
        stages = {stage["name"]: stage for stage in build_stages(self.args())}
        command = stages["train"]["command"]
        self.assertEqual(command[command.index("--action-horizon") + 1], "10")
        self.assertEqual(command[command.index("--samples-per-episode") + 1], "4")
        self.assertEqual(command[command.index("--gradient-accumulation") + 1], "4")


if __name__ == "__main__":
    unittest.main()
