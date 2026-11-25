import unittest
from VLA.smolvla.training import schedule_contract


class SmolVLAScheduleTest(unittest.TestCase):
    def test_short_repair_has_training_after_warmup(self):
        value = schedule_contract(1000, 8)
        self.assertEqual(value["warmup_optimizer_updates"], 100)
        self.assertEqual(value["warmup_microsteps"], 800)

    def test_long_run_keeps_original_preset(self):
        value = schedule_contract(30000, 4)
        self.assertEqual(value["warmup_microsteps"], 4000)
        self.assertEqual(value["decay_microsteps"], 120000)

    def test_explicit_update_units(self):
        value = schedule_contract(1000, 8, 50, 1000)
        self.assertEqual(value["warmup_microsteps"], 400)
        self.assertEqual(value["decay_microsteps"], 8000)

    def test_smoke_and_invalid_entire_run_warmup(self):
        self.assertEqual(schedule_contract(2, 8)["warmup_microsteps"], 0)
        with self.assertRaises(ValueError):
            schedule_contract(1000, 8, 1000)


if __name__ == "__main__":
    unittest.main()
