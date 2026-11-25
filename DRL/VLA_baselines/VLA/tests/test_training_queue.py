import unittest

from VLA.train_v6_queue import jobs


class HeavyModelQueueTest(unittest.TestCase):
    def test_all_heavy_smoke_checks_precede_first_long_training(self):
        queue = list(jobs())
        names = [name for name, command in queue]
        self.assertEqual(len(names), len(set(names)))
        first_training = min(i for i, name in enumerate(names) if name.endswith("_train"))
        for name in ("pi0_smoke", "pi0_fast_smoke", "openvla_smoke"):
            self.assertLess(names.index(name), first_training)
        for name, command in queue:
            if name.endswith("_evaluate"):
                self.assertIn("--development-only", command)
                self.assertNotIn("--final-only", command)


if __name__ == "__main__":
    unittest.main()
