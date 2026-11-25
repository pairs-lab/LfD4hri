import unittest
from VLA.replace_transport_demonstrations import validated_replacements


def proofs():
    visibility = {"all_episodes": True, "rows": [{"file": "bad.npz", "failures": ["occluded"]},
        {"file": "good.npz", "failures": []}]}
    repair = {"passed": True, "runtime_changed": False, "cameras_changed": False,
        "training_performed": False, "rows": [{"file": "bad.npz", "passed": True, "reset_errors": {"rgb": 0}}]}
    replay = {"passed": True, "mode": "both", "native_success_mode": "strict", "failures": [],
        "summaries": {mode: {"episodes": 1, "successes": 1, "initial_false_positives": 0}
            for mode in ("canonical", "native")}}
    return visibility, repair, replay


class TransportReplacementTest(unittest.TestCase):
    def test_all_and_only_camera_failures_replaced(self):
        self.assertEqual(validated_replacements(*proofs()), {"bad.npz"})

    def test_partial_collection_cannot_be_promoted(self):
        values = proofs()
        values[0]["all_episodes"] = False
        with self.assertRaises(ValueError): validated_replacements(*values)

    def test_extra_unrepaired_failure_stops(self):
        values = proofs()
        values[0]["rows"].append({"file": "new_failure.npz", "failures": ["occluded"]})
        with self.assertRaises(ValueError): validated_replacements(*values)

    def test_native_only_or_false_positive_rejected(self):
        values = proofs()
        values[2]["summaries"]["canonical"]["initial_false_positives"] = 1
        with self.assertRaises(ValueError): validated_replacements(*values)


if __name__ == "__main__":
    unittest.main()
