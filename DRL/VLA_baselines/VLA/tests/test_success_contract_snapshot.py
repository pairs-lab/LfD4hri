import unittest
from VLA.common.success_contract_snapshot import recorded_distance_thresholds, resolve_audit_thresholds


class HistoricalSuccessContractTest(unittest.TestCase):
    def row(self, touch=.015, goal=.02, streak=10):
        return {"success_contract": {"touch_distance_threshold_m": touch,
                                    "object_goal_threshold_m": goal, "required_consecutive_steps": streak}}

    def test_equal_live_is_default(self):
        self.assertEqual(resolve_audit_thresholds(self.row(), 10, (.015, .02)), (.015, .02))

    def test_live_mismatch_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "live reward"):
            resolve_audit_thresholds(self.row(), 10, (.025, .03))

    def test_explicit_stricter_snapshot_preserves_original_labels(self):
        item = self.row()
        self.assertEqual(resolve_audit_thresholds(item, 10, (.025, .03), True), (.015, .02))
        self.assertEqual(item, self.row())

    def test_looser_historical_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "looser"):
            resolve_audit_thresholds(self.row(), 10, (.01, .02), True)

    def test_wrong_streak_rejected(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            resolve_audit_thresholds(self.row(), 1, (.025, .03), True)

    def test_invalid_recorded_thresholds_rejected(self):
        for value in (None, True, float("nan"), float("inf"), 0, -.1, ".015"):
            with self.assertRaises(ValueError):
                recorded_distance_thresholds(self.row(touch=value))


if __name__ == "__main__":
    unittest.main()
