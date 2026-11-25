import copy
import unittest

from VLA.finish_native_touch_repair import failed_touch_files


class NativeTouchRepairContinuationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"file": "touch.npz", "task": {"name": "touch"}},
                     {"file": "pick.npz", "task": {"name": "pick"}}]
        self.report = {"all_episodes": True, "mode": "both", "native_success_mode": "strict",
                       "rows": [{"file": row["file"], "mode": mode,
                                 "success": not (mode == "native" and row["file"] == "touch.npz"),
                                 "initial_strict_success": False}
                                for row in self.rows for mode in ("canonical", "native")]}

    def test_only_all_exact_failed_touch_cases_selected(self):
        self.assertEqual(failed_touch_files(self.report, self.rows), ["touch.npz"])

    def test_incomplete_or_duplicate_evidence_is_rejected(self):
        for change in ("missing", "duplicate", "prefix", "semantic"):
            with self.subTest(change=change):
                report = copy.deepcopy(self.report)
                if change == "missing": report["rows"].pop()
                if change == "duplicate": report["rows"][0] = copy.deepcopy(report["rows"][1])
                if change == "prefix": report["all_episodes"] = False
                if change == "semantic": report["native_success_mode"] = "semantic"
                with self.assertRaises(ValueError): failed_touch_files(report, self.rows)

    def test_unsupported_failure_or_initial_false_positive_stops(self):
        for change in ("canonical", "pick", "initial", "paired", "no_failure"):
            with self.subTest(change=change):
                report, rows = copy.deepcopy(self.report), copy.deepcopy(self.rows)
                if change == "canonical": report["rows"][0]["success"] = False
                if change == "pick": report["rows"][3]["success"] = False
                if change == "initial": report["rows"][2]["initial_strict_success"] = True
                if change == "paired": rows[0]["paired_scene_source"] = "other.npz"
                if change == "no_failure": report["rows"][1]["success"] = True
                with self.assertRaises(ValueError): failed_touch_files(report, rows)


if __name__ == "__main__":
    unittest.main()
