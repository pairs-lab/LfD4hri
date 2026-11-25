import copy
import unittest
from VLA.compose_verified_physics_replays import remap_rows


class ReplayCompositionTests(unittest.TestCase):
    def fixture(self):
        item = {"file": "a.npz", "task": {"name": "touch"}, "steps": 2, "success": True}
        merged = {**item, "file": "component0_a.npz", "merge_origin": {"source": "parent", "file": "a.npz", "sha256": "digest"}}
        trace = {"file": "a.npz", "ever_strict_success": True, "initial_strict_success": False, "stored_steps": 2}
        return [merged], {"parent": {"a.npz": item}}, {"parent": {"rows": [{**trace, "mode": m} for m in ("canonical", "native")]}}, {"parent": {"a.npz": {"sha256": "digest"}}}

    def test_only_names_remapped_actual_metrics_preserved(self):
        args = self.fixture(); result = remap_rows(*args)
        self.assertEqual([r["file"] for r in result], ["component0_a.npz"]*2)
        self.assertTrue(all(r["ever_strict_success"] for r in result))
        self.assertEqual(result[0]["reused_parent_file"], "a.npz")

    def test_filter_duplicate_changed_task_digest_or_missing_mode_rejected(self):
        for mutate in (lambda a: a[0].clear(), lambda a: a[0].append(copy.deepcopy(a[0][0])),
                       lambda a: a[0][0].update(task={"name": "pick"}),
                       lambda a: a[0][0]["merge_origin"].update(sha256="changed"),
                       lambda a: a[2]["parent"]["rows"].pop()):
            args = list(self.fixture()); mutate(args)
            with self.assertRaises(ValueError): remap_rows(*args)


if __name__ == "__main__":
    unittest.main()
