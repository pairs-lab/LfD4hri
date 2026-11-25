import unittest

from VLA.queue_smolvla_postfit10 import fit_finished


class Postfit10QueueTests(unittest.TestCase):
    def state(self, status):
        return {"status": status, "updates": 10000, "run_training_requested": True}

    def test_collection_and_image_review_do_not_start_pilot(self):
        for status in ("waiting_for_complete_collection", "awaiting_actual_visual_review", "running"):
            with self.subTest(status=status):
                self.assertFalse(fit_finished(self.state(status), 10000))

    def test_only_completed_workflow_can_start_pilot(self):
        self.assertTrue(fit_finished(self.state("completed_diagnostic_only"), 10000))

    def test_old_budget_or_audit_only_workflow_is_rejected(self):
        state = self.state("completed_diagnostic_only")
        state["updates"] = 5000
        with self.assertRaises(ValueError): fit_finished(state, 10000)
        state["updates"] = 10000
        state["run_training_requested"] = False
        with self.assertRaises(ValueError): fit_finished(state, 10000)

    def test_stopped_upstream_is_not_retried_or_substituted(self):
        with self.assertRaises(RuntimeError):
            fit_finished(self.state("stopped_no_silent_retry"), 10000)


if __name__ == "__main__":
    unittest.main()
