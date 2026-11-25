import copy
import unittest

from VLA.run_privileged_priority_cycle import check_frozen_snapshot, merge_committed_rdt_prefix, ROOT, CROSS
from VLA.run_cross_vla120 import SEEDS, TASKS


class PrivilegedPriorityTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"model":"rdt", "checkpoint":"/exact/rdt5000", "adaptation_regime":"qualified_v9_postfit"}
        self.seed = SEEDS[1]
        self.state = {"status":"running", "pid":123}
        self.progress = {"status":"running", "pid":123, "current_model":"rdt", "current_seed":self.seed}
        self.path = ROOT / "VLA/rdt" / CROSS.name / f"test_seed{self.seed}"
        self.tokens = {456:[b"python",b"-m",b"VLA.rdt.server",b"/exact/rdt5000",b"5720"],
            789:[b"python",b"-m",b"VLA.cross_vla120_trial",str(self.path).encode(),b"rdt",str(self.seed).encode()]}

    def test_only_owned_server_and_inflight_block_are_adopted(self):
        info = check_frozen_snapshot(self.state,self.progress,[456,789],self.tokens,["456"],{"models":[self.spec]})
        self.assertEqual(info["trial_pid"],789)
        self.assertEqual(info["server_pid"],456)

    def test_no_training_child_can_be_stopped_by_handoff(self):
        self.tokens[789] = [b"python",b"-m",b"VLA.smolvla.train_simstate"]
        with self.assertRaises(ValueError):
            check_frozen_snapshot(self.state,self.progress,[456,789],self.tokens,["456"],{"models":[self.spec]})

    def test_unknown_GPU_owner_prevents_handoff(self):
        with self.assertRaises(ValueError):
            check_frozen_snapshot(self.state,self.progress,[456,789],self.tokens,["456","999"],{"models":[self.spec]})

    def test_changed_checkpoint_server_is_not_owned(self):
        self.tokens[456][3] = b"/other/checkpoint"
        with self.assertRaises(ValueError):
            check_frozen_snapshot(self.state,self.progress,[456,789],self.tokens,["456"],{"models":[self.spec]})

    def test_missing_finished_trial_can_be_adopted_from_receipt(self):
        info = check_frozen_snapshot(self.state,self.progress,[456],self.tokens,["456"],{"models":[self.spec]})
        self.assertIsNone(info["trial_pid"])

    def merge_fixture(self):
        rows = [{"seed":SEEDS[0],"task":t,"success":False,"error":None} for t in TASKS]
        progress = {"models":{"rdt":{"rows":rows}}}
        plan = {"handoff":{"rdt_spec":self.spec,"current_seed":self.seed,"trial_path":str(self.path)}}
        result = {"model":"rdt","seed":self.seed,"integration_error":None,
            "tasks":[{"name":t,"success":t=="pick","error":None} for t in TASKS]}
        return progress,plan,result

    def test_actual_failed_and_successful_receipts_retained_in_full_prefix(self):
        p,plan,result = self.merge_fixture()
        r = merge_committed_rdt_prefix(p,plan,result)
        self.assertEqual(r["completed_trials"],8)
        self.assertEqual(r["successes"],1)
        self.assertIsNone(r["success_rate_for_completed120"])
        self.assertEqual(len(p["models"]["rdt"]["rows"]),4)

    def test_no_duplicate_block_when_parent_committed_before_handoff(self):
        p,plan,result = self.merge_fixture()
        r = merge_committed_rdt_prefix(p,plan,result)
        p["models"]["rdt"]["rows"] = r["rows"]
        again = merge_committed_rdt_prefix(p,plan,result)
        self.assertEqual(again["completed_trials"],8)

    def test_partial_block_cannot_be_silently_replaced(self):
        p,plan,result = self.merge_fixture()
        p["models"]["rdt"]["rows"].append({"seed":self.seed,"task":"touch","success":False})
        with self.assertRaises(ValueError): merge_committed_rdt_prefix(p,plan,result)

    def test_integration_error_is_not_a_scored_failure(self):
        p,plan,result = self.merge_fixture()
        result["integration_error"] = "bad native action shape"
        with self.assertRaises(ValueError): merge_committed_rdt_prefix(p,plan,result)

    def test_duplicate_task_cannot_fill_a_missing_task(self):
        p,plan,result = self.merge_fixture()
        result["tasks"][-1] = copy.deepcopy(result["tasks"][0])
        with self.assertRaises(ValueError): merge_committed_rdt_prefix(p,plan,result)


if __name__ == "__main__": unittest.main()
