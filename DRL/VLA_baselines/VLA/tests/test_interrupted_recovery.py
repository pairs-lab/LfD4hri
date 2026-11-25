import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from VLA.common.provenance import sha256_file
from VLA.monitor_recovered_runs import owner_alive
from VLA.monitor_repair_cycles import model_snapshot
from VLA.resume_simstate_projection import reuse_prefix
from VLA.rdt.continue_interrupted_training import verify_saved_checkpoint


def prefix_fixture(root):
    source, previous, output = (root/name for name in ("source","previous","output"))
    source.mkdir();(previous/"states").mkdir(parents=True);(output/"states").mkdir(parents=True)
    file="touch0.npz"
    np.savez_compressed(source/file, canonical_actions=np.zeros((2,7),dtype=np.float32))
    np.savez_compressed(previous/"states"/file,state=np.zeros((2,18),dtype=np.float32))
    row={"file":file,"steps":2}
    proof={"file":file,"frames":2,"terminal_strict_success":True,
        "source_payload_sha256":sha256_file(source/file),"state_payload_sha256":sha256_file(previous/"states"/file),
        "sensor_maxima":{"joint_max_abs_error":0.,"measured_closure_abs_error":0.,"front_rgb_max_abs_error":0,
            "front_rgb_changed_pixel_fraction":0.,"wrist_rgb_max_abs_error":0,"wrist_rgb_changed_pixel_fraction":0.}}
    plan={"schema":"test","source":str(source),"source_index_sha256":"index","source_manifest_sha256":"manifest",
        "feature_builder_sha256":"features","source_audit_sha256":"audit","environment_runtime_fingerprint":{},
        "selected_files":[file],"split":"train"}
    (previous/"plan.json").write_text(json.dumps(plan))
    progress={"status":"extracting_causal_states_from_real_physics","rows":[proof],"frames":2}
    (previous/"progress.json").write_text(json.dumps(progress))
    return source,previous,output,[row],plan,progress


class RecoveryTests(unittest.TestCase):
    def test_dead_owner_is_not_reported_running_or_zero_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"status.json").write_text(json.dumps({"pid":2147483647,"status":"running","stage":"train","updates":5000}))
            value=model_snapshot(root,"smolvla")
            self.assertFalse(value["owner_alive"])
            self.assertEqual(value["status"],"interrupted_owner_missing")
            self.assertIsNone(value["completed_new_development_success_rate"])

    def test_live_pid_wrong_module_is_not_an_owned_trainer(self):
        import os
        self.assertFalse(owner_alive(os.getpid(),"VLA.nonexistent_owned_trainer"))
        self.assertFalse(owner_alive(2147483647,"VLA.recover_interrupted_cycles"))

    def test_sha_bound_prefix_reused_without_overwriting_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source,previous,output,rows,plan,_=prefix_fixture(Path(directory))
            original=sha256_file(previous/"states"/rows[0]["file"])
            receipts,report=reuse_prefix(previous,source,output,rows,plan)
            self.assertEqual(len(receipts),1)
            self.assertEqual(report["reused_episodes"],1)
            self.assertEqual(sha256_file(output/"states"/rows[0]["file"]),original)
            self.assertEqual(sha256_file(previous/"states"/rows[0]["file"]),original)

    def test_failed_prefix_cannot_be_silently_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            source,previous,output,rows,plan,progress=prefix_fixture(Path(directory))
            progress["status"]="failed_projection_NO_training"
            (previous/"progress.json").write_text(json.dumps(progress))
            with self.assertRaises(ValueError): reuse_prefix(previous,source,output,rows,plan)

    def test_changed_payload_or_frame_order_prevents_prefix_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            source,previous,output,rows,plan,_=prefix_fixture(Path(directory))
            np.savez_compressed(source/rows[0]["file"],canonical_actions=np.ones((2,7),dtype=np.float32))
            with self.assertRaises(ValueError): reuse_prefix(previous,source,output,rows,plan)

    def test_different_causal_feature_builder_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source,previous,output,rows,plan,_=prefix_fixture(Path(directory))
            changed={**plan,"feature_builder_sha256":"other"}
            with self.assertRaises(ValueError): reuse_prefix(previous,source,output,rows,changed)

    def test_saved_optimizer_count_not_last_log_count_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ("config.json","pybullet_rdt_config.json"):
                (root/name).write_text("{}")
            torch.save({"weight":torch.ones(1)},root/"pytorch_model.bin")
            state={"step":1000,"epoch":1,"optimizer":{"state":{0:{"step":torch.tensor(1000.),
                "exp_avg":torch.zeros(1),"exp_avg_sq":torch.zeros(1)}}}}
            torch.save(state,root/"training_state.pt")
            manifest={"step":1000,"steps":5000}
            for name,key in (("config.json","model_config_sha256"),("pytorch_model.bin","model_weights_sha256"),
                    ("training_state.pt","training_state_sha256"),("pybullet_rdt_config.json","target_contract_sha256")):
                manifest[key]=sha256_file(root/name)
            (root/"vla_checkpoint_manifest.json").write_text(json.dumps(manifest))
            saved,actual=verify_saved_checkpoint(root)
            self.assertEqual(actual["step"],1000)
            self.assertEqual(saved["steps"]-actual["step"],4000)
            (root/"training_state.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError): verify_saved_checkpoint(root)


if __name__=="__main__":
    unittest.main()
