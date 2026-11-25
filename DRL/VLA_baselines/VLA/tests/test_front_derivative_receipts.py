"""Transactional receipt tests; physics is mocked, indexed bytes are real.

These do not certify a camera, PyBullet renderer, or robot controller. They
exercise the immutable-file/provenance boundary around already-verified replay.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.front_derivative_worker import (
    bounded_process_replays, canonical_digest, coordinator_lock, derivative_metadata,
    produce_episode_receipt, receipt_path, replay_error_template,
    safe_destination, verify_episode_receipt,
)
from VLA.rerender_body_v9_front_candidate import generate_derivative


class FrontDerivativeReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root/"source", self.root/"output"
        self.source.mkdir(); self.output.mkdir()
        self.profile = json.loads((Path(__file__).resolve().parents[1]
            /"profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())
        self.identity = profile_hash(self.profile)
        self.rows = [{"file": f"ep{i}.npz", "steps": 1, "success": True,
                      "task": {"name": "touch", "seed": i},
                      "observation_profile_sha256": self.identity,
                      "success_contract": {"required_consecutive_steps": 10}}
                     for i in range(3)]
        self.manifest = {"robot": "ur5e", "image_size": 384, "seed": 42,
                         "required_success_steps": 10,
                         "observation_profile": self.profile,
                         "observation_profile_sha256": self.identity,
                         "environment_runtime_fingerprint":
                             environment_runtime_fingerprint(self.profile)}
        (self.source/"index.json").write_text(json.dumps(self.rows))
        (self.source/"dataset_manifest.json").write_text(json.dumps(self.manifest))
        for i, row in enumerate(self.rows):
            np.savez_compressed(self.source/row["file"], value=np.asarray([i]))
        self.binding = {"source": str(self.source),
            "source_index_sha256": sha256_file(self.source/"index.json"),
            "source_manifest_sha256": sha256_file(self.source/"dataset_manifest.json"),
            "source_payload_fingerprint": source_payload_fingerprint(self.source, self.rows),
            "environment_runtime_fingerprint": environment_runtime_fingerprint(self.profile),
            "observation_profile_sha256": self.identity, "image_size": 384,
            "required_success_steps": 10, "seed": 42, "canonical_reset_parents": {}}

    @staticmethod
    def unchanged_replay(env, item, arrays, maxima, reset_source=None):
        return arrays

    def produce(self, number=0, replay=None):
        return produce_episode_receipt(object(), self.rows[number], number,
            self.source, self.output, self.profile, self.binding,
            replay or self.unchanged_replay)

    def write_receipt(self, number, receipt, *, resign=False):
        receipt = copy.deepcopy(receipt)
        if resign:
            receipt.pop("receipt_sha256", None)
            receipt["receipt_sha256"] = canonical_digest(receipt)
        receipt_path(self.output, self.rows[number]["file"]).write_text(json.dumps(receipt))

    def run_coordinator(self, *, resume=False, worker=None):
        args = SimpleNamespace(output=self.output, resume=resume, workers=4)
        with patch("VLA.front_derivative_worker.bounded_process_replays", side_effect=worker):
            generate_derivative(args, self.manifest, self.rows, self.profile,
                                self.identity, self.binding)

    def test_successful_receipt_binds_output_and_exact_source_row(self):
        receipt = self.produce()
        self.assertEqual(verify_episode_receipt(self.output, self.rows[0], 0,
                                               self.binding, receipt), receipt)
        self.assertEqual(receipt["metadata"]["camera_derivative_output_sha256"],
                         sha256_file(self.output/"ep0.npz"))
        self.assertEqual(receipt["source_row_sha256"], canonical_digest(self.rows[0]))
        self.assertFalse(receipt_path(self.output, "ep0.npz").with_suffix(".json.tmp").exists())

    def test_returned_receipt_must_equal_exclusively_published_receipt(self):
        receipt = self.produce()
        returned = copy.deepcopy(receipt); returned["worker_pid"] += 1
        with self.assertRaisesRegex(RuntimeError, "Returned worker receipt differs"):
            verify_episode_receipt(self.output, self.rows[0], 0, self.binding, returned)

    def test_receipt_hash_and_semantic_provenance_are_both_checked(self):
        original = self.produce()
        corruptions = [({"worker_pid": -1}, False, "version/hash/source/binding"),
                       ({"version": True}, True, "version/hash/source/binding"),
                       ({"role": "not_replay"}, True, "version/hash/source/binding"),
                       ({"passed": False}, True, "version/hash/source/binding"),
                       ({"source_number": 1}, True, "version/hash/source/binding"),
                       ({"source_filename": "ep1.npz"}, True, "version/hash/source/binding"),
                       ({"source_row_sha256": "0"*64}, True, "version/hash/source/binding"),
                       ({"replay_verifier_sources_sha256": {}}, True, "verifier sources are stale"),
                       ({"output_bytes": original["output_bytes"] + 1}, True, "metadata/output payload hash"),
                       ({"exact_replay_errors": {**replay_error_template(),
                                                  "numeric_max_abs_error": .001}}, True,
                        "maxima exceed unchanged replay gates")]
        for changes, resign, message in corruptions:
            with self.subTest(changes=changes):
                self.write_receipt(0, {**original, **changes}, resign=resign)
                with self.assertRaisesRegex(RuntimeError, message):
                    verify_episode_receipt(self.output, self.rows[0], 0, self.binding)

    def test_stale_profile_runtime_source_row_and_reset_parent_bindings_fail(self):
        self.produce()
        bindings = [copy.deepcopy(self.binding) for _ in range(3)]
        bindings[0]["observation_profile_sha256"] = "stale_profile"
        bindings[1]["environment_runtime_fingerprint"]["sources_sha256"][
            "VLA/common/body_environment.py"] = "stale_runtime"
        bindings[2]["canonical_reset_parents"] = {"other_parent": "stale_parent"}
        for binding in bindings:
            with self.subTest(binding=binding):
                with self.assertRaisesRegex(RuntimeError, "version/hash/source/binding"):
                    verify_episode_receipt(self.output, self.rows[0], 0, binding)
        changed_row = copy.deepcopy(self.rows[0]); changed_row["task"]["seed"] = 999
        with self.assertRaisesRegex(RuntimeError, "version/hash/source/binding"):
            verify_episode_receipt(self.output, changed_row, 0, self.binding)

    def test_same_size_payload_corruption_is_not_hidden_by_unchanged_metadata(self):
        self.produce()
        payload = self.output/"ep0.npz"
        changed = bytearray(payload.read_bytes()); changed[-1] ^= 1
        payload.write_bytes(changed)
        with self.assertRaisesRegex(RuntimeError, "metadata/output payload hash"):
            verify_episode_receipt(self.output, self.rows[0], 0, self.binding)

    def test_source_metadata_and_payload_must_remain_unchanged_during_replay(self):
        original = (self.source/"index.json").read_bytes()
        def change_index(env, item, arrays, maxima, reset_source=None):
            (self.source/"index.json").write_bytes(original+b" ")
            return arrays
        with self.assertRaisesRegex(RuntimeError, "Worker source metadata changed"):
            self.produce(replay=change_index)
        self.assertFalse((self.output/"ep0.npz").exists())
        (self.source/"index.json").write_bytes(original)
        def change_payload(env, item, arrays, maxima, reset_source=None):
            path = self.source/item["file"]
            changed = bytearray(path.read_bytes()); changed[-1] ^= 1
            path.write_bytes(changed)
            return arrays
        with self.assertRaisesRegex(RuntimeError, "Worker source episode payload changed"):
            self.produce(replay=change_payload)
        self.assertFalse((self.output/"ep0.npz").exists())

    def test_worker_rejects_stale_runtime_or_canonical_parent_before_replay(self):
        original = copy.deepcopy(self.binding)
        self.binding["environment_runtime_fingerprint"]["sources_sha256"][
            "VLA/common/body_gripper.py"] = "stale"
        replay = unittest.mock.Mock(side_effect=self.unchanged_replay)
        with self.assertRaisesRegex(RuntimeError, "runtime fingerprint is stale"):
            self.produce(replay=replay)
        replay.assert_not_called()
        self.binding = original
        parent = self.root/"canonical_parent"; parent.mkdir()
        (parent/"index.json").write_text("[]")
        self.binding["canonical_reset_parents"] = {str(parent): "wrong_hash"}
        with self.assertRaisesRegex(RuntimeError, "Canonical reset parent changed"):
            self.produce(replay=replay)
        replay.assert_not_called()
        self.assertFalse((self.output/"ep0.npz").exists())

    def test_interruption_between_payload_and_receipt_preserves_unverified_payload(self):
        with patch("VLA.front_derivative_worker._write_exclusive_receipt",
                   side_effect=RuntimeError("receipt publication interrupted")):
            with self.assertRaisesRegex(RuntimeError, "receipt publication interrupted"):
                self.produce()
        path = self.output/"ep0.npz"
        original = path.read_bytes()
        self.assertFalse(receipt_path(self.output, "ep0.npz").exists())
        with self.assertRaisesRegex(FileExistsError, "refusing overwrite"):
            self.produce()
        self.assertEqual(path.read_bytes(), original)

    def test_existing_unverified_npz_or_partial_receipt_is_never_overwritten(self):
        for number in range(2):
            with self.subTest(kind=number):
                if number == 0:
                    artifact = self.output/self.rows[number]["file"]
                else:
                    receipt = receipt_path(self.output, self.rows[number]["file"])
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    artifact = receipt.with_name(receipt.name+".tmp")
                artifact.write_bytes(b"preserve interrupted contents")
                with self.assertRaisesRegex(FileExistsError, "preserved"):
                    self.produce(number)
                self.assertEqual(artifact.read_bytes(), b"preserve interrupted contents")

    def test_paired_reset_provenance_is_copied_and_bound_in_receipt_metadata(self):
        self.binding["paired_row_reset_provenance"] = {"ep0.npz": {
            "canonical_source": "/offline/canonical",
            "canonical_source_index_sha256": "a"*64,
            "paired_scene_source": "canonical_touch.npz",
            "paired_reset_strategy": "canonical_spawn_order_then_teacher_retarget"}}
        receipt = self.produce()
        provenance = self.binding["paired_row_reset_provenance"]["ep0.npz"]
        for key, value in provenance.items():
            self.assertEqual(receipt["metadata"][key], value)
        verify_episode_receipt(self.output, self.rows[0], 0, self.binding)
        receipt["metadata"]["paired_scene_source"] = "wrong.npz"
        self.write_receipt(0, receipt, resign=True)
        with self.assertRaisesRegex(RuntimeError, "metadata/output payload hash"):
            verify_episode_receipt(self.output, self.rows[0], 0, self.binding)

    def test_out_of_order_receipts_resume_into_ordered_prefix_without_rewriting(self):
        def interrupted(jobs, *args):
            self.assertEqual([number for number, _ in jobs], [0, 1, 2])
            yield 2, self.produce(2)
            yield 0, self.produce(0)
            raise RuntimeError("worker interruption")
        with self.assertRaisesRegex(RuntimeError, "worker interruption"):
            self.run_coordinator(worker=interrupted)
        self.assertEqual([row["file"] for row in json.loads(
            (self.output/"index.json").read_text())], ["ep0.npz"])
        preserved = {number: (self.output/self.rows[number]["file"]).read_bytes()
                     for number in (0, 2)}
        def resumed(jobs, *args):
            self.assertEqual([number for number, _ in jobs], [1])
            yield 1, self.produce(1)
        self.run_coordinator(resume=True, worker=resumed)
        self.assertEqual([row["file"] for row in json.loads(
            (self.output/"index.json").read_text())], ["ep0.npz", "ep1.npz", "ep2.npz"])
        for number, content in preserved.items():
            self.assertEqual((self.output/self.rows[number]["file"]).read_bytes(), content)
        state = json.loads((self.output/"collection_state.json").read_text())
        self.assertEqual(state["completed_episodes"], 3)
        self.assertEqual(state["status"], "completed_camera_only_candidate_audit_passed")

    def test_crash_after_index_publication_requires_receipt_for_new_suffix(self):
        receipts = [self.produce(number) for number in range(2)]
        (self.output/"index.json").write_text(json.dumps([r["metadata"] for r in receipts]))
        state = {"binding": self.binding, "completed_episodes": 1,
                 "exact_replay_errors": replay_error_template()}
        (self.output/"collection_state.json").write_text(json.dumps(state))
        def resumed(jobs, *args):
            self.assertEqual([number for number, _ in jobs], [2])
            yield 2, self.produce(2)
        self.run_coordinator(resume=True, worker=resumed)
        self.assertEqual(len(json.loads((self.output/"index.json").read_text())), 3)

    def test_legacy_committed_prefix_is_accepted_but_unverified_new_suffix_is_not(self):
        self.produce(0)
        receipt_path(self.output, "ep0.npz").unlink()
        metadata = derivative_metadata(self.rows[0], self.binding,
                                       sha256_file(self.output/"ep0.npz"))
        (self.output/"index.json").write_text(json.dumps([metadata]))
        state = {"binding": self.binding, "completed_episodes": 1,
                 "exact_replay_errors": replay_error_template()}
        (self.output/"collection_state.json").write_text(json.dumps(state))
        partial = self.output/"ep1.npz"; partial.write_bytes(b"unverified partial")
        with self.assertRaisesRegex(RuntimeError, "Successful replay receipt is missing"):
            self.run_coordinator(resume=True)
        self.assertEqual(partial.read_bytes(), b"unverified partial")
        self.assertEqual(json.loads((self.output/"index.json").read_text()), [metadata])

    def test_committed_index_suffix_after_state_crash_needs_successful_receipt(self):
        receipts = [self.produce(number) for number in range(2)]
        receipt_path(self.output, "ep1.npz").unlink()
        index = [r["metadata"] for r in receipts]
        (self.output/"index.json").write_text(json.dumps(index))
        state = {"binding": self.binding, "completed_episodes": 1,
                 "exact_replay_errors": replay_error_template()}
        (self.output/"collection_state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(RuntimeError, "Successful replay receipt is missing"):
            self.run_coordinator(resume=True)
        self.assertEqual(json.loads((self.output/"index.json").read_text()), index)

    def test_resume_rejects_nonprefix_mixed_metadata_and_stale_binding(self):
        receipt = self.produce(0)
        valid_state = {"binding": self.binding, "completed_episodes": 1,
                       "exact_replay_errors": replay_error_template()}
        original = receipt["metadata"]
        corruptions = [(dict(original, file="ep1.npz"), "not a committed source prefix"),
                       (dict(original, success=False), "mixed derivative")]
        for metadata, message in corruptions:
            with self.subTest(message=message):
                (self.output/"index.json").write_text(json.dumps([metadata]))
                (self.output/"collection_state.json").write_text(json.dumps(valid_state))
                with self.assertRaisesRegex(RuntimeError, message):
                    self.run_coordinator(resume=True)
        stale_state = copy.deepcopy(valid_state)
        stale_state["binding"]["observation_profile_sha256"] = "stale_profile"
        (self.output/"index.json").write_text(json.dumps([original]))
        (self.output/"collection_state.json").write_text(json.dumps(stale_state))
        with self.assertRaisesRegex(RuntimeError, "source/profile/runtime/payload identity differs"):
            self.run_coordinator(resume=True)

    def test_paths_and_coordinator_lock_prevent_aliasing_or_concurrent_ownership(self):
        for filename in ("../ep.npz", "/ep.npz", "nested/../ep.npz", "./ep.npz",
                         "nested//ep.npz", "ep.txt", "nested\\ep.npz"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                safe_destination(self.output, filename)
        (self.output/"escape").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "outside output"):
            safe_destination(self.output, "escape/ep0.npz")
        (self.output/"alias.npz").symlink_to(self.source/"ep0.npz")
        with self.assertRaisesRegex(RuntimeError, "is a symlink"):
            safe_destination(self.output, "alias.npz")
        with coordinator_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another derivative coordinator"):
                with coordinator_lock(self.output):
                    self.fail("Concurrent lock acquisition must fail")
        # Kernel ownership is released on context exit; no PID-file recovery.
        with coordinator_lock(self.output):
            pass

    def test_scheduler_bounds_inflight_jobs_and_shuts_down_on_wrong_episode_number(self):
        class Future:
            def __init__(self, executor, job):
                self.executor, self.job = executor, job
            def result(self):
                self.executor.inflight -= 1
                return self.job[0] + self.executor.number_offset, {"fake": self.job[0]}
        class Executor:
            number_offset = 0
            def __init__(self, **kwargs):
                self.options = kwargs
                self.inflight = self.maximum_inflight = 0
                self.submitted = []
                self.closed = None
            def submit(self, function, job):
                self.inflight += 1
                self.maximum_inflight = max(self.maximum_inflight, self.inflight)
                self.submitted.append(job[0])
                return Future(self, job)
            def shutdown(self, **kwargs):
                self.closed = kwargs
        executor = Executor()
        def finish_one(pending, return_when):
            first = next(iter(pending))
            return {first}, set(pending)-{first}
        with patch("VLA.front_derivative_worker.ProcessPoolExecutor", return_value=executor), \
             patch("VLA.front_derivative_worker.wait", side_effect=finish_one):
            results = list(bounded_process_replays(list(enumerate(self.rows)), self.source,
                self.output, self.manifest, self.profile, self.binding, workers=2))
        self.assertEqual([number for number, _ in results], [0, 1, 2])
        self.assertEqual(executor.maximum_inflight, 2)
        self.assertEqual(executor.submitted, [0, 1, 2])
        self.assertEqual(executor.closed, {"wait": True, "cancel_futures": True})
        executor = Executor(); executor.number_offset = 1
        with patch("VLA.front_derivative_worker.ProcessPoolExecutor", return_value=executor), \
             patch("VLA.front_derivative_worker.wait", side_effect=finish_one):
            with self.assertRaisesRegex(RuntimeError, "another episode number"):
                list(bounded_process_replays(list(enumerate(self.rows)), self.source,
                    self.output, self.manifest, self.profile, self.binding, workers=2))
        self.assertEqual(executor.closed, {"wait": True, "cancel_futures": True})


if __name__ == "__main__":
    unittest.main()
