"""Transactional, process-isolated workers for offline FRONT-only replay.

Workers never regenerate labels or commit the derivative index. A complete NPZ
and hashed, provenance-bound successful receipt must precede coordinator commit.
Unverified files are preserved and never silently replaced on resume.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import copy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint


RECEIPT_VERSION = 1
RECEIPT_ROLE = "front_camera_only_exact_episode_replay"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_worker_context = None


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def replay_verifier_fingerprint(binding):
    paths = ["VLA/rerender_body_v9_front_candidate.py", "VLA/front_derivative_worker.py",
             "VLA/dataset_replay_context.py", "VLA/common/success_contract_snapshot.py"]
    if binding.get("canonical_reset_parents"):
        paths += ["VLA/audit_training_readiness.py", "VLA/audit_counterfactual_collection.py"]
    return {name: sha256_file(PROJECT_ROOT/name) for name in paths}


def safe_destination(output, filename):
    root = Path(output).resolve()
    relative = PurePosixPath(filename)
    if (not filename or relative.is_absolute() or ".." in relative.parts or "\\" in filename
            or relative.as_posix() != filename or relative.suffix != ".npz"):
        raise ValueError("Derivative filename must be a canonical relative NPZ path")
    destination = root/filename
    if not destination.parent.resolve().is_relative_to(root) or destination.is_symlink():
        raise RuntimeError("Derivative output path resolves outside output or is a symlink")
    return destination


def receipt_path(output, filename):
    safe_destination(output, filename)
    directory = Path(output)/".front_replay_receipts"
    if directory.is_symlink() or not directory.resolve().is_relative_to(Path(output).resolve()):
        raise RuntimeError("Receipt directory must remain inside the derivative output")
    return directory/(hashlib.sha256(filename.encode()).hexdigest()+".json")


@contextmanager
def coordinator_lock(output):
    """Kernel-released lock survives interruption without stale-PID guessing."""
    path = Path(output)/".front_replay_coordinator.lock"
    if path.is_symlink():
        raise RuntimeError("Coordinator lock path cannot be a symlink")
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another derivative coordinator owns this output") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def replay_error_template():
    return {"numeric_max_abs_error": 0., "wrist_max_channel_lsb": 0,
            "wrist_max_changed_pixel_fraction": 0.}


def validate_replay_errors(errors):
    limits = {"numeric_max_abs_error": 1e-6, "wrist_max_channel_lsb": 1,
              "wrist_max_changed_pixel_fraction": .001}
    if not isinstance(errors, dict) or set(errors) != set(limits):
        raise RuntimeError("Replay receipt/state error maxima are malformed")
    for key, maximum in limits.items():
        value = errors[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= maximum:
            raise RuntimeError("Replay receipt/state maxima exceed unchanged replay gates: "+key)
    return errors


def merge_replay_errors(maxima, errors):
    validate_replay_errors(errors)
    for key in maxima:
        maxima[key] = max(maxima[key], errors[key])


def derivative_metadata(item, binding, output_digest):
    metadata = copy.deepcopy(item)
    metadata.update(binding.get("paired_row_reset_provenance", {}).get(item["file"], {}))
    metadata.update(observation_profile_sha256=binding["observation_profile_sha256"],
        camera_only_derivative_of=str(Path(binding["source"])/item["file"]),
        camera_derivative_source_sha256=binding["source_payload_fingerprint"]["payloads"][item["file"]]["sha256"],
        camera_derivative_output_sha256=output_digest)
    return metadata


def _verify_source_context(binding, profile, filename):
    source = Path(binding["source"])
    for name, key in (("index.json", "source_index_sha256"),
                      ("dataset_manifest.json", "source_manifest_sha256")):
        if sha256_file(source/name) != binding[key]:
            raise RuntimeError("Worker source metadata changed: "+name)
    if sha256_file(source/filename) != binding["source_payload_fingerprint"]["payloads"][filename]["sha256"]:
        raise RuntimeError("Worker source episode payload changed: "+filename)
    for parent, expected in binding["canonical_reset_parents"].items():
        if sha256_file(Path(parent)/"index.json") != expected:
            raise RuntimeError("Canonical reset parent changed during worker replay")
    verify_environment_runtime_fingerprint(binding["environment_runtime_fingerprint"], profile)


def _write_exclusive_receipt(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    # Both paths are private new artifacts. Any stale partial is preserved.
    if path.exists() or path.is_symlink():
        raise FileExistsError("Existing replay receipt preserved: "+str(path))
    with temporary.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # exclusive publication, never replace another receipt
    temporary.unlink()  # final link preserves the complete generated receipt


def produce_episode_receipt(env, item, number, source, output, profile, binding, replay):
    destination = safe_destination(output, item["file"])
    receipt_file = receipt_path(output, item["file"])
    if destination.exists() or destination.is_symlink() or receipt_file.exists() or receipt_file.with_name(receipt_file.name+".tmp").exists():
        raise FileExistsError("Uncommitted output artifact preserved; refusing overwrite: "+str(destination))
    verifier = replay_verifier_fingerprint(binding)
    _verify_source_context(binding, profile, item["file"])
    errors = replay_error_template()
    with np.load(Path(source)/item["file"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    result = replay(env, item, arrays, errors, source)
    validate_replay_errors(errors)
    _verify_source_context(binding, profile, item["file"])
    if verifier != replay_verifier_fingerprint(binding):
        raise RuntimeError("Offline replay verifier sources changed during the episode")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation leaves a partial NPZ intact if writing is interrupted.
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **result)
        stream.flush()
        os.fsync(stream.fileno())
    output_digest = sha256_file(destination)
    receipt = {"version": RECEIPT_VERSION, "role": RECEIPT_ROLE, "passed": True,
               "source_number": number, "source_filename": item["file"],
               "source_row_sha256": canonical_digest(item), "binding_sha256": canonical_digest(binding),
               "replay_verifier_sources_sha256": verifier,
               "metadata": derivative_metadata(item, binding, output_digest),
               "output_bytes": destination.stat().st_size, "exact_replay_errors": errors,
               "worker_pid": os.getpid()}
    receipt["receipt_sha256"] = canonical_digest(receipt)
    _write_exclusive_receipt(receipt_file, receipt)
    return receipt


def verify_episode_receipt(output, item, number, binding, receipt=None):
    path = receipt_path(output, item["file"])
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Successful replay receipt is missing/nonfile: "+item["file"])
    stored = json.loads(path.read_text())
    if receipt is not None and receipt != stored:
        raise RuntimeError("Returned worker receipt differs from its saved receipt")
    digest_body = dict(stored)
    digest = digest_body.pop("receipt_sha256", None)
    if (stored.get("version") != RECEIPT_VERSION or type(stored.get("version")) is not int
            or stored.get("role") != RECEIPT_ROLE or stored.get("passed") is not True
            or stored.get("source_number") != number or stored.get("source_filename") != item["file"]
            or stored.get("source_row_sha256") != canonical_digest(item)
            or stored.get("binding_sha256") != canonical_digest(binding)
            or digest != canonical_digest(digest_body)):
        raise RuntimeError("Replay receipt version/hash/source/binding mismatch: "+item["file"])
    if stored.get("replay_verifier_sources_sha256") != replay_verifier_fingerprint(binding):
        raise RuntimeError("Replay receipt verifier sources are stale: "+item["file"])
    validate_replay_errors(stored.get("exact_replay_errors"))
    destination = safe_destination(output, item["file"])
    if not destination.is_file() or destination.is_symlink():
        raise RuntimeError("Replay receipt payload is missing/nonfile: "+item["file"])
    digest = sha256_file(destination)
    if (destination.stat().st_size != stored.get("output_bytes")
            or stored.get("metadata") != derivative_metadata(item, binding, digest)):
        raise RuntimeError("Replay receipt metadata/output payload hash differs: "+item["file"])
    return stored


def initialize_worker(source, output, manifest, profile, binding):
    global _worker_context
    # One worker owns one independent DIRECT client; never PyBullet threads.
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    from VLA.common.body_environment import BodyGraspEnvironment
    env = BodyGraspEnvironment(profile, robot=manifest["robot"], image_size=manifest["image_size"],
        seed=manifest["seed"], required_success_steps=manifest["required_success_steps"], use_semantic_success=False)
    _worker_context = (env, source, output, profile, binding, limiter)
    atexit.register(env.close)


def worker_replay(job):
    from VLA.rerender_body_v9_front_candidate import replay_episode
    env, source, output, profile, binding, _ = _worker_context
    number, item = job
    return number, produce_episode_receipt(env, item, number, source, output, profile, binding, replay_episode)


def bounded_process_replays(jobs, source, output, manifest, profile, binding, workers):
    """Yield small receipts; at most N jobs/episode payloads are in flight."""
    pending_jobs = iter(jobs)
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker, initargs=(str(source), str(output), manifest, profile, binding))
    pending = {}
    try:
        for _ in range(workers):
            job = next(pending_jobs, None)
            if job is None:
                break
            pending[executor.submit(worker_replay, job)] = job[0]
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                expected_number = pending.pop(future)
                number, receipt = future.result()
                if number != expected_number:
                    raise RuntimeError("Worker returned another episode number")
                yield number, receipt
                job = next(pending_jobs, None)
                if job is not None:
                    pending[executor.submit(worker_replay, job)] = job[0]
    finally:
        # Completed/in-flight artifacts survive failure; unscheduled work never
        # runs. Running processes finish their bounded episode before shutdown.
        executor.shutdown(wait=True, cancel_futures=True)
