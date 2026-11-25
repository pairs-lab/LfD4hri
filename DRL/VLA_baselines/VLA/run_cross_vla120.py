"""Finite, serialized 30 episodes/action/model diagnostic queue.

Existing trainers are not interrupted. CPU reset auditing can run alongside
them; inference waits for their full owned cycles and a quiet GPU. A failed
model is retained and does not cancel the other declared models. No rerolls,
oracle replacement, threshold changes, historical-result adoption or silently
different physics. Not all adaptation regimes are comparable: report them.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("touch", "pick", "move", "pick_place")
SEEDS = list(range(20268001, 20268031))
SIM_PYTHON = ROOT / "VLA/third_party/lerobot/.venv/bin/python"
OPENPI_PYTHON = ROOT / "VLA/third_party/openpi/.venv/bin/python"
OPENVLA_PYTHON = ROOT / "VLA/.venvs/openvla/bin/python"
RDT_PYTHON = ROOT / "VLA/.venvs/rdt/bin/python"
PROFILE_PROTOCOL = ROOT / "VLA/evaluation_protocol_v9_postfit_10_per_action.json"
PROTECTED = [ROOT / "VLA/audits/development40_smolvla_r16_phase_share_20261013/status.json",
             ROOT / "VLA/audits/development40_rdt_r16_verified_guard_20261013/status.json"]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def model_specs():
    openvla = Path("/home/uet/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f")
    return [
        {"model": "rdt", "adaptation_regime": "qualified_v9_postfit", "port": 5720,
         "checkpoint": str(ROOT / "VLA/rdt/body_v9_staged_r16_verified_guard_20261013/test1/train/checkpoints/step-005000"),
         "minimum_free_gpu_mib": 11500},
        {"model": "smolvla", "adaptation_regime": "qualified_v9_postfit",
         "checkpoint": str(ROOT / "VLA/smolvla/body_v9_r16_phase_share_20261013/test1/train/checkpoints/080000/pretrained_model"),
         "minimum_free_gpu_mib": 10000},
        {"model": "pi0", "adaptation_regime": "official_zero_shot_v9", "port": 8020,
         "checkpoint": "gs://openpi-assets/checkpoints/pi0_base", "minimum_free_gpu_mib": 14000},
        {"model": "pi0_fast", "adaptation_regime": "official_zero_shot_v9", "port": 8021,
         "checkpoint": "gs://openpi-assets/checkpoints/pi0_fast_base", "minimum_free_gpu_mib": 14000},
        {"model": "openvla", "adaptation_regime": "official_zero_shot_v9", "port": 5721,
         "checkpoint": str(openvla), "unnorm_key": "berkeley_autolab_ur5", "minimum_free_gpu_mib": 12000},
        {"model": "cogact", "adaptation_regime": "official_zero_shot_v9", "port": 5520,
         "checkpoint": str(ROOT / "VLA/checkpoints/cogact-small/checkpoints/CogACT-Small.pt"),
         "unnorm_key": "berkeley_autolab_ur5", "minimum_free_gpu_mib": 14500,
         "memory_profile": "bf16_vlm_fp32_action_cpu_offload_no_quantization"},
    ]


def initialize(output):
    from VLA.common.observation_profile import profile_hash
    from VLA.common.provenance import sha256_file
    from VLA.common.runtime_provenance import environment_runtime_fingerprint
    if output.exists():
        raise FileExistsError("Fresh audit root required; prior queue preserved")
    template = json.loads(PROFILE_PROTOCOL.read_text())
    profile_path = Path(template["observation_profile_file"])
    profile = json.loads(profile_path.read_text())
    if profile_hash(profile) != template["observation_profile_sha256"]:
        raise ValueError("Stale common V9 sensor profile")
    reserved = json.loads((ROOT / template["reserved_final_protocol"]).read_text())
    if set(SEEDS) & set(reserved.get("final_test_seeds", [])):
        raise ValueError("Reserved paper-test seeds must remain unopened")
    runtime = environment_runtime_fingerprint(profile)
    paths = [Path(__file__).resolve(), ROOT / "VLA/cross_vla120_trial.py",
             ROOT / "VLA/cross_vla120_server.py", ROOT / "VLA/benchmark.py", PROFILE_PROTOCOL, profile_path]
    paths += [ROOT / name for name in (*runtime["sources_sha256"],
        "VLA/common/tasks.py", "VLA/common/spatial.py", "VLA/common/profile_binding.py",
        "VLA/common/recorder.py", "VLA/common/policy.py", "VLA/pi0/adapter.py", "VLA/pi0/server.py",
        "VLA/pi0/fast_tokenizer.py", "VLA/openvla/adapter.py", "VLA/openvla/quantized_loading.py",
        "VLA/cogact/adapter.py", "VLA/rdt/server.py", "VLA/rdt/adapter.py", "VLA/smolvla/adapter.py",
        "asset/params/reward.xml", "utils/random_env.py", "utils/camera.py")]
    plan = {"scope": "120_per_VLA_DEVELOPMENT_DIAGNOSTIC_NOT_FINAL_PAPER_TEST",
        "planned_trials_per_model": 120, "planned_trials_per_action_per_model": 30,
        "planned_trials_all_models": 720, "models": model_specs(), "tasks": list(TASKS), "seeds": SEEDS,
        "robot": "ur5e", "gripper": "Robotiq 2F-85", "scene_profile": "rl_random_env",
        "control_hz": 10, "replan_steps": 1, "success": template["success"],
        "observation_profile_file": str(profile_path), "observation_profile_sha256": profile_hash(profile),
        "environment_runtime_fingerprint": runtime,
        "frozen_files": {str(p): sha256_file(p) for p in paths},
        "scene_audit_root": str(output / "common_scene_resets"),
        "protected_status_files": list(map(str, PROTECTED)),
        "checkpoint_selection_rule": "R16 completed postfit for SmolVLA/RDT; official bases for others, NEVER choose by this cohort",
        "adaptation_regimes_are_NOT_matched": True, "accepted_fair_postfit_baseline": False,
        "zero_shot_cannot_establish_finetuned_model_weakness": True,
        "all_safety_aborts_are_task_failures": True, "integration_errors_are_NOT_task_failures": True,
        "final_test_used": False, "oracle_fallback": False, "outcome_seed_filter": False,
        "episodes_rerolled": 0, "video_seeds": [SEEDS[0]],
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True)
    atomic_json(output / "plan.json", plan)
    return plan


def verify_plan(plan):
    from VLA.common.provenance import sha256_file
    if plan["seeds"] != SEEDS or plan["planned_trials_per_model"] != 120:
        raise ValueError("Budget/seed drift")
    if any(sha256_file(Path(p)) != digest for p, digest in plan["frozen_files"].items()):
        raise ValueError("Frozen runtime changed; preserve the cohort and diagnose")


def audit_scenes(plan):
    import numpy as np
    import pybullet
    from PIL import Image
    from VLA.benchmark import _observation_telemetry
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.tasks import randomize_rl_scene, select_tasks
    from VLA.common.provenance import sha256_file
    profile = json.loads(Path(plan["observation_profile_file"]).read_text())
    directory = Path(plan["scene_audit_root"])
    if directory.exists():
        raise FileExistsError("Scene evidence is immutable; no hidden re-audit/replacement")
    directory.mkdir()
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tasks = [randomize_rl_scene(task, rng) for task in select_tasks("all")]
        env = environment_class_for_profile(profile)(observation_profile=profile,
            robot="ur5e", image_size=384, control_hz=10, seed=seed, assist_grasp=True,
            required_success_steps=1, success_threshold_scale=plan["success"]["reward_xml_threshold_scale"],
            use_semantic_success=True, semantic_success_thresholds=plan["success"]["semantic_thresholds"])
        try:
            for task in tasks:
                obs = env.reset(task)
                if env.is_semantic_success():
                    raise ValueError("Invalid initial-success scene; preserve, no reroll")
                target = directory / f"seed{seed}" / task.name
                target.mkdir(parents=True)
                for camera, pixels in (("front", obs.front_rgb), ("wrist", obs.wrist_rgb)):
                    Image.fromarray(pixels).save(target / f"{camera}.png")
                reset = {**_observation_telemetry(obs, env), "task": task.to_dict(),
                         "initial_semantic_success": False, "seed": seed}
                atomic_json(target / "reset.json", reset)
                rows.append({"seed": seed, "task": task.name, "prompt": task.prompt,
                             "directory": str(target), "reset_sha256": sha256_file(target / "reset.json")})
        finally:
            env.close()
        atomic_json(directory / "progress.json", {"completed_resets": len(rows), "planned_resets": 120,
                   "rows": rows, "pybullet_api_version": pybullet.getAPIVersion(),
                   "neural_episodes_executed": 0, "teacher_oracle_used": False})
        print(f"Common CPU scene resets {len(rows)}/120", flush=True)
    atomic_json(directory / "audit.json", {"passed": True, "resets": 120, "rows": rows,
        "neural_episodes_executed": 0, "teacher_oracle_used": False,
        "image_quality_or_model_grounding_not_certified_by_reset_parity": True})


def gpu_state():
    data = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    return {"free_mib": int(data.strip().splitlines()[0]), "compute_pids": active.splitlines() if active else []}


def protected_running(plan):
    rows = []
    for path in plan["protected_status_files"]:
        d = json.loads(Path(path).read_text())
        # A status marked running with an extinct owner is NOT an endless wait.
        alive = Path(f"/proc/{d.get('pid', 0)}").exists()
        if alive and d.get("status") == "running":
            rows.append({"file": path, "pid": d.get("pid"), "stage": d.get("stage")})
    return rows


def server_command(spec):
    model = spec["model"]
    if model in {"pi0", "pi0_fast"}:
        return [str(OPENPI_PYTHON), "-m", "VLA.pi0.server", "--model", model,
                "--checkpoint", spec["checkpoint"], "--port", str(spec["port"])]
    if model == "rdt":
        return [str(RDT_PYTHON), "-m", "VLA.rdt.server", "--checkpoint", spec["checkpoint"],
            "--port", str(spec["port"]), "--embeddings",
            str(ROOT / "VLA/audits/development40_rdt_r14_20261013/qualified_embeddings.pt"),
            "--vision", str(ROOT / "VLA/checkpoints/siglip-so400m-patch14-384")]
    if model in {"openvla", "cogact"}:
        return [str(OPENVLA_PYTHON), "-m", "VLA.cross_vla120_server", "--model", model,
                "--checkpoint", spec["checkpoint"], "--port", str(spec["port"]),
                "--unnorm-key", spec["unnorm_key"]]
    return None


def summarize_model(plan, spec, rows, status, error=None):
    from VLA.common.evaluation_statistics import wilson_interval
    valid = [r for r in rows if r.get("error") is None and isinstance(r.get("success"), bool)]
    n, k = len(valid), sum(r["success"] for r in valid)
    counts = Counter(r["task"] for r in valid)
    complete = n == 120 and counts == {task: 30 for task in TASKS}
    return {"model": spec["model"], "adaptation_regime": spec["adaptation_regime"],
        "status": status, "error": error, "planned_trials": 120, "completed_trials": n,
        "successes": k, "success_rate_for_completed120": k / 120 if complete else None,
        "partial_success_rate": k / n if n else None, "wilson_95_interval": wilson_interval(k, n) if n else None,
        "by_task": {task: {"completed": counts[task], "planned": 30,
            "successes": sum(r["success"] for r in valid if r["task"] == task),
            "success_rate_for_completed30": sum(r["success"] for r in valid if r["task"] == task) / 30
            if counts[task] == 30 else None} for task in TASKS},
        "safety_aborts": sum(bool(r.get("safety_violation")) for r in valid),
        "accepted_fair_postfit_baseline": False, "rows": rows}


def run_queue(output, plan, wait_hours, after_model=None):
    from VLA.cross_vla120_trial import validate_regime
    from VLA.fingerprint_checkpoint import verify_fingerprint
    progress_path = output / "progress.json"
    if progress_path.exists():
        raise FileExistsError("Queue already started; no silent resume/reroll")
    summaries = {s["model"]: summarize_model(plan, s, [], "queued") for s in plan["models"]}
    deadline = time.monotonic() + wait_hours * 3600
    def save(status, **extras):
        progress = {"status": status, "pid": os.getpid(),
            "planned_trials_per_model": 120, "planned_trials_all_models": 720,
            "completed_neural_trials": sum(s["completed_trials"] for s in summaries.values()),
            "models": summaries, "final_test_used": False, "oracle_fallback": False,
            "adaptation_regimes_are_NOT_matched": True, "accepted_fair_postfit_baseline": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extras}
        atomic_json(progress_path, progress)
        # A readable report must follow each actual result, not only the end
        # of a multi-day queue. This does not alter the scoring or denominator.
        from VLA.report_cross_vla120 import render
        target = output / "REPORT.md"
        temporary = target.with_suffix(".md.tmp")
        temporary.write_text(render(progress))
        temporary.replace(target)
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OPENPI_DATA_HOME": str(ROOT / "VLA/checkpoints/openpi"),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.82", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    environment.pop("JAX_PLATFORMS", None)
    profile = json.loads(Path(plan["observation_profile_file"]).read_text())
    save("cpu_scene_audit_running")
    audit_scenes(plan)
    for spec in plan["models"]:
        model, rows, server, stream = spec["model"], [], None, None
        model_output = output / model
        model_output.mkdir()
        try:
            while True:
                verify_plan(plan)
                gpu, protected = gpu_state(), protected_running(plan)
                if not protected and not gpu["compute_pids"] and gpu["free_mib"] >= spec["minimum_free_gpu_mib"]:
                    break
                summaries[model] = summarize_model(plan, spec, rows, "waiting_for_protected_cycles_and_quiet_gpu")
                save("waiting_gpu", current_model=model, gpu=gpu, protected_cycles=protected)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Finite GPU wait expired; no inference was substituted")
                time.sleep(30)
            validate_regime(model, spec["adaptation_regime"], spec["checkpoint"], profile)
            if spec["adaptation_regime"] == "qualified_v9_postfit":
                identity = verify_fingerprint(Path(spec["checkpoint"]), full=True)
                atomic_json(model_output / "checkpoint_identity.json", identity)
            command = server_command(spec)
            if command:
                if model == "cogact":
                    command += ["--audit-output", str(model_output / "runtime_qualification")]
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", spec["port"]))
                stream = (model_output / "server.log").open("x")
                server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                summaries[model] = summarize_model(plan, spec, rows, "starting_inference_server")
                save("running", current_model=model)
                startup_deadline = time.monotonic() + 1800
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Owned server exited {server.returncode}; inspect {model_output / 'server.log'}")
                    try:
                        with socket.create_connection(("127.0.0.1", spec["port"]), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() > startup_deadline:
                            raise TimeoutError("Owned server startup failed; no task score invented")
                        time.sleep(5)
            for seed in SEEDS:
                verify_plan(plan)
                trial = ROOT / "VLA" / model / output.name / f"test_seed{seed}"
                command = [str(SIM_PYTHON), "-m", "VLA.cross_vla120_trial", "--plan", str(output / "plan.json"),
                    "--model", model, "--seed", str(seed), "--output", str(trial)]
                if seed == SEEDS[0]:
                    command.append("--video")
                summaries[model] = summarize_model(plan, spec, rows, "testing")
                save("running", current_model=model, current_seed=seed)
                with (model_output / f"seed{seed}.log").open("x") as log:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=7200)
                path = trial / "results.json"
                if path.exists():
                    data = json.loads(path.read_text())
                    if data["model"] != model or data["seed"] != seed:
                        raise ValueError("Result identity differs")
                    rows.extend({"seed": seed, "task": t["name"], "success": t["success"],
                        "error": t.get("error"), "safety_violation": t.get("safety_violation"),
                        "result": str(path), "trajectory": str(trial / "trajectory.json"),
                        "video": str(trial / "rollout.mp4") if seed == SEEDS[0] else None} for t in data["tasks"])
                if result.returncode:
                    raise RuntimeError(f"Integration failed at seed{seed}; retain evidence, no reroll")
                if Counter(t["name"] for t in data["tasks"]) != {task: 1 for task in TASKS}:
                    raise ValueError("Incomplete scene block")
                summaries[model] = summarize_model(plan, spec, rows, "testing")
                save("running", current_model=model)
                print(f"{model}: {len(rows)}/120 completed learned episodes", flush=True)
            summaries[model] = summarize_model(plan, spec, rows, "completed120")
        except Exception as error:
            summaries[model] = summarize_model(plan, spec, rows, "integration_error_no_reroll", f"{type(error).__name__}: {error}")
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)
            if stream is not None:
                stream.close()
            atomic_json(model_output / "summary.json", summaries[model])
            save("running", current_model=model)
        # Optional orchestration boundary, AFTER the owned server is closed.
        # Errors here stop the supervisor; they must not become model failures.
        if after_model is not None:
            after_model(spec, summaries[model])
    complete = all(s["status"] == "completed120" for s in summaries.values())
    save("completed720" if complete else "finished_with_explicit_incomplete_models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--wait-hours", type=float, default=48)
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 72:
        raise ValueError("Finite wait in (0,72] hours required")
    output = args.output.resolve()
    plan = initialize(output) if args.initialize else json.loads((output / "plan.json").read_text())
    verify_plan(plan)
    try:
        run_queue(output, plan, args.wait_hours)
    except Exception as error:
        path = output / "progress.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        atomic_json(path, {**data, "status": "stopped_global_preflight_error_no_reroll",
                           "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
