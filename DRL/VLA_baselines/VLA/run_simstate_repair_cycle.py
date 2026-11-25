"""CPU causal-data audit -> real CUDA smoke -> bounded fit -> learned120.

Never modifies/kills existing training or the user's RL. Long fitting is
conditional on complete TRAIN/VAL projection, exact loader/normalization
qualification, a saved FP32-master smoke checkpoint, and a quiet owned GPU.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import gpu_state, ROOT
from VLA.simstate_feedback import SCHEMA, FEATURE_NAMES

SOURCE_ROOT = ROOT / "VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013"
BASE = ROOT / "VLA/smolvla/body_v9_r16_phase_share_20261013/test1/train/checkpoints/080000/pretrained_model"
CROSS_PROGRESS = ROOT / "VLA/audits/cross_vla120_per_model_20261013/progress.json"


def training_command(base, projection, output, updates, accumulation):
    rename = json.dumps({"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"})
    return [sys.executable, "-m", "VLA.smolvla.train_simstate", "--policy.type=smolvla",
        f"--policy.pretrained_path={base}", "--policy.load_vlm_weights=true", "--policy.chunk_size=5",
        "--policy.n_action_steps=1", "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=false",
        "--policy.train_state_proj=true", "--policy.optimizer_lr=0.000025", "--policy.scheduler_decay_lr=0.000000625",
        f"--policy.scheduler_warmup_steps={0 if updates==2 else 250*accumulation}",
        f"--policy.scheduler_decay_steps={max(updates, 1000)*accumulation}", "--policy.push_to_hub=false",
        "--dataset.repo_id=local/simstate_original_r13_qualified", f"--dataset.root={SOURCE_ROOT/'lerobot_dataset'}",
        "--dataset.image_transforms.enable=false", f"--rename_map={rename}", f"--output_dir={output}",
        f"--job_name=smolvla_simstate18_{updates}", f"--steps={updates*accumulation}", "--batch_size=2",
        "--num_workers=0", "--accelerator.mixed_precision=bf16",
        f"--accelerator.gradient_accumulation.steps={accumulation}", "--wandb.enable=false",
        "--env_eval_freq=0", "--eval_steps=0", f"--save_freq={updates*accumulation}", "--save_checkpoint=true", "--seed=20261013"]


def finalize_checkpoint(output, updates, accumulation, projection):
    from safetensors import safe_open
    cp = output / "checkpoints" / f"{updates*accumulation:06d}" / "pretrained_model"
    config = json.loads((cp / "config.json").read_text())
    if config["input_features"]["observation.state"]["shape"] != [18]:
        raise ValueError("Saved checkpoint lost its new observation schema")
    audit = json.loads((output / "master_dtype_audit.json").read_text())
    with safe_open(cp / "model.safetensors", framework="np") as weights:
        if any(weights.get_slice(name).get_dtype() != "F32" for name in audit["trainable_parameter_names"]):
            raise ValueError("New learned weights lost FP32 precision")
    with safe_open(cp.parent / "training_state/optimizer_state.safetensors", framework="np") as moments:
        if any(moments.get_slice(name).get_dtype() != "F32" for name in moments.keys() if "exp_avg" in name):
            raise ValueError("Adam moments are not FP32")
    atomic_json(cp / "fp32_master_contract.json", {**audit, "saved_weights_and_adam_moments_fp32": True})
    source = SOURCE_ROOT / "qualified_train"
    atomic_json(cp / "vla_checkpoint_manifest.json", {"source": str(source),
        "source_manifest_sha256": sha256_file(source / "dataset_manifest.json"),
        "privileged_model_input": True, "simstate_projection": str(projection),
        "simstate_projection_audit_sha256": sha256_file(projection / "audit.json")})
    atomic_json(cp / "simstate_contract.json", {"schema": SCHEMA, "feature_names": FEATURE_NAMES,
        "privileged_model_input": True, "oracle_controller_at_deployment": False,
        "feature_builder_sha256": sha256_file(ROOT / "VLA/simstate_feedback.py"),
        "projection_audit_sha256": sha256_file(projection / "audit.json"), "base_model": str(BASE),
        "new_optimizer_updates": updates, "unchanged_UR5e_objects_cameras_physics": True})
    return cp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wait-hours", type=float, default=72)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): raise FileExistsError("New cycle root required")
    if not 0 < args.max_wait_hours <= 72: raise ValueError("Finite wait required")
    output.mkdir(parents=True)
    pilot = json.loads((ROOT / "VLA/audits/simstate_feedback_repair_20261013/pilot_train/audit.json").read_text())
    if not pilot.get("passed") or not pilot.get("pilot_only") or len(pilot["rows"]) != 8:
        raise ValueError("Actual eight-case causal replay gate required before full preparation")
    paths = [Path(__file__), ROOT / "VLA/prepare_simstate_projection.py", ROOT / "VLA/simstate_feedback.py",
        ROOT / "VLA/smolvla/simstate_dataset.py", ROOT / "VLA/smolvla/train_simstate.py",
        ROOT / "VLA/smolvla/simstate_adapter.py", ROOT / "VLA/smolvla/audit_simstate_loader.py",
        ROOT / "VLA/smolvla/audit_simstate_updates.py", ROOT / "VLA/evaluate_simstate120.py"]
    frozen = {str(p): sha256_file(p) for p in paths}
    history = []
    def save(stage, status="running", error=None):
        atomic_json(output / "status.json", {"status": status, "stage": stage, "error": error,
            "pid": os.getpid(), "history": history, "frozen_files": frozen,
            "privileged_VLA_variant": True, "oracle_actions_at_deployment": False,
            "neural_success_established": False, "new_fit_optimizer_updates": 5000,
            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR": "1", "VLA_SMOLVLA_INITIAL_FRAME_PROB": ".1",
        "VLA_SMOLVLA_TARGET_BALANCED": "1", "VLA_SMOLVLA_GROUNDING_PROB": "0",
        "VLA_SMOLVLA_SAMPLING_SOURCE": str(SOURCE_ROOT / "qualified_train"),
        "VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB": "10000"}
    def run(stage, command, extra=None):
        if any(sha256_file(Path(p)) != digest for p,digest in frozen.items()):
            raise ValueError("Frozen new pipeline changed")
        save(stage)
        with (output / f"{stage}.log").open("x") as log:
            result = subprocess.run(list(map(str,command)), cwd=ROOT, env={**env, **(extra or {})},
                                    stdout=log, stderr=subprocess.STDOUT, timeout=86400)
        history.append({"stage": stage, "exit_code": result.returncode})
        if result.returncode: raise RuntimeError(f"{stage} failed; retain log and candidate, no silent retry")
    try:
        for split, source, audit in [("train", "qualified_train", "qualified_train_audit"),
                                     ("validation", "validation_candidate", "validation_audit")]:
            run(f"project_{split}", [sys.executable, "-m", "VLA.prepare_simstate_projection",
                "--source", SOURCE_ROOT/source, "--source-audit", SOURCE_ROOT/audit/"audit.json",
                "--output", output/split, "--split", split])
        from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
        validate_holdout(SOURCE_ROOT / "validation_candidate", SOURCE_ROOT / "qualified_train")
        # CPU input qualification uses already completed R14 weights only as
        # config/tokenizer metadata. R16 remains the declared fit warmstart.
        previous = ROOT / "VLA/smolvla/body_v9_staged_r14_development40_20261013/test1/train/checkpoints/160000/pretrained_model"
        run("actual_loader_normalization", [sys.executable, "-m", "VLA.smolvla.audit_simstate_loader",
            "--projection", output/"train", "--dataset", SOURCE_ROOT/"lerobot_dataset",
            "--base-model", previous, "--output", output/"loader_audit.json"])
        deadline = time.monotonic() + args.max_wait_hours * 3600
        while True:
            cross = json.loads(CROSS_PROGRESS.read_text())
            cross_running = Path(f"/proc/{cross['pid']}").exists() and cross["status"] not in {
                "completed720", "finished_with_explicit_incomplete_models", "stopped_global_preflight_error_no_reroll"}
            gpu = gpu_state()
            if not cross_running and not gpu["compute_pids"] and gpu["free_mib"] >= 11000:
                break
            save("qualified_CPU_data_waiting_existing_cycles_and_GPU")
            if time.monotonic() >= deadline: raise TimeoutError("Finite GPU wait expired; no fit success claimed")
            time.sleep(30)
        from VLA.fingerprint_checkpoint import verify_fingerprint
        verify_fingerprint(BASE, full=True)
        model_root = ROOT / "VLA/smolvla_simstate" / output.name / "test1"
        for label, updates, accumulation in [("smoke", 2, 2), ("train", 5000, 8)]:
            fit_output = model_root / label
            fit_env = {"VLA_SIMSTATE_TRAIN_PROJECTION": str(output/"train"),
                "VLA_SIMSTATE_OPTIMIZER_UPDATES": str(updates),
                "VLA_SMOLVLA_MASTER_AUDIT": str(fit_output / "master_dtype_audit.json")}
            run(label, training_command(BASE, output/"train", fit_output, updates, accumulation), fit_env)
            cp = finalize_checkpoint(fit_output, updates, accumulation, output/"train")
            run(label+"_precision", [sys.executable, "-m", "VLA.smolvla.audit_training_precision",
                "--checkpoint", cp, "--output", output/(label+"_precision.json")])
            run(label+"_fingerprint", [sys.executable, "-m", "VLA.fingerprint_checkpoint", cp])
            if label == "smoke":
                run("new_input_weight_updates", [sys.executable, "-m", "VLA.smolvla.audit_simstate_updates",
                    "--base", BASE, "--checkpoint", cp, "--output", output/"new_input_weight_updates.json"])
        run("learned120", [sys.executable, "-m", "VLA.evaluate_simstate120", "--checkpoint", cp, "--output", output/"development120"])
        save("complete_fit_and_learned120_requires_success_video_review", status="completed")
    except Exception as error:
        save("stopped_preserved_failure_NO_unsupported_longfit", status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
