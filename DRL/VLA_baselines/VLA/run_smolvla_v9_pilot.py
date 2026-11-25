"""Finite, configurable V9 SmolVLA fit pilot; dry plan unless --run is given.

No dataset is qualified here. The caller supplies immutable qualified TRAIN
and audited scene-disjoint VALIDATION. Fresh-base smoke and fit are separate;
validation is never used to normalize, optimize or select a checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FRESH_BASE = "lerobot/smolvla_base"


@dataclass
class Stage:
    name: str
    command: list[str]
    gpu: bool = False
    gate_report: str | None = None
    optimizer_updates: int | None = None


def checkpoint_path(output, updates, accumulation):
    microsteps = updates * accumulation
    return Path(output) / "checkpoints" / f"{microsteps:0{max(6, len(str(microsteps)))}d}" / "pretrained_model"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--prep-root", type=Path, help="Reuse immutable CPU preparation reports separately from fresh run evidence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--validation-audit", type=Path)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--experiment-scope", choices=("bounded_pipeline_pilot", "bounded_refinement_experiment"),
                        default="bounded_pipeline_pilot", help="Refinement requires a separately qualified larger source; old pilot cap remains1000")
    parser.add_argument("--action-horizon", type=int, choices=(5, 10), default=5)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--smoke-accumulation", type=int, default=8)
    parser.add_argument("--paired-grounding-probability", type=float, default=0.)
    parser.add_argument("--initial-frame-sampling-probability", type=float, default=.30,
                        help="Explicit base initial-frame mass; reduce excessive reset memorization in a separate recipe")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-python", type=Path, default=ROOT / "VLA/third_party/lerobot/.venv/bin/python")
    parser.add_argument("--sim-python", type=Path, help="Defaults to the model venv, which must also contain PyBullet/scipy/trimesh")
    parser.add_argument("--prepared", action="store_true", help="Reuse an already prepared conversion and its exact prep reports read-only")
    parser.add_argument("--run", action="store_true", help="Execute the finite plan after every qualification gate passes")
    args = parser.parse_args(argv)
    for name in ("source", "validation", "dataset_root", "audit_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    # A virtualenv's executable is usually a symlink to a shared base Python.
    # Resolving it bypasses pyvenv.cfg and silently loses installed packages.
    # Keep its absolute invocation path, unlike data/proof paths above.
    args.model_python = args.model_python.expanduser().absolute()
    args.sim_python = (args.sim_python or args.model_python).expanduser().absolute()
    args.prep_root = (args.prep_root or args.audit_root / "pilot_prep").resolve()
    args.source_audit = (args.source_audit or args.audit_root / "source/audit.json").resolve()
    args.validation_audit = (args.validation_audit or args.audit_root / "validation_source/audit.json").resolve()
    cap = 1000 if args.experiment_scope == "bounded_pipeline_pilot" else 20000
    if not 2 <= args.updates <= cap or args.accumulation < 1 or args.smoke_accumulation < 1:
        raise ValueError("Pilot updates must be 2..1000 and accumulations positive")
    if args.source == args.validation:
        raise ValueError("TRAIN and VALIDATION cannot be the same source")
    if not 0 <= args.paired_grounding_probability < 1:
        raise ValueError("Invalid optional grounding sampling mass")
    if not 0 < args.initial_frame_sampling_probability < 1 or not 0 < args.learning_rate < 1 or args.batch_size < 1:
        raise ValueError("Invalid sampling mass, learning rate or batch size")
    return args


def validate_inputs(args):
    from VLA.common.source_training_gate import require_training_source
    from VLA.evaluate_smolvla_v9_source import verify_source_audit
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
    train_audit, train_manifest = verify_source_audit(args.source, args.source_audit)
    val_audit, val_manifest = verify_source_audit(args.validation, args.validation_audit)
    require_training_source(args.source, model="smolvla", optimizer_updates=args.updates)
    require_training_source(args.source, model="smolvla", optimizer_updates=2)
    scope = train_manifest["training_qualification"]["scope"]
    if scope != getattr(args, "experiment_scope", "bounded_pipeline_pilot"):
        raise ValueError("Requested experiment scope does not match the frozen source qualification")
    if train_manifest.get("split") != "train" or val_manifest.get("split") not in {"val", "validation"}:
        raise ValueError("Only the frozen TRAIN/VALIDATION splits may be used")
    if train_manifest.get("final_test_used") or val_manifest.get("final_test_used"):
        raise ValueError("Final-test data is outside this pilot")
    validate_holdout(args.source, args.validation)
    for source in (args.source, args.validation):
        rows = json.loads((source / "index.json").read_text())
        if set(row["task"]["name"] for row in rows) != {"touch", "pick", "move", "pick_place"}:
            raise ValueError("Both splits must contain all four task primitives")
    return {"train": {"audit": str(args.source_audit), "index_sha256": train_audit["index_sha256"], "manifest_sha256": train_audit["manifest_sha256"]},
            "validation": {"audit": str(args.validation_audit), "index_sha256": val_audit["index_sha256"], "manifest_sha256": val_audit["manifest_sha256"]},
            "qualification_scope": scope, "allowed_updates": train_manifest["training_qualification"]["max_optimizer_updates"]}


def build_plan(args):
    model, sim = str(args.model_python), str(args.sim_python)
    source, validation, dataset = str(args.source), str(args.validation), str(args.dataset_root)
    prep = args.prep_root
    stats, roundtrip = prep / "stats.json", prep / "roundtrip.json"
    preprocess_png = prep / "smolvla_preprocess.png"
    preprocess = preprocess_png.with_suffix(".json")
    smoke_root, train_root = args.output / "smoke", args.output / "train"
    smoke_cp = checkpoint_path(smoke_root, 2, args.smoke_accumulation)
    cp = checkpoint_path(train_root, args.updates, args.accumulation)
    repo = "local/" + args.source.name
    stages = []
    def add(name, module, tokens=(), **kwargs):
        stages.append(Stage(name, [model, "-m", module, *map(str, tokens)], **kwargs))
    if not args.prepared:
        add("convert", "VLA.lerobot.convert_dataset", ["--source", source, "--audit-report", args.source_audit, "--output", dataset, "--repo-id", repo])
        add("exact_stats", "VLA.lerobot.recompute_exact_stats", ["--source", source, "--dataset-root", dataset, "--output", stats], gate_report=str(stats))
        add("roundtrip", "VLA.lerobot.audit_converted_dataset", ["--source", source, "--dataset-root", dataset, "--repo-id", repo, "--output", roundtrip], gate_report=str(roundtrip))
        add("preprocess", "VLA.audit_model_preprocessing", ["--model", "smolvla", "--dataset", source, "--output", preprocess_png], gate_report=str(preprocess))
    data_common = ["--source", source, "--dataset-root", dataset, "--audit-report", str(args.source_audit),
                   "--roundtrip-report", str(roundtrip), "--stats-report", str(stats), "--preprocess-report", str(preprocess),
                   "--base-model", FRESH_BASE, "--train-vlm", "--train-visual-connector", "--batch-size", str(args.batch_size),
                   "--learning-rate", str(args.learning_rate), "--initial-frame-sampling-probability", str(args.initial_frame_sampling_probability),
                   "--target-balanced-sampling", "--action-horizon", str(args.action_horizon)]
    if getattr(args, "paired_grounding_probability", 0.):
        data_common.extend(["--paired-grounding-probability", str(args.paired_grounding_probability)])
    train_count = ["--steps", str(args.updates), "--gradient-accumulation", str(args.accumulation),
                   "--warmup-updates", str(args.updates // 10), "--decay-updates", str(max(1000, args.updates))]
    add("validate", "VLA.smolvla.training", ["validate", *data_common, *train_count])
    # Run exact reset checks before spending GPU time, in the interpreter that
    # will deploy the neural policy (its PyBullet binary may differ from the
    # original collection interpreter). Use the same preselected rollout rows.
    for split, observed, audit in (("train", source, args.source_audit), ("heldout", validation, args.validation_audit)):
        report = args.audit_root / (split + "_reset_sensor_parity.json")
        stages.append(Stage(split + "_reset_sensor_parity", [sim, "-m", "VLA.audit_source_reset_observations",
            "--source", str(observed), "--audit-report", str(audit), "--episodes-per-task", "2", "--output", str(report)],
            gate_report=str(report)))
    # Check visible CUDA directly and stop on failure; no unbounded waits or
    # hidden retries. Root starts --run only after allocating this GPU.
    stages.append(Stage("cuda_preflight", [model, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in this execution context'; print(torch.cuda.get_device_name(0))"], gpu=True))
    add("smoke", "VLA.smolvla.training", ["smoke", *data_common, "--steps", "2", "--gradient-accumulation", args.smoke_accumulation,
        "--warmup-updates", "0", "--decay-updates", "1000", "--output", smoke_root], gpu=True)
    add("smoke_precision", "VLA.smolvla.audit_training_precision", ["--checkpoint", smoke_cp, "--output", args.audit_root / "smoke_precision.json"],
        gate_report=str(args.audit_root / "smoke_precision.json"), optimizer_updates=2)
    add("smoke_fingerprint", "VLA.fingerprint_checkpoint", [smoke_cp])
    add("smoke_input_parity", "VLA.smolvla.audit_train_infer_inputs", ["--checkpoint", smoke_cp, "--source", source,
        "--audit-report", args.source_audit, "--episodes-per-task", "2", "--output", args.audit_root / "smoke_input_parity.json"],
        gate_report=str(args.audit_root / "smoke_input_parity.json"))
    save_frequency = min(1000, args.updates) if getattr(args, "experiment_scope", "bounded_pipeline_pilot") == "bounded_refinement_experiment" else args.updates
    add("train", "VLA.smolvla.training", ["train", *data_common, *train_count, "--save-freq", save_frequency, "--output", train_root], gpu=True)
    add("train_precision", "VLA.smolvla.audit_training_precision", ["--checkpoint", cp, "--output", args.audit_root / "train_precision.json"],
        gate_report=str(args.audit_root / "train_precision.json"), optimizer_updates=args.updates)
    add("train_fingerprint", "VLA.fingerprint_checkpoint", [cp])
    add("train_input_parity", "VLA.smolvla.audit_train_infer_inputs", ["--checkpoint", cp, "--source", source,
        "--audit-report", args.source_audit, "--episodes-per-task", "2", "--output", args.audit_root / "train_input_parity.json"],
        gate_report=str(args.audit_root / "train_input_parity.json"))
    for split, observed, audit in (("train", source, args.source_audit), ("heldout", validation, args.validation_audit)):
        tokens = ["--checkpoint", cp, "--source", observed, "--audit-report", audit,
                  "--episodes-per-task", "2", "--seed", "20261009", "--output", args.audit_root / f"{split}_imitation.json"]
        if split == "heldout":
            tokens.extend(["--normalization-source", source, "--normalization-audit-report", args.source_audit])
        add(split + "_imitation", "VLA.smolvla.audit_checkpoint_predictions", tokens, gpu=True)
    for split, observed, audit, mode in (("heldout", validation, args.validation_audit, "heldout"),
                                        ("train", source, args.source_audit, "train_diagnostic")):
        stages.append(Stage(split + "_closedloop", [sim, "-m", "VLA.evaluate_smolvla_v9_source", "--checkpoint", str(cp),
            "--source", str(observed), "--audit-report", str(audit), "--split-mode", mode,
            "--training-source", source, "--training-audit-report", str(args.source_audit), "--episodes-per-task", "2",
            "--output", str(args.output / (split + "_closedloop"))], gpu=True))
    return stages


def run(args, stages, inputs):
    if args.output.exists() or (args.audit_root / "pilot_driver_status.json").exists():
        raise FileExistsError("Refusing to overwrite a previous pilot")
    if args.dataset_root.exists() and not args.prepared:
        raise FileExistsError("Choose a new conversion directory or explicitly use --prepared")
    if args.prepared:
        for name in ("stats.json", "roundtrip.json", "smolvla_preprocess.json"):
            if not json.loads((args.prep_root / name).read_text()).get("passed"):
                raise ValueError("Prepared CPU audit did not pass: " + name)
    args.audit_root.mkdir(parents=True, exist_ok=True)
    if not args.prepared:
        args.prep_root.mkdir(parents=True, exist_ok=True)
    logs = args.audit_root / "pilot_logs"
    logs.mkdir()
    args.output.mkdir(parents=True)
    status_path = args.audit_root / "pilot_driver_status.json"
    history = []
    contract = {"scope": getattr(args, "experiment_scope", "bounded_pipeline_pilot") + "_not_final_benchmark", "source": str(args.source), "validation": str(args.validation),
                "updates": args.updates, "smoke_updates": 2, "train_microsteps": args.updates * args.accumulation,
                "smoke_microsteps": 2 * args.smoke_accumulation, "fresh_base": FRESH_BASE,
                "validation_used_for_training_or_normalization": False, "final_test_used": False,
                "qualification": inputs, "driver_sha256": sha256_file(Path(__file__)),
                "harness_sha256": sha256_file(Path(__file__).with_name("evaluate_smolvla_v9_source.py")),
                "stages": [asdict(stage) for stage in stages]}
    (args.audit_root / "pilot_plan.json").write_text(json.dumps(contract, indent=2) + "\n")
    def status(state, stage, error=None):
        status_path.write_text(json.dumps({**contract, "status": state, "stage": stage,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "history": history,
            "error": error}, indent=2) + "\n")
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
                   "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
                   "TOKENIZERS_PARALLELISM": "false"}
    for stage in stages:
        try:
            # Re-check immutable source/proof bytes before every stage, including
            # held-out followups; no candidate can silently become training data.
            validate_inputs(args)
            status("running", stage.name)
            log = logs / (stage.name + ".log")
            with log.open("x") as stream:
                result = subprocess.run(stage.command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            history.append({"stage": stage.name, "exit_code": result.returncode, "log": str(log)})
            if result.returncode:
                raise RuntimeError(f"Stage exited {result.returncode}: {log}")
            if stage.gate_report:
                gate = json.loads(Path(stage.gate_report).read_text())
                if not gate.get("passed"):
                    raise RuntimeError("Stage report failed: " + stage.gate_report)
                if stage.optimizer_updates is not None and gate.get("optimizer_steps") != [float(stage.optimizer_updates)]:
                    raise RuntimeError("Saved Adam counters differ from intended optimizer-update count")
            status("stage_completed", stage.name)
        except Exception as exc:
            status("failed", stage.name, str(exc))
            raise
    validate_inputs(args)
    status("completed_diagnostic_only", "done")


def main():
    args = parse_args()
    stages = build_plan(args)
    if not args.run:
        # Planning never creates files or executes conversion/training. If the
        # source already exists, also refuse a plan beyond its advertised cap.
        manifest_path = args.source / "dataset_manifest.json"
        if manifest_path.exists():
            cap = json.loads(manifest_path.read_text()).get("training_qualification", {}).get("max_optimizer_updates")
            if type(cap) is not int or args.updates > cap:
                raise ValueError("Plan exceeds or lacks the source's bounded-pilot qualification")
        print(json.dumps({"run": False, "stages": [asdict(stage) for stage in stages], "scope": "plan_only_not_training_authorization"}, indent=2))
        return
    inputs = validate_inputs(args)
    run(args, stages, inputs)


if __name__ == "__main__":
    main()
