"""RDT+SimState via the unchanged native model/loss/optimizer trainer.

Only current-state input preparation and explicit metadata differ. A new
privileged two-update CUDA qualification is mandatory before any long fit.
Version2 prevents the ordinary STATE7 server from loading this checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from VLA.common.provenance import sha256_file
from VLA.rdt.simstate_inputs import RDTSimStateDataset, SCHEMA, EXTRA_SLOTS
from VLA.simstate_feedback import FEATURE_NAMES


def enrich_configuration(configuration, projection, proof, stats):
    configuration.update(privileged_pose_consumed=True, privileged_model_inputs=True,
        policy_inputs=["front_rgb", "wrist_rgb", "joint_positions", "gripper_open", "text",
                       "PRIVILEGED_pre_action_target_goal_feedback11"],
        simstate={"schema": SCHEMA, "projection": str(projection),
            "projection_audit_sha256": sha256_file(projection/"audit.json"),
            "source_index_sha256": proof["source_index_sha256"],
            "conditioning_slots_NOT_motor_DOFs": list(EXTRA_SLOTS), "state18_feature_names": FEATURE_NAMES,
            "train_only_state18_stats": stats,
            "input_implementation_sha256": sha256_file(Path(__file__).with_name("simstate_inputs.py")),
            "feature_builder_sha256": sha256_file(Path(__file__).resolve().parents[1]/"simstate_feedback.py"),
            "oracle_controller_at_deployment": False, "future_state_or_actions_in_input": False})
    return configuration


def require_new_qualification(args, path, projection):
    if args.steps <= 2:
        return
    if path is None:
        raise ValueError("RDT+SimState long fit requires its OWN real CUDA/reload/deployment qualification")
    proof = json.loads(path.read_text())
    if (proof.get("passed") is not True or proof.get("schema") != SCHEMA
            or proof.get("scope") != "RDT_SimState_actual_two_update_reload_four_task_integration_NOT_SR"
            or proof.get("optimizer_steps") != [2.] or proof.get("native_action_indices_unchanged") != [0,1,2,3,4,5,10]
            or proof.get("neural_success_claimed") is not False
            or proof.get("projection_audit_sha256") != sha256_file(projection/"audit.json")):
        raise ValueError("Incomplete/wrong privileged model qualification")
    for file, digest in proof["bound_files"].items():
        if sha256_file(Path(file)) != digest:
            raise ValueError("Privileged qualification changed: " + file)


def main():
    import torch
    import VLA.rdt.training as trainer
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--simstate-qualification", type=Path)
    extra, remainder = parser.parse_known_args()
    original_argv = sys.argv
    sys.argv = [original_argv[0], *remainder]
    try:
        args = trainer.parse_args()
    finally:
        sys.argv = original_argv
    if args.mode != "train" or args.max_episodes is not None:
        raise ValueError("Exact full-source training only; no hidden source subset")
    projection = extra.projection.resolve()
    require_new_qualification(args, extra.simstate_qualification, projection)
    limit = int(os.environ.get("VLA_RDT_SIMSTATE_GPU_MEMORY_LIMIT_MIB", "8500"))
    if not 6000 <= limit <= 10000:
        raise ValueError("Bounded RDT CUDA allocation required")
    torch.cuda.set_per_process_memory_fraction(limit*1024**2/torch.cuda.get_device_properties(0).total_memory)
    datasets = []
    def dataset(recipe):
        data = RDTSimStateDataset(recipe.source, projection, samples_per_episode=recipe.samples_per_episode,
            seed=recipe.seed, action_horizon=recipe.action_horizon)
        datasets.append(data)
        return data
    stock_target = trainer._target_contract
    def target(configuration, *, step=None):
        if len(datasets) != 1:
            raise ValueError("Actual full privileged dataset not instantiated exactly once")
        data = datasets[0]
        enrich_configuration(configuration, projection, data.simstate_audit, data.simstate_stats)
        # Native trainer writes its initial config just before this hook.
        # Persist the actual privileged recipe, not the ordinary default label.
        (args.output/"train_config.json").write_text(json.dumps(configuration, indent=2)+"\n")
        result = stock_target(configuration, step=step)
        return {**result, "format_version": 2, "model_family": "rdt_simstate",
                "privileged_model_inputs": True, "simstate": configuration["simstate"]}
    trainer.parse_args = lambda: args
    trainer._make_dataset = dataset
    trainer._target_contract = target
    # Native source, language, action/FK and precision guards still execute.
    raise SystemExit(trainer.main())


if __name__ == "__main__":
    main()
