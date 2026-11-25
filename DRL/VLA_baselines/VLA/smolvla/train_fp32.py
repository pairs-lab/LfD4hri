"""Official LeRobot trainer with a process-local FP32-master policy constructor."""

import json
from functools import partial
import os
from pathlib import Path

from VLA.smolvla.precision import fp32_policy_class, promote_trainable_masters


def main():
    memory_limit = os.environ.get("VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB")
    if memory_limit is not None:
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
        fraction = float(memory_limit) * 1024**2 / total
        if not 0 < fraction <= 1:
            raise ValueError("Invalid process-local CUDA memory budget")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    import lerobot.policies.factory as factory
    import lerobot.scripts.lerobot_train as trainer

    stock_class = factory.get_policy_class
    train_connector = os.environ.get("VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR", "0") == "1"
    factory.get_policy_class = lambda name: fp32_policy_class(train_connector) if name == "smolvla" else stock_class(name)
    stock_make = trainer.make_policy
    sampling_profile_contract = None

    def audited_make_policy(*args, **kwargs):
        policy = stock_make(*args, **kwargs)
        constructed = policy._fp32_master_audit
        loaded = promote_trainable_masters(policy)
        report = {**constructed, "loaded_trainable_dtype_counts": loaded["trainable_dtype_counts_before"],
                  "promoted_after_loading": loaded["promoted_parameter_values"]}
        if report["promoted_after_loading"]:
            raise RuntimeError("Checkpoint loader downcast FP32 master weights")
        path = Path(os.environ["VLA_SMOLVLA_MASTER_AUDIT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        if sampling_profile_contract is not None:
            (path.parent / "sampling_contract.json").write_text(json.dumps(sampling_profile_contract, indent=2) + "\n")
        print("FP32 masters verified: " + json.dumps({k: v for k, v in report.items() if k != "trainable_parameter_names"}), flush=True)
        return policy

    trainer.make_policy = audited_make_policy
    initial_probability = os.environ.get("VLA_SMOLVLA_INITIAL_FRAME_PROB")
    if initial_probability is not None:
        from VLA.smolvla.balanced_sampling import BalancedInitialSampler, sampling_contract

        source = Path(os.environ["VLA_SMOLVLA_SAMPLING_SOURCE"])
        target_balanced = os.environ.get("VLA_SMOLVLA_TARGET_BALANCED", "0") == "1"
        grounding_probability = float(os.environ.get("VLA_SMOLVLA_GROUNDING_PROB", "0"))
        contract = sampling_contract(source, float(initial_probability), target_balanced, grounding_probability)
        sampling_profile_contract = contract
        trainer.EpisodeAwareSampler = partial(
            BalancedInitialSampler, source_index=json.loads((source / "index.json").read_text()),
            initial_probability=float(initial_probability),
            target_balanced=target_balanced,
            grounding_probability=grounding_probability,
        )
        # Do not create output_dir before the official config validates its
        # immutable-output guard. Persist this alongside the master audit,
        # after make_policy is invoked by the validated trainer.
        print("Sampling ablation: " + json.dumps(contract), flush=True)
    trainer.main()


if __name__ == "__main__":
    main()
