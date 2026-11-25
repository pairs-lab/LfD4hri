"""Process-local explicit phase sampler; official policy/objective unchanged."""
import os

from VLA.smolvla import balanced_sampling, train_fp32
from VLA.smolvla.phase_balanced_sampling import PhaseBalancedSampler, load_phase_recipe


def main():
    if (os.environ.get("VLA_SMOLVLA_TARGET_BALANCED") != "1"
            or float(os.environ.get("VLA_SMOLVLA_GROUNDING_PROB", "0")) != 0):
        raise ValueError("Phase recipe requires target-balanced base without extra t=0 pair draws")
    _, contract = load_phase_recipe(os.environ["VLA_SMOLVLA_SAMPLING_SOURCE"], float(os.environ["VLA_SMOLVLA_INITIAL_FRAME_PROB"]))
    balanced_sampling.BalancedInitialSampler = PhaseBalancedSampler
    balanced_sampling.sampling_contract = lambda *_args, **_kwargs: contract
    train_fp32.main()


if __name__ == "__main__":
    main()
