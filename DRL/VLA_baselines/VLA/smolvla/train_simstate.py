"""Process-local state18 dataset wrapper; unchanged official SmolVLA loss/model."""
import os
from pathlib import Path


def main():
    import lerobot.scripts.lerobot_train as trainer
    from VLA.smolvla.simstate_dataset import SimStateDataset, load_projection
    from VLA.common.source_training_gate import require_training_source
    from VLA.smolvla import train_phase_balanced
    projection = Path(os.environ["VLA_SIMSTATE_TRAIN_PROJECTION"])
    _, audit, _ = load_projection(projection)
    require_training_source(audit["source"], model="smolvla", optimizer_updates=int(os.environ["VLA_SIMSTATE_OPTIMIZER_UPDATES"]))
    original = trainer.make_train_eval_datasets
    def make(cfg):
        dataset, validation = original(cfg)
        if validation is not None:
            raise ValueError("Use separately audited held-out diagnostics; do not accidentally train on validation")
        return SimStateDataset(dataset, projection), None
    trainer.make_train_eval_datasets = make
    # New state18 cannot inherit state7 normalizer tensors. Rebuild processors
    # from TRAIN-only stats; pretrained MODEL WEIGHTS still load normally.
    original_processors = trainer.make_pre_post_processors
    def processors(*args, **kwargs):
        kwargs["pretrained_path"] = None
        kwargs.pop("pretrained_revision", None)
        return original_processors(*args, **kwargs)
    trainer.make_pre_post_processors = processors
    train_phase_balanced.main()


if __name__ == "__main__":
    main()
