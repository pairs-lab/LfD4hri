"""Native SmolVLA objective/FP32 masters with full scene67 + replayed corrections."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from VLA.common.provenance import sha256_file


def main():
    import lerobot.scripts.lerobot_train as trainer
    from VLA.common.source_training_gate import require_training_source
    from VLA.smolvla.scene67_dataset import Scene67Dataset
    from VLA.scene64_dataset import load_scene_projection
    from VLA.smolvla import balanced_sampling, train_fp32
    projection = Path(os.environ["VLA_SCENE67_PROJECTION"])
    _, audit, _ = load_scene_projection(projection)
    updates = int(os.environ["VLA_SCENE67_UPDATES"])
    require_training_source(audit["source"], model="smolvla", optimizer_updates=updates)
    if updates > 2:
        proof = json.loads(Path(os.environ["VLA_SCENE67_SMOKE_QUALIFICATION"]).read_text())
        if not proof.get("passed") or proof.get("actual_optimizer_updates") != 2 or proof.get("actual_neural_controls") != 8:
            raise ValueError("Own new67 CUDA/saved-precision/exact reload/robot qualification required")
        for path, digest in proof["dependencies"].items():
            if sha256_file(path) != digest: raise ValueError("New67 smoke proof/source changed")
    created = []
    original_make = trainer.make_train_eval_datasets
    def make(cfg):
        base, validation = original_make(cfg)
        if validation is not None: raise ValueError("No validation/test samples in training")
        data = Scene67Dataset(base, projection, os.environ["VLA_SCENE67_RECOVERY_ADMISSION"])
        created.append(data); return data, None
    trainer.make_train_eval_datasets = make
    original_processors = trainer.make_pre_post_processors
    def processors(*args, **kwargs):
        kwargs["pretrained_path"] = None; kwargs.pop("pretrained_revision",None)
        return original_processors(*args, **kwargs)
    trainer.make_pre_post_processors = processors
    class RehearsalSampler(balanced_sampling.BalancedInitialSampler):
        def __init__(self, *args, **kwargs):
            # Retain the audited native seed/epoch/restart semantics and
            # original source-offset checks. Sampling must not consume the
            # policy's diffusion RNG or silently lose exact resume support.
            super().__init__(*args, **kwargs)
            if len(created) != 1: raise ValueError("Actual native dataset required before sampler")
            data = created[0]; source_rows = kwargs["source_index"]
            base = balanced_sampling.frame_weights(source_rows,.1,target_balanced=True)
            extra = np.zeros(len(data)-data.original_length,np.float64); offset=0
            tasks = {t: [r for r in data.corrections if r["task"]["name"]==t] for t in ("touch","pick","move","pick_place")}
            if any(not v for v in tasks.values()): raise ValueError("All four corrective tasks required")
            for row in data.corrections:
                n=row["steps"]; extra[offset:offset+n]=1/(4*len(tasks[row["task"]["name"]])*n);offset+=n
            self.weights=torch.from_numpy(np.r_[.8*base,.2*extra]); self.count=len(data)
        def __len__(self): return self.count
    balanced_sampling.BalancedInitialSampler = RehearsalSampler
    balanced_sampling.sampling_contract = lambda *_a,**_kw: {"profile":"scene67_TRAIN_rehearsal80_correction20_v1",
        "prefix_actions_supervised":False,"final_or_dev_data_used":False,"native_action_objective_unchanged":True}
    train_fp32.main()


if __name__ == "__main__": main()
