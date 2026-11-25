"""Goal/feedback projection-only transfer inside the unchanged SmolVLA model.

Zero-initialize ONLY previously unused state columns7..17, preserving the
original state7 policy exactly at initialization. Freeze every other learned
weight. Use ordinary teacher supervision; never substitute a controller.
"""
import json
import os
from pathlib import Path


def main():
    import torch
    import lerobot.scripts.lerobot_train as trainer
    from VLA.smolvla.precision import promote_trainable_masters
    from VLA.smolvla import train_simstate
    original=trainer.make_policy
    def make(*args,**kwargs):
        policy=original(*args,**kwargs)
        weight=policy.model.state_proj.weight
        if weight.shape[1]!=32:raise ValueError("Native SmolVLA state projection capacity changed")
        # Pretrained/new unused columns must not inject unrelated bimanual
        # features into a previously7-state adapted controller.
        before=weight.detach().clone()
        with torch.no_grad():weight[:,7:18].zero_()
        if not torch.equal(weight[:,:7],before[:,:7]) or not torch.equal(weight[:,18:],before[:,18:]):
            raise ValueError("Original learned state projection columns changed")
        for p in policy.parameters():p.requires_grad_(False)
        weight.requires_grad_(True)
        def keep_only_new_columns(grad):
            result=torch.zeros_like(grad);result[:,7:18]=grad[:,7:18];return result
        weight.register_hook(keep_only_new_columns)
        audit=promote_trainable_masters(policy)
        policy._fp32_master_audit={**audit,"train_visual_connector":False,
            "projection_only_new_columns":list(range(7,18)),
            "original_state7_policy_UNCHANGED_at_initialization":True}
        path=Path(os.environ["VLA_SMOLVLA_MASTER_AUDIT"]).parent/"projection_initialization.json"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({"only_parameter":"model.state_proj.weight",
            "gradient_columns":list(range(7,18)),"original7_and_padding_columns_unchanged":True,
            "new_conditioning_columns_initialized_zero":True,"architecture_changed":False,
            "all_other_model_weights_frozen":True,"handwritten_controller_or_action_override":False},indent=2)+"\n")
        return policy
    trainer.make_policy=make
    train_simstate.main()


if __name__=="__main__":main()
