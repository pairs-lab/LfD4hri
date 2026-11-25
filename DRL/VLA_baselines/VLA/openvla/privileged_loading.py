"""Target-fitted QLoRA reload with the SAME FP32 nonquantized base recipe.

Training's prepare_model_for_kbit_training casts non-4bit parameters to FP32.
Skipping it at inference downcasts auto-saved embedding/lm-head tensors into
a BF16 base. This isolated loader leaves historical plain adapters unchanged.
"""
from pathlib import Path
from unittest.mock import patch
import torch
from VLA.openvla.adapter import OpenVLAPolicy


class PrivilegedOpenVLA(OpenVLAPolicy):
    def _load(self):
        if self.model is not None:return
        if not (Path(self.checkpoint)/'adapter_config.json').is_file():
            raise ValueError('Actual fitted adapter required; no official-weight substitute')
        from peft import PeftModel,prepare_model_for_kbit_training
        stock=PeftModel.from_pretrained
        def aligned(base,*args,**kwargs):
            base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
            return stock(base,*args,**kwargs)
        with patch.object(PeftModel,'from_pretrained',aligned):super()._load()

    def predict(self,observation):
        self._load()
        # The training-aligned FP32 nonquantized vision modules consume the
        # same BF16 image tensors under the same autocast compute policy.
        with torch.autocast('cuda',dtype=torch.bfloat16):return super().predict(observation)


def verify_adapter_values(model,checkpoint):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    saved=load_file(str(Path(checkpoint)/'adapter_model.safetensors'),device='cpu')
    actual=get_peft_model_state_dict(model)
    missing=sorted(set(saved)-set(actual));extra=sorted(set(actual)-set(saved));mismatches=[]
    for key in sorted(set(saved)&set(actual)):
        value=actual[key].detach().cpu()
        if value.dtype!=saved[key].dtype or not torch.equal(value,saved[key]):
            mismatches.append({'name':key,'saved_dtype':str(saved[key].dtype),'loaded_dtype':str(value.dtype),
                'shape_equal':list(value.shape)==list(saved[key].shape),
                'maximum_error':float((value.float()-saved[key].float()).abs().max()) if value.shape==saved[key].shape else None})
    return {'passed':not missing and not extra and not mismatches,'missing_keys':missing,'extra_keys':extra,
        'mismatches':mismatches,'tensors':len(saved),'values':sum(t.numel() for t in saved.values()),
        'all_saved_adapter_values_exact':not missing and not extra and not mismatches}
