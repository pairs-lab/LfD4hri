"""Full-scene SmolVLA: actual neural commands only, no recovery controller."""
import json
from pathlib import Path
import torch
from VLA.common.provenance import sha256_file
from VLA.smolvla.adapter import SmolVLAPolicyAdapter
from VLA.scene64_conditioning import NUMERIC_SCHEMA, NUMERIC_NAMES, model_input


class Scene67SmolVLA(SmolVLAPolicyAdapter):
    name="smolvla_scene67";comparable=False
    def __init__(self,env,checkpoint,**kwargs):
        root=Path(checkpoint);contract=json.loads((root/"scene67_contract.json").read_text())
        cfg=json.loads((root/"config.json").read_text())
        if (contract.get("schema")!=NUMERIC_SCHEMA or contract.get("feature_names")!=NUMERIC_NAMES
            or contract.get("oracle_controller_at_deployment") is not False
            or cfg["input_features"]["observation.state"]["shape"] != [67] or cfg["max_state_dim"]!=67):
            raise ValueError("A separately trained67 native checkpoint is required")
        for path,digest in contract["input_bound_files"].items():
            if sha256_file(path)!=digest:raise ValueError("Native scene input implementation changed")
        super().__init__(env,checkpoint=checkpoint,**kwargs)
    def _load(self):
        if self.model is not None:return
        super()._load()
        from safetensors import safe_open
        contract=json.loads((Path(self.checkpoint)/"fp32_master_contract.json").read_text())
        loaded=dict(self.model.named_parameters())
        with safe_open(Path(self.checkpoint)/"model.safetensors",framework="pt",device="cpu") as saved:
            for name in contract["trainable_parameter_names"]:
                if not torch.equal(saved.get_tensor(name),loaded[name].detach().cpu()):
                    raise ValueError("A learned native67 value changed on inference reload")
        self.exact_trainable_reload_passed=True
    def build_sensor_batch(self,observation):
        batch=super().build_sensor_batch(observation)
        self.last_input=model_input(self.env,observation)
        batch["observation.state"]=torch.from_numpy(self.last_input["numeric67"]).unsqueeze(0)
        return batch
