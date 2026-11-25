"""Memory-conscious OpenVLA inference adapter.

The base checkpoint has Berkeley Autolab UR5 action statistics, so the smoke
test uses that key.  Results are explicitly zero-shot until a checkpoint
fine-tuned on this rendered scene is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.policy import Policy


class OpenVLAPolicy(Policy):
    name = "openvla"
    input_modalities = ("front_rgb", "text")

    def __init__(
        self,
        checkpoint: str | None = None,
        unnorm_key: str | None = None,
        **_kwargs,
    ) -> None:
        self.checkpoint = checkpoint or "openvla/openvla-7b"
        self.unnorm_key = unnorm_key
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from VLA.openvla.quantized_loading import quantized_loading_compatibility
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        checkpoint_path = Path(self.checkpoint)
        is_adapter = checkpoint_path.is_dir() and (checkpoint_path / "adapter_config.json").is_file()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_source = self.checkpoint
        if is_adapter:
            adapter_config = json.loads((checkpoint_path / "adapter_config.json").read_text())
            model_source = adapter_config["base_model_name_or_path"]
        # A locally cached HF snapshot can have all original weight shards
        # and processor assets but omit the dynamic model/config Python
        # files. Do NOT treat that loader failure as a robot/model failure.
        # Use the SAME official classes from the pinned upstream clone,
        # reading config/weights from the declared checkpoint unchanged.
        local_source = Path(model_source)
        model_loader = AutoModelForVision2Seq
        model_config = {}
        if local_source.is_dir() and not (local_source / "configuration_prismatic.py").is_file():
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            model_loader = OpenVLAForActionPrediction
            model_config["config"] = OpenVLAConfig.from_pretrained(model_source)
        with quantized_loading_compatibility():
            base_model = model_loader.from_pretrained(
                model_source, quantization_config=quantization, device_map="auto",
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                trust_remote_code=True, attn_implementation="eager",
                **model_config,
            )
        statistics_path = checkpoint_path / "dataset_statistics.json"
        local_statistics = json.loads(statistics_path.read_text()) if statistics_path.is_file() else None
        if local_statistics is not None:
            base_model.norm_stats = local_statistics
        if is_adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, self.checkpoint).eval()
        else:
            self.model = base_model.eval()
        if local_statistics is not None:
            self.model.norm_stats = local_statistics
            if hasattr(self.model, "get_base_model"):
                self.model.get_base_model().norm_stats = local_statistics
        if self.unnorm_key is None:
            keys = tuple(self.model.norm_stats)
            if len(keys) == 1:
                self.unnorm_key = keys[0]
            elif "berkeley_autolab_ur5" in keys:
                self.unnorm_key = "berkeley_autolab_ur5"
            else:
                raise ValueError(f"Specify --unnorm-key; checkpoint contains {keys}")

    def predict(self, observation: VLAObservation) -> np.ndarray:
        self._load()
        import torch
        from PIL import Image

        prompt = (
            "In: What action should the robot take to "
            # The trailing space token (29871) is part of OpenVLA's training
            # template.  Without it, remote predict_action appends input_ids
            # but not attention_mask, producing a one-token causal-mask skew.
            f"{observation.prompt.lower()}?\nOut: "
        )
        image = Image.fromarray(observation.front_rgb)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        native = np.asarray(
            self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if native.size != 7 or not np.isfinite(native).all():
            raise ValueError(f"OpenVLA returned invalid actions {native.shape}; expected seven finite values")
        # OXE/OpenVLA convention: last dimension is 1=open, 0=closed.
        return np.concatenate((native[:6], np.array((1.0 - 2.0 * native[6],), dtype=np.float32)))
