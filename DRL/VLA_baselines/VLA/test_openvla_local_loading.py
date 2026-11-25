"""No-CUDA regression for complete weights but missing HF dynamic Python."""
from contextlib import nullcontext
import tempfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch
import torch  # Import before sys.modules mocks; never unload/re-import Torch.

from VLA.openvla.adapter import OpenVLAPolicy


class LocalLoadingTests(unittest.TestCase):
    def run_case(self, source, expect_official):
        model = SimpleNamespace(norm_stats={"berkeley_autolab_ur5": {}}, eval=lambda: model)
        processor = SimpleNamespace()
        auto = Mock(); auto.from_pretrained.return_value = model
        official = Mock(); official.from_pretrained.return_value = model
        config = Mock(); config.from_pretrained.return_value = object()
        hf = ModuleType("transformers")
        hf.AutoModelForVision2Seq = auto
        hf.AutoProcessor = Mock(); hf.AutoProcessor.from_pretrained.return_value = processor
        hf.BitsAndBytesConfig = Mock()
        cm = ModuleType("prismatic.extern.hf.configuration_prismatic"); cm.OpenVLAConfig = config
        mm = ModuleType("prismatic.extern.hf.modeling_prismatic"); mm.OpenVLAForActionPrediction = official
        with patch.dict("sys.modules", {"transformers": hf, cm.__name__: cm, mm.__name__: mm}), \
             patch("VLA.openvla.quantized_loading.quantized_loading_compatibility", return_value=nullcontext()):
            policy = OpenVLAPolicy(checkpoint=source); policy._load()
        chosen, other = (official, auto) if expect_official else (auto, official)
        chosen.from_pretrained.assert_called_once(); other.from_pretrained.assert_not_called()
        self.assertEqual(chosen.from_pretrained.call_args.args[0], source)
        self.assertIs(policy.model, model)
        self.assertEqual(policy.unnorm_key, "berkeley_autolab_ur5")
        if expect_official:
            config.from_pretrained.assert_called_once_with(source)
            self.assertIn("config", chosen.from_pretrained.call_args.kwargs)

    def test_missing_local_dynamic_model_uses_same_official_class_and_weights(self):
        with tempfile.TemporaryDirectory() as directory: self.run_case(directory, True)

    def test_complete_local_dynamic_files_keep_original_path(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "configuration_prismatic.py").touch()
            self.run_case(directory, False)

    def test_remote_identifier_keeps_original_path(self):
        self.run_case("openvla/openvla-7b", False)


if __name__ == "__main__": unittest.main()
