from types import SimpleNamespace
import unittest
from VLA.openvla.quantized_loading import hooked_quantized_dispatch, quantized_loading_compatibility


class QuantizedLoadingTest(unittest.TestCase):
    def test_hooks_only_for_quantized_models(self):
        dispatch = hooked_quantized_dispatch(lambda model, **kwargs: kwargs)
        for bits in (4, 8):
            model = SimpleNamespace(**{f"is_loaded_in_{bits}bit": True})
            self.assertTrue(dispatch(model, device_map={"": 0})["force_hooks"])
        self.assertNotIn("force_hooks", dispatch(SimpleNamespace(), device_map={"": 0}))

    def test_patch_restored_after_success_and_exception(self):
        original = lambda *args, **kwargs: None
        module = SimpleNamespace(dispatch_model=original)
        for failure in (False, True):
            try:
                with quantized_loading_compatibility(module, original):
                    self.assertIsNot(module.dispatch_model, original)
                    if failure:
                        raise ValueError("expected")
            except ValueError:
                pass
            self.assertIs(module.dispatch_model, original)

    def test_multimodal_outer_method_without_language_flags(self):
        dispatch = hooked_quantized_dispatch(lambda model, **kwargs: kwargs)
        for method in ("bitsandbytes", SimpleNamespace(value="bitsandbytes")):
            self.assertTrue(dispatch(SimpleNamespace(quantization_method=method))["force_hooks"])
        self.assertNotIn("force_hooks", dispatch(SimpleNamespace(quantization_method="gptq")))


if __name__ == "__main__":
    unittest.main()
