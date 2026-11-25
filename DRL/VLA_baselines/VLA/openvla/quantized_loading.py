"""Scoped compatibility for old Transformers + recent bitsandbytes/Accelerate.

4.40 forbids quantized model.to(), whereas recent Accelerate assumes new BNB
modules are movable. Use supported device hooks for already quantized models;
do not suppress errors, change quantization or patch unquantized models.
"""
from contextlib import contextmanager
from functools import wraps


def hooked_quantized_dispatch(dispatch):
    @wraps(dispatch)
    def wrapped(model, *args, **kwargs):
        method = getattr(model, "quantization_method", None)
        method = getattr(method, "value", method)
        # Old multimodal wrappers expose the quantization METHOD on the outer
        # model but put is_loaded_in_4bit only on their language submodule.
        if (getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
                or (getattr(model, "is_quantized", False) and method == "bitsandbytes")
                or method == "bitsandbytes"):
            kwargs["force_hooks"] = True
        return dispatch(model, *args, **kwargs)
    return wrapped


@contextmanager
def quantized_loading_compatibility(modeling_utils=None, dispatch=None):
    if modeling_utils is None:
        import transformers.modeling_utils as modeling_utils
    if dispatch is None:
        from accelerate.big_modeling import dispatch_model as dispatch
    original = modeling_utils.dispatch_model
    modeling_utils.dispatch_model = hooked_quantized_dispatch(dispatch)
    try:
        yield
    finally:
        modeling_utils.dispatch_model = original
