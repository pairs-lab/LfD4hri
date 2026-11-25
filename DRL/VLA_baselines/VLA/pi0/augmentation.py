"""Opt-in process-local colour-name-safe augmentation; upstream stays intact."""
from functools import wraps


def colour_safe_factory(original):
    @wraps(original)
    def create(*args, **kwargs):
        args = list(args)
        if len(args) > 3:
            args[3] = 0.0
            kwargs.pop("hue", None)
        else:
            kwargs["hue"] = 0.0
        return original(*args, **kwargs)
    create._vla_colour_safe = True
    return create


def install_colour_safe_augmentation(module=None):
    if module is None:
        import augmax as module
    if not getattr(module.ColorJitter, "_vla_colour_safe", False):
        module.ColorJitter = colour_safe_factory(module.ColorJitter)
    return {"profile": "language_colour_safe_hue0", "hue_jitter": 0.0,
            "brightness": .3, "contrast": .4, "saturation": .5,
            "upstream_files_changed": False}
