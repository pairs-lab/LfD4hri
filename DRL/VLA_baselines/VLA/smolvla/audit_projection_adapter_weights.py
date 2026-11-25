"""Byte-exact proof that projection adaptation preserves all other weights."""
import argparse
from safetensors import safe_open


def identical_bytes(left, right):
    """Compare storage bits on CPU, including BF16 and signed floating zero."""
    import torch
    return (left.shape == right.shape and left.dtype == right.dtype and
            torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                        right.contiguous().reshape(-1).view(torch.uint8)))


def audit_weights(base_path, checkpoint_path):
    name = "model.state_proj.weight"
    unchanged = 0
    # NumPy cannot decode native frozen BF16 tensors. Keep their actual bits;
    # converting to float32 could mask a changed dtype or storage encoding.
    with safe_open(base_path, framework="pt", device="cpu") as base, safe_open(
            checkpoint_path, framework="pt", device="cpu") as saved:
        if set(base.keys()) != set(saved.keys()):
            raise ValueError("Architecture/weight names changed")
        if name not in base.keys():
            raise ValueError("Native state projection missing")
        for key in base.keys():
            if base.get_slice(key).get_dtype() != saved.get_slice(key).get_dtype():
                raise ValueError("Frozen weight was silently downcast: " + key)
            a, b = base.get_tensor(key), saved.get_tensor(key)
            if a.shape != b.shape:
                raise ValueError("Model parameter shape changed: " + key)
            if key == name:
                if a.ndim != 2 or a.shape[1] != 32:
                    raise ValueError("Native state projection capacity changed")
                if not identical_bytes(a[:, :7], b[:, :7]) or not identical_bytes(a[:, 18:], b[:, 18:]):
                    raise ValueError("Original state/padding columns changed")
            elif not identical_bytes(a, b):
                raise ValueError("A supposedly frozen model parameter changed: " + key)
            else:
                unchanged += a.numel()
    return {"passed": True, "all_non_new_projection_weights_BYTE_exact": True,
            "frozen_weight_values_unchanged": int(unchanged), "original_state7_columns_BYTE_exact": True,
            "native_unused_padding_columns_BYTE_exact": True, "architecture_changed": False,
            "controller_changed": False, "native_BF16_storage_compared_without_casting": True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",required=True);parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True);args=parser.parse_args()
    from pathlib import Path
    from VLA.cross_vla120_trial import atomic_json
    output=Path(args.output)
    if output.exists():raise FileExistsError("Keep each full-weight preservation proof")
    atomic_json(output, audit_weights(Path(args.base)/"model.safetensors", Path(args.checkpoint)/"model.safetensors"))


if __name__=="__main__":main()
