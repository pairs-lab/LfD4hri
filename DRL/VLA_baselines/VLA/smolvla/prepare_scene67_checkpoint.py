"""Expand ONLY the native state projection, preserving learned action weights.

Zero new columns, preserve first18 and bias. No optimistic optimizer/reload
qualification is fabricated: this is a warmstart, not a trained67 policy.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def expanded(weight):
    if weight.ndim != 2 or weight.shape[1] != 32 or weight.dtype != torch.float32:
        raise ValueError("Audited inherited32/F32 state projection required")
    new = torch.zeros(weight.shape[0],67,dtype=weight.dtype)
    new[:,:18] = weight[:,:18]
    return new


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError("Immutable architecture warmstart")
    config=json.loads((a.base/"config.json").read_text())
    if config["input_features"]["observation.state"]["shape"] != [18] or config["max_state_dim"] != 32:
        raise ValueError("Expected the existing learned18 checkpoint")
    digest=sha256_file(a.base/"model.safetensors");weights=load_file(str(a.base/"model.safetensors"),device="cpu")
    weights["model.state_proj.weight"]=expanded(weights["model.state_proj.weight"])
    config["input_features"]["observation.state"]["shape"]=[67];config["max_state_dim"]=67
    a.output.mkdir(parents=True);save_file(weights,str(a.output/"model.safetensors"))
    atomic_json(a.output/"config.json",config)
    if sha256_file(a.base/"model.safetensors") != digest: raise ValueError("Original learned weights changed")
    atomic_json(a.output/"architecture_warmstart.json",{"passed":True,"original_weights":str(a.base.resolve()),
        "original_model_sha256":digest,"only_changed_tensor":"model.state_proj.weight",
        "old_unused_padding_columns_NOT_reused_for_new_feedback":True,"new67_optimizer_updates":0,
        "new67_model_success_established":False,"optimizer_reset_for_new_input_architecture":True})


if __name__ == "__main__":main()
