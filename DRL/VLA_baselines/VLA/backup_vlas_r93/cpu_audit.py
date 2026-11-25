"""Native-source CPU checks and explicit integration hazards, never CUDA/FT/SR."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch import nn

from .contracts import canonical_from_seer_local, pad_state, pose, seer_local_from_canonical, temporal_indices

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "VLA/audits/backup_vlas_r93_20261022"
SEER = ROOT / "VLA/third_party/seer"
GROOT = ROOT / "VLA/third_party/gr00t_n15"


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    return namespace


def main():
    torch.set_num_threads(2)
    torch.manual_seed(93)
    record = {"UTC": datetime.now(timezone.utc).isoformat(), "scope": "CPU source/representation tests only",
              "full_native_VLA_forward_backward": False, "weights_loaded": False,
              "closed_loop_SR_measured": False, "checks": [], "hazards": []}
    rng = np.random.default_rng(93)
    maximum = 0.0
    for _ in range(512):
        current = pose(rng.uniform(-0.5, 0.5, 3), R.random(random_state=rng).as_matrix())
        previous = current @ pose(rng.uniform(-0.015, 0.015, 3),
                                 R.from_euler("xyz", rng.uniform(-0.08, 0.08, 3)).as_matrix())
        action = np.r_[rng.uniform(-0.04, 0.04, 3), rng.uniform(-0.15, 0.15, 3), rng.choice([-1., 1.])]
        for anchor in (None, previous):
            native = seer_local_from_canonical(action, current, anchor)
            decoded = canonical_from_seer_local(native, current, anchor)
            error = float(np.max(np.abs(decoded - action)))
            maximum = max(maximum, error)
            assert error < 1e-10
            assert native[6] == -action[6]
    record["checks"].append({"name": "seer_world_local_previous_command_roundtrip_and_gripper", "passed": True,
        "cases": 1024, "max_absolute_error": maximum,
        "does_not_qualify": "Dataset normalization/clipping, actual plant replay or native network performance"})

    namespace = extract(SEER/"utils/real_ft_data.py",
        ["_6d_to_pose", "pose_to_6d", "compute_delta_action"], {"np": np, "R": R})
    rows = [{"gripper_pose": np.array([0., 0., .3, 0., 0., 0.]),
             "action_gripper_pose": np.array([.01, 0., .3, 0., 0., 0., 1.])}]
    returned = namespace["compute_delta_action"](rows)
    assert returned is None
    record["checks"].append({"name": "reproduce_native_seer_helper_missing_return", "passed": True,
                              "actual_return": None})
    record["hazards"].append("Native compute_delta_action builds a list but returns None. Do not use its documented return as labels.")

    model_tree = ast.parse((SEER/"models/seer_model.py").read_text())
    agent = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "SeerAgent")
    forward = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    counts = {len(node.value.elts) for node in ast.walk(forward)
              if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple)}
    controller = ast.parse((SEER/"real_controller/controller.py").read_text())
    targets = []
    for node in ast.walk(controller):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = node.value.func
            if isinstance(callee, ast.Attribute) and callee.attr == "ddp_model":
                targets.extend(len(target.elts) for target in node.targets if isinstance(target, ast.Tuple))
    assert counts == {6} and targets == [5]
    record["checks"].append({"name": "native_seer_forward_return_vs_controller_unpack", "passed": True,
                              "model_return_arity": 6, "controller_unpack_arity": 5})
    record["hazards"].append("Native real controller unpacks five values, but SeerAgent.forward returns six. An unmodified wrapper would fail.")
    record["hazards"].append("Seer native state path slices six pose fields and one gripper field; appending state67 to that tensor is not a valid privileged encoder.")

    values = rng.normal(size=(3, 74)).astype(np.float32)
    padded = pad_state(values, 80)
    assert np.array_equal(padded[:, :74], values) and np.all(padded[:, 74:] == 0)
    try:
        pad_state(values, 64)
        raise AssertionError("Truncating state was accepted")
    except ValueError:
        pass
    record["checks"].append({"name": "state74_80_padding_no_truncation_failfast64", "passed": True})

    ns = extract(GROOT/"gr00t/model/action_head/flow_matching_action_head.py",
                 ["CategorySpecificLinear", "CategorySpecificMLP"], {"torch": torch, "nn": nn, "F": torch.nn.functional})
    native_cls = ns["CategorySpecificMLP"]
    base = native_cls(2, 64, 16, 8)
    expanded = native_cls(2, 80, 16, 8)
    with torch.no_grad():
        expanded.layer1.W[:, :64].copy_(base.layer1.W)
        expanded.layer1.b.copy_(base.layer1.b)
        expanded.layer2.load_state_dict(base.layer2.state_dict())
    x = torch.randn(3, 1, 64)
    ids = torch.tensor([0, 1, 0])
    assert torch.allclose(base(x, ids), expanded(torch.nn.functional.pad(x, (0, 16)), ids), atol=1e-7)
    with torch.no_grad():
        expanded.layer1.W.fill_(0.1)
        expanded.layer2.W.fill_(0.1)
    state = torch.ones(3, 1, 80, requires_grad=True)
    y = expanded(state, ids)
    y.sum().backward()
    assert torch.all(state.grad[..., :74] != 0)
    changed = state.detach().clone()
    changed[..., 73] += 0.25
    assert not torch.equal(expanded(changed, ids), y.detach())
    record["checks"].append({"name": "tiny_native_category_MLP_64to80_prefix_parity_and_state74_gradient", "passed": True,
        "scope": "Actual native small CPU state projector only, not full VLA or checkpoint load"})

    from peft import LoraConfig, get_peft_model
    lora_ns = extract(GROOT/"gr00t/utils/peft.py", ["_wrap_forward", "get_lora_model"],
                      {"torch": torch, "LoraConfig": LoraConfig, "get_peft_model": get_peft_model})

    class TinyNativeTopology(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"model_type": "r93_cpu_topology", "tie_word_embeddings": False}
            self.backbone = nn.Linear(8, 8)
            self.action_head = nn.Module()
            self.action_head.state_encoder = native_cls(2, 80, 8, 8)
            self.action_head.action_decoder = native_cls(2, 8, 8, 7)
            self.action_head.model = nn.Module()
            self.action_head.model.to_q = nn.Linear(8, 8)
            self.action_head.model.to_v = nn.Linear(8, 8)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return kwargs

    tiny = lora_ns["get_lora_model"](TinyNativeTopology(), rank=2)
    projector = [p for name, p in tiny.named_parameters() if "state_encoder" in name or "action_decoder" in name]
    assert projector and not any(p.requires_grad for p in projector)
    assert any(p.requires_grad for name, p in tiny.named_parameters() if "lora_" in name)
    record["checks"].append({"name": "actual_native_GR00T_LoRA_helper_freezes_nonattention_projectors", "passed": True,
                              "topology_only_not_full_model": True})
    record["hazards"].append("Stock LoRA helper targets attention q/k/v Linear modules, freezing state/action projectors. New robot/privileged inputs require explicitly trained and saved projectors, not unqualified stock LoRA.")

    for frame in (0, 1, 11):
        past, future, valid = temporal_indices(frame, 12, 7, 3)
        assert past.max() == frame and past.min() >= 0
        assert future.min() == frame and future.max() <= 11
        assert np.array_equal(valid, np.arange(frame, frame + 3) < 12)
    record["checks"].append({"name": "causal_history_and_real_future_label_mask", "passed": True})
    record["CUDA_initialized"] = torch.cuda.is_initialized()
    assert not record["CUDA_initialized"]
    record["passed"] = True
    record["new_model_SR"] = None
    record["source_hashes"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SEER/"utils/real_ft_data.py", SEER/"models/seer_model.py",
                     SEER/"real_controller/controller.py", GROOT/"gr00t/utils/peft.py",
                     GROOT/"gr00t/model/action_head/flow_matching_action_head.py", Path(__file__),
                     Path(__file__).with_name("contracts.py")]}
    (OUT/"CPU_AUDIT.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
