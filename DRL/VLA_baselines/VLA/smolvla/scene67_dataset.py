"""SmolVLA native67 overlay with a real batched-fetch implementation."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch
from VLA.scene64_dataset import load_scene_projection
from VLA.scene64_conditioning import NUMERIC_NAMES, encode_numeric, normalization
from VLA.common.provenance import sha256_file


class Scene67Dataset(torch.utils.data.Dataset):
    def __init__(self, base, projection, recovery_admission=None):
        self.base = base
        self.raw64, self.audit, rows = load_scene_projection(projection)
        if len(base) != len(self.raw64): raise ValueError("Native source and full projection counts differ")
        self.original_length = len(base); self.corrections = []; self.correction_ends = []
        self.recovery_contract = None
        self.parent_rows = rows
        self.parent_offsets = {}; offset = 0
        for row in rows: self.parent_offsets[row["file"]] = offset; offset += row["steps"]
        if recovery_admission is not None:
            path = Path(recovery_admission); proof = json.loads(path.read_text())
            if not proof.get("passed") or "smolvla" not in proof["allowed_models"] or proof.get("native_joint_training_qualified") is not False:
                raise ValueError("Own exact Cartesian correction admission required")
            for name, digest in proof["dependencies"].items():
                if sha256_file(name) != digest: raise ValueError("Corrective admission dependency changed")
            root = Path(proof["candidate"]); index = json.loads((root / "index.json").read_text())
            by_file = {r["file"]: r for r in index if r.get("file")}
            parts = [self.raw64]; total = 0
            for item in proof["rows"]:
                row = by_file[item["file"]]; file = root / item["file"]
                if sha256_file(file) != item["payload_sha256"]: raise ValueError("Corrective payload changed")
                with np.load(file, allow_pickle=False) as data: states = data["audit_pre_action_PRIVILEGED_state64"].copy()
                if row["source_file"] not in self.parent_offsets: raise ValueError("Not an original TRAIN scene")
                original = next(r for r in rows if r["file"] == row["source_file"])
                if row["task"] != original["task"]: raise ValueError("On-policy correction changed requested task/scene")
                self.corrections.append({**row, "path": str(file)})
                parts.append(states); total += len(states); self.correction_ends.append(total)
            if total != proof["frames"]: raise ValueError("Incomplete corrective coverage")
            self.raw64 = np.concatenate(parts)
            self.recovery_contract = {"admission": str(path.resolve()), "admission_sha256": sha256_file(path),
                "episodes": len(self.corrections), "frames": total, "all40_attempts_preserved": True}
        self.states = np.asarray([encode_numeric(v) for v in self.raw64], np.float32)
        stats, self.normalization_contract = normalization(self.raw64)
        if self.corrections:
            reference = json.loads((Path(projection) / "stats.json").read_text())
            for name in ("mean", "std", "min", "max"): stats[name][:18] = reference[name][:18]
            self.normalization_contract.update(first18_empirical_preserved=False,
                first18_reference_checkpoint_preserved=True, first18_reference_stats_sha256=sha256_file(Path(projection)/"stats.json"),
                corrective_admission=self.recovery_contract,
                action_normalization_reference="original qualified TRAIN dataset, not validation/test")
        self.meta = deepcopy(base.meta)
        self.meta.info.features["observation.state"] = {"dtype": "float32", "shape": [67], "names": NUMERIC_NAMES}
        self.meta.stats["observation.state"] = {k: np.asarray(v, np.int64 if k == "count" else np.float32) for k, v in stats.items()}

    def __len__(self): return len(self.states)
    def __getattr__(self, name):
        if name == "base": raise AttributeError(name)
        return getattr(self.base, name)
    @property
    def features(self): return self.meta.features
    def __getitem__(self, index):
        if index >= self.original_length: return self._correction(index)
        sample = self.base[index]
        if int(sample["index"]) != index: raise ValueError("Native source frame remapping forbidden")
        if not torch.allclose(sample["observation.state"].reshape(-1), torch.from_numpy(self.states[index, :7]), atol=1e-6, rtol=0):
            raise ValueError("Original measured joint/closure observation differs")
        shape = sample["observation.state"].shape
        return {**sample, "observation.state": torch.from_numpy(self.states[index].copy()).reshape(*shape[:-1], 67)}
    def __getitems__(self, indices): return [self[int(i)] for i in indices]

    @lru_cache(maxsize=1)
    def _correction_payload(self, episode):
        row = self.corrections[episode]
        with np.load(row["path"], allow_pickle=False) as data:
            values = {k: data[k] for k in ("front_rgb", "wrist_rgb", "canonical_actions", "prompt")}
        template = self.base[self.parent_offsets[row["source_file"]]]
        if template["task"] != str(values["prompt"].item()): raise ValueError("Native task index/prompt differs")
        return values, template

    def _correction(self, index):
        from bisect import bisect_right
        local = index-self.original_length
        if local < 0 or local >= self.correction_ends[-1]: raise IndexError(index)
        ei = bisect_right(self.correction_ends, local); fi = local-(self.correction_ends[ei-1] if ei else 0)
        data, template = self._correction_payload(ei); sample = dict(template)
        # Reuse the REAL temporal action shape and camera names, not a new
        # action format or an unchecked first-row-only loader approximation.
        horizon = template["action"].shape[-2]
        ids = fi+np.arange(horizon); n = len(data["canonical_actions"])
        sample["action"] = torch.from_numpy(data["canonical_actions"][np.minimum(ids,n-1)].copy())
        if "action_is_pad" in sample: sample["action_is_pad"] = torch.from_numpy(ids >= n)
        for camera in self.meta.camera_keys:
            role = camera.rsplit(".",1)[1]
            if role not in ("front", "wrist"): raise ValueError("Unqualified native camera slot")
            image = torch.from_numpy(data[role+"_rgb"][fi].copy()).permute(2,0,1).float().div_(255)
            # Native observation_delta_indices=[0] retains a singleton TIME
            # axis even for the current image. Keep pixels unchanged; do not
            # resize/crop or silently broadcast across historical frames.
            if template[camera].shape == (1, *image.shape): image = image.unsqueeze(0)
            if image.shape != template[camera].shape: raise ValueError("Native image transform shape differs")
            sample[camera] = image
            if camera+"_is_pad" in sample: sample[camera+"_is_pad"] = torch.zeros_like(template[camera+"_is_pad"],dtype=torch.bool)
        sample["observation.state"] = torch.from_numpy(self.states[index].copy()).reshape(*template["observation.state"].shape[:-1],67)
        if "observation.state_is_pad" in sample: sample["observation.state_is_pad"] = torch.zeros_like(template["observation.state_is_pad"],dtype=torch.bool)
        for name, value in (("index",index), ("frame_index",fi), ("episode_index", len(self.parent_rows)+ei), ("timestamp",fi/10)):
            if name in sample: sample[name] = torch.as_tensor(value,dtype=template[name].dtype)
        return sample
