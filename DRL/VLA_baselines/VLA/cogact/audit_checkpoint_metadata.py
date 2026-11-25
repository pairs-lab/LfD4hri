"""Inspect official Torch ZIP tensor metadata without downloading tensor payloads.

Only HTTP byte ranges and a strict metadata-only pickle whitelist are allowed.
This does not instantiate CogACT or claim that its inference fits in VRAM.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import io
import json
import math
from pathlib import Path
import pickle
import subprocess
import zipfile

import requests
from huggingface_hub import HfApi, hf_hub_url


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid ZIP seek")
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        if size > 2 * 1024 * 1024:
            raise ValueError("Refusing an unbounded or tensor-sized network read")
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        start, end = self.position, self.position + size - 1
        with requests.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                          stream=True, timeout=30) as response:
            response.raise_for_status()
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("Server ignored/mismatched range; full download refused")
            data = response.raw.read(size)
        if len(data) != size:
            raise IOError("Truncated checkpoint metadata range")
        self.position += size
        self.bytes_read += size
        return data


class TensorMetadata:
    def __init__(self, storage, offset, shape, stride, *args):
        del offset, stride, args
        self.storage, self.shape = storage, tuple(shape)


def rebuild_parameter(value, *args):
    del args
    return value


def rebuild_tensor_subclass(func, new_type, args, state):
    del new_type, state
    if func not in (TensorMetadata, rebuild_parameter):
        raise pickle.UnpicklingError("Unsupported tensor reconstruction function")
    return func(*args)


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return TensorMetadata
        if module == "torch._utils" and name == "_rebuild_parameter":
            return rebuild_parameter
        if (module, name) == ("torch._tensor", "_rebuild_from_type_v2"):
            return rebuild_tensor_subclass
        if (module, name) in {("torch", "Tensor"), ("torch.nn.parameter", "Parameter")}:
            return TensorMetadata
        if module == "torch" and name in {"FloatStorage", "HalfStorage", "BFloat16Storage", "LongStorage", "IntStorage", "BoolStorage", "ByteStorage", "DoubleStorage"}:
            return name
        raise pickle.UnpicklingError(f"Non-metadata pickle global refused: {module}.{name}")

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage":
            raise pickle.UnpicklingError("Unsupported persistent object")
        _, dtype, key, location, count = value
        return {"dtype": dtype, "key": str(key), "location": location, "count": int(count)}


def summarize(value):
    tensors = []
    def visit(node):
        if isinstance(node, TensorMetadata):
            tensors.append(node)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)
        elif isinstance(node, (tuple, list)):
            for child in node:
                visit(child)
    visit(value)
    counts = {}
    storages = {}
    for tensor in tensors:
        dtype = tensor.storage["dtype"]
        counts[dtype] = counts.get(dtype, 0) + math.prod(tensor.shape)
        storages[tensor.storage["key"]] = tensor.storage
    return {"tensor_count": len(tensors), "logical_elements_by_storage_dtype": counts,
            "unique_storage_elements": sum(item["count"] for item in storages.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="CogACT/CogACT-Small")
    parser.add_argument("--revision", default="ca302a300542393a0b06e291465076a7d9105c23")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    item, = [item for item in info.siblings if item.rfilename.endswith(".pt")]
    remote = HTTPRangeReader(hf_hub_url(args.repo, item.rfilename, revision=info.sha), item.size)
    cache = args.output.parent / ("cogact_" + info.sha + "_pickle_metadata.bin")
    if cache.exists():
        metadata = cache.read_bytes()
    else:
        with zipfile.ZipFile(remote) as archive:
            metadata_name, = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            metadata = archive.read(metadata_name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(metadata)
    state = MetadataUnpickler(io.BytesIO(metadata)).load()
    model = state["model"]
    report = {
        "repo": args.repo, "revision": info.sha, "checkpoint_file": item.rfilename,
        "checkpoint_bytes": item.size, "network_bytes_read": remote.bytes_read,
        "metadata_cache": str(cache.resolve()),
        "checkpoint_is_torch_zip": True,
        "top_level_keys": list(state), "model_components": {key: summarize(value) for key, value in model.items()},
        "overall": summarize(model),
        "current_gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], text=True).strip(),
        "scope": "Tensor metadata only; model load, quantization, inference and fine-tuning NOT validated",
        "next_step": "Validate mmap/streamed conversion plus LLM quantization in an isolated CogACT port; do not score before precision/action equivalence checks.",
    }
    report["logical_weights_if_all_bf16_mib"] = sum(report["overall"]["logical_elements_by_storage_dtype"].values()) * 2 / 1024**2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
