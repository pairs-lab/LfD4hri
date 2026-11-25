"""Compare the real RDT training dataset/collator and HTTP sensor preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor

from VLA.common.provenance import sha256_file
from VLA.rdt.adapter import RDTPolicy
from VLA.rdt.audit_checkpoint_predictions import observation_at
from VLA.rdt.server import prepare_sensor_tensors
from VLA.rdt.training_data import PyBulletRDTDataset, RDTBatchCollator, validate_lineage, validate_language_embeddings


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--action-contract-report", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    lineage = validate_lineage(source, args.audit_report, args.action_contract_report)
    dataset = PyBulletRDTDataset(source, action_horizon=10, samples_per_episode=4, seed=20261007)
    embeddings, _ = validate_language_embeddings(args.embeddings, args.embeddings.with_suffix(".manifest.json"),
                                                 source / "index.json", dataset.prompts)
    processor = AutoImageProcessor.from_pretrained(args.vision)
    collator = RDTBatchCollator(processor, embeddings)
    adapter = RDTPolicy(env=None, seed=20261007)
    errors = {key: 0. for key in ("images", "states", "state_elem_mask", "lang_tokens", "lang_attn_mask", "ctrl_freqs")}
    rows = []
    for task in ("touch", "pick", "move", "pick_place"):
        numbers = [i for i, item in enumerate(dataset.index) if item["task"]["name"] == task]
        selected = [numbers[int(j)] for j in np.linspace(0, len(numbers) - 1, 8, dtype=int)]
        for number in selected:
            samples = dataset[number]
            train = collator([samples])
            with np.load(source / samples.episode_file, allow_pickle=False) as episode:
                for sample_id, frame in enumerate(samples.step_ids):
                    current = observation_at(episode, frame)
                    previous = observation_at(episode, max(0, frame - 1))
                    payload = adapter.build_sensor_request(current, previous)
                    infer = prepare_sensor_tensors(payload, processor, embeddings, target_domain=True)
                    for key in errors:
                        actual, expected = infer[key], train[key][sample_id:sample_id + 1]
                        if actual.shape != expected.shape:
                            raise RuntimeError("Different train/infer shape: " + key)
                        difference = float((actual.float() - expected.float()).abs().max())
                        errors[key] = max(errors[key], difference)
                    rows.append({"episode": samples.episode_file, "frame": int(frame)})
            print(f"{samples.episode_file}: checked {len(rows)} observations", flush=True)
    report = {"passed": all(value == 0 for value in errors.values()), "samples": len(rows), "errors": errors,
              "source": str(source), "source_index_sha256": lineage["source_index_sha256"],
              "server_sha256": sha256_file(Path(__file__).with_name("server.py")),
              "adapter_sha256": sha256_file(Path(__file__).with_name("adapter.py")),
              "training_data_sha256": sha256_file(Path(__file__).with_name("training_data.py")),
              "methods": ["actual PyBulletRDTDataset/RDTBatchCollator", "actual deployed sensor request", "actual server prepare_sensor_tensors"],
              "dtype_note": "CPU prepared tensors compared exactly before the shared BF16 vision/model compute conversion",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if not report["passed"]:
        raise RuntimeError("RDT train/infer sensor preparation mismatch")


if __name__ == "__main__":
    main()
