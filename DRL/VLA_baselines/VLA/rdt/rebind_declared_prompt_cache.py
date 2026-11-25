"""Reuse immutable official T5 public-grammar embeddings with explicit lineage.

No T5 recomputation/finetuning or simulator observation is performed. Verify
the OLD binding and complete declared grammar before binding a NEW source.
"""
import argparse
import json
import os
from pathlib import Path
from VLA.rdt.training_data import sha256, validate_language_embeddings


def verified_metadata(old, new_index, old_manifest_hash, old_embedding_hash):
    if (old.get("additional_prompts_scope") != "declared_task_vocabulary_no_observations"
            or old.get("encoder_weight_precision") != "bfloat16"
            or old.get("load_in_4bit") is not False or old.get("load_in_8bit") is not False):
        raise ValueError("Reuse requires the complete declared public grammar and unquantized official T5 cache")
    rows = json.loads(new_index.read_text())
    prompts = {row["task"]["prompt"] for row in rows}
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("Missing target-source prompts")
    return {**old, "source_index": str(new_index.resolve()), "source_index_sha256": sha256(new_index),
            "training_prompt_count": len(prompts), "reused_cached_embeddings_not_recomputed": True,
            "reused_cache_manifest_sha256": old_manifest_hash, "reused_embeddings_sha256": old_embedding_hash,
            "original_source_index": old["source_index"],
            "original_source_index_sha256": old["source_index_sha256"],
            "scope": "Text-only frozen public grammar cache reuse; NOT model training or dataset qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--new-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_path = args.cache.with_suffix(".manifest.json")
    old = json.loads(old_path.read_text())
    old_index = Path(old["source_index"])
    old_prompts = [row["task"]["prompt"] for row in json.loads(old_index.read_text())]
    _, verified = validate_language_embeddings(args.cache, old_path, old_index, old_prompts)
    manifest = verified_metadata(verified, args.new_index, sha256(old_path), sha256(args.cache))
    target_manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or target_manifest.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.link(args.cache, args.output)
    target_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    prompts = [row["task"]["prompt"] for row in json.loads(args.new_index.read_text())]
    values, _ = validate_language_embeddings(args.output, target_manifest, args.new_index, prompts)
    print(json.dumps({"verified": True, "cached_prompts": len(values), "target_source_prompts": len(set(prompts)),
                      "text_encoder_recomputed": False, "source_observations_read": False,
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
