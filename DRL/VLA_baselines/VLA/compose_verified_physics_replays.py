"""Reuse actual completed replays ONLY for byte-identical merged episodes.

This does not execute a new robot rollout. It verifies all parent payload
hashes, frozen physics, strict-success traces and ALL-ROWS merge lineage, then
renames the actual parent trace rows. No simulated metrics are synthesized.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_source_environment_runtime
from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint


def remap_rows(index, originals, reports, payloads):
    expected = Counter((source, filename) for source, rows in originals.items() for filename in rows)
    actual = Counter((row["merge_origin"]["source"], row["merge_origin"]["file"]) for row in index)
    if expected != actual:
        raise ValueError("Reuse requires ALL parent episodes exactly once, not a filtered subset")
    result = []
    for row in index:
        origin = row["merge_origin"]
        source, filename = origin["source"], origin["file"]
        original = originals[source][filename]
        digest = payloads[source][filename]["sha256"]
        if (origin["sha256"] != digest or row["task"] != original["task"]
                or row["steps"] != original["steps"] or row["success"] != original["success"]):
            raise ValueError("Reused payload or task labels changed")
        traces = [trace for trace in reports[source]["rows"] if trace["file"] == filename]
        if Counter(trace["mode"] for trace in traces) != {"canonical": 1, "native": 1}:
            raise ValueError("Missing or duplicated canonical/native physical replay row")
        for trace in traces:
            if (not trace.get("ever_strict_success") or trace.get("initial_strict_success")
                    or trace.get("stored_steps") != original["steps"]):
                raise ValueError("Failed, initially successful or mismatched actual replay trace")
            result.append({**trace, "file": row["file"], "reused_parent_source": source,
                           "reused_parent_file": filename})
    return result


def compose(source, paths, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    index = json.loads((source/"index.json").read_text())
    if not manifest.get("merge_components") or manifest.get("candidate_only") is not True:
        raise ValueError("Only immutable candidate ALL-ROWS merges are eligible")
    reports, originals, payloads, proofs = {}, {}, {}, []
    adapter = Path(__file__).parent/"pi0/adapter.py"
    for path in paths:
        path = Path(path).resolve()
        report = json.loads(path.read_text())
        parent = Path(report["source"]).resolve()
        key = str(parent)
        if (key in reports or not report.get("passed") or report.get("failures")
                or report.get("all_episodes") is not True or report.get("mode") != "both"
                or report.get("native_success_mode") != "strict" or report.get("native_actions_root") is not None
                or report.get("adapter_sha256") != sha256_file(adapter)
                or report["source_index_sha256"] != sha256_file(parent/"index.json")
                or report["source_manifest_sha256"] != sha256_file(parent/"dataset_manifest.json")):
            raise ValueError("Incomplete/stale/duplicate replay or unsupported external codec")
        verify_source_environment_runtime(report, parent)
        verify_source_payload_fingerprint(report["source_payload_fingerprint"], parent)
        if report["environment_runtime_fingerprint"] != manifest["environment_runtime_fingerprint"]:
            raise ValueError("Physics/controller sources differ from completed replay")
        parent_manifest = json.loads((parent/"dataset_manifest.json").read_text())
        if parent_manifest["observation_profile_sha256"] != manifest["observation_profile_sha256"]:
            raise ValueError("Camera/home/controller profile differs from replay")
        reports[key] = report
        originals[key] = {r["file"]: r for r in json.loads((parent/"index.json").read_text())}
        payloads[key] = report["source_payload_fingerprint"]["payloads"]
        proofs.append({"path": str(path), "sha256": sha256_file(path), "source": key,
                       "role": "completed_actual_full_canonical_and_native_physics_replay"})
    if set(reports) != {component["source"] for component in manifest["merge_components"]}:
        raise ValueError("Every merge component must have exactly one verified completed replay")
    merged_payloads = source_payload_fingerprint(source)
    for row in index:
        origin = row["merge_origin"]
        if merged_payloads["payloads"][row["file"]] != payloads[origin["source"]][origin["file"]]:
            raise ValueError("Merged NPZ is not byte-identical to physically replayed parent")
    rows = remap_rows(index, originals, reports, payloads)
    summaries = {}
    for mode in ("canonical", "native"):
        selected = [r for r in rows if r["mode"] == mode]
        summaries[mode] = {"episodes": len(selected), "successes": sum(bool(r["ever_strict_success"]) for r in selected),
                           "initial_false_positives": sum(bool(r["initial_strict_success"]) for r in selected)}
    exemplar = next(iter(reports.values()))
    result = {"passed": True, "source": str(source), "source_index_sha256": sha256_file(source/"index.json"),
        "source_manifest_sha256": sha256_file(source/"dataset_manifest.json"),
        "source_payload_fingerprint": merged_payloads, "environment_runtime_fingerprint": manifest["environment_runtime_fingerprint"],
        "environment_sha256": exemplar["environment_sha256"], "adapter_sha256": sha256_file(adapter),
        "mode": "both", "native_success_mode": "strict", "all_episodes": True,
        "scope": "SHA_verified_composition_of_actual_completed_replays_NOT_fresh_rollouts",
        "new_physics_rollouts_executed": False, "parent_replay_proofs": proofs,
        "native_actions_root": None, "codec_payload_fingerprint": None,
        "summaries": summaries, "failures": [], "rows": rows}
    for proof in proofs:
        if sha256_file(proof["path"]) != proof["sha256"]:
            raise ValueError("Parent replay proof changed during composition")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.source, args.parent_replay, args.output), indent=2))


if __name__ == "__main__":
    main()
