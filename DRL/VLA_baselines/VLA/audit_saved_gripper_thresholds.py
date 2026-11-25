"""Re-score existing predictions using the controller's real hysteresis."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.provenance import sha256_file
from VLA.smolvla.audit_checkpoint_predictions import gripper_diagnostic_fields, summarize


def rescore_rows(report):
    """Use actuator request history, NOT measured aperture, for hysteresis.

    A partially closed hand holding an object can still have CLOSE active.
    At recovery t=0 the active request comes from startup actions. Neither
    state can be recovered by thresholding measured FK closure at 0.5.
    """
    source = Path(report["source"])
    index_path, manifest_path = source / "index.json", source / "dataset_manifest.json"
    frozen = {path: sha256_file(path) for path in (index_path, manifest_path)}
    index = json.loads(index_path.read_text())
    by_file = {item["file"]: item for item in index}
    if len(by_file) != len(index):
        raise ValueError("Ambiguous source episode identity")
    manifest = json.loads(manifest_path.read_text())
    body_v9 = (manifest.get("observation_profile") or {}).get("gripper_controller") == "body_contact_v9"
    if report.get("source_index_sha256", frozen[index_path]) != frozen[index_path]:
        raise ValueError("Saved prediction source index changed")
    cache, rows = {}, []
    for row in report["rows"]:
        name = row["episode"]
        if name not in by_file or (source / name).resolve().parent != source.resolve():
            raise ValueError("Prediction episode absent from exact source")
        if name not in cache:
            with np.load(source / name, allow_pickle=False) as data:
                keys = ["gripper_closure", "next_grasped_object"]
                if body_v9:
                    keys.append("next_requested_gripper_closure")
                cache[name] = {key: data[key].copy() for key in keys}
        frame = row["frame"]
        if type(frame) is not int or not 0 <= frame < len(cache[name]["gripper_closure"]):
            raise ValueError("Prediction frame outside source episode")
        fields = gripper_diagnostic_fields(cache[name], by_file[name], frame, body_v9=body_v9)
        rows.append({**row, **fields})
    if any(sha256_file(path) != digest for path, digest in frozen.items()):
        raise RuntimeError("Source metadata changed during rescoring")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    rows = rescore_rows(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "Re-scoring saved imitation predictions, NOT new rollout success",
        "prediction_report_sha256": sha256_file(args.report), "overall": summarize(rows),
        "by_task": {task: summarize([r for r in rows if r["task"] == task]) for task in {r["task"] for r in rows}},
        "original_report_modified": False, "final_test_used": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
