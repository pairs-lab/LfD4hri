"""RDT-specific report from completed learned-only privileged120 traces."""
import argparse
from collections import Counter
import json
from pathlib import Path

from VLA.analyze_simstate120 import audit_trace
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve error audit")
    progress = json.loads(args.progress.read_text()); rows = progress["rows"]
    if (progress["model"] != "rdt_simstate" or progress["status"] != "completed120_privileged_input_VLA"
            or len(rows) != 120 or len({(r["seed"], r["task"]) for r in rows}) != 120
            or Counter(r["task"] for r in rows) != {t: 30 for t in ("touch","pick","move","pick_place")}):
        raise ValueError("Do not substitute an incomplete cohort or another model")
    traces = {}; reports = []; evidence = {}
    for row in rows:
        path = Path(row["trajectory"])
        if path not in traces:
            traces[path] = json.loads(path.read_text()); evidence[str(path)] = sha256_file(path)
        reports.append(audit_trace(row, traces[path][row["task"]]))
    successes = sum(r["success"] for r in reports)
    stats = success_statistics(rows)
    args.output.mkdir(parents=True)
    atomic_json(args.output/"analysis.json", {"model": "RDT+SimState", "complete120": True,
        "trials": 120, "successes": successes, "success_rate": successes/120, "statistics": stats,
        "privileged_model_inputs": True, "NOT_plain_VLA_or_autonomous_visual_grounding": True,
        "rows": reports, "trace_evidence": evidence,
        "failure_signatures_overlap": dict(Counter(f for r in reports for f in r["failure_signatures"])),
        "root_cause_not_established_by_signatures": True, "all_video_frames_visually_reviewed": False,
        "development_seeds_not_final_paper_test": True, "stable_high_success_established": False})
    lines = ["# RDT+SimState: completed learned development120", "",
        "Privileged pre-action simulator inputs; NOT plain RDT or an oracle score.", "",
        f"Actual successes: {successes}/120 ({100*successes/120:.1f}%). Safety aborts remain failures.", ""]
    for task, value in stats["by_task"].items():
        lines.append(f"- {task}: {value['successes']}/{value['trials']}")
    lines.extend(["", "Failure signatures do not establish intrinsic model weakness. Videos and held-out audits still require review."])
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
