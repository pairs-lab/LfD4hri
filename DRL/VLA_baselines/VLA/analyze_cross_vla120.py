"""Audit all 120 actual standard-policy trials; signatures are not causes."""
from collections import Counter
import json
from pathlib import Path

from VLA.analyze_diagnostic50 import analyze_trial
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import SEEDS, TASKS


def analyze(summary_path, output):
    summary_path, output = Path(summary_path), Path(output)
    if output.exists():
        raise FileExistsError("Keep each measured analysis immutable")
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    if (summary["status"] != "completed120" or len(rows) != 120
            or {(r["seed"], r["task"]) for r in rows} != {(s, t) for s in SEEDS for t in TASKS}
            or any(r.get("error") or not isinstance(r["success"], bool) for r in rows)):
        raise ValueError("Every declared 30/action trial is required; incomplete is not 0%")
    cached, evidence, analyzed = {}, {}, []
    for row in rows:
        result_path, trace_path = Path(row["result"]), Path(row["trajectory"])
        if result_path not in cached:
            result = json.loads(result_path.read_text())
            manifest = json.loads((result_path.parent / "manifest.json").read_text())
            if (result["model"] != summary["model"] or result["integration_error"] is not None
                    or manifest["privileged_object_pose_given_to_vla"] is not False
                    or manifest["oracle_fallback"] is not False):
                raise ValueError("Result/model/input regime differs")
            cached[result_path] = (result, json.loads(trace_path.read_text()))
            for file in (result_path, trace_path, result_path.parent / "manifest.json"):
                evidence[str(file)] = sha256_file(file)
        result, traces = cached[result_path]
        task = next(t for t in result["tasks"] if t["name"] == row["task"])
        if result["seed"] != row["seed"] or task["success"] != row["success"] or task.get("error"):
            raise ValueError("Summary disagrees with actual physics trace")
        analyzed.append({"seed": row["seed"], **analyze_trial(task, traces[row["task"]]),
            "trajectory": str(trace_path), "snapshots": str(result_path.parent)})
    flags = dict(Counter(f for r in analyzed for f in r["failure_signatures"]))
    report = {"model": summary["model"], "adaptation_regime": summary["adaptation_regime"],
        "completed_trials": 120, "successes": sum(r["success"] for r in analyzed),
        "statistics": success_statistics(analyzed), "failure_signatures_overlap": flags,
        "rows": analyzed, "evidence": evidence, "summary_sha256": sha256_file(summary_path),
        "videos": sorted({r["video"] for r in rows if r.get("video")}),
        "all_video_frames_visually_reviewed": False, "root_causes_not_proven_by_signatures_alone": True,
        "no_privileged_policy_input": True, "oracle_fallback": False, "final_test_used": False,
        "scoring_or_controller_modified": False,
        "limitations": ["DEVELOPMENT seeds, not untouched paper-test generalization.",
            "Closed-empty recovery data do not certify wrong-object or dropped-object recovery.",
            "Official zero-shot rows are not matched fine-tuned baselines."]}
    output.mkdir(parents=True)
    atomic_json(output / "analysis.json", report)
    lines = [f"# {summary['model']}: actual120-trial failure audit", "",
        f"Success: {report['successes']}/120 ({100*report['successes']/120:.2f}%).", "",
        f"Regime: {summary['adaptation_regime']}. Safety aborts remain failures; no rerolls.", "",
        "Overlapping signatures (not established root causes):", "",
        "| Signature | Trials |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in flags.items()]
    lines += ["", "Actual traces and raw camera snapshots are linked in analysis.json.",
        "All video frames have NOT been manually certified; reserved final seeds remain unused.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return report
