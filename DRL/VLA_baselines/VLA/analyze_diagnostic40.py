"""Complete40 development error analysis; no metric changes or outcome filter."""
import argparse
from collections import Counter
import json
from pathlib import Path

from VLA.analyze_diagnostic50 import analyze_trial
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.provenance import sha256_file
from VLA.run_diagnostic40 import SEEDS, TASKS


def analyze(progress_path, output):
    progress_path, output = Path(progress_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Keep immutable complete40 analysis")
    p = json.loads(progress_path.read_text())
    rows = p["rows"]
    if (p["status"] != "completed40_development_trials" or len(rows) != 40
            or {(r["seed"], r["task"]) for r in rows} != {(s, t) for s in SEEDS for t in TASKS}):
        raise ValueError("All fixed10/action trials required, including every failure")
    by_path, reports, evidence = {}, [], {}
    for row in rows:
        path = Path(row["result"])
        if path not in by_path:
            by_path[path] = (json.loads(path.read_text()), json.loads(Path(row["trajectory"]).read_text()))
            for f in (path, Path(row["trajectory"]), path.parent / "manifest.json"):
                evidence[str(f)] = sha256_file(f)
                if p["result_hashes"].get(str(f)) != evidence[str(f)]:
                    raise ValueError("Evaluated result/trace changed")
        result, traces = by_path[path]
        task = next(t for t in result["tasks"] if t["name"] == row["task"])
        if task.get("error") or task["success"] != row["success"]:
            raise ValueError("Integration failure or corrupted progress is not task failure")
        reports.append({"seed": row["seed"], **analyze_trial(task, traces[row["task"]])})
    counts = Counter(flag for r in reports for flag in r["failure_signatures"])
    report = {"scope": "40_trial_DEVELOPMENT_error_analysis_NOT_paper_test", "complete40": True,
        "checkpoint_content_sha256": p["checkpoint_content_sha256"], "completed_trials": 40,
        "successes": sum(r["success"] for r in reports), "statistics": success_statistics(reports),
        "failure_signatures_overlap": dict(counts), "rows": reports, "evidence": evidence,
        "progress_sha256": sha256_file(progress_path), "no_privileged_model_input": True,
        "final_test_used": False, "root_causes_not_proven_by_signatures_alone": True,
        "observed_failures_requiring_postfit_recheck": [name for name in (
            "low_approach_offset_over5cm", "no_close_request_in_grasp_task", "safety_abort",
            "close_and_lift_without_hold_after_step", "held_wrong_object") if counts[name]],
        "repairs_applied_before_new_fit": [
            "Recollected unsafe/fragile/occluded demonstration cases with original scenes and native strict replay.",
            "Enlarged training source to independent multi-object layouts, target-text pairs and physically measured closed-empty/near-miss OPEN recovery labels.",
            "Fixed standalone recovery versus counterfactual-pair readiness classification; preserved physical/text/feedback checks.",
            "Recomputed exact TRAIN-only normalization and train/inference parity; fresh base language/visual connector adaptation, FP32 optimizer, reset sampling0.1."],
        "limitations": ["No oracle action override or false-grasp suppression based on hidden state.",
            "Wrong-object hold/drop recovery is not certified by closed-empty resets.",
            "Repeated development-cycle scores are not untouched generalization scores.",
            "No claimed high/stable success until new learned closed-loop trials exist."]}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"successes": report["successes"], "trials": 40, "failure_signatures_overlap": dict(counts)}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.progress, args.output)


if __name__ == "__main__":
    main()
