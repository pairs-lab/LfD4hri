"""Rescore recorded prefixes, NOT counterfactual full neural rollouts.

The old evaluator stops on its first hit. Tightening its metric censors the
unobserved continuation. This module never fills that continuation with fake
actions/outcomes, alters the frozen protocol, or claims five measured seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import betabinom

from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "VLA/audits/dita_terminal800_r95_20261023"
OUTPUT = ROOT / "VLA/audits/dita_minus1cm_sensitivity_r96_20261023"
TASKS = ("touch", "pick", "move", "pick_place")
EXPECTED_COUNTS = dict(zip(TASKS, (164, 165, 168, 160)))
GEOMETRY_EPS = 1e-7  # Conservative envelope for float32 AABB extent rounding.


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduced_thresholds(old):
    if len(old) != 8 or any(not name.endswith("_m") for name in old):
        raise ValueError("Only the eight distance thresholds may be reduced")
    result = {name: round(float(value) - .01, 12) for name, value in old.items()}
    if any(value <= 0 for value in result.values()):
        raise ValueError("Subtraction produced a nonpositive distance")
    return result


def geometry(task, step, initial_bottom):
    after = step["after"]
    raw = step["post_action_scene64"]
    center = after["object_center_xyz_m"]
    com, goal = after["object_base_xyz_m"], after["goal_xyz_m"]
    top = [center[0], center[1], center[2] + .5 * raw[26]]
    return {
        "distance": math.dist(after["pad_grasp_center_xyz_m"], top),
        "lift": center[2] - .5 * raw[26] - initial_bottom,
        # goal = COM + (reward_target - AABB_bottom). Thus the AABB term
        # cancels, leaving bottom - place_reference = COM - goal + offset.
        "xy": math.dist(com[:2], goal[:2]),
        "vertical": com[2] - goal[2] + (.15 if task == "move" else .015),
        "bilateral_correct_hold": bool(after["correct_object_held"] and raw[28] and raw[29]),
        "pad_contact": bool(raw[28] or raw[29]),
    }


def predicate(task, values, thresholds, any_gripper_contact=False):
    if task == "touch":
        return bool(any_gripper_contact or values["distance"] <= thresholds["touch_distance_m"])
    if task == "pick":
        return bool(values["bilateral_correct_hold"] and values["lift"] >= thresholds["pick_min_lift_m"])
    return bool(values["xy"] <= thresholds[task + "_xy_radius_m"]
        and -thresholds[task + "_vertical_below_m"] <= values["vertical"]
        <= thresholds[task + "_vertical_above_m"])


def forecast(k, n=200, draws=500_000, rng_seed=20261023):
    """One observed + four unrun cohorts; same frozen checkpoint.

    A single latent p per scenario preserves posterior parameter uncertainty
    shared by future cohorts. No synthetic seed outcomes are saved as data.
    """
    a, b = k + .5, n - k + .5
    mu, variance = a / (a+b), a*b / ((a+b)**2 * (a+b+1))
    epq = a*b / ((a+b) * (a+b+1))
    expected_mean_pct = 100 * (k + 4*n*mu) / (5*n)
    expected_sample_variance_pct2 = (100/n)**2 * (
        .8*n*epq + .2*n*n*variance + .2*(k-n*mu)**2)
    rng = np.random.default_rng(rng_seed)
    p = rng.beta(a, b, size=draws)
    future = rng.binomial(n, p[:, None], size=(draws, 4))
    samples = np.column_stack((np.full(draws, k), future)) * (100/n)
    std = samples.std(axis=1, ddof=1)
    if not np.isclose(float(np.mean(std**2)), expected_sample_variance_pct2, rtol=.012, atol=.01):
        raise ValueError("Predictive sample variance does not match its analytic moment")
    exact_q = betabinom.ppf([.025, .975], 4*n, a, b)
    return {
        "observed_successes": k, "observed_n": n,
        "expected_five_cohort_mean_percent": expected_mean_pct,
        "median_predictive_sample_std_percentage_points": float(np.median(std)),
        "predictive_mean_95_percent": ((k+exact_q)/(5*n)*100).tolist(),
        "predictive_std_95_percentage_points": np.quantile(std, [.025, .975]).tolist(),
        "expected_sample_variance_percentage_points_squared": expected_sample_variance_pct2,
        "monte_carlo_mean_sample_variance": float(np.mean(std**2)),
        "posterior": {"a": a, "b": b}, "statistical_draws_NOT_robot_episodes": draws,
        "statistical_rng_seed_NOT_robot_seed": rng_seed,
        "prior_sensitivity_uniform_expected_mean_percent": 100*(k+4*n*(k+1)/(n+2))/(5*n),
        "same_checkpoint_not_training_seed_variance": True,
        "five_measured_cohorts": False,
    }


def replay_touch(seed, trace, protocol, profile):
    """Recover missing ANY-link contact; require exact logged state parity."""
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.tasks import select_tasks, randomize_rl_scene
    from VLA.scene64_conditioning import model_input

    rng = np.random.default_rng(seed)
    tasks = [randomize_rl_scene(t, rng) for t in select_tasks("all")]
    if tasks[0].name != "touch":
        raise ValueError("Touch is no longer the first task; prior reset history is required")
    env = environment_class_for_profile(profile)(observation_profile=profile,
        robot="ur5e", image_size=384, control_hz=10, assist_grasp=True, seed=seed,
        required_success_steps=1, use_semantic_success=True,
        success_threshold_scale=protocol["success"]["reward_xml_threshold_scale"],
        semantic_success_thresholds=protocol["success"]["semantic_thresholds"],
        render_observations=False)
    result = []
    try:
        obs = env.reset(tasks[0])
        for entry in trace:
            pre = model_input(env, obs)["scene64"]
            if not np.array_equal(pre, np.asarray(entry["pre_action_scene64"], np.float32)):
                raise ValueError(f"Replay diverged BEFORE action: {seed}/{entry['step']}")
            obs, _, done, info = env.step(np.asarray(entry["action"], np.float32))
            post = model_input(env, obs)["scene64"]
            if not np.array_equal(post, np.asarray(entry["post_action_scene64"], np.float32)):
                raise ValueError(f"Replay diverged AFTER action: {seed}/{entry['step']}")
            if bool(info["success"]) != bool(entry["success"]) or info.get("safety_violation") != entry["safety_violation"]:
                raise ValueError("Replay outcome/safety diverged")
            if done != (entry is trace[-1]):
                raise ValueError("Replay stop timing differs")
            result.append({"step": entry["step"],
                "any_gripper_contact": bool(env.gripper.is_touching(env.pick_id)),
                "exact_touch_distance_m": env.ee_object_distance()})
    finally:
        env.close()
    return {"seed": seed, "steps": len(trace), "all64_pre_and_post_bitwise_equal": True,
        "original_stop_and_outcome_equal": True, "policy_queries": 0,
        "new_robot_trials": 0, "contact_log": result}


def main(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refuse to overwrite a prior sensitivity analysis")
    output.mkdir(parents=True)
    started = time.monotonic()
    final = SOURCE / "attempt00/final800"
    progress, plan = read(final/"progress.json"), read(final/"PLAN.json")
    certificate = read(SOURCE/"CERTIFIED_RESULT.json")
    if not certificate["audit_passed"] or progress["completed_trials"] != 800:
        raise ValueError("Source benchmark is not certified complete")
    protocol = read(plan["protocol"])
    old = plan["actual_primary_success"]["semantic_thresholds"]
    if old != protocol["success"]["semantic_thresholds"]:
        raise ValueError("Frozen source and current protocol differ")
    profile = read(protocol["observation_profile_file"])
    from VLA.common.observation_profile import profile_hash
    if profile_hash(profile) != protocol["observation_profile_sha256"]:
        raise ValueError("Source profile changed")
    new = reduced_thresholds(old)
    sources = {str(path): sha(path) for path in (final/"progress.json", final/"PLAN.json",
        SOURCE/"CERTIFIED_RESULT.json", SOURCE/"FROZEN_METHOD.json", Path(plan["protocol"]),
        Path(protocol["observation_profile_file"]))}
    traces, rows, replay_seeds, margins = {}, [], set(), []
    for seed in sorted({r["seed"] for r in progress["rows"]}):
        path = final/f"seed{seed}"/"trajectory.json"
        sources[str(path)] = sha(path)
        traces[seed] = read(path)
    if len(traces) != 200 or len({(r["seed"], r["task"]) for r in progress["rows"]}) != 800:
        raise ValueError("Expected 200 unique scenes and 800 unique task outcomes")
    for row in progress["rows"]:
        trace = traces[row["seed"]][row["task"]]
        if len(trace) != row["steps"] or bool(trace[-1]["success"]) != row["success"]:
            raise ValueError("Row does not agree with saved trajectory")
        initial = trace[0]["before"]["object_center_xyz_m"][2] - .5*trace[0]["pre_action_scene64"][26]
        for step in trace:
            if step["safety_violation"]:
                continue
            values = geometry(row["task"], step, initial)
            if row["task"] == "touch" and step["success"] and not values["pad_contact"]:
                # Pad flags do not capture all links or weak contacts. Replay
                # every case in the tolerance annulus; do not assume no contact.
                if new["touch_distance_m"]-GEOMETRY_EPS <= values["distance"] <= old["touch_distance_m"]+GEOMETRY_EPS:
                    replay_seeds.add(row["seed"])
    # Include one known physical-contact case as an independent replay check.
    replay_seeds.add(min(traces))
    replays = {}
    atomic_json(output/"PLAN.json", {"old_thresholds_m": old, "minus1cm_thresholds_m": new,
        "literal_pick_5_to_4_cm_easier": True,
        "new_closed_loop_evaluation": False, "unobserved_continuation_not_imputed": True,
        "contact_ambiguities_plus_positive_control_to_replay": sorted(replay_seeds),
        "replay_uses_original_actions_and_thresholds_not_model_or_Oracle": True,
        "no_frozen_protocol_or_checkpoint_changes": True})
    for seed in sorted(replay_seeds):
        result = replay_touch(seed, traces[seed]["touch"], protocol, profile)
        replays[seed] = result
        atomic_json(output/"touch_contact_replay"/f"seed{seed}.json", result)
        atomic_json(output/"status.json", {"stage": "verifying_missing_contact_with_exact_action_replay",
            "completed_replays": len(replays), "planned_replays": len(replay_seeds),
            "elapsed_s": time.monotonic()-started, "new_neural_episodes": 0})
        print(f"contact replay {len(replays)}/{len(replay_seeds)} seed={seed}", flush=True)
    for row in progress["rows"]:
        seed, task = row["seed"], row["task"]
        trace = traces[seed][task]
        initial = trace[0]["before"]["object_center_xyz_m"][2] - .5*trace[0]["pre_action_scene64"][26]
        first_hit, old_hits = None, []
        for index, entry in enumerate(trace):
            if entry["safety_violation"]:
                if entry["success"]:
                    raise ValueError("Safety-aborted step cannot be a hit")
                continue
            values = geometry(task, entry, initial)
            if task == "touch":
                if seed in replays:
                    contact = replays[seed]["contact_log"][index]["any_gripper_contact"]
                    values["distance"] = replays[seed]["contact_log"][index]["exact_touch_distance_m"]
                elif values["pad_contact"]:
                    contact = True
                elif values["distance"] > old["touch_distance_m"]+GEOMETRY_EPS:
                    # A logged success outside the old distance tolerance
                    # proves the other branch (any-link contact) was true.
                    contact = bool(entry["success"])
                elif values["distance"] < new["touch_distance_m"]-GEOMETRY_EPS:
                    contact = False  # Unknown contact cannot change either predicate.
                elif not entry["success"]:
                    contact = False  # Old predicate false proves no contact.
                else:
                    raise ValueError("Unresolved any-link contact; refuse a point estimate")
            else:
                contact = False
            old_hit = predicate(task, values, old, contact)
            new_hit = predicate(task, values, new, contact)
            if old_hit != bool(entry["success"]):
                raise ValueError(f"Old predicate does not reproduce source: {seed}/{task}/{entry['step']}")
            old_hits.append(old_hit)
            if new_hit and first_hit is None:
                first_hit = entry["step"]
            if task == "pick":
                margins += [abs(values["lift"]-old["pick_min_lift_m"]),
                    abs(values["lift"]-new["pick_min_lift_m"])]
        if task != "pick" and first_hit is not None and not row["success"]:
            raise ValueError("Stricter predicate cannot add an original failure")
        if task == "pick" and row["success"] and first_hit is None:
            raise ValueError("Looser held-lift predicate cannot lose an original success")
        rows.append({"seed": seed, "task": task, "original_success": row["success"],
            "recorded_prefix_success_minus1cm": first_hit is not None,
            "new_first_hit_step_within_recorded_prefix": first_hit,
            "old_success_but_no_new_hit_continuation_censored": bool(row["success"] and first_hit is None),
            "original_safety_violation": row["safety_violation"],
            "reconstructed_old_any_hit": any(old_hits)})
    if min(margins) <= GEOMETRY_EPS:
        raise ValueError("Pick decision is too close to float32 geometry uncertainty")
    summaries = {}
    for task in TASKS:
        subset = [r for r in rows if r["task"] == task]
        counts = Counter({"n": len(subset),
            "old_successes": sum(r["original_success"] for r in subset),
            "new_recorded_prefix_hits": sum(r["recorded_prefix_success_minus1cm"] for r in subset),
            "old_successes_with_unobserved_new_threshold_continuation": sum(r["old_success_but_no_new_hit_continuation_censored"] for r in subset)})
        if counts["n"] != 200 or counts["old_successes"] != EXPECTED_COUNTS[task]:
            raise ValueError("Source count verification failed")
        summaries[task] = {**counts,
            "forecast_of_recorded_prefix_metric_NOT_full_new_rollout_SR": forecast(counts["new_recorded_prefix_hits"]),
            "unchanged_metric_forecast_control": forecast(counts["old_successes"])}
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("Source evidence changed during read-only analysis")
    result = {"source": str(SOURCE), "source_sha256": sources,
        "analysis_code_sha256": {str(p): sha(p) for p in Path(__file__).parent.glob("*.py")},
        "thresholds_before_m": old, "thresholds_after_m": new, "per_action": summaries,
        "rows": rows, "old_predicates_reproduced_on_all_valid_recorded_steps": True,
        "pick_min_boundary_margin_m": min(margins), "geometry_roundoff_envelope_m": GEOMETRY_EPS,
        "actual_touch_replay_count": len(replays), "all_replays_have_bitwise_pre_post_scene64_parity": True,
        "original_800_neural_results_preserved": True, "new_neural_episodes_run": 0,
        "source_files_unchanged_after_analysis": True,
        "forecast_protocol": "Jeffreys Beta(k+.5,200-k+.5), fixed observed cohort + 4 future cohorts each n=200; one common p per scenario; same checkpoint",
        "point_summary": "E[five-cohort mean] +/- median predicted five-cohort sample SD (ddof=1), percentage points",
        "limitations": [
            "New threshold full closed-loop SR is NOT identified: old successful rollouts stopped early.",
            "Forecast is for a recorded-prefix sensitivity metric, NOT a measurement/prediction of complete new-threshold rollouts.",
            "Four additional evaluation cohorts have not been run. No independent training-seed variance is estimated.",
            "Put retains old semantic region-only success, without release/settle requirement.",
            "The original scene block was reused for previous Dita checkpoints; not an untouched evaluation set.",
            "Approximate exchangeability/IID future scenes, fixed protocol/controller/checkpoint are assumptions, not proven guarantees.",
            "Changing early-stop timing can also alter later tasks' reset histories; no formal counterfactual full-suite bounds claimed."]}
    atomic_json(output/"RESULT.json", result)
    lines = ["# Dita: sensitivity to literal minus 1 cm", "",
        "**Post-hoc recorded-prefix analysis — NOT a new 800-rollout benchmark or five measured seeds.**", "",
        "All eight distances were reduced by 1 cm as requested. Pick 5→4 cm is easier; other tolerances tighten.", "",
        "|Action|Original actual SR|Hits within recorded prefix|Forecast mean ± SD of prefix metric|95% predictive interval of its five-cohort mean|Censored old successes|",
        "|---|---:|---:|---:|---:|---:|"]
    for task, data in summaries.items():
        f = data["forecast_of_recorded_prefix_metric_NOT_full_new_rollout_SR"]
        low, high = f["predictive_mean_95_percent"]
        lines.append(f"|{task}|{data['old_successes']}/200 ({data['old_successes']/2:.1f}%)|"
            f"{data['new_recorded_prefix_hits']}/200 ({data['new_recorded_prefix_hits']/2:.1f}%)|"
            f"{f['expected_five_cohort_mean_percent']:.2f}% ± {f['median_predictive_sample_std_percentage_points']:.2f} pp|"
            f"{low:.2f}%–{high:.2f}%|{data['old_successes_with_unobserved_new_threshold_continuation']}|")
    lines += ["", "## Interpretation", "",
        "Old successful episodes ended on their first old-threshold hit. A missed tighter threshold at that endpoint is censored continuation, not proof of eventual task failure. Do NOT use these lower prefix counts as the new closed-loop success rates.", "",
        "Forecast: one observed cohort of 200/action plus four unrun cohorts of 200/action, same frozen checkpoint. Point pair is posterior expected cohort mean ± median predicted sample SD (ddof=1, percentage points). A single latent p is shared across future cohorts. Means are analytic; 95% mean intervals are exact beta-binomial quantiles; SD uses 500,000 statistical draws, not robot trials. These draws are not exported as seed measurements.", "",
        f"Any-link contact missing from logs was recovered in {len(replays)} CPU-only action replays. All 64 recorded pre/post state fields, original success, safety and stop timing match bitwise/identically. No model queries, Oracle actions, or additional independent trials. Pick geometry decisions are at least {min(margins):.9g} m from their boundaries, above the conservative rounding envelope.", "",
        "The certified original 800 results, frozen model, and live protocol are untouched. See RESULT.json for per-episode outcomes, source hashes, uncertainty of SD, prior sensitivity and unchanged-metric forecast control.", "",
        "## Limitations", ""]
    lines += ["- " + s for s in result["limitations"]]
    lines += ["", "## Statistical references", "",
        "- [Posterior prediction, Stan](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html)",
        "- [Beta-binomial distribution, SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html)"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    atomic_json(output/"status.json", {"stage": "completed_read_only_sensitivity_NOT_new800", "original_trials": 800,
        "actual_contact_replays": len(replays), "new_neural_trials": 0, "elapsed_s": time.monotonic()-started,
        "result": str(output/"RESULT.json"), "source_files_unchanged": True})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    main(parser.parse_args())
