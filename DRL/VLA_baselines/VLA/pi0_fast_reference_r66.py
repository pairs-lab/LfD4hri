"""Internal reference scenario only: five UNRUN FAST cohorts IF equal to pi0.

No FAST performance has been observed. This assumption is NOT established by
pi0 data. No files from actual experiments or official model reports change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VLA/audits/pi0_fast_cancel_reference_20261019'
SOURCE = ROOT / 'VLA/audits/cross_vla800_r62_20261018/pi0'
TASKS = ('touch', 'pick', 'move', 'pick_place')
SEED = 20261019


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out, draws):
    target = out / 'PI0_REFERENCE_ONLY_NOT_FAST_RESULTS.json'
    if target.exists():
        raise FileExistsError('Preserve the existing pre-run reference scenario')
    if draws < 1000:
        raise ValueError('At least 1000 statistical draws required')
    results = json.loads((SOURCE / 'RESULTS.json').read_text())
    audit = json.loads((SOURCE / 'FINAL_AUDIT.json').read_text())
    progress = json.loads((SOURCE / 'episodes/progress.json').read_text())
    if not audit['passed'] or audit['actual_trials'] != 800:
        raise ValueError('The pi0 reference must have a complete audited actual cohort')
    grouped = {}
    for r in progress['rows']:
        group = grouped.setdefault(r['seed'], {})
        if r['task'] in group:
            raise ValueError('Duplicate reference outcome')
        group[r['task']] = bool(r['success'])
    if len(grouped) != 200 or any(set(g) != set(TASKS) for g in grouped.values()):
        raise ValueError('Need 200 complete scene blocks with all four actions')
    observed = np.array([[grouped[s][t] for t in TASKS] for s in sorted(grouped)], dtype=int)
    bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(int)
    patterns = observed @ (1 << np.arange(4))
    counts = np.bincount(patterns, minlength=16)
    by_task = results['statistics']['by_task']
    if any(observed[:, i].sum() != by_task[t]['successes'] for i, t in enumerate(TASKS)):
        raise ValueError('Reference counts differ from audited results')
    # Jeffreys marginal Beta(k+.5,200-k+.5), preserving four-task block
    # correlations. THIS POSTERIOR DESCRIBES pi0, not measured FAST behavior.
    rng = np.random.default_rng(SEED)
    p = rng.dirichlet(counts + 1 / 16, size=draws)
    future_rates = np.empty((draws, 5, 4), dtype=np.float64)
    for cohort in range(5):
        future_rates[:, cohort, :] = (rng.multinomial(200, p) @ bits) / 200
    means = future_rates.mean(axis=1)
    sds = future_rates.std(axis=1, ddof=1)
    rows = {}
    for i, t in enumerate(TASKS):
        mu = np.quantile(means[:, i], [.025, .5, .975])
        sd = np.quantile(sds[:, i], [.025, .5, .975])
        rows[t] = {'pi0_actual_successes': int(observed[:, i].sum()), 'pi0_actual_trials': 200,
                   'pi0_actual_SR': float(observed[:, i].mean()),
                   'FAST_actual_trials': 0, 'FAST_SR_identified_from_data': False,
                   'reference_scenario_mean_median': float(mu[1]),
                   'reference_scenario_mean_95_interval': mu[[0, 2]].tolist(),
                   'reference_scenario_sample_SD_median': float(sd[1]),
                   'reference_scenario_sample_SD_95_interval': sd[[0, 2]].tolist()}
    evidence_paths = [SOURCE / name for name in ('RESULTS.json', 'FINAL_AUDIT.json', 'episodes/progress.json')]
    output = {'label': 'INTERNAL_REFERENCE_SCENARIO_NOT_MEASURED_PI0_FAST_RESULTS',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'FAST_actual_robot_trials': 0, 'FAST_actual_finetuning_seeds_evaluated': 0,
        'five_FAST_cohorts_all_UNRUN_not_one_observed_plus_four': True,
        'reference_model_only': 'pi0', 'target_model_for_assumption_only': 'pi0_fast',
        'unverified_transfer_assumption': 'FAST joint four-task success probabilities EQUAL those of the measured pi0 checkpoint',
        'does_NOT_identify_actual_FAST_performance': True,
        'source_pi0_trial_count': 800, 'source_pi0_scene_blocks': 200,
        'native_architecture_speed_or_tokenizer_improvement_NOT_assumed_to_improve_SR': True,
        'method': 'pi0 reference 16-pattern Dirichlet(counts+1/16); shared p per draw; five new Multinomial(200,p) cohorts',
        'random_seed_preselected_not_searched_for_favorable_results': SEED,
        'statistical_draws_NOT_robot_rollouts': draws, 'sample_SD_ddof': 1,
        'frozen_checkpoint_per_scenario': True, 'training_seed_variance_NOT_estimated': True,
        'model_transfer_bias_NOT_estimated_or_covered_by_intervals': True,
        'metric': 'existing pi0 protocol; PUT goal-region first hit, NOT release/settle; MOVE/PUT do not require prior verified grasp',
        'actual_experiment_results_edited': False, 'by_task': rows,
        'reference_evidence_sha256': {str(p): sha(p) for p in evidence_paths},
        'generator_code_sha256': sha(Path(__file__))}
    atomic_json(target, output)
    lines = ['# INTERNAL REFERENCE SCENARIO — NOT pi0-FAST experiment results', '',
        '**FAST has zero measured episodes here. These numbers do not demonstrate its capabilities.**', '',
        'User cancelled queued pi0-FAST; JEPA remains untouched. See CANCEL_REQUEST.json and GUARD_STATUS.json.', '',
        'Unverified assumption for planning only: FAST has the same joint success distribution as the measured pi0 checkpoint. '
        'No improvement is assumed because the model is called FAST. Actual FAST may be better or worse; pi0 alone cannot identify that difference.', '',
        'All FIVE target cohorts are UNRUN, 200 episodes/action/cohort, one frozen checkpoint; '
        'this is NOT one observed FAST cohort plus four predictions and NOT five independent training seeds.', '',
        '|Action|Actual pi0 reference|Conditional scenario: median(mean) ± median(sample std)|95% scenario interval for mean|',
        '|---|---:|---:|---:|']
    for t in TASKS:
        r = rows[t]
        lo, hi = r['reference_scenario_mean_95_interval']
        lines.append(f'|{t}|{r["pi0_actual_successes"]}/200 ({100*r["pi0_actual_SR"]:.1f}%)|'
                     f'{100*r["reference_scenario_mean_median"]:.2f}% ± '
                     f'{100*r["reference_scenario_sample_SD_median"]:.2f} pp|{100*lo:.2f}%–{100*hi:.2f}%|')
    lines += ['', 'Mean units: percent. SD units: percentage points. Marginal medians summarized separately; '
              'no five-seed experimental table is fabricated.', '',
              f'Method: {draws:,} CPU statistical draws; fixed RNG seed {SEED}, no favorable-seed search. '
              'Whole four-task scene blocks and their correlations are preserved.', '',
              'Intervals capture reference-data uncertainty and future sampling noise ONLY under the equal-model assumption. '
              'They do NOT cover unknown FAST-vs-pi0 transfer bias, different finetuning quality or training-seed variance. '
              'Do not insert these into an empirical baseline table or call them measured performance.', '',
              'PUT is goal-region first hit, not complete release/settle. Existing measured reports and forecasts remain unchanged.']
    (out / 'REFERENCE_SCENARIO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'label': output['label'], 'by_task': rows}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--draws', type=int, default=200000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.output, args.draws)
