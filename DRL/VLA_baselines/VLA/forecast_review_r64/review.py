"""Audit R62 forecasts without changing frozen experiments or predictions.

The independent marginal calculation integrates Beta uncertainty exactly for
the future-count sum. It independently simulates the five-cohort sample SD.
No simulated seed score is exported or counted as an experiment.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'VLA/audits/cross_vla800_r62_20261018'
R59 = ROOT / 'VLA/audits/xvla800_r59_20261017'
MODELS = ('xvla', 'smolvla', 'pi0', 'vla_adapter')
TASKS = ('touch', 'pick', 'move', 'pick_place')
SEEDS = tuple(range(20271001, 20271201))
N = 200
FUTURE = 4
DRAWS = 500_000
MC_SEED = 2026101864
MC_FAILURE_BOUND = 1e-6
REFERENCES = {
    'Wilson confidence interval, NIST':
        'https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm',
    'Posterior predictive uncertainty, Stan':
        'https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html',
    'Beta-binomial PMF, SciPy':
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_text(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def counts(rows):
    expected = {(s, t) for s in SEEDS for t in TASKS}
    keys = [(r['seed'], r['task']) for r in rows]
    if len(keys) != 800 or len(set(keys)) != 800 or set(keys) != expected:
        raise ValueError('Require exactly the declared 800 unique actual trials')
    if any(type(r['success']) is not bool for r in rows):
        raise ValueError('Require boolean measured outcomes, not estimates')
    return {t: sum(r['success'] for r in rows if r['task'] == t) for t in TASKS}


def beta_binomial_pmf(m, a, b):
    """Independent log-gamma finite-support PMF, no forecast-code import."""
    if type(m) is not int or m < 0 or min(a, b) <= 0:
        raise ValueError('Invalid beta-binomial parameters')
    norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    logs = np.array([
        math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1)
        + math.lgamma(a + j) + math.lgamma(b + m - j)
        - math.lgamma(a + b + m) - norm for j in range(m + 1)
    ])
    p = np.exp(logs - logs.max())
    return p / p.sum()


def ppf(pmf, qs):
    cdf = np.cumsum(pmf)
    return np.searchsorted(cdf, qs, side='left').clip(0, len(pmf) - 1)


def exact(k, prior=.5):
    if type(k) is not int or not 0 <= k <= N:
        raise ValueError('This action has 200 actual outcomes, not 800')
    a, b = k + prior, N - k + prior
    total = N * (FUTURE + 1)
    all_future = beta_binomial_pmf(N * FUTURE, a, b)
    one_future = beta_binomial_pmf(N, a, b)
    qs = np.array([.025, .5, .975])
    means = (k + ppf(all_future, qs)) / total
    expected = (k + N * FUTURE * a / (a + b)) / total
    mu = a / (a + b)
    vp = a * b / ((a + b)**2 * (a + b + 1))
    epq = a * b / ((a + b) * (a + b + 1))
    # Future cohort rates share one posterior p. The observed rate is fixed.
    expected_sample_variance = (FUTURE * epq / N + vp + (k / N - mu)**2) / (FUTURE + 1)
    return {
        'prior_alpha_beta': [prior, prior],
        'posterior_alpha_beta': [a, b],
        'five_cohort_mean_expectation': expected,
        'five_cohort_mean_median': float(means[1]),
        'five_cohort_mean_95_prediction_interval': means[[0, 2]].tolist(),
        'one_future_cohort_SR_95_prediction_interval': (ppf(one_future, [.025, .975]) / N).tolist(),
        'expected_five_cohort_sample_variance': expected_sample_variance,
        'sample_variance_definition': 'sample variance across 5 cohort rates, ddof=1; first rate observed and fixed',
    }, all_future


def sd_draws(k, draws, seed, prior=.5):
    rng = np.random.default_rng(seed)
    # One shared p per draw, NOT an independent p for each future cohort.
    p = rng.beta(k + prior, N - k + prior, size=draws)
    future = rng.binomial(N, p[:, None], size=(draws, FUTURE)) / N
    five = np.column_stack((np.full(draws, k / N), future))
    return five.std(axis=1, ddof=1)


def discrete_quantile_check(pmf, reported, target, offset, denominator, epsilon):
    # Linear sample quantiles may fall between lattice points. Check both
    # bracketing CDFs, allowing a DKW bound for Monte Carlo error.
    x = reported * denominator - offset
    if not math.isfinite(x) or x < -1e-8 or x > len(pmf) - 1 + 1e-8:
        return False
    lo = max(0, math.floor(x + 1e-9))
    hi = min(len(pmf) - 1, math.ceil(x - 1e-9))
    cdf = np.cumsum(pmf)
    left = float(cdf[lo - 1]) if lo > 0 else 0.
    right = float(cdf[hi]) if hi >= 0 else 0.
    return left <= target + epsilon and right >= target - epsilon


def empirical_quantile_check(samples, reported, target, epsilon):
    return (np.mean(samples < reported - 1e-12) <= target + epsilon
            and np.mean(samples <= reported + 1e-12) >= target - epsilon)


def action_review(k, q, old_draws, index, draws=DRAWS):
    if old_draws <= 0 or draws < 1000:
        raise ValueError('Insufficient declared Monte Carlo draws')
    analytic, pmf = exact(k)
    sd = sd_draws(k, draws, MC_SEED + index)
    alternative, _ = exact(k, prior=1.)
    sd_uniform = sd_draws(k, draws, MC_SEED + index, prior=1.)
    # Each bound is for one CDF comparison, not a global physical-validity claim.
    eps_old = math.sqrt(math.log(4 / MC_FAILURE_BOUND) / (2 * old_draws))
    eps_new = math.sqrt(math.log(4 / MC_FAILURE_BOUND) / (2 * draws))
    mean_values = [q['forecast_five_cohort_mean_95_interval'][0],
                   q['forecast_five_cohort_mean_median'],
                   q['forecast_five_cohort_mean_95_interval'][1]]
    sd_values = [q['forecast_five_cohort_sample_SD_95_interval'][0],
                 q['forecast_five_cohort_sample_SD_median'],
                 q['forecast_five_cohort_sample_SD_95_interval'][1]]
    levels = [.025, .5, .975]
    mean_checks = [bool(discrete_quantile_check(pmf, v, p, k, N * 5, eps_old))
                   for p, v in zip(levels, mean_values)]
    sd_checks = [bool(empirical_quantile_check(sd, v, p, eps_old + eps_new))
                 for p, v in zip(levels, sd_values)]
    variance = sd**2
    variance_mean = float(variance.mean())
    variance_mean_se = float(variance.std(ddof=1) / math.sqrt(draws))
    moment_check = abs(variance_mean - analytic['expected_five_cohort_sample_variance']) <= 7 * variance_mean_se + 1e-12
    matched = math.isclose(q['measured_one_cohort_SR'], k / N, abs_tol=1e-12)
    sd_q = np.quantile(sd, levels)
    sd_uniform_q = np.quantile(sd_uniform, levels)
    passed = matched and all(mean_checks) and all(sd_checks) and moment_check
    return {
        'passed': bool(passed), 'actual_successes': k, 'actual_trials': N,
        'actual_evaluation_cohorts': 1, 'future_cohorts_actually_run': 0,
        'independent_training_seed_variance_estimable': False,
        'unchanged_R62_reported_forecast': q,
        'independent_exact_Jeffreys_check': analytic,
        'independent_sample_SD_simulation': {
            'draws': draws, 'rng_seed': MC_SEED + index,
            'median': float(sd_q[1]), '95_prediction_interval': sd_q[[0, 2]].tolist(),
            'mean_sample_variance': variance_mean, 'MC_SE_of_mean_sample_variance': variance_mean_se,
            'analytic_sample_variance_moment_check_passed': bool(moment_check),
            'simulation_is_NOT_robot_rollouts': True,
        },
        'Monte_Carlo_validation': {
            'mean_quantiles_pass_exact_CDF_check': mean_checks,
            'SD_quantiles_pass_independent_empirical_CDF_check': sd_checks,
            'DKW_CDF_tolerance_existing_draws': eps_old,
            'DKW_CDF_tolerance_independent_draws': eps_new,
            'per_comparison_failure_probability_bound': MC_FAILURE_BOUND,
            'not_a_test_of_scene_IID_or_hardware_or_model_validity': True,
        },
        'prior_sensitivity_NOT_favorable_method_selection': {
            'Uniform_Beta_1_1_exact': alternative,
            'Uniform_SD_median': float(sd_uniform_q[1]),
            'Uniform_SD_95_prediction_interval': sd_uniform_q[[0, 2]].tolist(),
            'change_in_expected_mean_pp': 100 * (alternative['five_cohort_mean_expectation'] - analytic['five_cohort_mean_expectation']),
            'change_in_SD_median_pp': 100 * float(sd_uniform_q[1] - sd_q[1]),
            'primary_method_retained_Jeffreys_no_selection_from_sensitivity': True,
        },
    }


def review_model(base, out, model, draws=DRAWS):
    dest = base / model
    progress = R59 / 'episodes/progress.json' if model == 'xvla' else dest / 'episodes/progress.json'
    cert = R59 / 'FINAL_AUDIT.json' if model == 'xvla' else dest / 'FINAL_AUDIT.json'
    files = [progress, cert, dest / 'RESULTS.json', dest / 'FORECAST_NOT_MEASURED_5_SEEDS.json']
    if not all(p.is_file() for p in files):
        return {'status': 'pending_actual800_and_audit_NOT_zero_SR'}
    source_hashes = {str(p): sha(p) for p in files}
    proof, result, forecast = read(cert), read(files[2]), read(files[3])
    if not proof.get('passed') or result.get('actual_completed_trials') != 800:
        return {'status': 'pending_actual800_and_audit_NOT_zero_SR'}
    if (forecast.get('actual_additional_cohorts_run') != 0
            or forecast.get('actual_evaluation_cohorts') != 1
            or forecast.get('training_seed_variance_estimable') is not False
            or not forecast.get('label', '').startswith('FORECAST_ONLY_')
            or forecast.get('frozen_checkpoints') != 1):
        raise ValueError('Prediction must not be presented as measured five-seed data')
    outcomes = counts(read(progress)['rows'])
    by_task = {}
    for i, task in enumerate(TASKS):
        reported = result['statistics']['by_task'][task]
        if reported['successes'] != outcomes[task] or reported['trials'] != N:
            raise ValueError('Measured summary inconsistent with actual source rows')
        by_task[task] = action_review(outcomes[task], forecast['by_task'][task],
                                     forecast['posterior_draws'], i, draws)
    if source_hashes != {str(p): sha(p) for p in files}:
        raise ValueError('Source evidence changed during independent calculation')
    doc = {
        'model': model, 'passed': all(v['passed'] for v in by_task.values()),
        'status': 'independent_statistical_check_complete',
        'source_sha256': source_hashes, 'by_action': by_task,
        'scope': 'forecast calculation validation ONLY; not proof of empirical future accuracy',
        'source_experiments_weights_controller_and_R62_forecasts_edited': False,
        'retraining_seed_variation_identifiable': False,
    }
    write_json(out / (model + '.json'), doc)
    return doc


def report(out, models):
    lines = ['# Kiểm tra độc lập dự báo — RIÊNG TỪNG ACTION', '',
        'Một cohort đo thật: 800 lượt/model = 200 lượt/action. Bốn cohort tiếp theo CHƯA CHẠY.',
        'Mọi mean ± std dưới đây là ước lượng có điều kiện, KHÔNG phải kết quả đo 5 seed.',
        'Std là sample std của 5 cohort, ddof=1, đơn vị điểm phần trăm; không phải CI/SE của mean.', '',
        '## Đo thật và dự báo đã chốt (không chỉnh lại số để trông đẹp hơn)', '',
        '|Model|Action|Đo thật|Dự báo mean ± std (5 cohort)|Khoảng dự đoán 95% của mean|Kiểm tra số học|',
        '|---|---|---:|---:|---|---|']
    for model in MODELS:
        doc = models[model]
        if 'by_action' not in doc:
            lines.append(f'|{model}|cả 4|chờ đủ 800 và audit|chưa ước lượng|—|PENDING|')
            continue
        for task, q in doc['by_action'].items():
            r = q['unchanged_R62_reported_forecast']
            mean, sd = r['forecast_five_cohort_mean_median'], r['forecast_five_cohort_sample_SD_median']
            lo, hi = r['forecast_five_cohort_mean_95_interval']
            lines.append(f"|{model}|{'put' if task == 'pick_place' else task}|{q['actual_successes']}/200 ({q['actual_successes']/2:.1f}%)|{100*mean:.2f}% ± {100*sd:.2f} điểm %|{100*lo:.2f}%–{100*hi:.2f}%|{'PASS' if q['passed'] else 'FAIL'}|")
    lines += ['', '## Cách tính và phản chứng', '',
        '1. Giữ nguyên 200 scene-seed block, mỗi block gồm bốn task; không coi 800 lượt là 800 mẫu/action.',
        '2. R62 dùng Dirichlet trên 16 mẫu kết quả bốn task, alpha=1/16 mỗi mẫu. Marginal của từng action là Jeffreys Beta(0.5,0.5).',
        '3. Với k/200 thành công, posterior p ~ Beta(k+0.5,200-k+0.5). Mỗi dự báo lấy một p chung cho bốn cohort mới, rồi sinh bốn count Binomial(200,p). Cohort quan sát k/200 giữ cố định.',
        '4. Kiểm tra độc lập: tổng success của bốn cohort mới có phân phối Beta–Binomial(800,k+0.5,200-k+0.5). Mean 5 cohort = (k+tổng success tương lai)/1000. CDF/quantile được tính bằng tổng hữu hạn, không dùng lại hàm forecast R62.',
        f'5. Std được kiểm tra bằng {DRAWS:,} mô phỏng thống kê độc lập/action, moment sample variance giải tích, và kiểm tra CDF có dung sai Monte Carlo. Đây KHÔNG phải lượt chạy robot.',
        '6. Kiểm tra độ nhạy với prior Uniform Beta(1,1); không chọn prior cho số đẹp hơn. Không ghi thêm bốn kết quả seed giả. JSON lưu cả khoảng dự đoán của std và khoảng của một cohort tương lai.',
        '7. Giữ nguyên forecast R62 ban đầu. Các sai khác nhỏ so với quantile giải tích là sai số Monte Carlo, không được giải thích là model cải thiện.', '',
        '## Những điều KHÔNG suy ra được từ 800 lượt', '',
        '- Dự báo chỉ áp dụng khi checkpoint, native inference RNG convention, controller, metric và phân phối scene không đổi; các scene mới phải xấp xỉ IID như cohort gốc. Chưa có bằng chứng bảo đảm giả định này đúng tuyệt đối.',
        '- Một checkpoint không đủ ước lượng std do finetune lại bằng training seed khác. Bốn cohort evaluation mới không phải bốn lần training mới.',
        '- Không thể bảo đảm dự báo sát mọi kết quả tương lai hoặc không bị phản biện. CI/interval phải đi cùng point estimate; không chỉ đưa một con số std nhỏ.',
        '- Không thể dùng kết quả này để bảo đảm SR >50% cho các model chưa đủ 800 lượt. Không dùng thử nghiệm synthetic/statistical như neural rollouts.',
        '- PUT metric hiện hành là first hit vùng đích, không bắt buộc thả/settle; MOVE/PUT không bắt buộc prior grasp. Đây không phải full pick-and-place SR.', '',
        '## Cách trình bày trong paper', '',
        'Bảng chính nên ghi success/200, SR đo thật và Wilson 95% CI cho từng action. Để mean ± std ước lượng ở bảng riêng với nhãn “posterior-predictive estimate; one observed + four unrun evaluation cohorts; same checkpoint”.',
        'Không được viết “mean ± std over 5 evaluated seeds” trước khi thực sự chạy đủ năm cohort. Nếu reviewer yêu cầu std giữa các training seed thì phải retrain độc lập; các estimate này không thay thế được.', '',
        '## Tự động hóa', '',
        'Reviewer này chỉ đọc kết quả/audit; tự kiểm tra từng model sau khi đủ 800 lượt. Nó không sửa pipeline finetune, checkpoint, action, môi trường, metric hay prediction đã chốt.',
        'Bốn model trong hàng đợi: X-VLA, SmolVLA, pi0, VLA-Adapter. X-VLA giữ cohort đã hoàn thành; ba model còn lại tiếp tục qualification/fit và 800 lượt qua owner R62.', '',
        '## Tài liệu gốc', '']
    lines += [f'- [{title}]({url})' for title, url in REFERENCES.items()]
    write_text(out / 'REPORT.md', '\n'.join(lines) + '\n')


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.review.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_sha = sha(__file__)
        deadline = time.monotonic() + args.hours * 3600
        done = {}
        while True:
            if sha(__file__) != source_sha:
                raise ValueError('Running read-only statistical reviewer source changed')
            for model in MODELS:
                if model in done and 'by_action' in done[model]:
                    # Completed evidence is immutable; detect subsequent edits.
                    for path, digest in done[model]['source_sha256'].items():
                        if sha(path) != digest:
                            raise ValueError('Previously reviewed result changed: ' + path)
                    continue
                done[model] = review_model(args.base, args.output, model)
            report(args.output, done)
            failed = any(v.get('passed') is False for v in done.values())
            complete = all('by_action' in v for v in done.values())
            state = 'calculation_check_failed_no_model_mutation' if failed else 'completed_all_four_statistical_checks' if complete else 'watching_read_only_waiting_actual800'
            write_json(args.output / 'reviewer_status.json', {
                'status': state, 'pid': os.getpid(), 'source_sha256': source_sha,
                'updated_UTC': datetime.now(timezone.utc).isoformat(),
                'models': {m: {'status': v['status'], 'passed': v.get('passed')} for m, v in done.items()},
                'frozen_training_and_benchmark_files_modified': False,
            })
            if failed:
                raise ValueError('Forecast arithmetic validation failed; original evidence preserved')
            if not args.watch or complete:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Statistical watch expired; model owners are not stopped')
            time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=BASE / 'statistical_review_r64')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--hours', type=float, default=72)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / 'reviewer.log').open('x') as stream:
            child = subprocess.Popen([
                sys.executable, '-u', '-m', 'VLA.forecast_review_r64.review', '--watch',
                '--base', str(args.base.resolve()), '--output', str(args.output.resolve()),
                '--interval', str(args.interval), '--hours', str(args.hours),
            ], cwd=ROOT, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'reviewer_pid': child.pid, 'output': str(args.output.resolve())}))
    else:
        main(args)
