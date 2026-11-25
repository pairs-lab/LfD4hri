"""Measured SR, honest scene-seed uncertainty and labelled future-cohort predictions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.evaluation_statistics import wilson_interval
from VLA.common.provenance import sha256_file
from .common import OUT, CP, TASKS, SEEDS, read


def forecast(rows):
    if len(rows)!=800 or {(r['seed'],r['task']) for r in rows}!={(s,t) for s in SEEDS for t in TASKS}:
        raise ValueError('Forecast only after all800 unique locked trials')
    rng=np.random.default_rng(202610178)
    result={'interpretation':'Predictions for4 more independent environment-cohort draws of200 scenes/task, using ONE frozen checkpoint. NOT observed seeds and NOT training-seed variance.',
        'assumptions':['same task/scene distribution','unchanged checkpoint/controller/thresholds','fresh IID scene-seed blocks','no test-based tuning or scene selection'],
        'monte_carlo_draws':10000,'by_task':{}}
    matrix=np.asarray([[bool(next(r for r in rows if r['seed']==s and r['task']==t)['success']) for t in TASKS] for s in SEEDS],float)
    for i,task in enumerate(TASKS):
        s=int(matrix[:,i].sum());p=s/200
        true_rate=rng.beta(s+.5,200-s+.5,size=10000)
        future=rng.binomial(200,true_rate[:,None],size=(10000,4))/200
        five=np.column_stack((np.full(10000,p),future))
        result['by_task'][task]={'measured_current_rate':p,
            'plug_in_expected_between_200_trial_cohort_SD':math.sqrt(p*(1-p)/200),
            'NOT_a_measured_five_seed_mean_std':True,
            'Jeffreys_Beta_half_half_posterior_predictive':{
                'future_single_cohort_rate_95_prediction_interval':np.quantile(future[:,0],[.025,.975]).tolist(),
                'predicted_five_cohort_mean_median_and95': [float(np.median(five.mean(1))),*np.quantile(five.mean(1),[.025,.975]).tolist()],
                'predicted_five_cohort_sample_SD_median_and95':[float(np.median(five.std(1,ddof=1))),*np.quantile(five.std(1,ddof=1),[.025,.975]).tolist()]},
            'all_success_or_all_failure_has_uncertainty_despite_zero_plugin_SD':s in (0,200)}
    rates=matrix.mean(1)
    future=np.column_stack([rates[rng.integers(0,200,size=(10000,200))].mean(1) for _ in range(4)])
    five=np.column_stack((np.full(10000,rates.mean()),future))
    result['overall_scene_block_bootstrap']={'retains_cross_task_correlation':True,
        'current_mean':float(rates.mean()),
        'plug_in_expected_between_800_trial_cohort_SD':float(rates.std(ddof=1)/math.sqrt(200)),
        'bootstrap_conditional_on_empirical_scene_distribution_not_unknown_distribution_shift':True,
        'predicted_five_cohort_mean_95':np.quantile(five.mean(1),[.025,.975]).tolist(),
        'predicted_five_cohort_sample_SD_median_and95':[float(np.median(five.std(1,ddof=1))),*np.quantile(five.std(1,ddof=1),[.025,.975]).tolist()]}
    return result


def audit_finished(p):
    receipts=[json.loads(line) for line in (OUT/'episodes.neural_receipts.jsonl').read_text().splitlines()]
    by_input={r['input_sha256']:r for r in receipts}
    controls=full=partial=0
    for seed in SEEDS:
        traces=read(OUT/f'episodes/test_seed{seed}/trajectory.json')
        for task in TASKS:
            trace=traces[task]
            summary=next(r for r in p['rows'] if r['seed']==seed and r['task']==task)
            if len(trace)!=summary['steps'] or bool(trace[-1]['success'])!=summary['success']:
                raise ValueError('Trace/summary outcome differs')
            groups={}
            for row in trace:
                groups.setdefault(row['chunk_query_step'],[]).append(row)
            for chunk in groups.values():
                receipt=by_input[chunk[0]['neural_chunk_input_sha256']]
                if receipt['actual_native_visual_receipt']['actual_views']!=3:
                    raise ValueError('Three-view input receipt missing')
                a=np.asarray([r['native_neural_action'] for r in chunk],np.float32)
                if not np.isfinite(a).all() or a.shape[1:]!=(20,):raise ValueError('Malformed native action trace')
                if len(a)==30:
                    if hashlib.sha256(a.tobytes()).hexdigest()!=receipt['action_sha256_float32']:
                        raise ValueError('Full native chunk differs from actual server receipt')
                    full+=1
                else:partial+=1
            controls+=len(trace)
    if controls!=p['neural_controls'] or len(receipts)!=p['queries']:
        raise ValueError('Neural query/control count mismatch')
    if list((OUT/'episodes').rglob('*.mp4')):raise ValueError('Unexpected video in no-video800')
    atomic_json(OUT/'FINAL_AUDIT.json', {'passed':True,'unique_trials':800,'trials_each_action':200,
        'all_true_and_false_task_outcomes_retained':True,'all3view_queries_verified':len(receipts),
        'native_controls':controls,'full30_chunks_hash_matched':full,
        'partial_terminal_chunks_not_claimed_full_hash_verified':partial,'no_video_saved':True,
        'checkpoint':str(CP),'terminal_release_or_oracle_action_override':False})


def main():
    path=OUT/'episodes/progress.json'
    if not path.exists():return
    p=read(path);rows=p['rows'];complete=len(rows)==800
    lines=['# X-VLA R59 — 200 episodes/action, 800 tổng', '',
        f"Đã chạy thật: **{len(rows)}/800**. Hoàn tất: **{complete}**.", '',
        'Một frozen X-VLA checkpoint step014500 + simulator GT, native prefix30, UR5e + Robotiq2F-85. '
        'Không Oracle fallback, không supervisor thả cuối, không video, không điều chỉnh model trên tập800.', '',
        'Inference RNG cố định202643 như protocol cũ. “Thêm4seeds” ở phần dự đoán nghĩa là4cohort scene-seed mới; '
        'không phải4training seeds hoặc4lần chạy lại cùngscene với inference RNG khác. '
        'Nếu chia sẻ GPU với finetune, latency wall-clock được log nhưng không đại diện latency khi GPU độc quyền.', '',
        '**PUT primary vẫn là goal-region first-hit của benchmark cũ, không phải full release/settle của demo qualitative.**', '',
        '|Action|Success/trials|SR|Wilson95|', '|---|---:|---:|---|']
    stats={}
    for t in TASKS:
        rr=[r for r in rows if r['task']==t];n=len(rr);s=sum(bool(r['success']) for r in rr)
        if n:
            ci=wilson_interval(s,n);stats[t]={'successes':s,'trials':n,'SR':s/n,'Wilson95':ci}
            lines.append(f'|{t}|{s}/{n}|{s/n:.1%}|{ci[0]:.1%}–{ci[1]:.1%}|')
        else:lines.append(f'|{t}|0/0|chưa có|—|')
    result={'complete':complete,'actual_trials':len(rows),'per_action':stats,
        'integration_errors_are_not_task_failure_counts':True}
    if complete:
        audit_finished(p)
        f=forecast(rows);atomic_json(OUT/'FUTURE_COHORT_PREDICTION_NOT_MEASURED_SEEDS.json',f)
        overall=f['overall_scene_block_bootstrap'];result['overall_SR']=overall['current_mean']
        result['future_cohort_prediction']=f
        lines += ['',f"Overall SR: **{overall['current_mean']:.2%}**.", '',
            '## Nếu chạy thêm4 cohort seeds (ƯỚC LƯỢNG, chưa phải số đo5 seeds)', '',
            'Mỗi cohort mới có200 scene-seed blocks ×4 actions. Giả định giữ nguyên phân phối và checkpoint. '
            'Không suy ra được std giữa các training seeds từ một checkpoint.', '',
            '|Action|Mean dự kiến gần SR hiện tại|SD lấy mẫu dự kiến giữa các cohort200 trials|',
            '|---|---:|---:|']
        for t in TASKS:
            q=f['by_task'][t]
            lines.append(f"|{t}|{q['measured_current_rate']:.1%}|{100*q['plug_in_expected_between_200_trial_cohort_SD']:.2f} điểm %|")
        lines += ['', f"Overall SD lấy mẫu theo200 seed-blocks: khoảng **{100*overall['plug_in_expected_between_800_trial_cohort_SD']:.2f} điểm %**.",
            'Wilson95 và Jeffreys posterior-predictive intervals trong JSON giữ lại bất định ngay cả khi quan sát200/200. '
            'Các số dự đoán không được chép thành mean±std5seed đo thật trong paper. Muốn report5seed measured phải chạy thêm đủ4cohort.', '']
    else:
        lines += ['', '**Kết quả đang chạy, không coi các tỷ lệ tạm thời là kết quả cuối.** '
            'Chưa xuất mean±std dự đoán5cohort khi800trial chưa hoàn tất.', '']
    atomic_json(OUT/'RESULTS.json',result)
    (OUT/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
