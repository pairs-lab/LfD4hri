"""Measured single-cohort results and explicitly unmeasured future predictions."""
import argparse
import math
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.evaluation_statistics import success_statistics
from .common import OUT,R59,MODELS,TASKS,SEEDS,read,matrix,utc


def forecast(rows,draws=10000):
    a=matrix(rows);rng=np.random.default_rng(2026101862)
    # One block is the same scene RNG seed's four tasks. A Dirichlet prior
    # of1/16 per binary pattern gives Jeffreys Beta(.5,.5) marginal priors,
    # retains task dependence and leaves uncertainty at0/200 and200/200.
    patterns=((np.arange(16)[:,None]>>np.arange(4))&1).astype(float)
    ids=(a.astype(int)*(1<<np.arange(4))).sum(1)
    posterior=rng.dirichlet(np.bincount(ids,minlength=16)+1/16,size=draws)
    future_counts=np.stack([rng.multinomial(200,p) for p in posterior for _ in range(4)]).reshape(draws,4,16)
    future=future_counts@patterns/200
    current=a.mean(0);five=np.concatenate((np.broadcast_to(current,(draws,1,4)),future),axis=1)
    out={'label':'FORECAST_ONLY_four_unrun_environment_cohorts_NOT_measured_five_seeds',
         'actual_additional_cohorts_run':0,'frozen_checkpoints':1,
         'actual_evaluation_cohorts':1,'independent_retraining_runs_executed_here':0,
         'training_seed_variance_estimable':False,'posterior_draws':draws,
         'assumptions':['same scene distribution','same frozen checkpoint and native RNG policy',
                        'same physics/controller/success','fresh IID200 scene-seed blocks per future cohort',
                        'no selection based on the800outcomes'],
         'method':'16-pattern Dirichlet-multinomial; alpha=1/16 each; four tasks kept in each scene block',
         'by_task':{}}
    for i,t in enumerate(TASKS):
        mean=five[:,:,i].mean(1);sd=five[:,:,i].std(1,ddof=1);p=current[i]
        out['by_task'][t]={'measured_one_cohort_SR':float(p),
            'expected_sampling_SD_per_200_trial_cohort':math.sqrt(p*(1-p)/200),
            'forecast_five_cohort_mean_median':float(np.median(mean)),
            'forecast_five_cohort_mean_95_interval':np.quantile(mean,[.025,.975]).tolist(),
            'forecast_five_cohort_sample_SD_median':float(np.median(sd)),
            'forecast_five_cohort_sample_SD_95_interval':np.quantile(sd,[.025,.975]).tolist(),
            'future_one_cohort_SR_95_prediction_interval':np.quantile(future[:,0,i],[.025,.975]).tolist()}
    block=a.mean(1);fm=five.mean(2)
    out['overall']={'measured_SR':float(a.mean()),
        'expected_sampling_SD_per_800_trial_cohort':float(block.std(ddof=1)/np.sqrt(200)),
        'forecast_five_cohort_mean_median':float(np.median(fm.mean(1))),
        'forecast_five_cohort_mean_95_interval':np.quantile(fm.mean(1),[.025,.975]).tolist(),
        'forecast_five_cohort_sample_SD_median':float(np.median(fm.std(1,ddof=1))),
        'forecast_five_cohort_sample_SD_95_interval':np.quantile(fm.std(1,ddof=1),[.025,.975]).tolist(),
        'future_one_cohort_SR_95_prediction_interval':np.quantile(future[:,0].mean(1),[.025,.975]).tolist()}
    return out


def measured(model,rows,dest,source):
    matrix(rows);stats=success_statistics(rows);f=forecast(rows)
    result={'model':model,'actual_completed_trials':800,'actual_trials_per_action':200,
            'actual_evaluation_cohorts':1,'four_additional_cohorts_not_run':True,
            'source_progress':str(source),'statistics':stats,
            'overall_SR':sum(r['success'] for r in rows)/800,
            'PUT_primary':'goal-region first hit; no release/settle requirement',
            'privileged_simulator_VLA_NOT_RGB_text_only':True}
    atomic_json(dest/'RESULTS.json',result)
    atomic_json(dest/'FORECAST_NOT_MEASURED_5_SEEDS.json',f)
    return result


def refresh(out):
    lines=['# Four simulator-privileged VLAs — one measured800episode cohort each','',
        'Same200scene-seed blocks ×touch/pick/move/put, UR5e +Robotiq2F-85, frozen common physics/safety/success.',
        'X-VLA800 is preserved from R59, not rerun or refit. Other families finish R56 qualification and freeze first.',
        'PUT is goal-region first-hit, not full release/settle. Safety aborts remain failures; integration errors are NOT zero SR.',
        'RGB/numeric input modes and native chunk lengths can differ by admitted recipe and are disclosed per method.',
        'No new Oracle policy, geometric action override or terminal release is introduced. Existing shared contact assist stays unchanged.','',
        '|Model|Actually completed|Touch|Pick|Move|Put region|Overall|','|---|---:|---:|---:|---:|---:|---:|']
    results={}
    for model in ('xvla',*MODELS):
        folder=out/model;p=folder/'episodes/progress.json';r=folder/'RESULTS.json'
        if r.exists():
            v=read(r);results[model]=v;b=v['statistics']['by_task']
            lines.append('|'+model+'|800/800|'+ '|'.join(f"{b[t]['successes']}/200 ({b[t]['success_rate']:.1%})" for t in TASKS)+f"|{v['overall_SR']:.2%}|")
        else:
            n=read(p)['completed_trials'] if p.exists() else 0
            lines.append(f'|{model}|{n}/800|pending|pending|pending|pending|pending|')
    lines+=['','## Forecast only — four additional cohorts have NOT been run','',
        'This is not measured mean±std across five training/evaluation seeds. No four fabricated per-seed scores are generated.',
        'Below: measured SR ± expected between-cohort sampling SD, in percentage points; NOT a confidence interval.',
        'Posterior-predictive distributions of the future five-cohort mean and sample SD are saved separately.','',
        '|Model|Touch forecast|Pick forecast|Move forecast|Put forecast|Overall forecast|',
        '|---|---:|---:|---:|---:|---:|']
    for model in results:
        f=read(out/model/'FORECAST_NOT_MEASURED_5_SEEDS.json')
        cells=[f"{f['by_task'][t]['measured_one_cohort_SR']*100:.1f}% ± {f['by_task'][t]['expected_sampling_SD_per_200_trial_cohort']*100:.2f} pp" for t in TASKS]
        q=f['overall'];cells.append(f"{q['measured_SR']*100:.2f}% ± {q['expected_sampling_SD_per_800_trial_cohort']*100:.2f} pp")
        lines.append('|'+model+'|'+'|'.join(cells)+'|')
    lines+=['','### Modelled five-cohort mean and sample SD — NOT observed','',
        'Cells are posterior medians of the five-cohort mean and the five-cohort sample SD, respectively.',
        'These are predictions conditional on one observed cohort plus four UNRUN cohorts;95% prediction intervals are in JSON.',
        '|Model|Touch|Pick|Move|Put region|Overall|','|---|---:|---:|---:|---:|---:|']
    for model in results:
        f=read(out/model/'FORECAST_NOT_MEASURED_5_SEEDS.json')
        fields=[f['by_task'][t] for t in TASKS]+[f['overall']]
        cells=[f"{q['forecast_five_cohort_mean_median']*100:.2f}% ± {q['forecast_five_cohort_sample_SD_median']*100:.2f} pp" for q in fields]
        lines.append('|'+model+'|'+'|'.join(cells)+'|')
    lines+=['','At all-success/all-failure the plug-in SD can be zero, but the posterior predictive interval remains nondegenerate.',
            'Only repeating actual finetuning with independent training seeds can measure retraining variance.',
            'Only after running four further complete cohorts may their measured five-cohort mean±std be reported.','',
            'Live orchestration: [owner_status.json](owner_status.json). Last update UTC: '+utc()]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'SUMMARY.json',{'actual_results':results,'updated_UTC':utc(),
        'all_four_measured800_complete':len(results)==4,'forecasts_are_not_measured_seeds':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();refresh(a.output)
