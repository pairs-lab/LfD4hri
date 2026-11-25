"""Separate measured200/action, preliminary30/action, and UNRUN-cohort forecasts."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
from io import StringIO
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from VLA.common.artifact_io import atomic_json

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'VLA/audits/cross_vla800_r62_20261018'
R59=ROOT/'VLA/audits/xvla800_r59_20261017'
R56=ROOT/'VLA/audits/recovery_r56_20261017'
MODELS=('xvla','smolvla','pi0','vla_adapter')
TASKS=('touch','pick','move','pick_place')
NAMES={'xvla':'X-VLA','smolvla':'SmolVLA','pi0':'π0','vla_adapter':'VLA-Adapter'}
SLUGS={'touch':'touch','pick':'pick','move':'move','pick_place':'put'}
VN=ZoneInfo('Asia/Ho_Chi_Minh')


def read(p):return json.loads(Path(p).read_text())


def atomic_text(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(value);temp.replace(path)


def wilson(s,n):
    if not n:return None
    z=1.959963984540054;p=s/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,c-h),min(1.,c+h)]


def per_action(rows,allowed_seeds,per_task):
    allowed=set(allowed_seeds);keys=[]
    for r in rows:
        key=(r['seed'],r['task']);keys.append(key)
        if key[0] not in allowed or key[1] not in TASKS or type(r['success']) is not bool:
            raise ValueError('Unplanned scene/task or nonboolean measured outcome')
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate trial; cannot report it twice')
    result={}
    for t in TASKS:
        rr=[r for r in rows if r['task']==t];n=len(rr);s=sum(r['success'] for r in rr)
        if n>per_task:raise ValueError('Too many trials for this action')
        result[t]={'completed':n,'planned':per_task,'successes':s,
                   'measured_SR':s/n if n else None,'measured_Wilson95':wilson(s,n)}
    return result


def forecast_for_task(doc,task,measured,audited):
    if not audited or measured['completed']!=200:return None
    if (doc.get('actual_additional_cohorts_run')!=0 or doc.get('actual_evaluation_cohorts')!=1
        or doc.get('training_seed_variance_estimable') is not False
        or not doc.get('label','').startswith('FORECAST_ONLY_')):
        raise ValueError('Refuse to turn unqualified predictions into measured seed statistics')
    q=doc['by_task'][task]
    if not math.isclose(q['measured_one_cohort_SR'],measured['measured_SR'],abs_tol=1e-12):
        raise ValueError('Forecast belongs to a different measured result')
    return {'label':'ESTIMATE_ONLY_one_observed_plus_four_UNRUN_environment_cohorts',
            'actual_additional_cohorts_run':0,'training_seed_variance_measured':False,
            'mean_percent':100*q['forecast_five_cohort_mean_median'],
            'sample_SD_percentage_points':100*q['forecast_five_cohort_sample_SD_median'],
            'mean_95_prediction_interval_percent':[100*x for x in q['forecast_five_cohort_mean_95_interval']],
            'sample_SD_95_prediction_interval_pp':[100*x for x in q['forecast_five_cohort_sample_SD_95_interval']],
            'point_summary_is_posterior_medians_of_mean_and_SD_NOT_observed_five_seed_mean_SD':True}


def measured_model(base,model,seeds):
    root=base/model;progress=R59/'episodes/progress.json' if model=='xvla' else root/'episodes/progress.json'
    p=read(progress) if progress.exists() else {'rows':[],'status':'not_started'}
    rows=p.get('rows',[]);counts=per_action(rows,seeds,200)
    results=root/'RESULTS.json';cert=R59/'FINAL_AUDIT.json' if model=='xvla' else root/'FINAL_AUDIT.json'
    audited=False
    if results.exists() and cert.exists():
        r=read(results);proof=read(cert)
        audited=bool(proof.get('passed')) and len(rows)==800 and r.get('actual_completed_trials')==800
        if audited:
            for t in TASKS:
                actual=r['statistics']['by_task'][t];v=counts[t]
                if actual['successes']!=v['successes'] or actual['trials']!=v['completed']:
                    raise ValueError('Final summary differs from measured action rows')
    predictions=root/'FORECAST_NOT_MEASURED_5_SEEDS.json'
    doc=read(predictions) if audited and predictions.exists() else None
    for t in TASKS:
        counts[t].update(final800_audit_passed=audited,
            status='audited_complete' if audited else 'complete_pending_audit' if len(rows)==800 else p.get('status','pending'),
            source_progress=str(progress),
            forecast=forecast_for_task(doc,t,counts[t],audited) if doc else None)
    return counts


def preliminary():
    records=[]
    jobs=[('R56 main frozen120',R56/'automatic_v1',m,m+'_final120') for m in MODELS]
    jobs.append(('R56 Adapter refit frozen120',R56/'adapter_stage_restart_v1','vla_adapter','final120'))
    for scope,root,model,folder in jobs:
        p=root/folder/'progress.json'
        if not p.exists():continue
        value=read(p);rows=value.get('rows',[])
        seed_set={r['seed'] for r in rows}
        # These are different, explicitly labelled cohorts. NEVER pool them
        # with800or use them to predict the uncompleted800model's final SR.
        counts=per_action(rows,seed_set,30)
        for task,v in counts.items():
            records.append({'scope':scope,'model':model,'task':task,**v,
                            'source_progress':str(p),'status':value.get('status'),
                            'not_part_of_the800':True,'forecast_suppressed':True})
    return records


def refresh(base,out):
    out.mkdir(parents=True,exist_ok=True)
    declaration=read(base/'PREDECLARATION.json');seeds=declaration['seeds']
    if len(seeds)!=200 or len(set(seeds))!=200:raise ValueError('Need the actual declared200scene blocks')
    owner=read(base/'owner_status.json');models={m:measured_model(base,m,seeds) for m in MODELS}
    prelim=preliminary();now=datetime.now(timezone.utc).astimezone(VN).isoformat(timespec='seconds')
    result={'updated_VN':now,'scope':'separate_actions_ONLY_no_pooled_or_cross_action_average',
            'future_four_cohorts_are_UNRUN_forecasts_not_measured_seeds':True,
            'by_action':{},'pipeline_stage':owner.get('stage'),'pipeline_status':owner.get('status')}
    main=['# Báo cáo riêng từng action','',f'Cập nhật giờ Việt Nam: {now}','',
          'Mỗi model: một cohort800episode =200/action. Không có SR trung bình gộp bốn action trong bộ báo cáo này.',
          'Mean ± std dự báo = một cohort đã đo + bốn cohort môi trường CHƯA CHẠY, cùng checkpoint.',
          'Không được gọi các dự báo này là kết quả5seed đo thật hoặc std giữa các lần retraining.','',
          '|Action|Báo cáo|','|---|---|']
    csv_measured=[];csv_forecast=[]
    for task in TASKS:
        label=SLUGS[task];main.append(f'|{label}|[{label}.md]({label}.md)|')
        lines=[f'# {label} — báo cáo riêng','',f'Cập nhật giờ Việt Nam: {now}','',
            '## Đo thật — cohort200episode cho action này','',
            '|Model|Thành công / đã chạy|Tiến độ|SR đo thật|Wilson95|Trạng thái|',
            '|---|---:|---:|---:|---|---|']
        block={}
        for model in MODELS:
            v=models[model][task];block[model]=v
            sr='chưa có' if v['measured_SR'] is None else f"{v['measured_SR']:.1%}"
            ci=v['measured_Wilson95'];interval='—' if ci is None else f'{ci[0]:.1%}–{ci[1]:.1%}'
            state='đã audit đủ800' if v['final800_audit_passed'] else 'CHƯA FINAL'
            lines.append(f"|{NAMES[model]}|{v['successes']}/{v['completed']}|{v['completed']}/200|{sr}|{interval}|{state}|")
            csv_measured.append([label,model,v['successes'],v['completed'],200,
                '' if v['measured_SR'] is None else v['measured_SR']*100,
                '' if ci is None else ci[0]*100,'' if ci is None else ci[1]*100,
                v['final800_audit_passed'],v['status'],v['source_progress']])
        lines+=['','## ƯỚC LƯỢNG — thêm4cohort chưa chạy, KHÔNG phải đo5seed','',
            'Cột dưới là posterior median của mean5cohort và posterior median của sample std5cohort.',
            'Std tính bằng điểm phần trăm; không phải sai số tương đối hoặc khoảng tin cậy.','',
            '|Model|Mean ± std DỰ BÁO|Khoảng dự đoán95% của mean|',
            '|---|---:|---|']
        for model in MODELS:
            q=block[model]['forecast']
            if q:
                lo,hi=q['mean_95_prediction_interval_percent']
                lines.append(f"|{NAMES[model]}|{q['mean_percent']:.2f}% ± {q['sample_SD_percentage_points']:.2f} điểm %|{lo:.2f}%–{hi:.2f}%|")
                csv_forecast.append([label,model,q['mean_percent'],q['sample_SD_percentage_points'],lo,hi,
                    *q['sample_SD_95_prediction_interval_pp'],'ESTIMATE_ONLY_4_UNRUN_COHORTS'])
            else:lines.append(f'|{NAMES[model]}|chờ đủ800và audit|—|')
        lines+=['','## Tiến độ cohort120 RIÊNG — không gộp với800','',
            '|Cohort|Model|Thành công / đã chạy action này|SR|', '|---|---|---:|---:|']
        for v in prelim:
            if v['task']!=task:continue
            sr='—' if not v['completed'] else f"{v['measured_SR']:.1%}"
            lines.append(f"|{v['scope']}|{NAMES[v['model']]}|{v['successes']}/{v['completed']}|{sr}|")
        lines+=['','Kết quả chưa đủ200/action hoặc chưa qua audit là tiến độ, không phải kết quả cuối để so sánh trong paper.',
            'Chỉ tiêu PUT giữ nguyên: vào vùng đích, không bắt buộc prior grasp/release/settle; MOVE cũng không bắt buộc prior grasp.',
            'Đây là simulator-privileged VLA, giữ common controller/contact assist/safety. Không có Oracle policy mới.']
        atomic_text(out/(label+'.md'),'\n'.join(lines)+'\n')
        result['by_action'][label]={'models':block,
            'separate120_progress':[v for v in prelim if v['task']==task]}
    main+=['','Pipeline: `'+str(owner.get('stage'))+'`.','',
        'Checkpoint chỉ chọn từ development/confirmation; không sửa recipe dựa trên800episode.',
        'Lỗi tích hợp hoặc trial chưa chạy không được điền thành0% success.','',
        '[CSV đo thật](MEASURED_BY_ACTION.csv) · [CSV dự báo](FORECAST_BY_ACTION.csv) · [JSON](PER_ACTION.json)']
    atomic_text(out/'REPORT.md','\n'.join(main)+'\n')
    for name,header,rows in (
        ('MEASURED_BY_ACTION.csv',['action','model','successes','completed','planned','measured_SR_percent',
            'Wilson95_low_percent','Wilson95_high_percent','full800_audit_passed','status','source_progress'],csv_measured),
        ('FORECAST_BY_ACTION.csv',['action','model','forecast_mean_percent','forecast_sample_SD_pp',
            'forecast_mean95_low_percent','forecast_mean95_high_percent','forecast_SD95_low_pp','forecast_SD95_high_pp','label'],csv_forecast)):
        stream=StringIO();writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
        atomic_text(out/name,stream.getvalue())
    atomic_json(out/'PER_ACTION.json',result)
    return owner


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'.report.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from VLA.common.provenance import sha256_file
        source_sha=sha256_file(__file__);deadline=time.monotonic()+a.hours*3600
        while True:
            try:
                if sha256_file(__file__)!=source_sha:raise ValueError('Running report-only source changed')
                owner=refresh(a.base,a.output)
                done=owner.get('status')=='completed_all_four_actual800'
                state='completed_all_action_reports' if done else 'watching_read_only'
                atomic_json(a.output/'reporter_status.json',{'status':state,'pid':os.getpid(),
                    'source_sha256':source_sha,'source':str(Path(__file__).resolve()),
                    'updated_VN':datetime.now(timezone.utc).astimezone(VN).isoformat(),
                    'running_benchmark_or_model_files_modified':False})
                if not a.watch or done:break
                if time.monotonic()>deadline:raise TimeoutError('Finite reporting watch expired; pipeline itself is NOT stopped')
                time.sleep(a.interval)
            except Exception as e:
                atomic_json(a.output/'REPORTING_ERROR.json',{'error':f'{type(e).__name__}: {e}',
                    'benchmark_not_stopped_or_changed':True,'last_good_report_preserved':True})
                raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--output',type=Path,default=BASE/'per_action');p.add_argument('--watch',action='store_true')
    p.add_argument('--interval',type=float,default=30);p.add_argument('--hours',type=float,default=72)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'reporter.log').open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-u','-m','VLA.action_reports_r63.report','--watch',
                '--base',str(a.base.resolve()),'--output',str(a.output.resolve()),
                '--interval',str(a.interval),'--hours',str(a.hours)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'reporter_pid':proc.pid,'output':str(a.output.resolve())}))
    else:main(a)
