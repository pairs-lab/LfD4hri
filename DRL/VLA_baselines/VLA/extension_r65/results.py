"""Read-only six-model view, plus native action audit before any new forecast."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.cross_vla800_r62.common import matrix,SEEDS
from .common import R62,TASKS,read,utc,verify


def audit(folder):
    freeze=read(folder/'FROZEN_BEFORE_800.json');verify(freeze['checkpoint_sha256'])
    p=read(folder/'episodes/progress.json');matrix(p['rows'])
    if p['checkpoint']!=freeze['checkpoint']:raise ValueError('Measured wrong checkpoint')
    journal=[json.loads(s) for s in (folder/'episodes/NEURAL_QUERY_RECEIPTS.jsonl').read_text().splitlines()]
    query_index=controls=0;by_task={t:Counter() for t in TASKS}
    lookup={(r['seed'],r['task']):r for r in p['rows']}
    for seed in SEEDS:
        traces=read(folder/f'episodes/test_seed{seed}/trajectory.json')
        for task in TASKS:
            trace=traces[task];row=lookup[seed,task]
            if len(trace)!=row['steps'] or not trace or trace[-1]['success']!=row['success']:
                raise ValueError('Outcome differs from genuine saved trajectory')
            by_task[task]['trials']+=1;by_task[task]['successes']+=row['success']
            by_task[task]['safety_aborts']+=bool(row['safety_violation'])
            groups=[]
            for entry in trace:
                q=entry['neural_receipt']['query_step']
                if not groups or q!=groups[-1][0]:groups.append((q,[]))
                groups[-1][1].append(entry)
            for _,chunk in groups:
                receipt=journal[query_index];query_index+=1
                if receipt['seed']!=seed or receipt['task']!=task or receipt['native_views']!=3 or not receipt['GT_map_SHA256']:
                    raise ValueError('Chronological actual input receipt mismatch')
                native=receipt['actual_native_input_receipt']
                if freeze['model']=='vla_jepa':
                    if len(native['actual_Qwen_image_grid_thw'])!=3 or native['actual_DiT_state_dimensions']!=67:
                        raise ValueError('Actual frozen JEPA privileged input consumption differs')
                elif (native['actual_native_views']!=3 or native['actual_state_dimensions']!=32
                      or not native['complete_scene64_text_verified']):
                    raise ValueError('Actual frozen FAST privileged input consumption differs')
                actions=np.asarray(receipt['actions'],np.float32)
                for i,entry in enumerate(chunk):
                    if (entry['neural_receipt']['input_sha256']!=receipt['input_sha256']
                        or not np.array_equal(np.asarray(entry['action'],np.float32),actions[i])):
                        raise ValueError('Executed command not from actual native VLA chunk')
                controls+=len(chunk)
    if query_index!=len(journal) or query_index!=p['native_queries'] or controls!=p['motor_controls']:
        raise ValueError('Native query/control counts differ')
    if any((folder/'episodes').rglob('*.mp4')) or any((folder/'episodes').rglob('*.png')):
        raise ValueError('Unexpected media in final800')
    atomic_json(folder/'FINAL_AUDIT.json',{'passed':True,'actual_trials':800,'actual_per_action':200,
        'actual_native_queries':query_index,'actual_motor_controls':controls,
        'all_executed_neural_prefixes_matched_chronological_RPC':True,
        'all_safety_aborts_retained':True,'by_task':{t:dict(v) for t,v in by_task.items()},'UTC':utc()})
    from VLA.cross_vla800_r62.report import measured
    measured(freeze['model'],p['rows'],folder,folder/'episodes/progress.json')


def refresh(out,status):
    model_roots={m:R62/m for m in ('xvla','smolvla','pi0','vla_adapter')}
    model_roots.update({m:out/m for m in ('vla_jepa','pi0_fast')})
    lines=['# R65 — sáu native VLA families, báo cáo riêng action','',
        f'Cập nhật UTC: {utc()}', '', 'Existing R62 runs unchanged. New models are NOT measured until their800+audit complete.',
        'PUT=goal-region first hit, NOT release/settle. All methods are simulator-privileged; input modes/PEFT differ.', '',
        '|Model|800 hoàn thành|Touch|Pick|Move|Put region|','|---|---:|---:|---:|---:|---:|']
    for model,root in model_roots.items():
        result=root/'RESULTS.json';progress=root/'episodes/progress.json'
        if result.exists():
            r=read(result);b=r['statistics']['by_task']
            cells=[f"{b[t]['successes']}/200 ({100*b[t]['success_rate']:.1f}%)" for t in TASKS]
            lines.append('|'+model+'|800/800|'+'|'.join(cells)+'|')
        else:
            d=read(progress) if progress.exists() else {'completed_trials':0}
            lines.append(f"|{model}|{d['completed_trials']}/800|pending|pending|pending|pending|")
    lines+=['','## Forecast ONLY — 1 observed +4 UNRUN cohorts, same checkpoint','',
        'Not measured five-seed results or retraining variance. Std units=percentage points, sample ddof=1.',
        '|Model|Touch|Pick|Move|Put region|','|---|---:|---:|---:|---:|']
    for model,root in model_roots.items():
        p=root/'FORECAST_NOT_MEASURED_5_SEEDS.json'
        if p.exists():
            q=read(p)['by_task'];cells=[f"{q[t]['forecast_five_cohort_mean_median']*100:.2f}% ± {q[t]['forecast_five_cohort_sample_SD_median']*100:.2f} pp" for t in TASKS]
            lines.append('|'+model+'|'+'|'.join(cells)+'|')
    lines+=['','Pipeline: `'+status.get('stage','pending')+'`.',
        'Errors are preserved in owner_status.json, NOT interpreted as0% model SR.',
        'VLA-JEPA: frozen pretrained Qwen + trained native DiT/state67; world auxiliary loss off. Not full paper retraining.',
        'pi0-FAST: official32×32 codec,7 physical EEF channels; own gate/reload before finite native LoRA fit.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
