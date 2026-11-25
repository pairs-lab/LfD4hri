"""Live counts based on real output and actual signed visual reviews only."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import time
from VLA.common.artifact_io import atomic_json
from .common import OUT,read


def safe(path,default):
    try:return read(path)
    except (FileNotFoundError,json.JSONDecodeError):return default


def mechanism(evidence):
    if evidence.get('full_put_complete'):return 'Full PUT / released and supported'
    events=evidence.get('events',[])
    if not events:return 'No complete PUT; inspect actual evidence'
    event=max(events,key=lambda e:e.get('object_displacement_within250ms_m',0))
    mode=event['mode']
    if mode=='contact_confirmed_impact_displacement_NOT_automatically_elastic_bounce':
        return f"Impact/ejection: {100*event['object_displacement_within250ms_m']:.1f} cm / 250 ms"
    if mode=='verified_in_transit_drop':return f"Verified transit drop: {100*event['measured_fall_m']:.1f} cm fall"
    if mode=='empty_gripper_lift_after_failed_acquisition_no_object_lift':return f"Failed acquisition / empty lift {100*event['TCP_lift_m']:.1f} cm"
    if mode=='controlled_Oracle_transport_timeout':return 'Controlled slow-transport timeout / 22 s'
    if mode=='controlled_Oracle_failed_acquisition_empty_closure':return 'Controlled failed acquisition / closed jaws'
    return mode.replace('_',' ')


def publish():
    assignments=safe(OUT/'assignments.json',[])
    produced=sorted(safe(OUT/'produced_pairs.json',[]),key=lambda row:(bool(row['backup']),row['scene_id']))
    reviews=[]
    for row in produced:
        if not row.get('paired_output'):continue
        check=safe(Path(row['paired_output'])/'FINAL_VISUAL_REVIEW.json',{})
        if check.get('approved_for_delivery'):reviews.append(row)
    counts={name:{'required_pairs':n,'CPU_selected':0,'media_pairs':0,'visually_certified_pairs':0} for name,n in (
        ('01_RL_succeed_VLA_succeed',2),('02_RL_fail_VLA_succeed',3),
        ('03_RL_succeed_VLA_fail',3),('04_RL_fail_VLA_fail',3),('05_backups',4))}
    for row in assignments:counts[row['group']]['CPU_selected']+=1
    for row in produced:
        if row.get('paired_output'):counts[row['group']]['media_pairs']+=1
    for row in reviews:counts[row['group']]['visually_certified_pairs']+=1
    done=safe(OUT/'production_done.json',{})
    complete=(all(v['visually_certified_pairs']>=v['required_pairs'] for v in counts.values())
        and bool(done) and not done.get('errors') and len(reviews)==len(assignments)==len(produced)
        and safe(OUT/'FINAL_AUDIT.json',{}).get('passed',False))
    stamp=datetime.now(timezone.utc).isoformat()
    summary={'updated_UTC':stamp,'groups':counts,'delivery_complete':complete,
        'certified_main_principal_videos':2*sum(v['visually_certified_pairs'] for k,v in counts.items() if k!='05_backups'),
        'certified_backup_principal_videos':2*counts['05_backups']['visually_certified_pairs'],
        'principal_aliases_NOT_independent_additional_trials':True}
    atomic_json(OUT/'COVERAGE.json',summary)
    lines=['# R58 —22principal videos + backup, actual coverage','',f'Updated UTC: {stamp}.',
        '',f'Delivery complete: **{complete}**. Certified main videos: {summary["certified_main_principal_videos"]}/22. Certified backup videos: {summary["certified_backup_principal_videos"]}.',
        '', '|Group|Required pairs|CPU selected|7-camera media pairs|Actual visual review passed|',
        '|---|---:|---:|---:|---:|']
    for key,v in counts.items():lines.append(f'|{key}|'+ '|'.join(str(x) for x in v.values())+'|')
    lines+=['','## Actual scene outcomes and measured mechanisms','',
        'All tasks are PUT on the paired UR5e + Robotiq2F-85 plant. B-prefix means backup, not another principal scene ID.',
        '', '|ID|Actual instruction|Oracle observation|X-VLA + GT observation|', '|---|---|---|---|']
    for row in produced:
        if not row.get('paired_output'):continue
        folder=Path(row['paired_output']);task=read(folder/'XVLA_terminal_release/initial_scene.json')['task']
        oracle=read(folder/'Scripted_Oracle/content_evidence.json');vla=read(folder/'XVLA_terminal_release/content_evidence.json')
        label=('B' if row['backup'] else '')+f'{row["scene_id"]:02d}'
        lines.append(f'|{label}|{task["prompt"]}|{mechanism(oracle)}|{mechanism(vla)}|')
    lines+=['','## Provenance', '',
        '- RL folder is privileged feedback Oracle with disclosed controlled faults, not learned RL.',
        '- A fault name is not an outcome: Backup04 Oracle actually recovers/succeeds despite its persistent_miss offset; both backup videos are correctly success-labeled.',
        '- X-VLA + simulator GT step014500: genuine saved neural-action physical replay; not vanilla X-VLA, new NN inference or a new SR trial. Original front/wrist RGB + third GT diagram + native proprio/67 numeric fields. Opening-only terminal supervisor is disclosed.',
        '- Physics, policy observations and action decoding remain on the pinned original PyBullet binding. A separate NumPy render replica is gated by lossless RGB AND segmentation against pinned golden images for each rollout. The initial21-image probe was pixel-exact.',
        '- Seven streams per rollout:6wide1920×1080 plus handeye1280×720 at30FPS. Paired common layout, all-stream full decode/count/physical-tick/inset sync audited.',
        '- Selected qualitative scenes do not establish a quantitative SR, memory loss, wrong-object understanding or an intrinsic architecture ceiling.',
        '', '## Current owners', '']
    for folder in ('capture_v1','produce_v1'):
        s=safe(OUT/folder/'status.json',{})
        lines.append(f'- {folder}: {s.get("status","not started")} / PID{s.get("pid")} / child{s.get("child")} / {s.get("active_label",s.get("active",""))}')
    lines+=['','## Produced cases','']
    for row in produced:
        visual=bool(row.get('paired_output') and safe(Path(row['paired_output'])/'FINAL_VISUAL_REVIEW.json',{}).get('approved_for_delivery'))
        state='actual_encoded_visual_review_PASSED' if visual else row['status']
        lines.append(f'- {row["scene_id"]} / seed{row["seed"]} / backup={row["backup"]}: {state}'+(' — '+row['error'] if row.get('error') else ''))
    lines+=['','## Audit and interpretation','',
        '- [Evidence and counterarguments](AUDIT_AND_LIMITATIONS.md): controlled Oracle faults versus observed VLA behavior; what these clips cannot establish.',
        '- [Rejected camera-clipped banana and its validated replacement](audit/replacement_scene11_camera_and_content.json). The rejected candidate remains in the ledger.',
        '- [Independent whole-stream packet timestamps](audit/whole_stream_packet_PTS.json). This is separate from whole-stream decoding and sampled visual inspection.',
        '- [Actual THREE-view input receipts](audit/third_privileged_visual_input_receipts.json):68 query GT-image hashes and53 full native action-chunk hashes verified;15 partial terminal chunks explicitly not full-hash-verified.',
        '- Each produced pair contains EXPLANATION.md, the actual encoded before/event/after/terminal images, seven camera streams per method, and exact replay/repeat proofs.','']
    lines+=['Only the five numbered outcome/backup groups contain delivered scenes. `_captures`, `_renders` and `_pre_renders` retain internal/raw/rejected artifacts and are not additional reportable trials. The delivery manifest excludes incomplete staging movies.','']
    (OUT/'REPORT.md.tmp').write_text('\n'.join(lines)+'\n');(OUT/'REPORT.md.tmp').replace(OUT/'REPORT.md')
    parts=['<!doctype html><meta charset="utf-8"><title>R58 paired PUT qualitative</title>',
        '<style>body{font:16px system-ui;margin:24px auto;max-width:1440px;background:#f4f6f9;color:#18212a}section{background:#fff;padding:18px;margin:18px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}video{width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}</style>',
        '<h1>PUT — scripted Oracle / X-VLA + simulator GT + terminal release</h1>',
        f'<p>Certified main videos: {summary["certified_main_principal_videos"]}/22; backups: {summary["certified_backup_principal_videos"]}. Delivery complete: {complete}.</p>',
        '<p>Oracle is not trained RL. Controlled faults and terminal opening are disclosed. Outcome-selected scenes are not quantitative success-rate samples.</p>',
        '<p><a href="REPORT.md">Report</a> · <a href="COVERAGE.json">Coverage</a> · <a href="PLAN.md">Plan</a> · <a href="candidate_ledger.json">All attempted candidates</a> · <a href="AUDIT_AND_LIMITATIONS.md">Audit / counterarguments</a></p>',
        '<table><tr><th>Group</th><th>Required pairs</th><th>Selected</th><th>Media</th><th>Visual pass</th></tr>']
    for key,v in counts.items():parts.append('<tr><td>'+key+'</td>'+''.join(f'<td>{x}</td>' for x in v.values())+'</tr>')
    parts.append('</table>')
    for row in produced:
        if not row.get('paired_output'):continue
        folder=Path(row['paired_output']);audit=safe(folder/'PAIR_AUDIT.json',{})
        approved=safe(folder/'FINAL_VISUAL_REVIEW.json',{}).get('approved_for_delivery',False)
        label='backup' if row['backup'] else 'scene'
        parts.append(f'<section><h2>{html.escape(row["group"])} — {label}{row["scene_id"]:02d}, seed{row["seed"]}</h2><p>Visual certification: {approved}; Oracle fault: {html.escape(row["oracle_fault"])}. <a href="{folder.relative_to(OUT)}/PAIR_AUDIT.json">Audit + event frames</a></p><div class="pair">')
        for method in ('Scripted_Oracle','XVLA_terminal_release'):
            entry=audit.get('outcomes',{}).get(method,{})
            alias=Path(entry['delivery_alias']).relative_to(OUT)
            parts.append(f'<div><h3>{method} — full PUT: {entry.get("full_put_complete")}</h3><video controls preload="metadata" src="{alias}"></video><p><a href="{alias.with_suffix(".txt")}">Notes/caption</a></p></div>')
        relative=folder.relative_to(OUT)
        parts.append(f'</div><p><a href="{relative}/EXPLANATION.md">Scene explanation / alternative interpretations</a> · <a href="{relative}/index.html">Full pair evidence</a></p>')
        if approved:
            parts.append(f'<details><summary>Actual encoded before / event / after / terminal (Oracle top, VLA bottom)</summary><img style="width:100%" src="{relative}/review/wide_sequence.jpg"><img style="width:100%" src="{relative}/review/handeye_sequence.jpg"></details>')
        parts.append('</section>')
    (OUT/'index.html.tmp').write_text('\n'.join(parts));(OUT/'index.html.tmp').replace(OUT/'index.html')
    print({'coverage':counts,'delivery_complete':complete},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');a=parser.parse_args()
    deadline=time.monotonic()+36*3600
    while True:
        publish()
        if not a.watch or time.monotonic()>deadline:break
        time.sleep(20)
