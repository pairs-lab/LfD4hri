"""Browse extracted PNGs; prepare group sheets for actual human inspection."""
import argparse
from datetime import datetime,timezone
import csv
import html
from pathlib import Path
import cv2
import numpy as np
from .export import OUTPUT,GROUPS,METHODS,CAMERAS,read,write,sha


def event_index(evidence,n):
    if evidence['full_put_complete']:return min(n-1,evidence['stable_completion_frame'])
    events=evidence.get('events',[])
    if events:return int(np.clip(max(events,key=lambda e:e.get('object_displacement_within250ms_m',0))['event_frame'],0,n-1))
    return n//2


def tile(path,label):
    image=cv2.imread(str(path))
    if image is None:raise ValueError('Cannot read exported PNG: '+str(path))
    result=np.full((204,320,3),245,np.uint8)
    result[:180]=cv2.resize(image,(320,180),interpolation=cv2.INTER_AREA)
    cv2.putText(result,label,(5,197),cv2.FONT_HERSHEY_SIMPLEX,.37,(20,20,20),1,cv2.LINE_AA)
    return result


def prepare(out):
    data=read(out/'INVENTORY.json');assert read(out/'FINAL_AUDIT.json')['passed']
    directory=out/'review';directory.mkdir(exist_ok=False);records=[]
    top=['<!doctype html><meta charset="utf-8"><title>qualitative_rl_vla frames</title>',
         '<style>body{font:16px sans-serif;margin:24px}td,th{padding:7px;border:1px solid #ccc}table{border-collapse:collapse}code{background:#eee}</style>',
         '<h1>qualitative_rl_vla — full-resolution frames</h1>',
         f'<p>{data["frames"]:,} PNG frames · {data["stream_count"]} camera slots · ONLY views 2, 3, 4, 6 · 6 groups.</p>',
         '<p>26 principal clips in the main set (13 paired scenes); each selected camera shows the same rollout, not an extra trial. Backups and the duplicated 06_new group are included separately.</p>',
         '<p>Parent scene name encodes Oracle outcome; both Oracle and X-VLA are inside. RL alias means scripted feedback Oracle, NOT trained RL. All frame files are lossless, unannotated original-resolution images.</p>',
         '<p><a href="README.md">Structure and numbering</a> · <a href="REPORT.md">Report</a> · <a href="FINAL_AUDIT.json">Audit</a></p>']
    table=[]
    for group in GROUPS:
        panels=[];top.append(f'<h2>{html.escape(group)}</h2><table><tr><th>Scene</th><th>Instruction</th><th>Oracle PNG viewer</th><th>X-VLA PNG viewer</th></tr>')
        for pair in [p for p in data['pairs'] if p['group']==group]:
            viewer_links=[]
            for method in METHODS:
                rows=[r for r in data['streams'] if r['output_scene']==pair['output_scene'] and r['method']==method]
                dest=out/pair['output_scene']/method;n=rows[0]['frames']
                evidence=read(dest/'content_evidence.json');index=event_index(evidence,n)
                cameras=list(CAMERAS);thumbs=[];sources=[]
                for camera in cameras:
                    p=dest/camera/f'frame{index+1:06d}.png'
                    thumbs.append(tile(p,f'{Path(pair["output_scene"]).name} {method[:1]} {camera} f{index+1}'))
                    sources.append({'camera':camera,'frame_1_based':index+1,'PNG':str(p.relative_to(out)),
                                    'PNG_sha256':sha(p)})
                panels.append(np.hstack(thumbs))
                records.append({'group':group,'scene':pair['output_scene'],'method':method,'frames':sources,
                    'frame_scope':'one actual exported event/settle frame from ALL4 requested cameras',
                    'actual_visual_review_pending':True})
                label=html.escape(pair['instruction']+' — '+method)
                options=''.join(f'<option value="{c}">{c}</option>' for c in cameras)
                document=f'''<!doctype html><meta charset="utf-8"><title>{label}</title>
<style>body{{font:16px sans-serif;margin:20px}}img{{max-width:100%;height:auto}}input[type=range]{{width:65%}}code{{background:#eee}}</style>
<h2>{label}</h2><p>{n} frames, original30FPS. PNG numbering starts at1; video index starts at0.</p>
<select id="view">{options}</select> <input id="seek" type="range" min="1" max="{n}" value="1">
<input id="number" type="number" min="1" max="{n}" value="1"><p id="info"></p><p><a id="full" target="_blank">Open full-resolution PNG</a></p><img id="image">
<script>const n={n};const view=document.getElementById('view'),seek=document.getElementById('seek'),number=document.getElementById('number');
function draw(value){{let k=Math.max(1,Math.min(n,Number(value)||1));seek.value=k;number.value=k;let path=view.value+'/frame'+String(k).padStart(6,'0')+'.png';document.getElementById('image').src=path;document.getElementById('full').href=path;document.getElementById('info').textContent=path+' | video index '+(k-1)+' | encoded PTS '+((k-1)/30).toFixed(6)+'s | simulation '+(k/30).toFixed(6)+'s';}}
view.onchange=()=>draw(number.value);seek.oninput=()=>draw(seek.value);number.oninput=()=>draw(number.value);draw(1);</script>'''
                (dest/'index.html').write_text(document)
                viewer_links.append(str((dest/'index.html').relative_to(out)))
            top.append(f'<tr><td>{html.escape(Path(pair["output_scene"]).name)}</td><td>{html.escape(pair["instruction"])}</td>'+''.join(f'<td><a href="{s}">Browse all views/frames</a></td>' for s in viewer_links)+'</tr>')
            table.append([group,Path(pair['output_scene']).name,pair['seed'],pair['Oracle_success'],pair['VLA_success'],pair['instruction'],pair['source_scene']])
        top.append('</table>')
        grid=np.vstack(panels);name=group+'_all_methods_all_views.jpg'
        cv2.imwrite(str(directory/name),grid,[cv2.IMWRITE_JPEG_QUALITY,96])
        top.append(f'<p><a href="review/{name}">Review sheet: all scenes, BOTH methods, ALL4 requested cameras</a></p>')
    (out/'index.html').write_text('\n'.join(top))
    with (out/'SCENE_MAP.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['group','output_scene','seed','Oracle_success','VLA_success','instruction','source_scene']);writer.writerows(table)
    readme=f'''# qualitative_rl_vla

{data['frames']:,} full-resolution lossless PNG frames, all six groups, {len(data['pairs'])} paired scene slots.

Example:

    02_RL_fail_VLA_succeed/
      scene03_RL_failure/
        Scripted_Oracle/
          view_2.mp4
          view2/frame000001.png ... frame000306.png
          view2.timestamps.tsv
          view2.sha256
          view2.audit.json
          view3/ ... view4/ ... view6/
        XVLA_terminal_release/
          view2/ ... view3/ ... view4/ ... view6/

Every scene contains BOTH methods. Parent folder success/failure is the Oracle outcome, not the VLA outcome.
The 06_new contents are preserved as found (currently four backup pairs duplicated from05_backups).
Decoded duplicates are copied into each folder; PNGs do NOT share hardlink inodes, so editing one does not edit another.

frame000001.png = original video frame0. No frames skipped, no resize, no crop, no added text, no FPS change.
Only views2/3/4/6 are included, as requested. PNGs are1920×1080. Source videos are copied without re-encoding.
Encoded PTS0 corresponds to source simulation tick8 (0.033333s); exact per-frame mapping is in each .timestamps.tsv.
Use index.html to select any method/view/frame, or open the PNG directories directly.

Checksums protect the ORIGINAL exported PNGs. Save annotated figures under a different filename/folder;
in-place edits will correctly invalidate that frame's audit checksum.

RL is a folder alias for a scripted feedback Oracle, NOT trained RL. These are camera views and selected
qualitative examples, not additional independent trials. Source R60 videos/metadata and training/checkpoints remain untouched.
'''
    (out/'README.md').write_text(readme)
    write(directory/'EXTRACTION.json',{'records':records,'images':{
        p.name:sha(p) for p in directory.glob('*.jpg')},'actual_visual_review_pending':True,
        'labels_and_resizing_ONLY_on_review_sheets_NOT_original_exported_PNGs':True})
    print({'review_sheets':6,'scene_method_slots':len(records),'sampled_exported_frames':sum(len(r['frames']) for r in records)},flush=True)


def approve(out,note):
    if not note:raise ValueError('Actual image review findings required')
    review=read(out/'review/EXTRACTION.json')
    for name,h in review['images'].items():
        if sha(out/'review'/name)!=h:raise ValueError('Review sheet changed')
    audit=read(out/'FINAL_AUDIT.json');assert audit['passed']
    audit.update(actual_visual_contact_sheet_review_pending=False,
        actual_visual_review={'opened_group_sheets':list(review['images']),
            'all21_scene_slots_both_methods_all4_requested_cameras_sampled':True,'findings':note,
            'UTC':datetime.now(timezone.utc).isoformat(),
            'continuous_every_frame_human_viewing_NOT_claimed':True})
    write(out/'FINAL_AUDIT.json',audit)
    write(out/'PROGRESS.json',{'status':'complete_all_frames_and_automated_audit_and_sampled_visual_review',
        'streams_completed':audit['video_stream_slots'],'streams_total':audit['video_stream_slots'],
        'frames_completed':audit['PNG_frames'],'frames_total':audit['PNG_frames'],
        'UTC':datetime.now(timezone.utc).isoformat()})
    print({'complete':True,'frames':audit['PNG_frames'],'streams':audit['video_stream_slots']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--approve',action='store_true');p.add_argument('--note');a=p.parse_args()
    if a.approve:approve(a.output,a.note)
    else:prepare(a.output)
