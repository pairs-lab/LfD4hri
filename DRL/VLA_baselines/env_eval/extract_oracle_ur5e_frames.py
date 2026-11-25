"""Export every source frame from the user's UR5e views3/4/6, without simulation.

python -m env_eval.extract_oracle_ur5e_frames all
Reuses the already audited R61 PNG/decoder helpers, without changing R61.
One low-priority CPU worker; no GPU, environment, policy or physics imports.
"""
import argparse
import csv
import fcntl
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np

from VLA.frame_export_r61 import export as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper/UR5e_Robotiq2F'
OUTPUT = ROOT/'VLA/qualitative/qualitative_oracle_ur5e_views346'
ACTIONS = ('touch','pick','move','put')
VIEWS = (3,4,6)
RESERVE = 30 * 1024**3


def space(out, needed=0):
    free=shutil.disk_usage(out).free
    if free-needed < RESERVE:
        raise RuntimeError('Keep30GiB free for the existing VLA pipeline; partial extraction preserved')
    return free


def command(video, target):
    cmd=shared.png_encode_command(video,target)
    return cmd[:1]+['-filter_threads','1']+cmd[1:]


def expected_paths(source):
    return {source/action/f'config{config}'/f'view_{v}.mp4'
            for action in ACTIONS for config in range(1,5) for v in VIEWS}


def inventory(source,out):
    actual={p for p in source.rglob('*.mp4') if p.stem in {f'view_{v}' for v in VIEWS}}
    if actual != expected_paths(source):
        raise ValueError('Missing/unexpected requested source videos: '+str(actual ^ expected_paths(source)))
    rows=[];seal={};episodes=[]
    for action in ACTIONS:
        for config in range(1,5):
            folder=source/action/f'config{config}';trajectory=shared.read(folder/'trajectory.json')
            infos=[shared.metadata(folder/f'view_{v}.mp4') for v in VIEWS]
            if any(x != infos[0] for x in infos):raise ValueError('Source views are not synchronized')
            info=infos[0]
            if (info['width'],info['height'],info['fps'])!=(1920,1080,30.):
                raise ValueError('Unexpected native source format')
            if len(trajectory)!=info['frames']:
                raise ValueError('Video frame count differs from original simulation trajectory')
            times=np.asarray([f['time'] for f in trajectory],np.float64)
            np.testing.assert_allclose(times,np.arange(len(times))/30,rtol=0,atol=1e-8)
            for name in ('trajectory.json','audit.json','visibility.json','presentation_audit.json'):
                p=folder/name
                if p.exists():seal[str(p)]=shared.sha(p)
            episodes.append({'action':action,'config':config,'frames':info['frames'],
                'source_trajectory':str(folder/'trajectory.json'),
                'simulation_times_s':times.tolist(),
                'phases':[r['phase'] for r in trajectory]})
            for v in VIEWS:
                video=folder/f'view_{v}.mp4';digest=shared.sha(video);seal[str(video)]=digest
                rows.append({'id':len(rows)+1,'action':action,'config':config,'view':v,
                    'camera':f'view{v}','source':str(video),'source_sha256':digest,
                    'source_bytes':video.stat().st_size,'output_parent':f'{action}/config{config}',
                    'output_frames':f'{action}/config{config}/view{v}',**info})
    return {'source':str(source),'output':str(out),'selected_views':list(VIEWS),
        'episodes':episodes,'streams':rows,'stream_count':len(rows),
        'frames':sum(r['frames'] for r in rows),'source_files_sha256':seal,
        'CPU_only_single_worker':True,'original_source_read_only':True,'UTC':shared.utc()}


def plan(source,out):
    if (out/'INVENTORY.json').exists():
        data=shared.read(out/'INVENTORY.json')
        if data['source']!=str(source) or data['selected_views']!=list(VIEWS):
            raise ValueError('Existing inventory belongs to another source/scope')
        for path,digest in data['source_files_sha256'].items():
            if shared.sha(path)!=digest:raise ValueError('Inventoried source changed')
        if not shared.read(out/'PILOT_AND_ADMISSION.json')['passed']:raise ValueError('Pilot incomplete')
        return data
    data=inventory(source,out);shared.write(out/'INVENTORY.json',data)
    pilot=[]
    for row in data['streams']:
        space(out);indices=sorted({0,row['frames']//2,row['frames']-1})
        images=shared.raw_samples(row['source'],indices,row['width'],row['height'])
        folder=out/'_pilot'/row['output_frames'];folder.mkdir(parents=True,exist_ok=False)
        cmd=shared.png_encode_command(None,folder,raw=(row['width'],row['height']))
        p=subprocess.run(cmd[:1]+['-filter_threads','1']+cmd[1:],input=images.tobytes(),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=shared.THREAD_ENV)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
        paths=sorted(folder.glob('*.png'))
        if len(paths)!=len(indices):raise ValueError('Pilot frame count mismatch')
        for path,expected in zip(paths,images,strict=True):
            np.testing.assert_array_equal(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB),expected)
        sizes=[p.stat().st_size for p in paths]
        pilot.append({'stream':row['id'],'indices_0_based':indices,'mean_bytes':sum(sizes)/len(sizes),
            'maximum_bytes':max(sizes),'RGB_bit_exact':True})
        shared.write(out/'PROGRESS.json',{'status':'CPU_PNG_pilot','pilot_streams':len(pilot),
            'streams_total':len(data['streams']),'UTC':shared.utc()})
    estimate=sum(p['mean_bytes']*r['frames'] for p,r in zip(pilot,data['streams'],strict=True))
    conservative=sum(p['maximum_bytes']*r['frames']*1.3 for p,r in zip(pilot,data['streams'],strict=True))
    required=int(conservative+sum(r['source_bytes'] for r in data['streams'])+512*1024**2)
    free=space(out,required)
    shared.write(out/'PILOT_AND_ADMISSION.json',{'passed':True,'streams_sampled':len(pilot),'samples':pilot,
        'PNG_expected_GiB':estimate/1024**3,'PNG_conservative_GiB':conservative/1024**3,
        'free_GiB':free/1024**3,'reserve_GiB':30,'source_RGB_samples_bit_exact':True})
    shared.write(out/'EXTRACTOR_BINDING.json',{'python':sys.executable,
        'sources':{str(p):shared.sha(p) for p in (Path(__file__).resolve(),Path(shared.__file__).resolve())},
        'ffmpeg_binary':shared.FFMPEG,'ffmpeg_sha256':shared.sha(shared.FFMPEG)})
    print({'plan_passed':True,'streams':len(data['streams']),'frames':data['frames'],
        'PNG_GiB_estimate':round(estimate/1024**3,2)},flush=True)
    return data


def binding(out):
    b=shared.read(out/'EXTRACTOR_BINDING.json')
    for path,digest in {**b['sources'],b['ffmpeg_binary']:b['ffmpeg_sha256']}.items():
        if shared.sha(path)!=digest:raise ValueError('Bound extractor/decoder changed')


def run(out):
    data=shared.read(out/'INVENTORY.json');pilot=shared.read(out/'PILOT_AND_ADMISSION.json');binding(out)
    if not pilot['passed']:raise ValueError('Pilot gate not passed')
    results=[];done=0;started=time.monotonic()
    episodes={(r['action'],r['config']):r for r in data['episodes']}
    def progress(row=None,phase='extracting_all_frames',current=0):
        shared.write(out/'PROGRESS.json',{'status':phase,'pid':os.getpid(),
            'streams_completed':len(results),'streams_total':data['stream_count'],
            'frames_completed':done,'frames_total':data['frames'],
            'active_stream':None if row is None else row['output_frames'],
            'active_frames_encoded':current,'elapsed_s':time.monotonic()-started,
            'free_GiB':space(out)/1024**3,'UTC':shared.utc()})
    for row,sample in zip(data['streams'],pilot['samples'],strict=True):
        dest=out/row['output_frames'];parent=dest.parent;parent.mkdir(parents=True,exist_ok=True)
        audit_path=parent/(row['camera']+'.audit.json')
        if audit_path.exists():
            audit=shared.read(audit_path)
            if not audit['passed'] or audit['source_sha256']!=row['source_sha256'] or not dest.is_dir():
                raise ValueError('Existing output differs, will not overwrite')
            results.append(audit);done+=row['frames'];progress(row);continue
        if dest.exists():raise FileExistsError('Preserve unsealed existing output: '+str(dest))
        if shared.sha(row['source'])!=row['source_sha256']:raise ValueError('Source changed after pilot')
        space(out,int(sample['maximum_bytes']*row['frames']*1.3+row['source_bytes']))
        video=parent/f'view_{row["view"]}.mp4'
        if not video.exists():shutil.copy2(row['source'],video)
        if shared.sha(video)!=row['source_sha256']:raise ValueError('Copied source video differs')
        stage=parent/('.'+row['camera']+'.extracting_'+uuid.uuid4().hex[:10]);stage.mkdir()
        cmd=command(video,stage)
        with (parent/(row['camera']+'.ffmpeg.log')).open('ab') as log:
            process=subprocess.Popen(cmd[:-1]+['-progress','pipe:1','-nostats',cmd[-1]],
                stdout=subprocess.PIPE,stderr=log,text=True,env=shared.THREAD_ENV)
            try:
                for line in process.stdout:
                    if line.startswith('frame='):progress(row,current=int(line.split('=',1)[1]))
                code=process.wait()
            except BaseException:
                process.terminate();process.wait();raise
        if code:raise RuntimeError('FFmpeg error; log and partial frames preserved')
        paths=sorted(stage.glob('*.png'));names=[f'frame{i:06d}.png' for i in range(1,row['frames']+1)]
        if [p.name for p in paths]!=names:raise ValueError('Frame count or continuous numbering differs')
        checksum=parent/(row['camera']+'.sha256');total_bytes=0
        with checksum.open('x') as f:
            for p in paths:
                digest,size=shared.check_png(p,row['width'],row['height']);total_bytes+=size
                f.write(digest+'  '+p.name+'\n')
        n=shared.independent_count(video)
        if n!=row['frames']:raise ValueError('Independent full decode frame count differs')
        indices=sorted({0,n//2,n-1});images=shared.raw_samples(video,indices,row['width'],row['height'])
        for i,pixels in zip(indices,images,strict=True):
            actual=cv2.cvtColor(cv2.imread(str(stage/names[i])),cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(actual,pixels)
        timing=parent/(row['camera']+'.timestamps.tsv');episode=episodes[row['action'],row['config']]
        with timing.open('x') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow([
                'file','frame_1_based','source_video_index_0_based','encoded_time_CFR_s',
                'source_trajectory_time_s','source_phase'])
            for i,name in enumerate(names):writer.writerow([
                name,i+1,i,i/row['fps'],episode['simulation_times_s'][i],episode['phases'][i]])
        audit={'passed':True,**row,'output_frames':str(dest),'PNG_bytes':total_bytes,
            'PNG_checksum_index':str(checksum),'PNG_checksum_index_sha256':shared.sha(checksum),
            'timestamps':str(timing),'timestamps_sha256':shared.sha(timing),
            'independently_decoded_source_frames':n,'sample_RGB_bit_exact':True,
            'sample_indices_0_based':indices,'all_PNG_CRC_dimensions_RGB8_checked':True,
            'no_resize_crop_filter_FPS_conversion_or_annotation':True,'UTC':shared.utc()}
        shared.publish_dir(stage,dest);shared.write(audit_path,audit)
        results.append(audit);done+=n;shared.write(out/'STREAM_RESULTS.json',results);progress(row)
        print(f'{len(results)}/48 {row["output_frames"]}: {n} PNGs',flush=True)
    progress(phase='all_frames_extracted_final_verify_pending')


def verify(out):
    binding(out);data=shared.read(out/'INVENTORY.json');records=[];sizes=0;by_action={t:0 for t in ACTIONS}
    for row in data['streams']:
        folder=out/row['output_frames'];parent=folder.parent;audit=shared.read(parent/(row['camera']+'.audit.json'))
        lines=Path(audit['PNG_checksum_index']).read_text().splitlines();files=sorted(folder.iterdir())
        if not audit['passed'] or len(files)!=row['frames'] or len(lines)!=len(files):
            raise ValueError('Final count/audit mismatch')
        for i,(p,line) in enumerate(zip(files,lines,strict=True),1):
            expected,name=line.split('  ',1)
            digest,_=shared.check_png(p,1920,1080)
            if p.name!=f'frame{i:06d}.png' or name!=p.name or digest!=expected:
                raise ValueError('Missing/corrupt/reordered frame')
            # Independently decode every PNG, not only its header/CRC.
            image=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
            if image is None or image.shape!=(1080,1920,3) or image.dtype!=np.uint8:
                raise ValueError('An exported PNG cannot be fully decoded')
            if p.stat().st_nlink!=1:raise ValueError('Shared mutable hardlinks are forbidden')
        if shared.sha(parent/f'view_{row["view"]}.mp4')!=row['source_sha256']:
            raise ValueError('Copied source video changed')
        if shared.sha(audit['timestamps'])!=audit['timestamps_sha256']:
            raise ValueError('Frame timing index changed')
        if shared.sha(audit['PNG_checksum_index'])!=audit['PNG_checksum_index_sha256']:
            raise ValueError('Frame checksum index changed')
        by_action[row['action']]+=len(files);sizes+=audit['PNG_bytes'];records.append(audit)
        shared.write(out/'PROGRESS.json',{'status':'independent_full_PNG_verification',
            'streams_verified':len(records),'streams_total':48,'frames_total':data['frames'],'UTC':shared.utc()})
    for path,digest in data['source_files_sha256'].items():
        if shared.sha(path)!=digest:raise ValueError('Original source video/metadata changed')
    for episode in data['episodes']:
        parent=out/episode['action']/f'config{episode["config"]}'
        if {p.name for p in parent.iterdir() if p.is_dir()}!={f'view{v}' for v in VIEWS}:
            raise ValueError('Wrong/missing/extra exported view folder')
    result={'passed':True,'video_streams':len(records),'episodes':len(data['episodes']),
        'selected_views':list(VIEWS),'PNG_frames':sum(by_action.values()),'frames_by_action':by_action,
        'PNG_GiB':sizes/1024**3,'native_resolution':[1920,1080],'native_fps':30,
        'every_PNG_fully_decoded_CRC_SHA256_and_numbering_checked':True,
        'independent_full_source_decode_count_each_stream':True,
        'initial_middle_final_RGB_bit_exact_each_stream':True,
        'source_videos_and_metadata_unchanged':True,'GPU_not_used':True,
        'visual_review_pending':True,'UTC':shared.utc()}
    shared.write(out/'FINAL_AUDIT.json',result)
    print(result,flush=True)


def review(out):
    data=shared.read(out/'INVENTORY.json');audit=shared.read(out/'FINAL_AUDIT.json')
    if not audit['passed']:raise ValueError('Automated final audit required')
    review_dir=out/'review';review_dir.mkdir(exist_ok=True);sheets={};records=[]
    top=['<!doctype html><meta charset="utf-8"><title>UR5e Oracle frames — views3/4/6</title>',
        '<style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}img{max-width:100%}</style>',
        '<h1>UR5e Oracle — 10,044 original-resolution PNG frames</h1>',
        '<p>Only views3/4/6. Scripted feedback Oracle, not trained RL. Original camera/inset preserved.</p>',
        '<p><a href="REPORT.md">Audit report</a> · <a href="FINAL_AUDIT.json">Machine audit</a></p>']
    for action in ACTIONS:
        panels=[];top.append(f'<h2>{action}</h2><table><tr><th>Config</th><th>Browse / image folders</th></tr>')
        for episode in [r for r in data['episodes'] if r['action']==action]:
            config=episode['config'];parent=out/action/f'config{config}';n=episode['frames']
            for i in sorted({0,n//2,n-1}):
                tiles=[]
                for view in VIEWS:
                    p=parent/f'view{view}'/f'frame{i+1:06d}.png';img=cv2.imread(str(p))
                    tile=np.full((204,320,3),245,np.uint8);tile[:180]=cv2.resize(img,(320,180),interpolation=cv2.INTER_AREA)
                    cv2.putText(tile,f'{action}/config{config} view{view} frame{i+1}',(4,198),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
                    tiles.append(tile);records.append({'PNG':str(p.relative_to(out)),'sha256':shared.sha(p)})
                panels.append(np.hstack(tiles))
            options=''.join(f'<option value="{v}">view{v}</option>' for v in VIEWS)
            viewer=f'''<!doctype html><meta charset="utf-8"><title>{action}/config{config}</title>
<style>body{{font:16px sans-serif;margin:20px}}img{{max-width:100%}}input[type=range]{{width:65%}}</style>
<h2>{action}/config{config}: {n} frames per view, 1920×1080, 30FPS</h2>
<select id="view">{options}</select><input id="seek" type="range" min="1" max="{n}" value="1">
<input id="number" type="number" min="1" max="{n}" value="1"><p id="info"></p>
<a id="full" target="_blank">Open original-resolution PNG</a><br><img id="image">
<script>const n={n},v=document.getElementById('view'),s=document.getElementById('seek'),k=document.getElementById('number');
function draw(x){{let i=Math.max(1,Math.min(n,Number(x)||1));s.value=i;k.value=i;let p='view'+v.value+'/frame'+String(i).padStart(6,'0')+'.png';document.getElementById('image').src=p;document.getElementById('full').href=p;document.getElementById('info').textContent=p+' | source index '+(i-1)+' | time '+((i-1)/30).toFixed(6)+'s';}}
v.onchange=()=>draw(k.value);s.oninput=()=>draw(s.value);k.oninput=()=>draw(k.value);draw(1);</script>'''
            (parent/'index.html').write_text(viewer)
            links=f'<a href="{action}/config{config}/index.html">Frame viewer</a> · '+ ' · '.join(
                f'<a href="{action}/config{config}/view{v}/">view{v} PNGs</a>' for v in VIEWS)
            top.append(f'<tr><td>config{config}</td><td>{links}</td></tr>')
        top.append('</table>');name=action+'_initial_middle_final_all_views.jpg'
        cv2.imwrite(str(review_dir/name),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,96])
        sheets[name]=shared.sha(review_dir/name)
        top.append(f'<p><a href="review/{name}">Contact sheet: all4 configs ×3 times ×3 requested views</a></p>')
    (out/'index.html').write_text('\n'.join(top))
    shared.write(review_dir/'REVIEW_MANIFEST.json',{'sheets_sha256':sheets,'actual_PNG_samples':records,
        'annotation_and_resizing_ONLY_on_review_sheets':True,'actual_visual_review_pending':True})
    text=f'''# UR5e Oracle — frames from views3/4/6

Source: `{data['source']}`.

Exported **{audit['PNG_frames']:,} lossless PNG frames from48 videos /16 scenes**.
Native1920×1080 and30FPS; no resize, crop, skipped/interpolated frames, overlays or FPS conversion.
Existing front-gripper inset is part of the source video and is preserved exactly.

    touch|pick|move|put/
      config1|config2|config3|config4/
        view_3.mp4 / view_4.mp4 / view_6.mp4   (byte-identical source copies)
        view3/frame000001.png ...
        view4/frame000001.png ...
        view6/frame000001.png ...
        view3.audit.json / view3.sha256 / view3.timestamps.tsv
        index.html

`frame000001.png` = source video index0, time0. Source trajectory timing/phase is indexed in each TSV.
Frame index is preserved without dropping or interpolating any decoded source frame.
Touch:{audit['frames_by_action']['touch']:,}; pick:{audit['frames_by_action']['pick']:,};
move:{audit['frames_by_action']['move']:,}; put:{audit['frames_by_action']['put']:,} PNGs (three views combined).

Automated audit PASS: full source decode counts, every PNG decode/CRC/SHA256/dimensions/sequence,
initial/middle/final RGB bit-exact against source decoder for every video, all original source hashes unchanged.
Actual sampled visual review is recorded separately after opening the4 contact sheets.

Open `index.html` to browse frames. Annotate separate copies, not checksum-protected original PNGs.
These are scripted feedback Oracle demonstrations, **not trained RL** and not extra independent trials.
CPU-only, one low-priority worker. No camera, model, checkpoint, physics or running VLA process changed.
'''
    (out/'REPORT.md').write_text(text);(out/'README.md').write_text(text)


def approve(out,note):
    if not note:raise ValueError('Actual image inspection findings required')
    manifest=shared.read(out/'review/REVIEW_MANIFEST.json')
    for name,digest in manifest['sheets_sha256'].items():
        if shared.sha(out/'review'/name)!=digest:raise ValueError('Review sheet changed')
    result=shared.read(out/'FINAL_AUDIT.json')
    if not result['passed']:raise ValueError('Automated audit failed')
    result.update(visual_review_pending=False,actual_visual_review={
        'opened_contact_sheets':list(manifest['sheets_sha256']),'sampled_frames':len(manifest['actual_PNG_samples']),
        'findings':note,'every_frame_human_inspection_NOT_claimed':True,'UTC':shared.utc()})
    shared.write(out/'FINAL_AUDIT.json',result)
    shared.write(out/'PROGRESS.json',{'status':'completed_all_frames_and_audit_and_sampled_visual_review',
        'streams_completed':48,'frames_completed':result['PNG_frames'],'UTC':shared.utc()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('plan','run','verify','review','all','approve'))
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--note');a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
    if out==source or source in out.parents or out in source.parents:
        raise ValueError('Output must be separate from the original source tree')
    out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);os.nice(10)
    if shutil.which('ionice'):subprocess.run(['ionice','-c','2','-n','7','-p',str(os.getpid())],check=True)
    with (out/'.extract.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            stages=('plan','run','verify','review') if a.stage=='all' else (a.stage,)
            for stage in stages:
                if stage=='plan':plan(source,out)
                elif stage=='run':run(out)
                elif stage=='verify':verify(out)
                elif stage=='review':review(out)
                else:approve(out,a.note)
        except BaseException as error:
            shared.write(out/'LAST_ERROR.json',{'type':type(error).__name__,'error':str(error),
                'partial_outputs_preserved_original_source_unchanged':True,'UTC':shared.utc()})
            raise


if __name__=='__main__':main()
