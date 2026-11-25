"""Inventory, lossless full-frame export, and audit without robot/model imports.

Usage: python -m VLA.frame_export_r61.export plan|run|verify --output PATH
Original scenes/videos are read-only. One low-priority CPU worker; no GPU.
"""
import argparse
import contextlib
import csv
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
import uuid
import zlib

import cv2
import imageio_ffmpeg
import numpy as np

ROOT=Path('/work/LfD4hri/DRL')
SOURCE=ROOT/'VLA/qualitative/ur5e_xvla_put_r60_named_failures_20261017'
OUTPUT=ROOT/'VLA/qualitative/qualitative_rl_vla'
GROUPS=('01_RL_succeed_VLA_succeed','02_RL_fail_VLA_succeed',
        '03_RL_succeed_VLA_fail','04_RL_fail_VLA_fail','05_backups','06_new')
METHODS=('Scripted_Oracle','XVLA_terminal_release')
VIEWS=(2,3,4,6)
CAMERAS=tuple(f'view{i}' for i in VIEWS)
RESERVE=25*1024**3
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
THREAD_ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
cv2.setNumThreads(1)


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()


def space(out,needed=0):
    free=shutil.disk_usage(out).free
    if free-needed<RESERVE:
        raise RuntimeError(f'Disk reserve gate: {free/1024**3:.2f}GiB free, need {needed/1024**3:.2f}GiB, keep25GiB. No frames deleted/resized/skipped.')
    return free


@contextlib.contextmanager
def lock(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.export.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def publish_dir(stage,target):
    fn=ctypes.CDLL(None,use_errno=True).renameat2
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(stage),-100,os.fsencode(target),1):
        err=ctypes.get_errno();raise OSError(err,os.strerror(err),str(target))


def metadata(path):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Cannot open '+str(path))
        return {'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps':float(cap.get(cv2.CAP_PROP_FPS))}
    finally:cap.release()


def inventory(source,out):
    rows=[];pairs=[];seal={};inode_hash={};seen={}
    master=read(source/'MANIFEST.sha256.json') if (source/'MANIFEST.sha256.json').exists() else {}
    def digest(path):
        s=path.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in inode_hash:inode_hash[key]=sha(path)
        seal[str(path)]=inode_hash[key]
        return inode_hash[key]
    for group in GROUPS:
        directory=source/group
        if not directory.is_dir():raise ValueError('Requested group missing: '+group)
        dirs=[p for p in sorted(directory.iterdir()) if p.is_dir() and all((p/m).is_dir() for m in METHODS)]
        if not dirs:raise ValueError('No paired scenes in '+group)
        for scene in dirs:
            match=re.fullmatch(r'(scene|backup)(\d+)_seed(\d+)',scene.name)
            if not match:raise ValueError('Unrecognized local scene: '+str(scene))
            kind,sid,seed=match.groups();sid=int(sid);seed=int(seed)
            initial=[read(scene/m/'initial_scene.json') for m in METHODS]
            if initial[0]['measured_state_sha256']!=initial[1]['measured_state_sha256'] or initial[0]['task']!=initial[1]['task']:
                raise ValueError('Unpaired initial scene: '+str(scene))
            evidence=[read(scene/m/'content_evidence.json') for m in METHODS]
            ok=[bool(e['full_put_complete']) for e in evidence]
            expected={'01':(True,True),'02':(False,True),'03':(True,False),'04':(False,False)}.get(group[:2])
            if expected is not None and tuple(ok)!=expected:raise ValueError('Group outcome mismatch: '+str(scene))
            anchor=f'{kind}{sid:02d}_RL_{"success" if ok[0] else "failure"}'
            dest_pair=Path(group)/anchor
            pair={'group':group,'scene_id':sid,'kind':kind,'seed':seed,'source_scene':str(scene),
                  'output_scene':str(dest_pair),'Oracle_success':ok[0],'VLA_success':ok[1],
                  'instruction':initial[0]['task']['prompt'],
                  'initial_world_sha256':initial[0]['measured_state_sha256']}
            pairs.append(pair)
            for method in METHODS:
                folder=scene/method;fpath=folder/'frames.json';frames=read(fpath)
                for name in ('initial_scene.json','content_evidence.json','frames.json','layout.json','camera_audit.json'):
                    if (folder/name).exists():digest(folder/name)
                for name in ('PAIR_AUDIT.json','FINAL_VISUAL_REVIEW.json'):
                    if (scene/name).exists():digest(scene/name)
                cameras=[(f'view_{i}',f'view{i}',1920,1080) for i in VIEWS]
                for stem,dest,w,h in cameras:
                    video=folder/(stem+'.mp4')
                    if not video.is_file():raise ValueError('Missing canonical view: '+str(video))
                    info=metadata(video)
                    if info!={'frames':len(frames),'width':w,'height':h,'fps':30.0}:
                        raise ValueError(f'Video vs metadata/native format mismatch: {video} {info}')
                    for i,f in enumerate(frames):
                        if abs(f['encoded_pts_s']-i/30)>1e-6:raise ValueError('Noncanonical PTS: '+str(video))
                        if i and f['physics_tick']-frames[i-1]['physics_tick']!=8:raise ValueError('Noncontinuous source ticks')
                    digest_value=digest(video);relative=str(video.relative_to(source))
                    if relative in master and master[relative]!=digest_value:
                        raise ValueError('Certified source video changed: '+relative)
                    row={'id':len(rows)+1,'group':group,'output_scene':str(dest_pair),'method':method,
                         'source':str(video),'source_sha256':digest_value,'source_bytes':video.stat().st_size,
                         'frame_metadata':str(fpath),'output_method':str(dest_pair/method),
                         'output_frames':str(dest_pair/method/dest),'camera':dest,'source_camera':stem,
                         'certified_source_manifest_match':relative in master,**info}
                    row['duplicate_of_stream']=seen.get(digest_value)
                    seen.setdefault(digest_value,row['id']);rows.append(row)
                    write(out/'INVENTORY_PROGRESS.json',{'stage':'read_only_inventory','streams':len(rows),'pairs':len(pairs),'UTC':utc()})
    return {'source':str(source),'output':str(out),'UTC':utc(),'groups':list(GROUPS),'pairs':pairs,'streams':rows,
            'stream_count':len(rows),'unique_source_videos':len(seen),'frames':sum(r['frames'] for r in rows),
            'source_bytes_all_folder_copies':sum(r['source_bytes'] for r in rows),
            'source_files_sha256':seal,'no_source_files_modified':True,
            'selected_wide_views':list(VIEWS),'native_handeye_exported':False}


def raw_samples(path,indices,w,h):
    expr='+'.join(f'eq(n\\,{i})' for i in indices)
    cmd=[FFMPEG,'-nostdin','-v','error','-xerror','-err_detect','explode','-threads','1','-i',str(path),
         '-map','0:v:0','-vf','select='+expr,'-fps_mode','passthrough','-pix_fmt','rgb24',
         '-threads','1','-f','rawvideo','pipe:1']
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=THREAD_ENV)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    expected=len(indices)*w*h*3
    if len(p.stdout)!=expected:raise ValueError('Independent selected decode size mismatch')
    return np.frombuffer(p.stdout,np.uint8).reshape(len(indices),h,w,3)


def png_encode_command(video,target,*,raw=None):
    base=[FFMPEG,'-nostdin','-hide_banner','-v','error','-n','-threads','1']
    if raw is None:base+=['-xerror','-err_detect','explode','-i',str(video),'-map','0:v:0']
    else:base+=['-f','rawvideo','-pix_fmt','rgb24','-s',f'{raw[0]}x{raw[1]}','-r','30','-i','pipe:0']
    return base+['-an','-sn','-dn','-fps_mode','passthrough','-c:v','png','-pix_fmt','rgb24',
                 '-compression_level','3','-pred','sub','-threads','1','-start_number','1',str(target/'frame%06d.png')]


def plan(source,out):
    if (out/'INVENTORY.json').exists():raise FileExistsError('Plan already exists; use run/verify, do not replace it')
    write(out/'PROGRESS.json',{'status':'planning_read_only_inventory_then_small_PNG_pilot','UTC':utc()})
    data=inventory(source,out);write(out/'INVENTORY.json',data)
    results={};done=0
    for row in data['streams']:
        if row['source_sha256'] in results:continue
        space(out)
        indices=sorted({0,row['frames']//2,row['frames']-1})
        images=raw_samples(row['source'],indices,row['width'],row['height'])
        dest=out/'_pilot'/f'stream{row["id"]:04d}';dest.mkdir(parents=True,exist_ok=False)
        started=time.monotonic()
        p=subprocess.run(png_encode_command(None,dest,raw=(row['width'],row['height'])),
            input=images.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=THREAD_ENV)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
        paths=sorted(dest.glob('frame*.png'))
        if len(paths)!=len(indices):raise ValueError('Pilot frame loss')
        for actual,expected in zip(paths,images):
            if not np.array_equal(cv2.cvtColor(cv2.imread(str(actual)),cv2.COLOR_BGR2RGB),expected):
                raise ValueError('PNG pilot changed RGB pixels')
        sizes=[p.stat().st_size for p in paths]
        results[row['source_sha256']]={'stream':row['id'],'indices':indices,'PNG_bytes':sizes,
            'average_bytes':sum(sizes)/len(sizes),'max_bytes':max(sizes),'encode_wall_seconds':time.monotonic()-started,
            'sample_RGB_bit_exact':True}
        done+=1
        write(out/'PROGRESS.json',{'status':'small_stratified_PNG_pilot','unique_videos_sampled':done,
            'unique_videos':data['unique_source_videos'],'planned_frames':data['frames'],'UTC':utc()})
    estimate=sum(results[r['source_sha256']]['average_bytes']*r['frames'] for r in data['streams'])
    conservative=sum(results[r['source_sha256']]['max_bytes']*r['frames'] for r in data['streams'])*1.30
    required=int(conservative+data['source_bytes_all_folder_copies']+512*1024**2)
    free=space(out,required)
    write(out/'PILOT_AND_ADMISSION.json',{'passed':True,'sampled_unique_videos':len(results),
         'expected_PNG_GiB':estimate/1024**3,'conservative_PNG_GiB':conservative/1024**3,
         'free_GiB_before_full_export':free/1024**3,'reserve_GiB':25,'required_conservative_GiB':required/1024**3,
         'format':'lossless RGB24 PNG, native resolution, compression3/Sub predictor',
         'samples':results,'all_sample_RGB_bit_exact':True})
    write(out/'PROGRESS.json',{'status':'plan_and_pilot_passed_READY_for_full_export','streams_total':data['stream_count'],
        'frames_total':data['frames'],'UTC':utc()})
    print({'plan_pass':True,'pairs':len(data['pairs']),'streams':data['stream_count'],'frames':data['frames'],
        'PNG_GiB_estimate':round(estimate/1024**3,2),'PNG_GiB_conservative':round(conservative/1024**3,2)},flush=True)


def check_png(path,w,h):
    blob=path.read_bytes()
    if blob[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Invalid PNG signature: '+str(path))
    offset=8;first=True;ended=False
    while offset<len(blob):
        size=struct.unpack('>I',blob[offset:offset+4])[0];tag=blob[offset+4:offset+8]
        body=blob[offset+8:offset+8+size];crc=blob[offset+8+size:offset+12+size]
        if len(crc)!=4 or zlib.crc32(tag+body)&0xffffffff!=struct.unpack('>I',crc)[0]:
            raise ValueError('PNG CRC/truncation error: '+str(path))
        if first:
            if tag!=b'IHDR' or struct.unpack('>IIBB',body[:10])!=(w,h,8,2):raise ValueError('PNG native dimensions/RGB8 changed')
            first=False
        offset+=size+12
        if tag==b'IEND':ended=True;break
    if not ended or offset!=len(blob):raise ValueError('Invalid PNG end/trailing data')
    return hashlib.sha256(blob).hexdigest(),len(blob)


def independent_count(video):
    cmd=[FFMPEG,'-nostdin','-v','error','-xerror','-err_detect','explode','-threads','1','-i',str(video),
         '-map','0:v:0','-an','-sn','-dn','-fps_mode','passthrough','-threads','1',
         '-progress','pipe:1','-nostats','-f','null','-']
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=THREAD_ENV)
    if p.returncode:raise RuntimeError('Independent full source decode failed: '+p.stderr.decode(errors='replace'))
    values=[int(s.split('=',1)[1]) for s in p.stdout.decode().splitlines() if s.startswith('frame=')]
    if not values:raise ValueError('No independent decoded source count')
    return values[-1]


def setup_folders(data,out):
    for pair in data['pairs']:
        dest=out/pair['output_scene'];dest.mkdir(parents=True,exist_ok=True)
        write(dest/'SCENE.json',{**pair,'RL_alias_is_feedback_Oracle_NOT_trained_RL':True,
            'parent_name_encodes_Oracle_outcome_both_methods_still_present':True})
        scene=Path(pair['source_scene'])
        for name in ('PAIR_AUDIT.json','FINAL_VISUAL_REVIEW.json'):
            if (scene/name).exists() and not (dest/name).exists():shutil.copy2(scene/name,dest/name)
        for method in METHODS:
            method_out=dest/method;method_out.mkdir(exist_ok=True)
            for name in ('initial_scene.json','content_evidence.json','frames.json','layout.json'):
                if (scene/method/name).exists() and not (method_out/name).exists():shutil.copy2(scene/method/name,method_out/name)
    for row in data['streams']:
        target=out/row['output_method']/(row['source_camera']+'.mp4')
        if not target.exists():space(out,row['source_bytes']);shutil.copy2(row['source'],target)
        if sha(target)!=row['source_sha256']:raise ValueError('Copied source video mismatch: '+str(target))


def run(out):
    data=read(out/'INVENTORY.json');admission=read(out/'PILOT_AND_ADMISSION.json')
    if not admission['passed']:raise ValueError('Pilot/storage gate not passed')
    if {r['camera'] for r in data['streams']}!=set(CAMERAS):
        raise ValueError('Inventory is not restricted to the four requested views')
    os.nice(10);space(out);setup_folders(data,out)
    results=[];cache={};started=time.monotonic();done_frames=0
    def progress(row=None,phase='running',current=0):
        elapsed=time.monotonic()-started
        write(out/'PROGRESS.json',{'status':phase,'pid':os.getpid(),'streams_completed':len(results),
            'streams_total':len(data['streams']),'frames_completed':done_frames,'frames_total':data['frames'],
            'active_stream':row['output_frames'] if row else None,'active_stream_frames_encoded':current,
            'elapsed_seconds_current_process':elapsed,'free_GiB':shutil.disk_usage(out).free/1024**3,'UTC':utc()})
    for row in data['streams']:
        dest=out/row['output_frames'];parent=dest.parent
        audit_file=parent/(row['camera']+'.audit.json')
        if audit_file.exists():
            audit=read(audit_file)
            if not audit['passed'] or audit['source_sha256']!=row['source_sha256'] or not dest.is_dir():
                raise ValueError('Existing completed output differs; not overwritten')
            results.append(audit);done_frames+=row['frames'];cache.setdefault(row['source_sha256'],(dest,audit));progress(row);continue
        if dest.exists():raise FileExistsError('Unsealed existing output preserved: '+str(dest))
        if sha(row['source'])!=row['source_sha256']:raise ValueError('Source changed after inventory')
        space(out,int(admission['samples'][row['source_sha256']]['max_bytes']*row['frames']*1.30))
        stage=parent/('.'+row['camera']+'.extracting_'+uuid.uuid4().hex[:10]);stage.mkdir(exist_ok=False)
        copied=cache.get(row['source_sha256']);progress(row)
        snapshot=parent/(row['source_camera']+'.mp4')
        if copied:
            for i in range(1,row['frames']+1):
                if i%30==1:space(out);progress(row,'copying_exact_duplicate_frames',i-1)
                name=f'frame{i:06d}.png';shutil.copy2(copied[0]/name,stage/name)
        else:
            command=png_encode_command(snapshot,stage)
            with (parent/(row['camera']+'.ffmpeg.log')).open('ab') as log:
                process=subprocess.Popen(command[:-1]+['-progress','pipe:1','-nostats',command[-1]],
                    stdout=subprocess.PIPE,stderr=log,env=THREAD_ENV,text=True)
                try:
                    for line in process.stdout:
                        if line.startswith('frame='):
                            space(out);progress(row,'extracting_all_frames',int(line.split('=',1)[1]))
                    code=process.wait()
                except BaseException:
                    process.terminate();process.wait();raise
            if code:raise RuntimeError('FFmpeg failed; staged files/log preserved: '+str(stage))
        paths=sorted(stage.iterdir())
        names=[f'frame{i:06d}.png' for i in range(1,row['frames']+1)]
        if [p.name for p in paths]!=names:raise ValueError('Output frame count/name continuity mismatch')
        hashes=[];total_bytes=0
        for p in paths:
            digest,size=check_png(p,row['width'],row['height']);hashes.append((digest,p.name));total_bytes+=size
        checksums=parent/(row['camera']+'.sha256')
        with checksums.open('w') as f:
            for digest,name in hashes:f.write(digest+'  '+name+'\n')
        samples=sorted({0,row['frames']//2,row['frames']-1})
        if copied:
            original=(copied[0].parent/(row['camera']+'.sha256'))
            # Same video bytes may occur under a different camera filename;
            # compare against the actual source audit's recorded index.
            original=Path(copied[1]['PNG_checksum_index'])
            if checksums.read_bytes()!=original.read_bytes():raise ValueError('Duplicate PNG copying changed bytes')
            source_count=copied[1]['independently_decoded_source_frames']
        else:
            source_count=independent_count(snapshot)
            pixels=raw_samples(snapshot,samples,row['width'],row['height'])
            for idx,expected in zip(samples,pixels):
                actual=cv2.cvtColor(cv2.imread(str(stage/f'frame{idx+1:06d}.png')),cv2.COLOR_BGR2RGB)
                if not np.array_equal(actual,expected):raise ValueError('Sample RGB differs from source decoder')
        if source_count!=row['frames']:raise ValueError('Independent source count differs from PNG count')
        frames=read(row['frame_metadata']);times=parent/(row['camera']+'.timestamps.tsv')
        with times.open('w') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['file','frame_1_based','video_index_0_based','encoded_PTS_s','simulation_time_s','physics_tick','control_index'])
            for i,t in enumerate(frames):writer.writerow([names[i],i+1,i,t['encoded_pts_s'],t['simulation_time_s'],t['physics_tick'],t.get('control_index')])
        audit={'passed':True,'stream_id':row['id'],'source':row['source'],'source_sha256':row['source_sha256'],
            'output_frames':str(dest),'frames':row['frames'],'width':row['width'],'height':row['height'],'fps':row['fps'],
            'all_frames_contiguous_and_all_PNG_CRC_dimensions_RGB8_verified':True,
            'PNG_checksum_index':str(checksums),'PNG_checksum_index_sha256':sha(checksums),
            'timestamps_tsv':str(times),'timestamps_tsv_sha256':sha(times),
            'independently_decoded_source_frames':source_count,'sample_RGB_bit_exact':True,
            'sample_indices_0_based':samples,'duplicate_decoding_reused_from':str(copied[0]) if copied else None,
            'duplicate_frames_copied_NOT_shared_hardlinks':bool(copied),'PNG_bytes':total_bytes,
            'no_resize_no_crop_no_FPS_filter_no_missing_frames':True,'verified_UTC':utc()}
        publish_dir(stage,dest);write(audit_file,audit);cache.setdefault(row['source_sha256'],(dest,audit))
        results.append(audit);done_frames+=row['frames'];write(out/'STREAM_RESULTS.json',results);progress(row)
        print(f'{len(results)}/{len(data["streams"])} {row["output_frames"]}: {row["frames"]} frames',flush=True)
    write(out/'PROGRESS.json',{'status':'extraction_complete_independent_final_verification_REQUIRED',
          'streams_completed':len(results),'streams_total':len(results),'frames_completed':done_frames,'frames_total':data['frames'],'UTC':utc()})
    print({'all_extracted':True,'frames':done_frames,'streams':len(results)},flush=True)


def verify(out):
    data=read(out/'INVENTORY.json');counts={g:0 for g in GROUPS};records=[]
    for row in data['streams']:
        folder=out/row['output_frames'];audit=read(folder.parent/(row['camera']+'.audit.json'))
        if not audit['passed']:raise ValueError('Unpassed stream')
        lines=Path(audit['PNG_checksum_index']).read_text().splitlines()
        files=sorted(folder.glob('*.png'))
        if len(files)!=row['frames'] or len(lines)!=row['frames']:raise ValueError('Missing/excess frame')
        for i,(p,line) in enumerate(zip(files,lines),1):
            expected,name=line.split('  ',1)
            if p.name!=name or name!=f'frame{i:06d}.png':raise ValueError('Sequence naming error')
            actual,_=check_png(p,row['width'],row['height'])
            if actual!=expected:raise ValueError('Final PNG checksum mismatch')
        if sha(row['source'])!=row['source_sha256']:raise ValueError('Original video changed')
        if sha(folder.parent/(row['source_camera']+'.mp4'))!=row['source_sha256']:raise ValueError('Copied video changed')
        counts[row['group']]+=row['frames'];records.append(audit)
        write(out/'FINAL_VERIFY_PROGRESS.json',{'streams_checked':len(records),'streams_total':len(data['streams']),'UTC':utc()})
    for path,digest in data['source_files_sha256'].items():
        if sha(path)!=digest:raise ValueError('Source evidence changed: '+path)
    for pair in data['pairs']:
        for method in METHODS:
            subset=[r for r in data['streams'] if r['output_scene']==pair['output_scene'] and r['method']==method]
            if {r['camera'] for r in subset}!=set(CAMERAS):
                raise ValueError('Missing requested view or an extra unrequested view')
            folder=out/pair['output_scene']/method
            for extra in ('view1','view5','handeye','view_1.mp4','view_5.mp4','handeye.mp4'):
                if (folder/extra).exists():raise ValueError('Unrequested camera retained in final set: '+extra)
            if len({r['frames'] for r in subset})!=1:raise ValueError('Views are not synchronized')
    result={'passed':True,'UTC':utc(),'groups':counts,'paired_scene_slots':len(data['pairs']),
        'method_slots':len(data['pairs'])*2,'video_stream_slots':len(records),
        'unique_source_videos':data['unique_source_videos'],'PNG_frames':sum(counts.values()),
        'selected_wide_views':list(VIEWS),'native_handeye_exported':False,
        'all_PNG_rechecked_CRC_dimensions_SHA256_and_contiguous_names':True,
        'all_original_source_files_unchanged':True,'all_copied_videos_source_SHA_exact':True,
        'frame_count_verified_by_independent_full_source_decode':True,
        'no_annotation_side_effects_between_groups_shared_image_hardlinks_NOT_used':True,
        'actual_visual_contact_sheet_review_pending':True,
        'PNG_GiB':sum(r['PNG_bytes'] for r in records)/1024**3,'training_GPU_not_used':True}
    write(out/'FINAL_AUDIT.json',result)
    report=['# Full-frame export — qualitative_rl_vla','',f'Exported {result["PNG_frames"]:,} PNG frames from {len(records)} camera-stream slots.',
       f'Coverage: all six groups, {len(data["pairs"])} paired-scene slots, both methods, ONLY requested wide views 2, 3, 4, 6.',
       'PNG is lossless RGB8 at original1920×1080. No skipped/interpolated frames, resize, crop, overlays or FPS changes.',
       'frame000001.png is video frame0 (encoded PTS0); physical time starts at tick8/240=0.033333s. Per-view TSV preserves the distinction.',
       'RL folder naming denotes the scripted feedback Oracle, NOT a trained RL policy. Parent folder outcome belongs to Oracle; each parent contains BOTH methods.',
       '06_new is exported exactly as found. Exact duplicate videos reuse decoding, but output PNGs are independent file copies, not shared hardlinks.',
       'See INVENTORY.json, STREAM_RESULTS.json, per-view .audit.json/.sha256/.timestamps.tsv and FINAL_AUDIT.json.',
       'Actual sampled visual review is recorded separately after opening the generated contact sheets.']
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    write(out/'PROGRESS.json',{'status':'all_frames_extracted_and_automated_audit_passed_visual_review_pending',
         'streams_completed':len(records),'streams_total':len(records),'frames_completed':sum(counts.values()),'frames_total':sum(counts.values()),'UTC':utc()})
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('plan','run','verify'))
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();a.source=a.source.resolve();a.output=a.output.resolve()
    if a.output==a.source or a.source in a.output.parents:raise ValueError('Output must not modify certified source tree')
    with lock(a.output):
        try:
            if a.stage=='plan':plan(a.source,a.output)
            elif a.stage=='run':run(a.output)
            else:verify(a.output)
        except BaseException as e:
            write(a.output/'LAST_ERROR.json',{'stage':a.stage,'type':type(e).__name__,'error':str(e),'UTC':utc(),
                'partial_outputs_preserved_no_source_deleted':True});raise


if __name__=='__main__':main()
