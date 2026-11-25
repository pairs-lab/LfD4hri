"""Only PUT view2, configs1–4; separate eval folder and verified download ZIP.

python -m env_eval.extract_put_view2_frames run
python -m env_eval.extract_put_view2_frames approve-package --note 'actual review findings'
The frozen view3/4/6 extractor and its completed outputs remain unchanged.
"""
import argparse
import csv
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import zipfile

import cv2
import numpy as np

from VLA.frame_export_r61 import export as shared

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper/UR5e_Robotiq2F/put'
OUTPUT=ROOT/'eval/put_view2_frames'
RESERVE=30*1024**3


def space(out,needed=0):
    if shutil.disk_usage(out).free-needed < RESERVE:
        raise RuntimeError('Keep30GiB free for existing VLA jobs; no images deleted/downscaled')


def run(out):
    if (out/'INVENTORY.json').exists():raise FileExistsError('Preserve the existing export')
    if set(SOURCE.glob('config*/view_2.mp4'))!={SOURCE/f'config{c}/view_2.mp4' for c in range(1,5)}:
        raise ValueError('Exactly4 PUT view2 videos are required')
    rows=[]
    for config in range(1,5):
        video=SOURCE/f'config{config}/view_2.mp4';info=shared.metadata(video)
        if (info['width'],info['height'],info['fps'])!=(1920,1080,30.):raise ValueError('Unexpected source format')
        trace_path=video.parent/'trajectory.json';trace=shared.read(trace_path)
        source_audit=shared.read(video.parent/'audit.json')
        if source_audit['action']!='put' or source_audit['robot']!='ur5e':raise ValueError('Wrong source action/robot')
        if len(trace)!=info['frames']:raise ValueError('Source trajectory/video length differs')
        np.testing.assert_allclose([r['time'] for r in trace],np.arange(len(trace))/30,rtol=0,atol=1e-8)
        rows.append({'config':config,'source':str(video),'source_sha256':shared.sha(video),
            'trajectory':str(trace_path),'trajectory_sha256':shared.sha(trace_path),**info})
    shared.write(out/'INVENTORY.json',{'source':str(SOURCE),'action':'put','views':[2],
        'streams':rows,'frames':sum(r['frames'] for r in rows),'original_source_read_only':True})
    shared.write(out/'EXTRACTOR_BINDING.json',{'sources':{str(p):shared.sha(p) for p in (
        Path(__file__).resolve(),Path(shared.__file__).resolve())},
        'ffmpeg_binary':shared.FFMPEG,'ffmpeg_sha256':shared.sha(shared.FFMPEG),
        'CPU_only_one_low_priority_worker':True})
    results=[];panels=[];sample_records=[];started=time.monotonic()
    for row in rows:
        config=row['config'];video=Path(row['source']);n=row['frames'];dest=out/f'config{config}'
        if dest.exists():raise FileExistsError('Preserve existing frame folder')
        indices=sorted({0,n//2,n-1});expected=shared.raw_samples(video,indices,1920,1080)
        # Storage pilot uses the actual decoded source samples, without saving
        # additional PNGs that could be confused with final frame counts.
        sizes=[]
        for pixels in expected:
            ok,encoded=cv2.imencode('.png',cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3])
            if not ok:raise ValueError('Lossless PNG pilot failed')
            decoded=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB),pixels)
            sizes.append(encoded.size)
        # Include the remaining images and a second copy for the download ZIP.
        space(out,int(max(sizes)*sum(r['frames'] for r in rows)*3))
        stage=out/(f'.config{config}.extracting_'+uuid.uuid4().hex[:10]);stage.mkdir()
        cmd=shared.png_encode_command(video,stage)
        cmd=cmd[:1]+['-filter_threads','1']+cmd[1:]
        with (out/f'config{config}.ffmpeg.log').open('xb') as log:
            process=subprocess.Popen(cmd[:-1]+['-progress','pipe:1','-nostats',cmd[-1]],
                stdout=subprocess.PIPE,stderr=log,text=True,env=shared.THREAD_ENV)
            try:
                for line in process.stdout:
                    if line.startswith('frame='):
                        space(out)
                        shared.write(out/'PROGRESS.json',{'status':'extracting_only_PUT_view2',
                            'pid':os.getpid(),'configs_completed':len(results),'active_config':config,
                            'active_frames':int(line.split('=',1)[1]),'frames_total':sum(r['frames'] for r in rows),
                            'UTC':shared.utc()})
                code=process.wait()
            except BaseException:
                process.terminate();process.wait();raise
        if code:raise RuntimeError('FFmpeg failed; partial files/log preserved')
        files=sorted(stage.glob('*.png'));names=[f'frame{i:06d}.png' for i in range(1,n+1)]
        if [p.name for p in files]!=names:raise ValueError('Missing/excess/reordered frame')
        total_bytes=0
        with (stage/'frames.sha256').open('x') as index:
            for p in files:
                digest,size=shared.check_png(p,1920,1080);total_bytes+=size
                bitmap=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
                if bitmap is None or bitmap.shape!=(1080,1920,3) or bitmap.dtype!=np.uint8:
                    raise ValueError('Exported PNG cannot be fully decoded')
                index.write(digest+'  '+p.name+'\n')
        if shared.independent_count(video)!=n:raise ValueError('Independent full source decode count differs')
        tiles=[]
        for i,pixels in zip(indices,expected,strict=True):
            image=cv2.imread(str(stage/names[i]))
            np.testing.assert_array_equal(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),pixels)
            tile=np.full((204,320,3),245,np.uint8);tile[:180]=cv2.resize(image,(320,180),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,f'put/config{config} view2 frame{i+1}',(4,198),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
            tiles.append(tile);sample_records.append({'file':f'config{config}/{names[i]}','sha256':shared.sha(stage/names[i])})
        panels.append(np.hstack(tiles));trace=shared.read(row['trajectory'])
        with (stage/'timestamps.tsv').open('x') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['file','video_index_0_based','time_CFR_s','source_time_s','source_phase'])
            for i,name in enumerate(names):writer.writerow([name,i,i/30,trace[i]['time'],trace[i]['phase']])
        if shared.sha(video)!=row['source_sha256'] or shared.sha(row['trajectory'])!=row['trajectory_sha256']:
            raise ValueError('Original source changed')
        audit={'passed':True,**row,'PNG_bytes':total_bytes,'all_PNG_decoded_CRC_dimensions_SHA256_checked':True,
            'independent_full_source_frame_count_checked':True,'sample_RGB_bit_exact':True,
            'sample_indices_0_based':indices,'no_resize_crop_skip_interpolation_or_FPS_change':True}
        shared.write(stage/'AUDIT.json',audit);shared.publish_dir(stage,dest);results.append(audit)
        print(f'config{config}: {n} frames, audit PASS',flush=True)
    # A second path-independent checksum pass reads every final frame after
    # publication, before the result is eligible for packaging.
    for row in rows:
        folder=out/f'config{row["config"]}'
        for line in (folder/'frames.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            if shared.sha(folder/name)!=digest:raise ValueError('Final PNG checksum mismatch')
    cv2.imwrite(str(out/'CONTACT_SHEET.jpg'),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,96])
    shared.write(out/'VISUAL_SAMPLES.json',{'samples':sample_records,
        'contact_sheet_sha256':shared.sha(out/'CONTACT_SHEET.jpg'),
        'annotations_resizing_only_in_contact_sheet_NOT_original_PNGs':True})
    result={'passed':True,'action':'put','selected_views':[2],'configs':4,'PNG_frames':sum(r['frames'] for r in rows),
        'frames_by_config':{str(r['config']):r['frames'] for r in rows},'native_resolution':[1920,1080],
        'source_fps':30,'PNG_GiB':sum(r['PNG_bytes'] for r in results)/1024**3,
        'source_files_unchanged':True,'every_PNG_rechecked':True,'GPU_not_used':True,
        'visual_review_pending':True,'wall_seconds':time.monotonic()-started,'UTC':shared.utc()}
    shared.write(out/'FINAL_AUDIT.json',result)
    (out/'README.md').write_text('''# UR5e PUT — view2 only

Only put/config1–4, 1213 PNG frames total. Native1920×1080, full30FPS sequence.
No crop/resize/skipped/interpolated frames. Original front-gripper inset is preserved.

    config1/frame000001.png ... frame000308.png
    config2/frame000001.png ... frame000305.png
    config3/frame000001.png ... frame000300.png
    config4/frame000001.png ... frame000300.png

Frame000001 is source frame0/time0. Each config has checksums and timing/phase TSV.
CONTACT_SHEET.jpg contains12 annotated previews; original PNGs have no added text.
The source is scripted feedback Oracle, not a trained RL policy. Source videos,
the completed view3/4/6 export and running VLA jobs remain unchanged.
''')
    shared.write(out/'PROGRESS.json',{'status':'extraction_and_automated_audit_complete_visual_review_pending',
        'frames_completed':result['PNG_frames'],'UTC':shared.utc()})
    print(result,flush=True)


def package(out,note):
    if not note:raise ValueError('Actual sample image review findings required')
    result=shared.read(out/'FINAL_AUDIT.json');visual=shared.read(out/'VISUAL_SAMPLES.json')
    if not result['passed'] or result['action']!='put' or result['selected_views']!=[2] or result['PNG_frames']!=1213:
        raise ValueError('Incomplete or wrong-scope export')
    if shared.sha(out/'CONTACT_SHEET.jpg')!=visual['contact_sheet_sha256']:raise ValueError('Review sheet changed')
    for sample in visual['samples']:
        if shared.sha(out/sample['file'])!=sample['sha256']:raise ValueError('Reviewed sample changed')
    for path,digest in shared.read(out/'EXTRACTOR_BINDING.json')['sources'].items():
        if shared.sha(path)!=digest:raise ValueError('Extractor source changed')
    result.update(visual_review_pending=False,actual_visual_review={
        'samples':12,'findings':note,'continuous_all_frame_human_inspection_NOT_claimed':True,'UTC':shared.utc()})
    shared.write(out/'FINAL_AUDIT.json',result)
    shared.write(out/'PROGRESS.json',{'status':'completed_audit_and_sampled_visual_review',
        'frames_completed':1213,'UTC':shared.utc()})
    archive=out.with_suffix('.zip')
    if archive.exists():raise FileExistsError('Preserve existing download archive')
    files=sorted(p for p in out.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(out).parts))
    space(out,sum(p.stat().st_size for p in files)+16*1024**2)
    stage=archive.with_name('.'+archive.name+'.packaging_'+uuid.uuid4().hex[:10])
    with zipfile.ZipFile(stage,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in files:z.write(p,arcname=str(Path(out.name)/p.relative_to(out)))
    with zipfile.ZipFile(stage) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC verification failed')
        members=[x for x in z.namelist() if x.endswith('.png')]
        if len(members)!=1213:raise ValueError('Archive PNG count mismatch')
        for config in range(1,5):
            for line in (out/f'config{config}/frames.sha256').read_text().splitlines():
                digest,name=line.split('  ',1);member=f'{out.name}/config{config}/{name}'
                if hashlib.sha256(z.read(member)).hexdigest()!=digest:raise ValueError('Archived PNG differs')
    shared.publish_dir(stage,archive)
    shared.write(out.with_name(out.name+'.archive_audit.json'),{'passed':True,'archive':str(archive),
        'archive_sha256':shared.sha(archive),'archive_bytes':archive.stat().st_size,'PNG_frames':1213,
        'all_archived_PNG_SHA256_and_archive_CRC_verified':True,'UTC':shared.utc()})
    print({'download_archive':str(archive),'MiB':archive.stat().st_size/1024**2,'PNG_frames':1213},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('run','approve-package'))
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--note');a=p.parse_args();out=a.output.resolve()
    if SOURCE==out or SOURCE in out.parents or out in SOURCE.parents:raise ValueError('Output must not alter source tree')
    out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);os.nice(10)
    if shutil.which('ionice'):subprocess.run(['ionice','-c','2','-n','7','-p',str(os.getpid())],check=True)
    with (out/'.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(out) if a.stage=='run' else package(out,a.note)
        except BaseException as error:
            shared.write(out/'LAST_ERROR.json',{'error':str(error),'partial_outputs_preserved':True,'UTC':shared.utc()});raise


if __name__=='__main__':main()
