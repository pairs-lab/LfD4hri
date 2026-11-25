"""Prepare actual encoded evidence; approve ONLY after the agent opens it."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import decode_at
from .common import read


def prepare(folder):
    pair=read(folder/'PAIR_AUDIT.json');directory=folder/'review';directory.mkdir(exist_ok=True)
    selected={};panels={'wide':[],'handeye':[]};format_checks={}
    for method in ('Scripted_Oracle','XVLA_terminal_release'):
        src=folder/method;rows=read(src/'review/content_frames.json');evidence=read(src/'content_evidence.json')
        indices=[r['frame'] for r in rows];n=read(src/'camera_audit.json')['streams'][0]['frames']
        if evidence['full_put_complete']:
            stable=evidence['stable_completion_frame'];indices=[max(0,stable-45),max(0,stable-21),stable,n-1]
        elif evidence['events']:
            event=max(evidence['events'],key=lambda e:e.get('object_displacement_within250ms_m',0))
            indices=[event['before_frame'],event['event_frame'],event['after_frame'],n-1]
            if event['mode']=='controlled_Oracle_transport_timeout':
                actual=read(src/'frames.json');held=next((i for i,r in enumerate(actual) if r['correct_attached']),0)
                indices=[0,min(n-1,held+30),n//2,n-1]
            elif event['mode']=='controlled_Oracle_failed_acquisition_empty_closure':
                indices=[0,n//2,max(0,n-31),n-1]
            elif event['after_frame']==n-1 and event['event_frame']<n-60:
                indices[2]=(event['event_frame']+n-1)//2
        else:indices=[0,n//3,2*n//3,n-1]
        selected[method]=indices
        grid=[cv2.resize(decode_at(src/f'view_{i}.mp4',indices[1]),(640,360),interpolation=cv2.INTER_AREA) for i in range(1,7)]
        cv2.imwrite(str(directory/(method+'_six_views.jpg')),np.vstack((np.hstack(grid[:3]),np.hstack(grid[3:]))),[cv2.IMWRITE_JPEG_QUALITY,96])
        for key,name in (('wide','view_2'),('handeye','handeye')):
            panels[key].append(np.hstack([cv2.resize(decode_at(src/(name+'.mp4'),i),(640,360),interpolation=cv2.INTER_AREA) for i in indices]))
        checks=[]
        for name in ('handeye',*[f'view_{i}' for i in range(1,7)]):
            reader=imageio_ffmpeg.read_frames(str(src/(name+'.mp4')))
            try:meta=next(reader)
            finally:reader.close()
            expected=(1280,720) if name=='handeye' else (1920,1080)
            if meta['codec']!='h264' or 'yuv420p' not in meta['pix_fmt'] or meta['fps']!=30 or tuple(meta['source_size'])!=expected:
                raise ValueError('Independent codec/size/FPS gate failed: '+name)
            checks.append(meta)
        format_checks[method]=checks
    for key,rows in panels.items():cv2.imwrite(str(directory/(key+'_sequence.jpg')),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,96])
    atomic_json(directory/'sequence_manifest.json',{'row_order':['Scripted_Oracle','XVLA_terminal_release'],
        'source_frames':selected,'six_view_grid_frame':{method:indices[1] for method,indices in selected.items()},
        'each_panel_is_actual_encoded_frame':True,
        'no_overlay_or_timing_change_to_delivered_videos':True,'format_checks':format_checks})
    print({'pair':str(folder),'open_before_approve':[str(directory/'wide_sequence.jpg'),str(directory/'handeye_sequence.jpg')],
        'actual_frames':selected},flush=True)


def approve(a):
    pair=read(a.pair/'PAIR_AUDIT.json');manifest=read(a.pair/'review/sequence_manifest.json')
    if not a.note:raise ValueError('Actual visual findings required')
    for method,value in pair['outcomes'].items():
        folder=a.pair/method;audit=read(folder/'camera_audit.json')
        if not audit['media_decode_sync_pass'] or not audit['actual_tick_spacing_pass']:raise ValueError('Media integrity not passed')
        if read(folder/'content_evidence.json')['full_put_complete']!=value['full_put_complete']:raise ValueError('Outcome mismatch')
        if sha256_file(value['delivery_alias'])!=sha256_file(folder/'view_2.mp4'):raise ValueError('Principal not the verified camera stream')
    if not pair['same_measured_initial_state'] or not pair['Oracle_repeat_proof']['passed']:raise ValueError('Pair/repeat parity failed')
    reviewed_files=['wide_sequence.jpg','handeye_sequence.jpg',
        'Scripted_Oracle_six_views.jpg','XVLA_terminal_release_six_views.jpg']
    if not all((a.pair/'review'/f).exists() for f in reviewed_files):raise ValueError('Prepare all camera grids before actual viewing/approval')
    record={'approved_for_delivery':True,'reviewed_UTC':datetime.now(timezone.utc).isoformat(),
        'reviewer':'Codex after actual view_image inspection, not auto approval',
        'opened_encoded_sequence_files':['review/'+f for f in reviewed_files],
        'source_frames':manifest['source_frames'],'actual_visual_findings':a.note,
        'full_stream_decode_and_format_audited':True,'manual_full_frame_by_frame_playback_NOT_claimed':True,
        'no_claim_of_intrinsic_model_limit_or_memory_loss':True,'backup':pair['backup'],
        'review_artifact_SHA256':{name:sha256_file(a.pair/'review'/name) for name in reviewed_files}}
    atomic_json(a.pair/'FINAL_VISUAL_REVIEW.json',record)
    pair.update(main_certified=not pair['backup'],requires_actual_agent_visual_review=False,
        delivered_visual_review_scope='Actual encoded before/event/after/settle/terminal samples + whole-stream decode')
    for method in pair['outcomes']:
        pair['outcomes'][method].update(final_review_pending=False,manual_encoded_content_review_pending=False)
        folder=a.pair/method;audit=read(folder/'camera_audit.json');audit['manual_visual_review_pending']=False
        audit['manual_review_scope']='sampled encoded event sequences, not continuous every-frame manual viewing'
        atomic_json(folder/'camera_audit.json',audit)
        for path in dict.fromkeys([Path(pair['outcomes'][method]['delivery_alias']),Path(pair['outcomes'][method]['principal']),
                *[folder/f'view_{i}.mp4' for i in range(1,7)],folder/'handeye.mp4']):
            notes=path.with_suffix('.txt')
            text=notes.read_text().split('\nFINAL REVIEW:')[0].replace('Manual final encoded-content review remains required.',
                'Sampled encoded-event review completed; exact scope is in FINAL_VISUAL_REVIEW.json.')
            notes.write_text(text+'\nFINAL REVIEW: actual encoded event/terminal frames inspected; whole-stream decode passed. Findings: '+a.note+'\n')
    atomic_json(a.pair/'PAIR_AUDIT.json',pair)
    index=a.pair/'index.html'
    index.write_text(index.read_text().replace('Final visual review pending;',
        'Sampled encoded-event/terminal visual review passed;'))
    print({'approved':str(a.pair),'backup':pair['backup'],'actual_frame_review_not_full_manual_playback':True},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--approve',action='store_true');parser.add_argument('--note');a=parser.parse_args()
    if a.approve:approve(a)
    else:prepare(a.pair)
