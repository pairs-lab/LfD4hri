"""Package the paired seven-camera movies, real event sheets and two aliases."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_r57.package import main as package_pair
from VLA.qualitative_recorder_r51 import decode_at
from .common import read


def main(a):
    package_pair(SimpleNamespace(vla=a.vla,oracle=a.oracle,repeat_proof=a.repeat_proof,
        output=a.output,scene_id=a.scene_id))
    pair=read(a.output/'PAIR_AUDIT.json')
    for method in ('Scripted_Oracle','XVLA_terminal_release'):
        folder=a.output/method;evidence=read(folder/'content_evidence.json')
        audit=read(folder/'camera_audit.json');n=audit['streams'][0]['frames']
        index={0,n-1}
        if evidence['full_put_complete']:
            stable=evidence['stable_completion_frame'];index.update([max(0,stable-21),max(0,stable-15),stable])
        elif evidence['events']:
            # Strongest impact, or clearly measured acquisition/timeout/drop.
            events=sorted(evidence['events'],key=lambda e:e.get('object_displacement_within250ms_m',0),reverse=True)
            e=events[0];index.update([e['before_frame'],e['event_frame'],e['after_frame']])
        else:index.add(n//2)
        rows=[]
        for idx in sorted(index):
            idx=min(n-1,max(0,idx));panels=[]
            for stream in ('view_1','view_2','view_3','handeye'):
                panels.append(cv2.resize(decode_at(folder/(stream+'.mp4'),idx),(640,360),interpolation=cv2.INTER_AREA))
            image=np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
            file=folder/'review'/f'content_{idx:05d}.jpg';cv2.imwrite(str(file),image,[cv2.IMWRITE_JPEG_QUALITY,96])
            rows.append({'frame':idx,'physical_time_s':(idx+1)/30.,'file':str(file)})
        atomic_json(folder/'review/content_frames.json',rows)
        audit.update(scene_parity_vs_Oracle_NOT_yet_claimed=False,
            scene_parity_verified_in_PAIR_AUDIT=True,manual_visual_review_pending=True)
        atomic_json(folder/'camera_audit.json',audit)
        pair['outcomes'][method]['review_frames']=rows
        pair['outcomes'][method]['final_review_pending']=True
        alias_method='RL' if method=='Scripted_Oracle' else 'VLA'
        success=evidence['full_put_complete'];suffix='success' if success else 'failure'
        prefix='backup' if a.backup else 'scene'
        alias=a.output.parent/f'{prefix}{a.scene_id:02d}_{alias_method}_{suffix}.mp4'
        if alias.exists():raise FileExistsError('Do not overwrite delivered principal alias')
        os.link(folder/'view_2.mp4',alias)
        note=(folder/'view_2.txt').read_text()+(
            f'Actual requested fault/seed and selected slot: {a.scene_id}.\n'
            'Final encoded-event visual review is recorded separately; this label is observed outcome, not an SR claim.\n')
        alias.with_suffix('.txt').write_text(note)
        pair['outcomes'][method]['delivery_alias']=str(alias)
    pair.update(scene_id=a.scene_id,backup=bool(a.backup),source_pair=a.source_pair,
        main_certified=False,requires_actual_agent_visual_review=True)
    atomic_json(a.output/'PAIR_AUDIT.json',pair)
    print({'pair':str(a.output),'two_principal_aliases_created':True,'manual_review_pending':True},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('vla','oracle','repeat-proof','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scene-id',type=int,required=True);parser.add_argument('--backup',action='store_true')
    parser.add_argument('--source-pair',default='');main(parser.parse_args())
