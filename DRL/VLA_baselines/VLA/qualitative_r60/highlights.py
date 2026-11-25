"""Export untouched encoded frames suitable for the user's image highlights."""
from pathlib import Path
import html
import shutil
import cv2
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import decode_at
from VLA.qualitative_r57.common import read
from .prepare import OUT


def main():
    out=OUT/'HIGHLIGHT_FRAMES';out.mkdir(exist_ok=False);rows=[]
    for item in read(OUT/'produced_pairs.json'):
        pair=Path(item['paired_output']);sid=item['scene_id']
        selections=([('RL','Scripted_Oracle','Before Impact',124),
                     ('RL','Scripted_Oracle','Ground Slam',133),
                     ('RL','Scripted_Oracle','Ground Contact',157)] if sid==12 else
                    [('VLA','XVLA_terminal_release','Target Miss',177),
                     ('VLA','XVLA_terminal_release','Failed Recovery',599)])
        success_method='XVLA_terminal_release' if sid==12 else 'Scripted_Oracle'
        success=read(pair/success_method/'content_evidence.json')['stable_completion_frame']
        selections.append(('VLA' if sid==12 else 'RL',success_method,'Success',success))
        for alias,method,label,index in selections:
            frames=read(pair/method/'frames.json')
            for stream in (('view_2','view_1','handeye') if sid==13 else ('view_2','handeye')):
                source=pair/method/(stream+'.mp4')
                name=f'scene{sid:02d}_{alias}_{label.replace(" ","_")}_{stream}.png'
                frame=decode_at(source,index)
                cv2.imwrite(str(out/name),frame)
                rows.append({'file':name,'scene_id':sid,'method_alias':alias,'label':label,
                    'frame_index_0_based':index,'physical_time_s':frames[index]['simulation_time_s'],
                    'encoded_PTS_s':frames[index]['encoded_pts_s'],'source_video':str(source),
                    'source_video_sha256':sha256_file(source),'PNG_sha256':sha256_file(out/name),
                    'no_text_no_cropping_no_resizing_no_retiming':True,
                    'RL_alias_is_scripted_Oracle_NOT_trained_RL':True})
    native=OUT/'_captures/scene13_seed20271060/XVLA_verified/policy_inputs'
    for role in ('front','wrist'):
        shutil.copy2(native/f'reset_{role}.png',out/f'scene13_actual_policy_reset_{role}.png')
    atomic_json(out/'FRAME_MANIFEST.json',rows)
    blocks=['<!doctype html><meta charset="utf-8"><title>Untouched highlight frames</title>',
        '<style>body{font:16px sans-serif}img{max-width:900px;width:95%}</style>',
        '<h1>Frames for annotation</h1><p>Untouched decoded video frames. No labels added to source videos. RL alias = staged Oracle.</p>']
    for row in rows:
        blocks.append(f'<h2>Scene {row["scene_id"]}: {row["method_alias"]} — {html.escape(row["label"])}</h2>'
            f'<p>Frame {row["frame_index_0_based"]}, simulation {row["physical_time_s"]:.3f}s</p><img src="{row["file"]}">')
    (out/'index.html').write_text('\n'.join(blocks))
    print({'highlight_frames':len(rows),'policy_reset_frames':2},flush=True)


if __name__=='__main__':main()
