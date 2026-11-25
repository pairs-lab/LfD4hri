"""Serial, resumable render -> independent audit -> gallery for all 80 movies."""
import argparse
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from VLA.common.artifact_io import atomic_json
from .qualify import ROOT,OUT,ROBOTS,digest
from .render import render,VIEWS
from .audit import audit,motion_audit,read,SCHEMA
from .scenes import SCENES,display_command


def package(out):
    rows=[];videos=[]
    for robot,label in ROBOTS.items():
        for scene in SCENES:
            folder=out/label/f'scene{scene.sid:02d}'
            check=read(folder/'AUDIT.json') if (folder/'AUDIT.json').exists() else None
            if not check or not check['passed'] or check.get('schema')!=SCHEMA:continue
            row={'robot':robot,'scene':scene.sid,'folder':str(folder.relative_to(out)),
                'commands':[display_command(s) for s in scene.steps], 'frames':check['all_frames'],'audit_passed':True}
            rows.append(row)
            for view in VIEWS:
                path=folder/f'view_{view}.mp4';videos.append({'path':str(path.relative_to(out)),'sha256':digest(path)})
    atomic_json(out/'MANIFEST.json',{'audited_scene_robot_pairs':len(rows),'expected_pairs':20,
        'main_video_count':len(videos),'expected_main_videos':80,'scene_robot_pairs':rows,'videos':videos,
        'all_main_videos_complete':len(videos)==80,'executor':'scripted privileged feedback Oracle, not RL/VLA'})
    chunks=['<!doctype html><meta charset="utf-8"><title>ICCAS_demo</title>',
        '<style>body{font:16px system-ui;background:#10151e;color:#e8edf5;margin:24px} '
        '.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px} '
        'video{width:100%;background:#000} article{padding:15px;background:#1b2432;border-radius:8px} '
        'a{color:#95caff} select{padding:6px}</style>',
        '<h1>ICCAS_demo</h1><p>UR5e–Robotiq / UF850 · 10 scenes · views 2, 3, 4, 6 · front-gripper only.</p>',
        f'<p>{len(rows)}/20 scene/robot pairs independently audited; {len(videos)}/80 main videos. '
        'Scripted privileged feedback Oracle — not learned RL or VLA.</p><div class="grid">']
    for row in rows:
        folder=out/row['folder'];images=sorted((folder/'review').glob('*_initial_view_6.jpg'))
        poster=str(images[0].relative_to(out)) if images else ''
        title=html.escape(f'{ROBOTS[row["robot"]]} / scene {row["scene"]:02d}')
        opts=''.join(f'<option value="{v}" {"selected" if v==6 else ""}>view {v}</option>' for v in VIEWS)
        chunks.append(f'<article><h2>{title}</h2><p>{html.escape(" → ".join(row["commands"]))}</p>'
            f'<select onchange="this.nextElementSibling.src=\'{row["folder"]}/view_\'+this.value+\'.mp4\'">{opts}</select>'
            f'<video controls preload="none" poster="{poster}" src="{row["folder"]}/view_6.mp4"></video>'
            f'<p><a href="{row["folder"]}/AUDIT.json">Audit</a> · '
            f'<a href="{row["folder"]}/review/CONTACT_SHEET_view_6.jpg">Contact sheet</a></p></article>')
    chunks.append('</div>');(out/'index.html').write_text('\n'.join(chunks))
    return rows,videos


def main(out):
    with (out/'.batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=out/'BATCH_ERROR.json'
        if previous.exists():
            archive=out/'_maintenance_errors';archive.mkdir(exist_ok=True)
            previous.replace(archive/f'previous_error_{time.time_ns()}.json')
        start=time.monotonic()
        try:
            # Both robot pilots, apple/cup, and multi-task examples first;
            # never repeat an already independently audited scene.
            jobs=[(r,s) for s in (1,4,8,9,10,2,3,5,6,7) for r in ROBOTS]
            for robot,sid in jobs:
                folder=out/ROBOTS[robot]/f'scene{sid:02d}'
                existing=folder/'AUDIT.json'
                if existing.exists() and read(existing)['passed'] and read(existing).get('schema')==SCHEMA:continue
                if not (folder/'PHYSICS.json').exists():raise RuntimeError(f'No qualified physical scene: {robot}/{sid}')
                physical=read(folder/'PHYSICS.json');trajectory=read(Path(physical['qualified_directory'])/'trajectory.json')
                if not physical['result']['overall_pass'] or not motion_audit(trajectory)['passed']:
                    raise RuntimeError('Physical/motion gate failed; no false video success')
                done,videos=package(out)
                atomic_json(out/'PROGRESS.json',{'status':'rendering_then_independent_audit','robot':robot,'scene':sid,
                    'audited_scene_robot_pairs':len(done),'audited_main_videos':len(videos),'target_main_videos':80,
                    'pid':os.getpid(),'elapsed_wall_s':time.monotonic()-start})
                render(robot,sid,out)
                result=audit(robot,sid,out)
                if not result['passed']:raise RuntimeError(f'Encoded/physical audit failed: {robot}/{sid}; evidence retained')
                package(out)
            rows,videos=package(out)
            if len(rows)!=20 or len(videos)!=80:raise ValueError('Not all 20 cases / 80 main videos are audited')
            atomic_json(out/'PROGRESS.json',{'status':'completed_and_audited80','audited_scene_robot_pairs':20,
                'audited_main_videos':80,'target_main_videos':80,'pid':os.getpid(),
                'elapsed_wall_s':time.monotonic()-start})
            report='# ICCAS_demo — complete\n\n20 actual physical scene/robot executions, 80 main HD movies.\n\n'
            report+='Both robots: scenes 1–10, cinematic views 2/3/4/6, front-gripper inset only.\n'
            report+='Scenes 8–9 keep continuous object state and return home between subtasks.\n'
            report+='All outputs independently fully decoded, frame/timing/inset checks passed.\n'
            report+='Measured-state presentation replay has no extra physics steps and matches source TCP/FK.\n'
            report+='These are scripted privileged feedback Oracle demonstrations, NOT trained RL/VLA evidence.\n'
            (out/'REPORT.md').write_text(report)
        except BaseException as e:
            atomic_json(out/'BATCH_ERROR.json',{'error':str(e),'traceback':traceback.format_exc(),
                'NOT_complete80':True,'elapsed_wall_s':time.monotonic()-start})
            rows,videos=package(out)
            atomic_json(out/'PROGRESS.json',{'status':'needs_diagnosis_NOT_complete80','error':str(e),
                'audited_scene_robot_pairs':len(rows),'audited_main_videos':len(videos),'target_main_videos':80})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--launch',action='store_true');a=parser.parse_args()
    if a.launch:
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        logpath=a.output/('batch.log' if not (a.output/'batch.log').exists() else f'batch_resume_{time.time_ns()}.log')
        with logpath.open('x') as log:
            child=subprocess.Popen(['nice','-n','10',sys.executable,'-u','-m','env_eval.iccas_demo.batch',
                '--output',str(a.output)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'batch_pid':child.pid,'output':str(a.output),'log':str(logpath)}))
    else:main(a.output)
