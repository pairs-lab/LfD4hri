"""Scene05 only: real slower lift and six seconds of physical holding.

No repeated/looped final video frames, no change to the other nine scenes,
no edits to the original80 movies or to VLA training/evaluation code.
"""
import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from VLA.common.artifact_io import atomic_json
from .motion import RequestedRollout
from .qualify import OUT, ROBOTS, digest
from .audit import read, audit
from .render import render, VIEWS

LIFT_SECONDS = 4.5
FINAL_HOLD_SECONDS = 6.0
REVISION = OUT/'scene05_extended_pick'


def retime(curve, seconds):
    original_duration = float(curve.duration)
    if original_duration <= 0 or seconds < original_duration:
        raise ValueError('Only positive, slower physical lift curves are allowed')
    return SimpleNamespace(duration=float(seconds),
        sample=lambda elapsed: curve.sample(float(elapsed)*original_duration/seconds))


class ExtendedPick(RequestedRollout):
    def curve(self, curve, phase, command_start=-1., command_end=None):
        if phase == 'carry':
            if self.spec.sid != 5 or self.scene.action != 'pick':
                raise ValueError('The revision applies only to scene05 pick')
            curve = retime(curve, LIFT_SECONDS)
        return super().curve(curve, phase, command_start, command_end)

    def hold(self, center, q, duration, phase, command_start, command_end=None):
        if phase == 'final':
            duration = max(duration, FINAL_HOLD_SECONDS)
        return super().hold(center, q, duration, phase, command_start, command_end)


def extended_audit(robot, out):
    folder = out/ROBOTS[robot]/'scene05'
    selection = read(folder/'PHYSICS.json')
    source = Path(selection['qualified_directory'])
    samples = read(source/'trajectory.json')
    initial = read(source/'INITIAL.json')
    original = read(OUT/ROBOTS[robot]/'scene05/PHYSICS.json')
    original_initial = read(Path(original['qualified_directory'])/'INITIAL.json')
    if set(initial) != set(original_initial): raise ValueError('Original scene object list differs')
    initial_errors = [float(np.max(abs(np.asarray(initial[name]['base_pose'][0])-
        np.asarray(original_initial[name]['base_pose'][0])))) for name in initial]
    if max(initial_errors) > 1e-6: raise ValueError('Original scene spawn positions differ')
    final = [s for s in samples if s['phase']=='final']
    carry = [s for s in samples if s['phase']=='carry']
    if len(final) < 180 or len(carry) < 135:
        raise ValueError('The physical lift/hold was not actually extended')
    positions = np.asarray([s['objects']['red block']['position'] for s in final])
    lift = positions[:,2]-initial['red block']['base_pose'][0][2]
    held = all(s['held'] and s['bilateral'] for s in final)
    drift = float(np.max(np.linalg.norm(positions-positions[0],axis=1)))
    velocity = max(float(np.linalg.norm(s['objects']['red block']['velocity'])) for s in final)
    visible = read(out/'_camera_sources'/ROBOTS[robot]/'scene05/visibility.json')
    end_visibility = [v for v in visible if v['phase']=='final']
    pixel_counts = {'front':min(v['target_pixels_front'] for v in end_visibility),
        **{f'view_{v}':min(row['wide'][str(v)]['target_pixels'] for row in end_visibility) for v in VIEWS}}
    passed = bool(held and lift.min()>.13 and drift<.006 and velocity<.020
        and pixel_counts['front']>=100 and min(pixel_counts[f'view_{v}'] for v in VIEWS)>=8)
    result = {'passed':passed,'robot':robot,'scene':5,'physical_lift_seconds':len(carry)/30,
        'physical_final_hold_seconds':len(final)/30,'all_final_frames_held_and_bilateral':held,
        'minimum_final_object_base_lift_m':float(lift.min()),'final_object_drift_m':drift,
        'maximum_final_object_speed_m_s':velocity,'minimum_final_target_pixels':pixel_counts,
        'same_original_objects_and_spawn_positions':True,'max_initial_position_difference_m':max(initial_errors),
        'real_physics_samples_NOT_still_frame_extension':True,'fps':30,
        'original_video_seconds':read(OUT/ROBOTS[robot]/'scene05/RENDER.json')['frames']/30,
        'new_video_seconds':len(samples)/30,'source_trajectory_sha256':digest(source/'trajectory.json')}
    atomic_json(folder/'EXTENDED_HOLD_AUDIT.json',result)
    if not passed: raise ValueError('Held-object stability/visibility audit failed: '+str(result))
    return result


def package(out, results):
    videos=[]
    page=['<!doctype html><meta charset="utf-8"><title>ICCAS scene05 — extended pick</title>',
        '<style>body{font:16px system-ui;background:#10151e;color:#e8edf5;margin:24px}'
        'video{width:100%;background:#000}article{max-width:1100px;margin-bottom:30px}a{color:#95caff}</style>',
        '<h1>Scene05 — Pick red block up</h1><p>4.5-second physical lift + 6-second physical hold. '
        'UR5e and UF850, views 2/3/4/6, front-gripper inset only. '
        'Scripted privileged feedback Oracle, not trained RL/VLA.</p>']
    for robot in ROBOTS:
        folder=out/ROBOTS[robot]/'scene05'
        if not read(folder/'AUDIT.json')['passed'] or not read(folder/'EXTENDED_HOLD_AUDIT.json')['passed']:
            raise ValueError('Do not publish unaudited movies')
        page.append(f'<h2>{html.escape(ROBOTS[robot])}</h2>')
        for view in VIEWS:
            path=folder/f'view_{view}.mp4';relative=str(path.relative_to(out))
            videos.append({'path':relative,'sha256':digest(path)})
            page.append(f'<article><h3>View {view}</h3><video controls preload="none" src="{relative}"></video></article>')
    page.append('<p><a href="../index.html">Original 80-video gallery (unchanged movies)</a></p>')
    (out/'index.html').write_text('\n'.join(page))
    atomic_json(out/'MANIFEST.json',{'complete':True,'main_video_count':len(videos),'expected_main_videos':8,
        'robots':list(ROBOTS),'scene':5,'views':list(VIEWS),'videos':videos,'extended_hold_results':results,
        'old_movies_preserved':True,'executor':'scripted privileged feedback Oracle'})
    text='# Scene05 extended pick — complete\n\n'
    text+='Same six objects/spawn positions, two robots, four views (2/3/4/6).\n'
    text+='Actual lift is4.5s; actual final closed-gripper hold is6s. No repeated still frames or playback retiming.\n'
    for row in results:
        text+=f"\n{row['robot']}: {row['original_video_seconds']:.2f}s -> {row['new_video_seconds']:.2f}s; "
        text+=f"minimum final object-base lift {row['minimum_final_object_base_lift_m']*100:.2f}cm.\n"
    text+='\nFull stream decode, timing, synchronized inset, measured FK, physical holding and original-scene checks passed.\n'
    text+='Original80 movies remain untouched. This is scripted Oracle demonstration, not trained RL/VLA.\n'
    (out/'REPORT.md').write_text(text)


def main(a):
    from . import qualify as physical
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_json(a.output/'PLAN.json',{'scene':5,'robots':list(ROBOTS),'views':list(VIEWS),
        'lift_seconds':LIFT_SECONDS,'final_hold_seconds':FINAL_HOLD_SECONDS,
        'original_movies_preserved':True,'no_still_frame_extension':True,
        'physics_assets_cameras_encoding_unchanged':True,'source_sha256':digest(Path(__file__))})
    original_class=physical.RequestedRollout
    physical.RequestedRollout=ExtendedPick
    results=[]
    try:
        for robot in ROBOTS:
            atomic_json(a.output/'PROGRESS.json',{'stage':'actual_physical_execution','robot':robot})
            result=physical.qualify(robot,5,a.output)
            if not result['overall_pass']: raise ValueError('Extended physical run failed, evidence retained')
            atomic_json(a.output/'PROGRESS.json',{'stage':'rendering_four_HD_views','robot':robot})
            render(robot,5,a.output)
            atomic_json(a.output/'PROGRESS.json',{'stage':'full_decode_and_holding_audit','robot':robot})
            if not audit(robot,5,a.output)['passed']: raise ValueError('Independent media audit failed')
            results.append(extended_audit(robot,a.output))
        package(a.output,results)
        atomic_json(a.output/'PROGRESS.json',{'stage':'completed_and_audited8','main_videos':8,'robots':list(ROBOTS)})
    except Exception as error:
        import traceback
        atomic_json(a.output/'ERROR.json',{'error':str(error),'traceback':traceback.format_exc(),'NOT_complete':True})
        raise
    finally:
        physical.RequestedRollout=original_class


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=REVISION)
    main(p.parse_args())
