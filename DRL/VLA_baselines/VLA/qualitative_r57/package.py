"""Pair, common layout, full-decode audit, truthful notes and coverage ledger."""
import argparse
import html
import os
from pathlib import Path
import shutil

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_recorder_r51 import audit_videos,SETTINGS
from env_eval.camera_editions import composite,overlap
from .common import read


def common_layout(frames):
    layouts={}
    for view in range(1,7):
        found=None
        for width in (594,536,460,384):
            height=round(width*9/16)//2*2
            for side,x in (('left',35),('right',1920-width-35)):
                rect=['handeye',x,37,width,height]
                costs=[]
                for row in frames:
                    item=row['wide'][view-1]
                    boxes=[item['projected_pick_box']]
                    if item['projected_place_box'] is not None:boxes.append(item['projected_place_box'])
                    costs.append(sum(overlap(box,rect) for box in boxes))
                if not max(costs):
                    found={'mode':'handeye','side':side,'rects':[rect],
                        'scope':'ONE_shared_layout_checked_over_BOTH_full_trajectories',
                        'reference594x334_preserved':width==594,'maximum_pick_goal_AABB_occlusion':0}
                    break
            if found:break
        if found is None:raise ValueError(f'No common unobstructed layout view{view}; retain raw videos, do not hide failure')
        layouts[str(view)]=found
    return layouts


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    vi=read(a.vla/'initial_scene.json');oi=read(a.oracle/'initial_scene.json')
    if vi['measured_state_sha256']!=oi['measured_state_sha256']:raise ValueError('Not a paired initial physical world')
    repeat=read(a.repeat_proof)
    if not repeat.get('passed'):raise ValueError('Oracle repeat failed')
    vf=read(a.vla/'frames.json');of=read(a.oracle/'frames.json')
    layouts=common_layout(vf+of);settings=read(SETTINGS);outcomes={}
    for method,source in (('Scripted_Oracle',a.oracle),('XVLA_terminal_release',a.vla)):
        dest=a.output/method;dest.mkdir();(dest/'review').mkdir()
        for name in ('handeye.mp4','frames.json','initial_scene.json','renderer.json','content_evidence.json'):
            os.link(source/name,dest/name)
        atomic_json(dest/'layout.json',layouts)
        for view in range(1,7):composite(source,dest,view,layouts[str(view)],settings)
        if method=='XVLA_terminal_release':
            proof=read(source/'replay_proof.json')
            shutil.copy2(source/'replay_proof.json',dest/'replay_proof.json')
        else:
            outcome=read(source/'outcome.json')
            proof={'physics_ticks':outcome['physics_ticks'],
                'benchmark_region_success':None,'safety_violation':outcome['error'],
                'every_after_action_full_physical_state_and_action_exact':repeat['passed']}
            shutil.copy2(source/'protocol.json',dest/'protocol.json')
            shutil.copy2(source/'outcome.json',dest/'outcome.json')
        audit=audit_videos(dest,proof)
        if not audit['media_decode_sync_pass']:raise ValueError('Final paired stream full-decode/sync failed')
        evidence=read(source/'content_evidence.json');success=evidence['full_put_complete']
        principal=dest/(('success' if success else 'failure')+'_'+method.lower()+f'_scene{a.scene_id:02d}_view2.mp4')
        os.link(dest/'view_2.mp4',principal)
        note=(f'Method: {method}. Scene ID: {a.scene_id}. Task: put. Seed: {vi["seed"]}.\n'
            'This is selected qualitative evidence, NOT a quantitative SR sample.\n'
            f'Complete put (verified hold, support and0.5s settle): {success}.\n'
            f'Actual events: {evidence["events"]}.\n'
            'Event frame indices are0-based; source physics times are in content_evidence.json.\n'
            'Oracle is scripted privileged feedback, not learned RL. Explicit controller-fault variant is in protocol/outcome.\n'
            'X-VLA approach/carry uses genuine frozen native neural controls; any terminal release is a separate disclosed supervisor.\n'
            'No memory-loss or misidentification claim follows from camera visibility alone.\n'
            'Video has no overlay/speed-up/interpolated poses. All six views and native hand-eye share source physical ticks.\n'
            'See repeat/replay proofs, raw telemetry and dense event frames. Manual final encoded-content review remains required.\n')
        for path in [principal,*[dest/f'view_{i}.mp4' for i in range(1,7)],dest/'handeye.mp4']:
            path.with_suffix('.txt').write_text(note)
        atomic_json(dest/'manifest.sha256.json',{f.name:sha256_file(f) for f in dest.glob('*.mp4')})
        outcomes[method]={'full_put_complete':success,'media_decode_sync_pass':True,
            'event_modes':[e['mode'] for e in evidence['events']],'principal':str(principal),
            'manual_encoded_content_review_pending':True}
    atomic_json(a.output/'PAIR_AUDIT.json',{'same_measured_initial_state':True,
        'initial_state_sha256':vi['measured_state_sha256'],'Oracle_repeat_proof':repeat,
        'common_layout':layouts,'outcomes':outcomes,'principal_method_rollouts':2,
        'unique_camera_streams':14,'principal_aliases_are_NOT_additional_trials':True,
        'asymmetric_controlled_Oracle_fault_and_VLA_release_supervisor_disclosed':True,
        'manual_final_review_required':True})
    blocks=[]
    for method,result in outcomes.items():
        path=Path(result['principal']).relative_to(a.output)
        blocks.append(f'<h2>{html.escape(method)}</h2><video controls preload="metadata" width="850" src="{path}"></video>')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Put qualitative pair</title>'
        '<h1>Paired put — scripted Oracle / X-VLA + terminal release</h1>'
        '<p>Not learned RL, not quantitative SR. Final visual review pending; controller variants and event evidence are disclosed in notes.</p>'
        +''.join(blocks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('vla','oracle','repeat-proof','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scene-id',type=int,required=True)
    main(parser.parse_args())
