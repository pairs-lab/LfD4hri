"""Publish only manually-reviewed new pairs and immutable named old media."""
from collections import Counter
from datetime import datetime,timezone
import html
import os
from pathlib import Path
import shutil
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_r57.common import read
from .prepare import OUT,OLD


def copy_artifact(source,target):
    if Path(source).suffix.lower() in ('.mp4','.png','.jpg','.jpeg'):
        os.link(source,target)
    else:
        shutil.copy2(source,target)
    return target


def main():
    rows=read(OUT/'LABELS.json')
    if any(r['scene_id']>=12 and not r['backup'] for r in rows):
        raise ValueError('Already published: do not silently overwrite final label inventory')
    # Preserve directory structure and all original camera streams without
    # duplicating heavy bytes or changing any old JSON/text/media inode.
    for item in read(OLD/'produced_pairs.json'):
        source=Path(item['paired_output']);target=OUT/item['group']/source.name
        if target.exists():raise FileExistsError(target)
        shutil.copytree(source,target,copy_function=copy_artifact)
    old_review=Path('VLA/qualitative/ur5e_xvla_put_r59_independent_review_20261017/REVIEW_COMPLETE.json')
    assert read(old_review)['passed']
    new=read(OUT/'produced_pairs.json')
    if len(new)!=2:raise ValueError('Both requested new pairs are required')
    for item in new:
        pair=Path(item['paired_output']);review=read(pair/'FINAL_VISUAL_REVIEW.json')
        if not review['approved_for_delivery']:raise ValueError('Actual visual review absent')
        if not read(pair/'review/independent/actual_visual_review.json')['passed']:
            raise ValueError('Independent 12-frame visual review absent')
        proof=read(pair/'PAIR_AUDIT.json')
        assert proof['same_measured_initial_state'] and proof['Oracle_repeat_proof']['passed']
        labels=('Ground Slam','Success') if item['scene_id']==12 else ('Success','Target Miss')
        for method,folder,label in zip(('RL','VLA'),('Scripted_Oracle','XVLA_terminal_release'),labels):
            src=pair/folder/'view_2.mp4';dest=OUT/item['group']/f'scene{item["scene_id"]:02d}_{method}_{label.replace(" ","_")}.mp4'
            generic=Path(proof['outcomes'][folder]['delivery_alias'])
            if sha256_file(generic)!=sha256_file(src):raise ValueError('Principal alias changed')
            if dest.exists():raise FileExistsError(dest)
            generic.rename(dest)
            if generic.with_suffix('.txt').exists():generic.with_suffix('.txt').rename(dest.with_suffix('.txt'))
            proof['outcomes'][folder]['delivery_alias']=str(dest)
            row={'scene_id':item['scene_id'],'backup':False,'seed':item['seed'],'method_alias':method,
                'actual_method':'Scripted feedback Oracle, NOT trained RL' if method=='RL' else
                    'Frozen native X-VLA; no action modification',
                'label':label,'secondary_label':'Failed Recovery' if label=='Target Miss' else None,
                'success':proof['outcomes'][folder]['full_put_complete'],
                'video':str(dest.relative_to(OUT)),'sha256':sha256_file(src),
                'paired_scene':str(pair),'original_video':str(src),'same_bytes_hardlinked':True}
            best=OUT/'BEST_ANGLES'/dest.name;best.parent.mkdir(exist_ok=True)
            preferred=1 if item['scene_id']==13 else 2
            os.link(pair/folder/f'view_{preferred}.mp4',best)
            row.update(recommended_video=str(best.relative_to(OUT)),recommended_fixed_camera=preferred,
                recommended_video_sha256=sha256_file(best),
                recommended_camera_same_for_BOTH_methods=True,
                alternative_camera_alias_NOT_an_additional_episode=True)
            rows.append(row);atomic_json(dest.with_suffix('.json'),row)
        atomic_json(pair/'PAIR_AUDIT.json',proof)
        item.update(status='certified_full_stream_decode_and_sampled_encoded_visual_review',
            final_visual_review=str(pair/'FINAL_VISUAL_REVIEW.json'))
    atomic_json(OUT/'produced_pairs.json',new)
    for r in rows:
        task=read(Path(r['paired_scene'])/'Scripted_Oracle/initial_scene.json')['task']
        r.update(instruction=task['prompt'],requested_object=task['pick_name'],destination=task['place_name'])
        assert 1<=len(r['label'].split())<=2
        assert sha256_file(OUT/r['video'])==r['sha256']
        atomic_json((OUT/r['video']).with_suffix('.json'),r)
    atomic_json(OUT/'LABELS.json',rows)
    main_rows=[r for r in rows if not r['backup']];backup_rows=[r for r in rows if r['backup']]
    groups=Counter(Path(r['video']).parts[0] for r in main_rows if r['method_alias']=='RL')
    assert len(main_rows)==26 and len(backup_rows)==8
    table=['# English highlight labels','',
      'RL is a folder alias for scripted feedback Oracle, NOT trained RL. Oracle faults are staged and disclosed.',
      'These selected qualitative videos are not an unbiased quantitative failure distribution.','',
      '| Scene | Oracle (RL alias) | X-VLA |','|---|---|---|']
    for sid in range(1,14):
        q=[r for r in main_rows if r['scene_id']==sid]
        table.append(f'| {sid:02d} | '+ ' | '.join(next(r['label'] for r in q if r['method_alias']==m) for m in ('RL','VLA'))+' |')
    table+=['',
      '- Ground Slam: actual high-speed gripper-floor contact; staged wrong ground-height controller, not learned-RL evidence.',
      '- Target Miss: native X-VLA first close about 29 cm away from target; zero verified hold through 20 seconds.',
      '- Failed Recovery: optional secondary label for scene 13, not proof of memory loss or semantic confusion.',
      '- Object Ejection: contact-confirmed displacement; elastic vertical bouncing is not implied.',
      '- Empty Lift: the gripper rises without carrying the target; never label as a transit drop.',
      '- Transit Drop: verified holding precedes actual detachment and fall away from goal.',
      '- Grasp Miss: no verified acquisition. Timeout: explicit physical-time budget exhausted.']
    table+=['','BEST_ANGLES contains unchanged paired reference-camera clips: scene 12 uses view 2; scene 13 uses view 1 because the gray target is less occluded. Both methods in a scene always use the same selected camera. These are aliases, not new trials.']
    (OUT/'LABELS.md').write_text('\n'.join(table)+'\n')
    report=['# R60 delivery','',
      '26 main videos (13 same-scene pairs), plus 8 backup videos (4 pairs). Original 11 pairs remain unchanged.',
      'All tasks PUT, UR5e + Robotiq 2F-85. Each method has six 1920×1080 wide views and native 1280×720 hand-eye, H.264, 30 FPS.',
      'Principal named files are exact aliases of view_2, not extra episodes.',
      'BEST_ANGLES and the gallery recommend view_1 for BOTH sides of scene13: it keeps the stationary gray target visible while the gripper closes elsewhere. Camera extrinsics are unchanged. Scene12 uses view_2 for BOTH sides.',
      'Camera mode is hand-eye inset only; no UF850 or other inset modes are claimed.','',
      '## Added cases','',
      'Scene 12: staged feedback-Oracle Ground Slam / X-VLA Success. Incoming downward TCP speed ≈0.354 m/s; peak ground force ≈852 N. Original collision/linkage guard stays active. No external force, teleport, physics modification, or learned-RL failure claim.',
      'Scene 12 reuses the verified native X-VLA source rollout from scene 01, paired with a newly executed Oracle fault on that exact initial world. It is a counterfactual qualitative pairing, not another independent VLA trial. There are 238 delivered camera-stream slots, but 33 distinct source method rollouts (34 displayed slots, one reused X-VLA source).',
      'Scene 13: feedback-Oracle Success / native X-VLA Target Miss, followed by Failed Recovery. First close is 28.99 cm away in XY, target moves <0.001 mm, zero verified holding, 20 s actual rollout. Seven original RGB/proprio/GT RPC inputs and every saved native command are replay-verified; no fabricated neural failure. Both reset policy cameras show the gray target fully.','',
      'Scene 13 actual instruction: Put the gray block onto the bottom right corner. The pink bowl is a distractor, not the goal. The arm initially reaches above the gray target, then closes elsewhere; this does not prove semantic misidentification.','',
      '## Limits and fair interpretation','',
      'Oracle controller faults are explicitly scripted; this is not evidence of a trained RL policy failing. X-VLA scene 13 is selected post hoc from a completed frozen quantitative trace; selection does not add SR trials or change the quantitative cohort.',
      'Ground-truth privileged information remains available to X-VLA. The video does not prove semantic misidentification, intrinsic memory loss, or a fundamental model limitation.',
      'The original X-VLA successes use a disclosed terminal opening-only supervisor. No arm waypoint correction is inserted. Scene 13 never uses that supervisor.',
      'Manual review covers sampled decoded event/aftermath/all-view frames, not continuous every-frame human viewing. Full-stream automated decode, time spacing and synchrony audits are separate.',
      'Physical execution uses the pinned PyBullet binding. HD images use a non-stepping rendering replica, gated against pinned-binding lossless RGB and segmentation samples.','',
      'See LABELS.md, LABELS.json, HIGHLIGHT_FRAMES/index.html, each new pair’s PAIR_AUDIT.json, FINAL_VISUAL_REVIEW.json, review/independent, and source captures.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    parts=['<!doctype html><meta charset="utf-8"><title>R60 named paired qualitative videos</title>',
      '<style>body{font:16px sans-serif;background:#111;color:#eee;margin:24px}video{width:48%;max-width:850px}a{color:#9cf}section{margin:30px 0}.pair{display:flex;gap:14px}.method{width:49%}.method video{width:100%}</style>',
      '<h1>Paired PUT: Oracle / native X-VLA</h1>',
      '<p>26 main videos + 8 backups. RL alias = explicitly scripted feedback Oracle, not trained RL. Selected qualitative cases, not quantitative SR.</p>',
      '<p><a href="REPORT.md">Report</a> · <a href="LABELS.md">English highlight labels</a> · <a href="HIGHLIGHT_FRAMES/index.html">Frames for annotation</a></p>']
    for backup,sid in sorted({(r['backup'],r['scene_id']) for r in rows}):
        q=[r for r in rows if (r['backup'],r['scene_id'])==(backup,sid)]
        parts.append(f'<section><h2>{"Backup" if backup else "Scene"} {sid:02d}</h2>')
        parts.append('<p>Instruction: '+html.escape(q[0]['instruction'])+'</p>')
        if not backup and sid==12:
            parts.append('<p>Counterfactual Oracle fault on scene 01 initial world; verified X-VLA source rollout reused, not a new VLA trial.</p>')
        if not backup and sid>=12:
            parts.append('<p>Recommended fixed camera: '+('view 1 (both methods; target remains unobscured).' if sid==13 else 'view 2 (both methods; floor contact visible).')+'</p>')
        parts.append('<div class="pair">')
        for r in q:
            alias='Scripted Oracle (RL alias)' if r['method_alias']=='RL' else 'X-VLA'
            video=r.get('recommended_video',r['video'])
            parts.append(f'<div class="method"><p>{alias}: {html.escape(r["label"])}</p><video controls preload="none" src="{video}"></video></div>')
        parts.append('</div></section>')
    (OUT/'index.html').write_text('\n'.join(parts))
    status={'status':'complete_sampled_visual_and_full_stream_media_review',
       'updated_UTC':datetime.now(timezone.utc).isoformat(),'main_videos':26,'backup_videos':8,
       'main_pairs':13,'backup_pairs':4,'group_pair_counts':dict(groups),'delivered_camera_stream_slots':238,
       'distinct_source_method_rollouts':33,'displayed_method_rollout_slots':34,
       'scene12_VLA_source_reused_from_scene01_NOT_an_extra_trial':True,
       'original_videos_unchanged':True,'all_short_labels_1_or_2_words':True,
       'new_pairs_visual_review_pass':True,'original_R59_review_record':str(old_review)}
    original_integrity=read(OUT/'audit/original_integrity/independent_delivery_integrity.json')
    if not original_integrity['passed']:raise ValueError('Original certified delivery changed')
    if not read(OUT/'audit/CPU_AND_PROVENANCE_AUDIT.json')['passed']:raise ValueError('Native/plant provenance failed')
    from .highlights import main as highlights
    highlights()
    atomic_json(OUT/'FINAL_SOURCE_MANIFEST.json',{str(p):sha256_file(p)
        for p in Path(__file__).parent.iterdir() if p.is_file()})
    atomic_json(OUT/'status.json',status)
    queue=read(OUT/'queue_status.json')
    atomic_json(OUT/'queue_status.json',{'status':'complete_with_actual_visual_review',
        'automatic_media_stage_record':queue,'final_delivery_status':str(OUT/'status.json')})
    inventory={};inode_cache={}
    for p in OUT.rglob('*'):
        if not p.is_file() or p.name=='MANIFEST.sha256.json' or any(s.startswith('_') for s in p.relative_to(OUT).parts):continue
        st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in inode_cache:inode_cache[key]=sha256_file(p)
        inventory[str(p.relative_to(OUT))]=inode_cache[key]
    atomic_json(OUT/'MANIFEST.sha256.json',inventory)
    print({'published':str(OUT),'main':26,'backups':8,'manifest_entries':len(inventory)},flush=True)


if __name__=='__main__':main()
