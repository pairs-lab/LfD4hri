"""Lightweight gallery packaging for arbitrary future three-edition runs."""
import json
from pathlib import Path
from env_eval.package_camera_editions import html,showcase,digest
from env_eval.camera_editions import dump


def finalize_galleries(parent,name,modes,settings):
    for mode in modes:
        root=parent/(name+settings['folder_suffixes'][mode]);audits=[];rows=[];physical=[]
        for path in sorted(root.glob('*/*/*/camera_audit.json')):
            audit=json.loads(path.read_text());audits.append(audit)
            physics=json.loads(path.with_name('audit.json').read_text());physical.append(physics)
            rows.append(dict(robot=path.parts[-4],action=physics['action'],config=physics['config'],
                path=str(path.parent.relative_to(root)),prompt=physics['task']['prompt'],duration=physics['duration_seconds']))
        if not rows:continue
        if (root/'quality_summary.json').exists():continue # Preserve a completed, richer audit package.
        dump(root/'camera_settings.json',settings);dump(root/'results.json',physical)
        videos=[v for audit in audits for v in audit['videos']]
        dump(root/'presentation_audit.json',dict(scenes=len(audits),scenes_passed=sum(a['pass'] for a in audits),
            videos=len(videos),videos_passed=sum(v['pass_video'] for v in videos),rows=audits))
        for label in ('UF850','UR5e_Robotiq2F'):
            if sum(r['robot']==label for r in rows)==16:showcase(root,label)
        html(root,mode,rows,name)
        report=f'''# Camera edition: {mode}

{len(audits)} configurations; {len(videos)} per-configuration clips. All configured inset rectangles are fixed for the whole clip; stacked front/hand-eye panels have equal dimensions. Wide streams: 1920×1080 at 30 FPS; native front/hand-eye: 1280×720. No added video text.

[Open gallery](index.html) · [Camera settings](camera_settings.json) · [All encoded-video audits](presentation_audit.json).

Controller: privileged feedback Oracle with IK and post-contact grasp assist; not a trained RL/VLA policy. Exact comparison with an earlier baseline is reported only when a baseline was supplied. See per-scene `replay_proof.json`, `trajectory.json`, `camera_audit.json` and `layout.json`.
'''
        (root/'REPORT.md').write_text(report)
        dump(root/'manifest.json',dict(videos={str(p.relative_to(root)):digest(p) for p in root.rglob('*.mp4')}))
