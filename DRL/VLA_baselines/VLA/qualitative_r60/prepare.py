"""Create NEW delivery and short, evidence-bound labels; preserve originals."""
import html
import json
import os
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT=Path('/work/LfD4hri/DRL')
OUT=ROOT/'VLA/qualitative/ur5e_xvla_put_r60_named_failures_20261017'
OLD=ROOT/'VLA/qualitative/ur5e_xvla_put_r58_complete_20261017'
LABELS={1:('Success','Success'),2:('Success','Success'),3:('Transit Drop','Success'),
 4:('Grasp Miss','Success'),5:('Timeout','Success'),6:('Success','Object Ejection'),
 7:('Success','Object Ejection'),8:('Success','Object Ejection'),
 9:('Transit Drop','Object Ejection'),10:('Grasp Miss','Empty Lift'),
 11:('Transit Drop','Empty Lift')}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'PLAN.md').write_text((Path(__file__).parent/'PLAN.md').read_text())
    rows=[]
    for pair in json.loads((OLD/'produced_pairs.json').read_text()):
        backup=pair['backup']; sid=pair['scene_id']
        labels=(('Success' if pair['Oracle_full_put'] else
                 'Timeout' if pair['oracle_fault']=='slow_transport_timeout' else 'Transit Drop'),
                'Success' if pair['VLA_full_put'] else 'Failure') if backup else LABELS[sid]
        for method,label,ok in zip(('RL','VLA'),labels,(pair['Oracle_full_put'],pair['VLA_full_put'])):
            tag='backup' if backup else 'scene'
            source=OLD/pair['group']/f'{tag}{sid:02d}_{method}_{"success" if ok else "failure"}.mp4'
            target=OUT/pair['group']/f'{tag}{sid:02d}_{method}_{label.replace(" ","_")}.mp4'
            target.parent.mkdir(parents=True,exist_ok=True)
            os.link(source,target)
            row={'scene_id':sid,'backup':backup,'seed':pair['seed'],'method_alias':method,
                 'actual_method':'Scripted feedback Oracle, NOT trained RL' if method=='RL' else
                    'Native X-VLA with disclosed terminal release supervisor',
                 'label':label,'success':ok,'video':str(target.relative_to(OUT)),
                 'original_video':str(source),'sha256':sha256_file(source),
                 'paired_scene':pair['paired_output'],'same_bytes_hardlinked':True}
            rows.append(row)
            target.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n')
    atomic_json(OUT/'LABELS.json',rows)
    atomic_json(OUT/'status.json',{'existing_named_main_videos':22,'existing_named_backup_videos':8,
      'new_ground_slam_pair':'pending actual physical contact and visual audit',
      'new_target_miss_pair':'pending native replay and visual audit',
      'original_videos_unchanged':True})
    table=['# English highlight labels','',
      'RL in filenames is the user-requested alias for a scripted feedback Oracle, NOT trained RL.',
      'Native VLA symptoms are not proof of semantic confusion or memory loss.','',
      '| Scene | Oracle (RL alias) | X-VLA |','|---|---|---|']
    for sid,(ol,vl) in LABELS.items():table.append(f'| {sid:02d} | {ol} | {vl} |')
    table+=['','New Ground Slam and Target Miss pairs remain pending until physical and visual evidence passes.']
    (OUT/'LABELS.md').write_text('\n'.join(table)+'\n')
    gallery=['<!doctype html><meta charset="utf-8"><title>R60 named qualitative videos</title>',
      '<style>body{font:16px sans-serif;background:#111;color:#eee}video{width:47%;max-width:800px}section{margin:24px 0}</style>',
      '<h1>Named qualitative pairs</h1><p>RL = staged feedback Oracle, not trained RL. Original certified videos preserved.</p>',
      '<p>New Ground Slam / Target Miss cases: in preparation; not yet certified.</p>']
    for backup,sid in sorted({(r['backup'],r['scene_id']) for r in rows}):
        group=[r for r in rows if (r['backup'],r['scene_id'])==(backup,sid)]
        gallery.append(f'<section><h2>{"Backup" if backup else "Scene"} {sid:02d}</h2>')
        for row in group:
            gallery.append(f'<p>{row["method_alias"]}: {html.escape(row["label"])}</p><video controls preload="none" src="{row["video"]}"></video>')
        gallery.append('</section>')
    (OUT/'index.html').write_text('\n'.join(gallery))
    print(OUT,flush=True)


if __name__=='__main__':main()
