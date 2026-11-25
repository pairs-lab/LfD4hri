"""Report camera limitations honestly, separately from media/task success."""
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from .common import OUT,read


def main():
    rows=[]
    for item in read(OUT/'produced_pairs.json'):
        pair=Path(item['paired_output'])
        for method in ('Scripted_Oracle','XVLA_terminal_release'):
            folder=pair/method;audit=read(folder/'camera_audit.json')
            result={'scene':item['scene_id'],'backup':item['backup'],'seed':item['seed'],
                'method':method,'task_success':read(folder/'content_evidence.json')['full_put_complete'],
                'presentation_handeye':audit['visibility'],
                'camera_visibility_not_equated_with_task_success_or_media_decode':True}
            if method=='XVLA_terminal_release':
                controls=read(folder/'replay_proof.json')['controls']
                visibility={}
                for role in ('front','wrist'):
                    present=[c['actual_policy_visibility'][role] for c in controls]
                    absent=[c['control'] for c in controls if not c['actual_policy_visibility'][role]['target_pixels']]
                    queried=[c for c in controls if c['control']%30==0]
                    absent_queries=[c['control'] for c in queried if not c['actual_policy_visibility'][role]['target_pixels']]
                    box=present[0]['target_box']
                    visibility[role]={'initial_target_pixels':present[0]['target_pixels'],
                        'initial_target_bbox':box,'initial_margin_px':min(box[0],box[1],383-box[2],383-box[3]) if box else None,
                        'target_absent_control_count':len(absent),'control_count':len(controls),
                        'target_absent_at_actual_neural_queries':absent_queries,'query_count':len(queried),
                        'absence_alone_does_not_prove_memory_loss':True}
                result['actual_policy_observations']=visibility
            rows.append(result)
    atomic_json(OUT/'audit/camera_visibility_limits.json',{'rows':rows,
        'policy_camera_and_presentation_camera_are_distinct':True,
        'post_impact_or_policy_motion_can_remove_target_from_FOV':True,
        'no_claim_that_every_object_is_visible_through_every_failure_frame':True})
    lines=['# Camera coverage: policy versus presentation','',
        'Whole-stream media integrity is not a promise that a policy keeps its target in view after a failed interaction. These measured warnings are retained, not hidden or fixed by altering the trajectory.',
        '', '|Scene|Backup|Method|Hand-eye target absent frames|Minimum critical target pixels|Minimum critical margin px|',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        v=r['presentation_handeye']
        lines.append(f"|{r['scene']}|{r['backup']}|{r['method']}|{v['target_outside_presentation_handeye_frames']}|{v['minimum_critical_target_pixels']}|{v['minimum_critical_target_margin']}|")
    lines+=['','Raw per-query/initial policy visibility and pad-pixel measurements: `audit/camera_visibility_limits.json`.',
        'Thresholds (object100px, each pad20px, margin8px) are reference visibility diagnostics, not silently substituted task-success conditions.',
        'All selected VLA targets must initially be uncut in both policy images; the banana candidate failing this audit was rejected. This does not prove all distractors are visible at every instant.',
        'Do not infer that the neural network receives the presentation hand-eye inset. Its native wrist/front preprocessing and privileged input remain unchanged.','']
    (OUT/'CAMERA_VISIBILITY_FINDINGS.md').write_text('\n'.join(lines))
    print({'camera_visibility_rows':len(rows),'all_absences_reported':True},flush=True)


if __name__=='__main__':main()
