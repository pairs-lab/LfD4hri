"""Reuse native3-view audit at ACTUAL TRAIN interaction events, not just reset.

Event labels choose diagnostic frames only; never fed to policy/renderer.
All wrapped changes are local to this CPU audit interpreter.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.privileged_visual_r53.data import MapData
from VLA.privileged_visual_r53.scene_map import contract


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    data=MapData(args.data,verify_rgb_hashes=False)
    import VLA.privileged_visual_r53.audit_inputs as audit
    original_atomic=audit.atomic_json
    summaries=[]
    for task in ('touch','pick','move','pick_place'):
        anchors=[];events={}
        for kind in ('complete_original','wrist_tilt','closed_empty'):
            ei=next(i for i,r in enumerate(data.rows) if r['task']==task and
                (r['source_kind']=='complete_original' if kind=='complete_original' else r.get('recovery_kind')==kind))
            numeric=data.numeric(ei);raw,actions=numeric['scene64'],numeric['canonical7']
            selected={'initial_pre_action':0,'terminal_pre_action':len(raw)-1}
            buckets={'lateral_gt_motion_audit_anchor':np.linalg.norm(actions[:,:2],axis=1)>.005,
                'measured_closed_empty':(raw[:,6]>.8)&(raw[:,31]<.5),
                'target_pad_contact':(raw[:,28:30]>.5).any(axis=1),
                'correct_bilateral_hold':raw[:,17]>.5}
            for name,flag in buckets.items():
                ids=np.flatnonzero(flag)
                if len(ids):selected[name]=int(ids[0])
            for label,fi in selected.items():
                key=(ei,fi)
                if key not in events:anchors.append(key);events[key]=[]
                events[key].append(label)
        audit.gate_anchors=lambda unused,values=anchors:values
        def atomic(path,value):
            if Path(path).name=='audit.json':
                value={**value,'scope':'actual_NATIVE_TRAIN_interaction_phase_examples_NOT_model_SR_or_exhaustive_coverage',
                    'diagnostic_events_NEVER_policy_inputs':True,'audit_extension_sha256':sha256_file(Path(__file__))}
                for row in value['rows']:
                    row['diagnostic_event_labels_NOT_policy_input']=events[row['episode'],row['frame']]
            original_atomic(path,value)
        audit.atomic_json=atomic
        audit.main(SimpleNamespace(data=args.data,family=args.family,output=args.output/task))
        summaries.append({'task':task,'anchors':len(anchors),'audit':str(args.output/task/'audit.json')})
    atomic_json(args.output/'audit.json',{'passed':True,'family':args.family,'profile':contract(),
        'scope':'actual_TRAIN_camera_GT_view_preprocessing_at_multiple_physical_events_NOT_neural_trials',
        'rows':summaries,'no_physics_or_policy_actions_executed':True})
    print(json.dumps({'family':args.family,'actual_event_anchors':sum(r['anchors'] for r in summaries)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=('xvla','vla_adapter'),required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
