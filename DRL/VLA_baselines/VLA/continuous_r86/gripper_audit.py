"""Quantify actual old adapter/actuator semantic disagreement, without replay."""
from collections import Counter
import json
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


if __name__=='__main__':
    root=Path('VLA/audits/recovery_r85_20261021/vlanext_s/dev40_fit512')
    progress=json.loads((root/'progress.json').read_text())
    controls=deadband=mismatches=affected=0;counts=Counter();unsafe=0
    for row in progress['rows']:
        trace=json.loads(Path(row['trajectory']).read_text())[row['task']];this=0
        for entry in trace:
            controls+=1;g=entry['action'][6]
            in_deadband=abs(g)<=.2;deadband+=int(in_deadband)
            mismatch=bool(g>0)!=bool(entry['post_action_scene64'][10]>.5)
            if entry.get('safety_violation'):
                unsafe+=int(mismatch)
                continue  # rollback can restore command; do not call this a decoder bug
            if in_deadband and mismatch:
                mismatches+=1;this+=1;counts[row['task']]+=1
        affected+=int(this>0)
    source=Path('VLA/third_party/vlanext/src/evaluation/libero_bench/parallel_eval_utils.py')
    actuator=Path('VLA/common/robotiq.py')
    atomic_json(Path('VLA/audits/continuous_r86_20261021/GRIPPER_DECODER_BUG.json'),{
        'proven_adapter_semantic_mismatch':True,'actual_motor_controls':controls,
        'old_continuous_requests_in_actuator_deadband':deadband,
        'native_sign_vs_actual_latched_command_disagreements_excluding_safety_rollback':mismatches,
        'affected_episodes':affected,'per_action_disagree_steps':dict(counts),
        'safety_rollback_disagreements_excluded':unsafe,
        'official_decode':'>0 -> +1, else -1', 'shared_actuator':'close if >0.2; open if <-0.2; otherwise hold previous command',
        'fix':'native binarization in versioned R86 server before unchanged common actuator',
        'pose_weights_training_labels_physics_unchanged':True,
        'this_bug_explains_all_failures_or_predicts_SR_gain':False,
        'old_DEV_outcomes_retained_NOT_pooled_with_fixed_decoder':True,
        'source_sha256':{str(source):sha256_file(source),str(actuator):sha256_file(actuator)}})
