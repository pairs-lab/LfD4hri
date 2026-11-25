"""Versioned RGB-D admission with measured, bounded render quantization."""
from pathlib import Path
import os
import numpy as np
from PIL import Image,ImageDraw

from .common import ROOT,OUT,read,atomic_json,utc
from VLA.common.provenance import sha256_file
from VLA.recovery_quality_r90.data import Data
from VLA.recovery_quality_r90.audit import OUT as R90
from VLA.falcon_esm_r91.data import DEPTH
from VLA.falcon_esm_r91.depth import capture
from VLA.dataset_replay_context import audit_environment,reset_source_episode
from VLA.simstate_scene64 import scene_state

DEST=OUT/'falcon_esm'


def rgb_parity(actual,expected):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=(384,384,3) or b.shape!=a.shape or a.dtype!=np.uint8 or b.dtype!=np.uint8:
        raise ValueError('RGB schema mismatch; cannot call this quantization')
    delta=abs(a.astype(np.int16)-b.astype(np.int16));count=int(np.count_nonzero(delta))
    maximum=int(delta.max());fraction=count/delta.size
    if maximum>1 or fraction>1e-4:
        raise ValueError(f'RGB replay differs beyond declared quantization: max={maximum}, fraction={fraction}')
    return {'different_channel_values':count,'maximum_byte_error':maximum,'fraction':fraction,
        'bit_exact':count==0,'bounded_one_byte_quantization_only':True}


def main():
    os.nice(10);DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'DEPTH_ADMISSION.json').exists():
        proof=read(DEST/'DEPTH_ADMISSION.json')
        if sha256_file(DEST/'DEPTH_INDEX.json')!=proof['depth_index_sha256']:
            raise ValueError('Completed R100 depth index changed')
        print({'already_completed':True,**proof},flush=True);return
    data=Data();original=read(DEPTH/'AUDIT.json')
    if not original['passed_pairing_replay'] or original['episodes']!=2185 or original['frames']!=119495:
        raise ValueError('Original full TRAIN pairing proof missing')
    rows=[];old={row['qualified_row']:row for row in original['rows']}
    for ei in range(2185):
        entry,source=old[ei],data.rows[ei]
        if entry['source_sha256']!=source['source_sha256'] or entry['frames']!=source['frames'] or not Path(entry['file']).is_file():
            raise ValueError('Original RGB-D index/source/file coverage differs')
        rows.append({'file':entry['file'],'sha256':entry['sha256'],'frames':entry['frames'],
            'source_sha256':source['source_sha256'],'actual_pre_action_pair_replay_passed':True,
            'pairing_proof_reused_from_original_audit':True,
            'payload_hash_must_be_checked_at_every_actual_training_read':True})
    # Original immutable pairing receipts reused; no redundant enormous bank rewrite.
    atomic_json(DEST/'ORIGINAL_DEPTH_REUSE.json',{'UTC':utc(),'episodes':2185,'frames':119495,
        'source':str(DEPTH/'AUDIT.json'),'source_sha256':sha256_file(DEPTH/'AUDIT.json'),
        'all_original_payloads_rehashed_in_this_R100_scan':False,
        'actual_training_loader_must_verify_every_read_payload_hash':True})
    candidate=R90/'near_miss_sweep';plan=read(candidate/'plan.json');start_source=Path(plan['source'])
    originals={r['file']:r for r in read(start_source/'index.json')}
    corrections={str(candidate/r['file']):r for r in read(candidate/'index.json') if r.get('candidate_replay_pass')}
    folder=DEST/'recovery_depth';folder.mkdir(exist_ok=True);quantization=[];pilots={}
    for ei in range(2185,len(data.rows)):
        source=data.rows[ei];dest=folder/f'row{ei:04d}.npz';receipt=folder/f'row{ei:04d}.json'
        if receipt.exists():
            record=read(receipt)
            if (record['source_sha256']!=source['source_sha256'] or not dest.exists()
                or sha256_file(dest)!=record['sha256']):raise ValueError('Committed recovery depth changed')
            rows.append(record);quantization.extend(record['render_quantization']);continue
        if dest.exists():raise FileExistsError('Uncommitted depth retained for inspection: '+str(dest))
        meta=corrections[source['source']]
        if sha256_file(source['source'])!=source['source_sha256']:raise ValueError('Admitted recovery RGB source changed')
        env=audit_environment(start_source,robot='ur5e',image_size=384,control_hz=10,
            seed=0,required_success_steps=10000,use_semantic_success=False)
        depths=[];views=[];projections=[];evidence=[]
        try:
            obs=reset_source_episode(env,start_source,originals[meta['source_file']])
            for entry in meta['perturbation_actions_NOT_labels']:
                obs,_,_,info=env.step(np.asarray(entry['action'],np.float32))
                if info.get('safety_violation'):raise ValueError('Recovery intervention replay unsafe')
            env.step_count=env.success_streak=0;env._last_success=False;obs=env.observe()
            with np.load(source['source'],allow_pickle=False) as values:
                front=values['front_rgb'];wrist=values['wrist_rgb']
                raw=values['audit_pre_action_PRIVILEGED_state64'];actions=values['canonical_actions']
            for fi,action in enumerate(actions):
                np.testing.assert_allclose(scene_state(env,obs),raw[fi],rtol=0,atol=1e-6)
                for camera,current,reference in (('front',obs.front_rgb,front[fi]),('wrist',obs.wrist_rgb,wrist[fi])):
                    stats=rgb_parity(current,reference)
                    if not stats['bit_exact']:evidence.append({'row':ei,'frame':fi,'camera':camera,**stats})
                depth,view,projection,K,cv=capture(env,obs)
                depths.append(depth);views.append(view);projections.append(projection)
                if source['task'] not in pilots and fi==len(actions)//2:
                    display=np.clip((depth-.1)/1.4,0,1)
                    panel=Image.new('RGB',(768,256),'white')
                    for col,im in enumerate((front[fi],wrist[fi],np.repeat((display*255).astype(np.uint8)[...,None],3,axis=2))):
                        panel.paste(Image.fromarray(im).resize((256,256)),(col*256,0))
                    panel.save(DEST/(source['task']+'_RECOVERY_RGBD.png'))
                    pilots[source['task']]={'row':ei,'frame':fi,'depth_range_m':[float(depth.min()),float(depth.max())],
                        'K':K.tolist(),'world_to_OpenCV':cv.tolist()}
                obs,_,_,info=env.step(action)
                if info.get('safety_violation'):raise ValueError('Original admitted recovery action became unsafe')
        finally:env.close()
        tmp=dest.with_name(dest.stem+'.partial.npz')
        np.savez_compressed(tmp,front_depth_z_m=np.stack(depths),view_world_to_OpenGL_camera=np.asarray(views),
            projection_OpenGL=np.asarray(projections),source_frame_indices=np.arange(len(depths)))
        os.replace(tmp,dest)
        record={'file':str(dest),'sha256':sha256_file(dest),'frames':len(depths),'source_sha256':source['source_sha256'],
            'actual_pre_action_pair_replay_passed':True,'render_quantization':evidence}
        atomic_json(receipt,record);rows.append(record);quantization.extend(evidence)
        atomic_json(DEST/'DEPTH_STATUS.json',{'UTC':utc(),'status':'replaying_actual_TRAIN_recovery_depth',
            'rows':len(rows),'planned_rows':len(data.rows),'new_recovery_frames':sum(r['frames'] for r in rows[2185:]),
            'bounded_RGB_quantization_cases':len(quantization),'policy_or_neural_SR':False})
        print({'row':ei,'frames':len(depths),'one_byte_cases':len(evidence)},flush=True)
    atomic_json(DEST/'DEPTH_INDEX.json',rows)
    atomic_json(DEST/'DEPTH_ADMISSION.json',{'passed':True,'UTC':utc(),'split':'train','episodes':len(rows),
        'frames':sum(r['frames'] for r in rows),'depth_index_sha256':sha256_file(DEST/'DEPTH_INDEX.json'),
        'original_pair_proof_reused_not_every_old_payload_rehashed':True,
        'new_recovery_RGB_max_byte_difference':1,'new_recovery_RGB_max_channel_fraction':1e-4,
        'state64_replay_absolute_tolerance':1e-6,'quantization':quantization,'pilots':pilots,
        'no_new_expert_actions_or_evaluation_labels':True,'native_GPU_admission_still_required':True})


if __name__=='__main__':main()
