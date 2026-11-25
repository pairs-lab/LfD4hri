"""X-VLA frozen native action replay, with actual RPC/input/physics proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import model_input
from VLA.new_vlas.action_contracts import canonical_from_absolute20
from VLA.new_vlas.policy_outputs import native_gripper_command
from VLA.new_vlas.evaluate import png
from VLA.qualitative_recorder_r51 import physical_state,canonical_hash,observation_signature
from .common import CP,COHORTS,OUT,read,make_env,reset_put
from .capture import EvidenceRecorder,visibility


def run(candidate,output=None,baseline=None,terminal_release=False):
    seed=candidate['seed'];trace=read(candidate['trajectory'])['pick_place']
    if sha256_file(candidate['trajectory'])!=candidate['trajectory_sha256']:raise ValueError('Source trace changed')
    env=make_env(seed);recorder=None;rows=[];ticks=control=0
    step=p.stepSimulation
    try:
        task,obs=reset_put(env,seed)
        initial=physical_state(env);initial_hash=canonical_hash(initial)
        if baseline and initial_hash!=baseline['initial_state_sha256']:raise ValueError('Initial replay physics differs')
        if output is not None:
            atomic_json(output/'initial_scene.json',{'task':task.to_dict(),'seed':seed,'measured_state':initial,
                'measured_state_sha256':initial_hash,'observation_signature':observation_signature(env,obs)})
            recorder=EvidenceRecorder(env,output)
            folder=output/'policy_inputs';folder.mkdir()
            Image.fromarray(obs.front_rgb).save(folder/'reset_front.png')
            Image.fromarray(obs.wrist_rgb).save(folder/'reset_wrist.png')
        post=env.gripper.post_step
        def counted(*args,**kwargs):
            nonlocal ticks
            value=step(*args,**kwargs);ticks+=1;return value
        def sampled():
            try:return post()
            finally:
                if recorder is not None:
                    recorder.record_tick(ticks,control)
                    if ticks%8==0:recorder.capture(ticks,control)
        p.stepSimulation=counted;env.gripper.post_step=sampled
        maximum=0.;queries=0
        for control,source in enumerate(trace):
            state=model_input(env,obs)
            error=float(abs(state['scene64']-np.asarray(source['pre_action_scene64'],np.float32)).max())
            maximum=max(maximum,error)
            if error>1e-6:raise ValueError(f'Original neural state diverges at{control}: {error}')
            native=np.asarray(source['native_neural_action'],np.float32)
            action=canonical_from_absolute20(native,obs.ee_position,obs.ee_rotation_6d)
            action[6]=native_gripper_command(native[9],convention='xvla_closed01')
            np.testing.assert_array_equal(action,np.asarray(source['action'],np.float32))
            if source['task_execution_prefix']!=30 or source['action_index_in_chunk']!=control%30:
                raise ValueError('Not the frozen native30 execution schedule')
            if source['action_index_in_chunk']==0:
                payload={'instruction':obs.prompt,'scene64':state['scene64'].tolist(),
                    'ee_position':obs.ee_position.tolist(),'ee_rotation_6d':obs.ee_rotation_6d.tolist(),
                    'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb)}
                digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                if digest!=source['neural_chunk_input_sha256']:raise ValueError('Original neural RGB/proprio/scene RPC input differs')
                queries+=1
            signature=observation_signature(env,obs)
            policy_visibility=visibility(env)
            obs,_,done,info=env.step(action)
            if bool(info['success'])!=bool(source['success']) or info.get('safety_violation')!=source.get('safety_violation'):
                raise ValueError('Original saved outcome changed')
            np.testing.assert_array_equal(info['executed_canonical_action'],source['executed_canonical_action'])
            row={'control':control,'before_signature':signature,'after_signature':observation_signature(env,obs),
                'after_physical_sha256':canonical_hash(physical_state(env)),'physics_ticks':ticks,
                'requested_action':action.tolist(),'executed_action':info['executed_canonical_action'],
                'benchmark_region_success':bool(info['success']),'safety_violation':info.get('safety_violation'),
                'actual_policy_visibility':policy_visibility,
                'neural_chunk_input_sha256':source['neural_chunk_input_sha256']}
            if baseline and row!=baseline['controls'][control]:raise ValueError('Recorder changed real observation/state/action/timing')
            rows.append(row)
            if done!=(control==len(trace)-1):raise ValueError('Termination differs')
        boundary_ticks=ticks
        region_success=bool(info['success'])
        held_history=sum(bool(r['after']['correct_object_held']) for r in trace)>=1
        tail=None
        if terminal_release and region_success and not info.get('safety_violation'):
            from .release import run as release
            if recorder is not None:recorder.phase='deterministic_terminal_release_NOT_VLA'
            def control_index(index):
                nonlocal control
                control=len(trace)+index
            tail=release(env,held_history,control_index)
            if baseline and tail!=baseline['terminal_release_supervisor']:
                raise ValueError('Release supervisor differs between baseline and captured replay')
        proof={'passed_exact_replay':True,'model':'X-VLA','checkpoint':str(CP),'seed':seed,'task':'pick_place',
            'source_trajectory':candidate['trajectory'],'source_trajectory_sha256':candidate['trajectory_sha256'],
            'source_cohort':candidate['cohort'],'scope':'actual_neural_action_physical_replay_NOT_new_policy_inference_or_SR',
            'initial_state_sha256':initial_hash,'controls':rows,'physics_ticks':ticks,
            'maximum_original_pre_action_scene64_error':maximum,'actual_original_RPC_inputs_verified':queries,
            'every_after_action_full_physical_state_and_action_exact':baseline is not None,
            'raw_observation_segmentation_and_prompt_bit_exact_vs_baseline':baseline is not None,
            'new_NN_processed_tensor_or_rng_parity_claimed':False,
            'benchmark_region_success':region_success,'benchmark_boundary_physics_tick':boundary_ticks,
            'safety_violation':info.get('safety_violation'),
            'safety_diagnostics_before_rollback':info.get('safety_diagnostics_before_rollback'),
            'benchmark_termination_prevents_unseen_release_continuation':bool(info['success']),
            'terminal_release_supervisor':tail,
            'native_approach_and_transport_actions_unmodified':True,
            'user_authorized_qualitative_only_release':terminal_release,
            'physics_camera_safety_or_benchmark_success_modified':False}
        return proof
    finally:
        p.stepSimulation=step
        if recorder is not None:recorder.close()
        env.close()


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    candidates=read(a.shortlist)
    candidate=next(r for r in candidates if r['seed']==a.seed)
    atomic_json(a.output/'candidate.json',candidate)
    atomic_json(a.output/'status.json',{'status':'replaying_uninstrumented_native_baseline','seed':a.seed})
    baseline=run(candidate,terminal_release=a.terminal_release)
    atomic_json(a.output/'baseline_replay.json',baseline)
    atomic_json(a.output/'status.json',{'status':'capturing_real_30FPS_ticks','seed':a.seed})
    proof=run(candidate,a.output,baseline,terminal_release=a.terminal_release)
    atomic_json(a.output/'replay_proof.json',proof)
    atomic_json(a.output/'status.json',{'status':'completed_exact_CPU_replay_NOT_media_audit','seed':a.seed,
        'benchmark_region_success':proof['benchmark_region_success']})
    print({'seed':a.seed,'exact_replay':True,'controls':len(proof['controls']),'physics_ticks':proof['physics_ticks']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--terminal-release',action='store_true')
    parser.add_argument('--shortlist',type=Path,default=OUT/'audit/initial_trace_triage/shortlist.json')
    a=parser.parse_args()
    try:main(a)
    except Exception as error:
        if a.output.exists():atomic_json(a.output/'error.json',{'error':f'{type(error).__name__}: {error}','not_complete':True})
        raise
