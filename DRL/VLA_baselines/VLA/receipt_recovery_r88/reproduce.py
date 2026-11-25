"""Replay logged commands to reconstruct missing request, NOT a new policy trial."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.scene64_conditioning import model_input
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.new_vlas.evaluate import png
from VLA.privileged_visual_r53.scene_map import render

ROOT=Path('/work/LfD4hri/DRL')
OLD=ROOT/'VLA/audits/dita800_user_r87_20261021/final800'
OUT=ROOT/'VLA/audits/receipt_recovery_r88_20261021'


if __name__=='__main__':
    seed=20284061
    progress=json.loads((OLD/'progress.json').read_text())
    if progress['completed_trials']!=243:raise ValueError('This diagnostic is pinned to the actual failed prefix')
    protocol=json.loads(Path(PROFILE_PROTOCOL).read_text())
    profile=json.loads(Path(protocol['observation_profile_file']).read_text())
    rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
    refs=json.loads((OLD/f'seed{seed}/trajectory.json').read_text())
    queries=defaultdict(list)
    with (OLD/'NEURAL_QUERIES.jsonl').open() as stream:
        for line in stream:
            q=json.loads(line)
            if q['seed']==seed:queries[q['task']].append(q)
    env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
        control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
        success_threshold_scale=protocol['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=protocol['success']['semantic_thresholds'])
    matched=0;max_error=0.
    try:
        for task in tasks:
            obs=env.reset(task);discard_snapshot(OUT,task.name,obs,env)
            for q in queries[task.name]:
                raw=model_input(env,obs)['scene64']
                payload={'instruction':obs.prompt,'scene64':raw.tolist(),'ee_position':obs.ee_position.tolist(),
                    'ee_rotation_6d':obs.ee_rotation_6d.tolist(),'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                    'seed':seed,'inference_index':q['index']}
                digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                if digest!=q['input_sha256']:
                    raise ValueError(f'Replay request does not match actual log: {task.name}/{q["index"]}')
                matched+=1
                obs,_,done,info=env.step(np.asarray(q['actions'][0],np.float32))
                if task.name in refs:
                    ref=refs[task.name][q['index']]['post_action_scene64']
                    error=float(abs(model_input(env,obs)['scene64']-np.asarray(ref,np.float32)).max())
                    max_error=max(max_error,error)
                    if error>1e-7:raise ValueError('Known completed task state diverged in replay')
            if task.name=='pick_place':
                raw=model_input(env,obs)['scene64'];index=len(queries[task.name])
                payload={'instruction':obs.prompt,'scene64':raw.tolist(),'ee_position':obs.ee_position.tolist(),
                    'ee_rotation_6d':obs.ee_rotation_6d.tolist(),'front_png':png(obs.front_rgb),'wrist_png':png(obs.wrist_rgb),
                    'seed':seed,'inference_index':index}
                image=render(raw,task.name)
                atomic_json(OUT/'REPRODUCED_REQUEST.json',payload)
                Image.fromarray(image).save(OUT/'SIM_map_at_failure.png')
                atomic_json(OUT/'REPLAY_PROOF.json',{'passed':True,'matched_actual_request_hashes':matched,
                    'known_completed_tasks_max_state_error':max_error,'failure_query_index':index,
                    'raw64':raw.tolist(),'sim_numpy':np.__version__,
                    'SIM_map_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                    'command_replay_diagnostic_NOT_new_VLA_trial_or_success':True})
    finally:env.close()
