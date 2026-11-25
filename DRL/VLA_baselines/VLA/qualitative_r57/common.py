import json
from pathlib import Path
import numpy as np

from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.run_cross_vla120 import PROFILE_PROTOCOL

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VLA/qualitative/ur5e_xvla_put_r57_20261017'
R55=ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1'
CP=R55/'xvla_round1/fit1500/checkpoints/step-014500'
COHORTS=[R55/'xvla_new_r1_development40',R55/'xvla_new_r1_confirmation40',R55/'xvla_final120']


def read(path):return json.loads(Path(path).read_text())


def tasks(seed):
    rng=np.random.default_rng(int(seed))
    return [randomize_rl_scene(t,rng) for t in select_tasks('all')]


def protocol():
    from VLA.common.observation_profile import profile_hash
    value=read(PROFILE_PROTOCOL)
    profile=read(value['observation_profile_file'])
    if profile_hash(profile)!=value['observation_profile_sha256']:raise ValueError('Frozen physical/camera profile changed')
    from VLA.privileged_visual_r53.contract import require_visual
    cfg=require_visual(CP)
    if cfg['family']!='xvla' or cfg['actual_optimizer_updates']!=14500:raise ValueError('Wrong frozen VLA')
    saved=read(CP/'saved_values.json')
    if sha256_file(CP/'config.json')!=saved['config_sha256']:raise ValueError('Checkpoint metadata changed')
    return value,profile


def make_env(seed):
    from VLA.common.environment_factory import environment_class_for_profile
    value,profile=protocol()
    return environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
        control_hz=10,assist_grasp=True,seed=int(seed),required_success_steps=1,use_semantic_success=True,
        success_threshold_scale=value['success']['reward_xml_threshold_scale'],
        semantic_success_thresholds=value['success']['semantic_thresholds'])


def reset_put(env,seed):
    for task in tasks(seed):
        obs=env.reset(task)
        if task.name=='pick_place':return task,obs
    raise ValueError('Missing put task')
