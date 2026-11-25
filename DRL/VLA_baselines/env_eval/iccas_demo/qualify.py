"""CPU physical qualification and immutable measured frame-state recording."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import pybullet as p
from VLA.common.artifact_io import atomic_json
from env_eval.feedback_motion import MotionProfile
from .assets import ICCASEnvironment
from .motion import RequestedRollout
from .scenes import get_scene

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'eval/ICCAS_demo'
ROBOTS={'ur5e':'UR5e_Robotiq2F','uf850':'UF850'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(robot,sid,out=OUT):
    spec=get_scene(sid);base=out/'_qualification'/ROBOTS[robot]/f'scene{sid:02d}'
    base.mkdir(parents=True,exist_ok=True)
    index=1
    while (base/f'attempt{index:02d}').exists():index+=1
    folder=base/f'attempt{index:02d}';folder.mkdir()
    sources={str(path):digest(path) for path in Path(__file__).parent.glob('*.py')}
    atomic_json(folder/'SOURCES.json',sources)
    for path in Path(__file__).parent.glob('*.py'):
        target=folder/'sources'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    profile=MotionProfile(approach_scale=.94 if robot=='ur5e' else 1.)
    started=time.monotonic();rollout=None;result={}
    with ICCASEnvironment(robot,out/'_derived_assets'/ROBOTS[robot],False) as env:
        env.joint_jerk=profile.joint_jerk;env.reset(spec.task())
        initial={name:{'base_pose':p.getBasePositionAndOrientation(body),'AABB':p.getAABB(body)}
            for name,body in env.object_ids.items()}
        atomic_json(folder/'INITIAL.json',initial)
        atomic_json(folder/'ASSETS.json',env.asset_manifest)
        try:
            rollout=RequestedRollout(env,spec,folder,profile)
            result=rollout.run_sequence()
        except Exception as e:
            result={'success':False,'overall_pass':False,'error':str(e),'traceback':traceback.format_exc(),
                'subtasks':rollout.subtasks if rollout else []}
        finally:
            if rollout:
                atomic_json(folder/'trajectory.json',rollout.samples)
                arrays={key:np.stack([frame[key] for frame in rollout.frames])
                        for key in ('base','robot_joints','gripper_joints')} if rollout.frames else {}
                np.savez_compressed(folder/'frame_states.npz',**arrays)
                atomic_json(folder/'REPLAY_SCHEMA.json',{'tracked':rollout.tracked,
                    'robot':robot,'scene':sid,'physics_hz':240,'fps':30,
                    'profile':asdict(profile),'home_tcp':rollout.home_tcp.tolist(),
                    'home_q':rollout.home_q.tolist(),'home_quat':rollout.home_quat.tolist(),
                    'subtasks':rollout.subtasks,'home_checks':rollout.home_checks})
                rollout.close()
        result.update(robot=robot,scene=sid,wall_seconds=time.monotonic()-started,
            assets_creation_only_before_execution=True)
        atomic_json(folder/'PHYSICS_AUDIT.json',result)
    if result['overall_pass']:
        selected=out/ROBOTS[robot]/f'scene{sid:02d}';selected.mkdir(parents=True,exist_ok=True)
        atomic_json(selected/'PHYSICS.json',{'qualified_directory':str(folder),'result':result,
            'scene':asdict(spec),'frame_states_sha256':digest(folder/'frame_states.npz'),
            'trajectory_sha256':digest(folder/'trajectory.json'),
            'asset_source_sha256':digest(Path(__file__).with_name('assets.py'))})
    print(json.dumps({'robot':robot,'scene':sid,'passed':result['overall_pass'],
        'error':result.get('error'),'seconds':result['wall_seconds'],'folder':str(folder)}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--robots',default='ur5e,uf850')
    parser.add_argument('--scenes',default=','.join(str(s) for s in range(1,11)))
    parser.add_argument('--output',type=Path,default=OUT);a=parser.parse_args()
    for robot in a.robots.split(','):
        for sid in map(int,a.scenes.split(',')):
            qualify(robot,sid,a.output)


if __name__=='__main__':main()
