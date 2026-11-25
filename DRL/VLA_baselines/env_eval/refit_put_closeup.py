"""Replay identical physics; refit the macro and replace only the old inset.

The six encoded wide sources remain preserved. A state-by-state replay check
prevents combining a macro with a different trajectory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from env_eval.legacy_gripper_environment import LegacyGripperEnvironment
from env_eval.oracle_cinematic import CinematicRollout
from env_eval.oracle_demo import scenes
from utils.camera import CINEMATIC_VIEWS


class RefitRollout(CinematicRollout):
    def __init__(self,env,scene,output,source):
        self.source=source
        self.original_samples=json.loads((source/'trajectory.json').read_text())
        original_visibility=json.loads((source/'visibility.json').read_text())
        self.original_views={i:[v for v in original_visibility if v['view']==i] for i in range(1,7)}
        self.readers={}
        for i in range(1,7):
            self.readers[i]=cv2.VideoCapture(str(source/f'view_{i}.mp4'))
        self.position_error=self.joint_error=0.
        super().__init__(env,scene,output,True,1920,1080,30)

    def frame(self,cfg,width,height):
        if (width,height)==self.macro_size:return super().frame(cfg,width,height)
        i=next(i for i,c in enumerate(CINEMATIC_VIEWS,1) if c['name']==cfg['name'])
        ok,frame=self.readers[i].read()
        if not ok:raise RuntimeError(f'Cannot decode original wide view {i}')
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),np.full((height,width),-1,dtype=np.int64)

    def inset(self,wide,macro,seg,view_index):
        i=view_index+1;index=len(self.samples)-1
        x,y,w,h=self.original_views[i][index]['inset_rect']
        cv2.rectangle(wide,(x-3,y-3),(x+w+3,y+h+3),(28,32,38),-1)
        cv2.rectangle(wide,(x-2,y-2),(x+w+2,y+h+2),(226,231,236),2)
        wide[y:y+h,x:x+w]=cv2.resize(macro,(w,h),interpolation=cv2.INTER_AREA)
        return wide,(x,y,w,h)

    def capture(self):
        if getattr(self,'initializing',True):return
        super().capture()
        index=len(self.samples)-1
        if index>=len(self.original_samples):raise RuntimeError('Replay has extra frames')
        actual=self.samples[index];original=self.original_samples[index]
        if actual['phase']!=original['phase'] or abs(actual['time']-original['time'])>1e-10:
            raise RuntimeError(f'Replay timing differs at frame {index}')
        for field in ('tcp','object'):
            self.position_error=max(self.position_error,float(np.max(abs(np.array(actual[field])-original[field]))))
        self.joint_error=max(self.joint_error,float(np.max(abs(np.array(actual['joints'])-original['joints']))))
        self.joint_error=max(self.joint_error,float(np.max(abs(np.array(actual['finger_joints'])-original['finger_joints']))))
        if self.position_error>1e-5 or self.joint_error>1e-5:
            raise RuntimeError(f'Physics differs from wide video: {self.position_error} m, {self.joint_error} rad')
        for offset,i in enumerate(range(1,7)):
            self.visibility[-6+offset]=dict(self.original_views[i][index])

    def close(self):
        super().close()
        for reader in self.readers.values():reader.release()


def main():
    base=Path(__file__).resolve().parent/'demos'
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=base/'v2_refit_put_render_jobs_r3')
    parser.add_argument('--robots',default='uf850,ur5e')
    parser.add_argument('--configs',default='1,2,3,4')
    args=parser.parse_args();output=args.output
    sources={'uf850':base/'v2_seated_uf_render_jobs/uf_moveput/UF850/put',
             'ur5e':base/'v2_ur_original_delivery_jobs/ur_moveput/UR5e_Robotiq2F/put'}
    completed=set()
    robots=args.robots.split(',');configs={int(c) for c in args.configs.split(',')}
    expected=len(robots)*len(configs)
    for _ in range(300):
        for robot,root in sources.items():
            if robot not in robots:continue
            for scene in [s for s in scenes() if s.action=='put']:
                if scene.config not in configs:continue
                key=(robot,scene.config);source=root/f'config{scene.config}'
                target=output/('UF850' if robot=='uf850' else 'UR5e_Robotiq2F')/'put'/f'config{scene.config}'
                if key in completed:continue
                if (target/'audit.json').exists():completed.add(key);continue
                if not (source/'audit.json').exists():continue
                result=json.loads((source/'audit.json').read_text())
                if not result['overall_pass']:raise RuntimeError(f'Source task failed: {source}')
                target.mkdir(parents=True,exist_ok=False)
                rollout=None
                with LegacyGripperEnvironment(robot,output/'assets'/f'{robot}_{scene.config}',True) as env:
                    try:
                        env.reset(scene.task());rollout=RefitRollout(env,scene,target,source)
                        result=rollout.run()
                        if len(rollout.samples)!=len(rollout.original_samples):raise RuntimeError('Replay frame count differs')
                        result['macro_refit']={'state_replay_pass':True,'max_position_error_m':rollout.position_error,
                            'max_joint_error_rad':rollout.joint_error,'wide_source':str(source),
                            'wide_sha256':{f'view_{i}.mp4':hashlib.sha256((source/f'view_{i}.mp4').read_bytes()).hexdigest() for i in range(1,7)}}
                    finally:
                        if rollout:
                            rollout.close()
                            (target/'trajectory.json').write_text(json.dumps(rollout.samples))
                            (target/'visibility.json').write_text(json.dumps(rollout.visibility))
                (target/'audit.json').write_text(json.dumps(result,indent=2));completed.add(key)
                print(robot,scene.config,result['overall_pass'],result['macro_refit'],flush=True)
        if len(completed)==expected:break
        time.sleep(5)
    else:raise SystemExit('Missing source put videos')


if __name__=='__main__':main()
