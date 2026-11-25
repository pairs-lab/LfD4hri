"""Read-only camera pilot using the unchanged, qualified feedback controller."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pybullet as p
from VLA.common.environment import CAMERA_CONFIG
from env_eval.oracle_feedback import FeedbackRollout
from env_eval.feedback_environment import FeedbackEnvironment
from env_eval.feedback_motion import MotionProfile
from env_eval.oracle_demo import scenes
from env_eval.handeye_camera import handeye_geometry


class Probe(FeedbackRollout):
    def capture(self):
        super().capture()
        if self.initializing:return
        e=self.env;s=self.samples[-1]
        cfg=handeye_geometry(e,json.loads(Path('env_eval/camera_editions.json').read_text())['handeye'])
        image,seg=self.frame(cfg,900,900)
        ids=seg&((1<<24)-1);links=(seg>>24)-1
        self.visibility.append(dict(time=s['time'],phase=self.phase,progress=self.phase_progress,
            object_pixels=int((ids==e.pick_id).sum()),pad_pixels=[int(((ids==e.gripper.body)&(links==i)).sum()) for i in e.gripper.PAD_LINKS],
            camera=cfg,distance=s['tcp_object_distance']))
        stage=('start' if self.phase_progress<.05 else 'middle' if .48<=self.phase_progress<=.53 else 'end' if self.phase_progress>=.98 else None)
        key=(self.phase,stage)
        if stage and key not in self.phase_frames:
            self.phase_frames[key]=True
            cv2.imwrite(str(self.directory/f'{self.phase}_{stage}_handeye.jpg'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            inv=p.invertTransform(*p.getLinkState(e.robot_id,e.robot_spec.mount_link,computeForwardKinematics=True)[:2])
            local=p.multiplyTransforms(*inv,e.gripper.grasp_center(),(0,0,0,1))[0]
            if self.phase=='carry' and stage=='middle':print('local_tcp',local,flush=True)
            for name,eye,target,fov in [('center',( .07,.05,.04),(0,0,.20),80),
                ('outside',(.10,.08,.02),(0,0,.19),75),('side',(.12,0,.025),(0,0,.19),75),
                ('high',(.075,.06,-.015),(0,0,.20),75),
                ('jawA',(0,.12,.025),(0,0,.17),80),('jawB',(0,-.12,.025),(0,0,.17),80),
                ('diag',(.06,-.10,.01),(0,0,.17),80),('far',(.08,.12,-.08),(0,0,.17),75)]:
                setting=dict(eye_in_tool=eye,target_in_tool=target,up_axis_in_tool=(0,1,0),fov_degrees=fov)
                candidate,_=self.frame(handeye_geometry(e,setting),960,540)
                cv2.imwrite(str(self.directory/f'{self.phase}_{stage}_{name}.jpg'),cv2.cvtColor(candidate,cv2.COLOR_RGB2BGR))


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--robot',required=True)
    a.add_argument('--action',default='put');a.add_argument('--config',type=int,default=3);args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(Path('env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper/selected_profiles.json').read_text())
    with FeedbackEnvironment(args.robot,args.output/'assets',True) as e:
        profile=MotionProfile(**data[args.robot]);e.joint_jerk=profile.joint_jerk
        scene=next(s for s in scenes() if s.action==args.action and s.config==args.config)
        e.reset(scene.task());r=Probe(e,scene,args.output,profile=profile);result=r.run()
        (args.output/'trajectory.json').write_text(json.dumps(r.samples))
        (args.output/'visibility.json').write_text(json.dumps(r.visibility))
        (args.output/'audit.json').write_text(json.dumps(result,indent=2));r.close()
        interaction=[v for v in r.visibility if v['phase'] in ('grasp','carry','release','touch','touch_retract') or v['distance']<.04]
        print(json.dumps(dict(robot=args.robot,success=result['overall_pass'],frames=len(r.samples),
            minimum_object_pixels=min(v['object_pixels'] for v in interaction),minimum_pad_pixels=[min(v['pad_pixels'][i] for v in interaction) for i in (0,1)])),flush=True)


if __name__=='__main__':main()
