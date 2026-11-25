"""Render one clean wide camera when the previous inset cannot be fully reused."""
import argparse,json
from pathlib import Path
from env_eval.camera_editions import writer,dump,ENGINE_FILES
from env_eval.oracle_feedback import FeedbackRollout
from env_eval.feedback_environment import FeedbackEnvironment
from env_eval.feedback_motion import MotionProfile
from env_eval.oracle_demo import scenes
from utils.camera import CINEMATIC_VIEWS


class PatchRollout(FeedbackRollout):
    def __init__(self,*args,view,**kwargs):
        self.patch_writer=None;self.patch_view=view
        super().__init__(*args,**kwargs)
        path=self.directory/f'clean_wide_{view}.mp4'
        if path.exists():raise FileExistsError(path)
        self.patch_writer=writer(path,(self.width,self.height),self.fps)

    def capture(self):
        super().capture()
        if self.initializing or self.patch_writer is None:return
        frame,_=self.frame(CINEMATIC_VIEWS[self.patch_view-1],self.width,self.height)
        self.patch_writer.send(frame)

    def close(self):
        if self.patch_writer:self.patch_writer.close();self.patch_writer=None
        super().close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--view',type=int,required=True);a=p.parse_args()
    label,action,config=a.source.parts[-3:];robot='uf850' if label=='UF850' else 'ur5e'
    scene=next(s for s in scenes() if s.action==action and s.config==int(config.removeprefix('config')))
    metadata=json.loads((a.source.parents[2]/'run_sources.json').read_text());profile=MotionProfile(**metadata['profile'][robot])
    with FeedbackEnvironment(robot,a.source/'patch_assets',True) as env:
        env.joint_jerk=profile.joint_jerk;env.reset(scene.task());r=PatchRollout(env,scene,a.source,view=a.view,profile=profile)
        try:result=r.run()
        finally:r.close()
        old=json.loads((a.source/'trajectory.json').read_text())
        proof=dict(pass_exact=bool(old==r.samples and result['overall_pass']),frame_count=len(r.samples),
                   fields_compared=list(old[0]),method='Exact equality of every saved telemetry field on every frame')
        dump(a.source/f'clean_wide_{a.view}_proof.json',proof)
        if not proof['pass_exact']:raise RuntimeError('Clean-wide replay diverged')
        print('CLEAN WIDE EXACT',a.source,a.view,flush=True)


if __name__=='__main__':main()
