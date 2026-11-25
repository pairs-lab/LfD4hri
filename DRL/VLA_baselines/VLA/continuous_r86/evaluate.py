"""Unchanged R85 neural evaluator + read-only actual-observation timelines.

No extra render/physics/policy calls. Stored images are COPIES of the RGB already
returned by env.step/reset. Final800 intentionally has no image recorder.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from VLA.common.artifact_io import atomic_json
from VLA.recovery_r85.common import read


def observed_class(base, output, enabled, raw_fn):
    class ObservedEnvironment(base):
        def __init__(self,*args,**kwargs):
            self._review_ready=False
            super().__init__(*args,**kwargs)
            self._review_enabled=bool(enabled and kwargs.get('seed') in enabled)
            self._review_root=Path(output)/f"seed{kwargs.get('seed')}"
            self._review_ready=True

        def reset(self,*args,**kwargs):
            obs=super().reset(*args,**kwargs)
            if not self._review_ready or not self._review_enabled:return obs
            task=args[0] if args else kwargs.get('task')
            self._review_task=task.name
            self._review_frames=[];self._review_closed=False;self._review_hold=False
            self._review_contact=False;self._review_closest=float('inf')
            self._review_capture(obs,['reset'])
            return obs

        def _review_capture(self,obs,events):
            from VLA.privileged_visual_r53.scene_map import render
            raw=raw_fn(self,obs)
            self._review_closest=min(self._review_closest,float(np.linalg.norm(raw[11:14])))
            images=[np.asarray(obs.front_rgb).copy(),np.asarray(obs.wrist_rgb).copy(),render(raw,self._review_task)]
            self._review_frames.append({'step':int(obs.step),'events':events,'scene64':raw.tolist(),
                'images':images,'RGB_sha256':[hashlib.sha256(v.tobytes()).hexdigest() for v in images]})

        def step(self,action):
            result=super().step(action)
            if not self._review_ready or not self._review_enabled:return result
            obs,reward,done,info=result
            raw=raw_fn(self,obs)
            contact=bool((raw[28:30]>.5).any());held=bool(raw[17]>.5);closed=bool(action[6]>0)
            events=[]
            if int(obs.step)%20==0:events.append('periodic')
            if closed and not self._review_closed:events.append('close_requested')
            if contact and not self._review_contact:events.append('target_contact')
            if held and not self._review_hold:events.append('correct_hold')
            if self._review_hold and not held:events.append('verified_hold_lost')
            if info.get('safety_violation'):events.append('safety_abort')
            if info.get('success'):events.append('actual_success')
            if done:events.append('terminal')
            if events:self._review_capture(obs,events)
            self._review_closed=closed;self._review_contact=contact;self._review_hold=held
            if done:self._review_save()
            return result

        def _review_save(self):
            rows=self._review_frames
            canvas=Image.new('RGB',(768,280*len(rows)),'white');draw=ImageDraw.Draw(canvas)
            for ri,row in enumerate(rows):
                draw.text((3,ri*280+3),f"t{row['step']}: {','.join(row['events'])}",fill='black')
                for col,im in enumerate(row['images']):
                    canvas.paste(Image.fromarray(im).resize((256,256)),(256*col,280*ri+24))
            self._review_root.mkdir(parents=True,exist_ok=True)
            canvas.save(self._review_root/(self._review_task+'_INFERENCE_TIMELINE.png'))
            atomic_json(self._review_root/(self._review_task+'_INFERENCE_TIMELINE.json'),{
                'read_only_copies_of_actual_policy_observations':True,
                'extra_policy_or_physics_or_Bullet_render_calls':0,
                'display_resize_is_not_a_change_to_model_preprocessing':True,
                'actual_observations':[{k:v for k,v in r.items() if k!='images'} for r in rows]})
    return ObservedEnvironment


def main(a):
    import VLA.autonomy_r83.evaluate as old
    from VLA.recovery_r85.evaluate import main as safe_main
    cohort=read(a.cohort)
    enabled=set(cohort['seeds'][:2]) if cohort['role']=='development_selection' else set()
    factory=old.environment_class_for_profile
    def observed_factory(profile):
        return observed_class(factory(profile),a.output,enabled,
            lambda env,obs:old.model_input(env,obs)['scene64'])
    old.environment_class_for_profile=observed_factory
    try:safe_main(a)
    finally:old.environment_class_for_profile=factory


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('method','cohort','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
