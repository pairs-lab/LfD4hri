"""Shared camera render, then three paper-video layouts, without changing control.

Default invocation emits all three editions; completed videos are immutable.
Raw clean views are retained so layouts can be reproduced without rerunning physics.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np,pybullet as p
from env_eval.oracle_feedback import FeedbackRollout
from env_eval.feedback_environment import FeedbackEnvironment
from env_eval.feedback_motion import MotionProfile
from env_eval.oracle_demo import scenes
from env_eval.handeye_camera import handeye_geometry
from utils.camera import CINEMATIC_VIEWS

SETTINGS=Path(__file__).with_name('camera_editions.json')
ENGINE_FILES=('env_eval/oracle_feedback.py','env_eval/feedback_environment.py','env_eval/feedback_motion.py',
    'env_eval/oracle_cinematic.py','env_eval/legacy_gripper_environment.py','env_eval/oracle_demo.py',
    'env_eval/oracle_style.py','utils/Robotiq2F85.py','utils/Robotiq2F85UF850.py','utils/camera.py')


def dump(path,value):
    path.write_text(json.dumps(value,indent=2))


def rectangles(mode,fraction,side,settings):
    width,height=settings['wide_resolution'];w=int(width*fraction)//2*2
    margin=round(width*settings['layout']['margin_fraction']);top=margin+2
    left=margin if side=='left' else width-w-margin;gap=settings['layout']['gap_pixels']
    front_h=round(w*9/16)//2*2
    if mode=='front':return [('front',left,top,w,front_h)]
    if mode=='handeye':return [('handeye',left,top,w,front_h)]
    return [('front',left,top,w,front_h),('handeye',left,top+front_h+gap,w,front_h)]


def candidates(settings):
    return {f'{mode}:{fraction}:{side}':dict(mode=mode,fraction=fraction,side=side,
            rects=rectangles(mode,fraction,side,settings))
        for mode in settings['modes'] for fraction in settings['layout'][mode+'_width_fractions'] for side in ('left','right')}


def writer(path,size,fps):
    w=imageio_ffmpeg.write_frames(str(path),size,fps=fps,codec='libx264',quality=8,macro_block_size=1,
        ffmpeg_log_level='error',output_params=['-preset','fast','-tune','zerolatency','-rc-lookahead','0',
            '-bf','0','-refs','1','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart'])
    w.send(None);return w


def mask_stats(seg,env):
    ids=seg&((1<<24)-1);mask=ids==env.pick_id;ys,xs=np.where(mask)
    links=(seg>>24)-1
    return dict(object_pixels=int(mask.sum()),object_border_margin_px=int(min(xs.min(),seg.shape[1]-1-xs.max(),ys.min(),seg.shape[0]-1-ys.max())) if len(xs) else -1,
        object_box=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
        pad_pixels=[int(((ids==env.gripper.body)&(links==i)).sum()) for i in env.gripper.PAD_LINKS],
        gripper_pixels=int((ids==env.gripper.body).sum()))


def reused_layouts(rect,settings):
    x,y,w,h=rect;gap=settings['layout']['gap_pixels']
    result={'reuse:front':dict(mode='front',fraction=w/settings['wide_resolution'][0],side='original',rects=[('front',x,y,w,h)])}
    result['reuse:handeye']=dict(mode='handeye',fraction=w/settings['wide_resolution'][0],side='original',rects=[('handeye',x,y,w,h)])
    result['reuse:stack:1.0']=dict(mode='front_handeye',fraction=w/settings['wide_resolution'][0],side='original',
            rects=[('front',x,y,w,h),('handeye',x,y+h+gap,w,h)])
    return result


def projected_box(env,body,cfg,width,height):
    """Conservative screen bounds of the complete collision/visual AABB.

    Zero intersection proves the inset cannot hide any part of the object;
    this is stricter than counting only visible segmentation pixels.
    """
    low,high=p.getAABB(body)
    corners=np.array([[x,y,z,1] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
    view=np.array(p.computeViewMatrix(cfg['eye'],cfg['target'],cfg.get('up',(0,0,1)))).reshape(4,4,order='F')
    projection=np.array(p.computeProjectionMatrixFOV(cfg['fov'],width/height,.015,4.)).reshape(4,4,order='F')
    clip=corners@(projection@view).T
    if np.any(clip[:,3]<=0):return (0,0,width,height)
    ndc=clip[:,:2]/clip[:,3,None]
    points=np.column_stack([(ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2])
    return (float(points[:,0].min()-2),float(points[:,1].min()-2),float(points[:,0].max()+2),float(points[:,1].max()+2))


def overlap(box,rect):
    _,x,y,w,h=rect;a,b,c,d=box
    return max(0,min(c,x+w+4)-max(a,x-3))*max(0,min(d,y+h+4)-max(b,y-3))


class ReusedWideRollout(FeedbackRollout):
    """Keep the completed front edition byte-identical; render only hand-eye.

    An equal-sized panel replaces the old inset in handeye mode. In stacked mode
    the old inset is retained exactly and hand-eye is placed below it. No
    content is erased or reconstructed. All added rectangles are checked
    against projected whole-object bounds on every frame.
    """
    def __init__(self,*args,settings,baseline,**kwargs):
        self.camera_writers={};self.settings=settings;self.baseline=baseline
        rows=json.loads((baseline/'visibility.json').read_text())
        self.baseline_front=[r for r in rows if r['view']=='gripper_closeup']
        self.baseline_wide={i:[r for r in rows if r['view']==i] for i in range(1,7)}
        super().__init__(*args,**kwargs)
        self.camera_writers['handeye']=writer(self.directory/'handeye.mp4',settings['handeye_resolution'],self.fps)
        for i in range(1,7):link(baseline/f'view_{i}.mp4',self.directory/f'wide_{i}.mp4')
        link(baseline/'gripper_closeup.mp4',self.directory/'front.mp4')
        dump(self.directory/'source_layout.json',dict(wide_contains_front=True,baseline=str(baseline)))

    def capture(self):
        super().capture()
        if self.initializing or not self.camera_writers:return
        e=self.env;s=self.samples[-1];index=len(self.samples)-1
        old=self.baseline_front[index]
        if abs(old['time']-s['time'])>1e-8:raise ValueError('Baseline/native camera timestamp mismatch')
        cfg=handeye_geometry(e,self.settings['handeye']);frame,seg=self.frame(cfg,*self.settings['handeye_resolution'])
        self.camera_writers['handeye'].send(frame)
        record=dict(time=s['time'],phase=self.phase,phase_progress=self.phase_progress,distance=s['tcp_object_distance'],
            handeye={**mask_stats(seg,e),'camera':cfg},front=dict(object_pixels=old['visible_pick_pixels'],pad_pixels=old['pad_pixels'],
                object_border_margin_px=old['object_border_margin_px'],camera=old['camera']),wide=[])
        for i,view in enumerate(CINEMATIC_VIEWS,1):
            old_wide=self.baseline_wide[i][index];rect=old_wide['inset_rect']
            pick_box=projected_box(e,e.pick_id,view,self.width,self.height)
            place_box=projected_box(e,e.place_id,view,self.width,self.height) if e.place_id is not None else None
            costs={}
            for key,layout in reused_layouts(rect,self.settings).items():
                area=sum(overlap(pick_box,r) for r in layout['rects'] if r[0]=='handeye')
                if place_box is not None:area+=sum(overlap(place_box,r) for r in layout['rects'] if r[0]=='handeye')
                # The unchanged original inset is already certified by the baseline.
                area+=old_wide['pick_pixels_covered_by_inset']
                costs[key]=[float(area),0]
            record['wide'].append(dict(object_pixels=old_wide['visible_pick_pixels'],candidate_costs=costs,
                baseline_inset_rect=rect,projected_pick_box=pick_box,projected_place_box=place_box))
        self.visibility.append(record)
        stage=('start' if self.phase_progress<.05 else 'middle' if .48<=self.phase_progress<=.53 else 'end' if self.phase_progress>=.98 else None)
        if stage and (self.phase,stage,'handeye') not in self.phase_frames:
            (self.directory/'frames').mkdir(exist_ok=True)
            cv2.imwrite(str(self.directory/'frames'/f'{self.phase}_{stage}_handeye.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            self.phase_frames[self.phase,stage,'handeye']=True

    def close(self):
        for w in self.camera_writers.values():w.close()
        self.camera_writers={};super().close()


class CameraSourceRollout(FeedbackRollout):
    def __init__(self,*args,settings,**kwargs):
        self.camera_writers={};self.settings=settings;self.layout_candidates=candidates(settings)
        super().__init__(*args,**kwargs)
        self.macro_size=tuple(settings['front_resolution'])
        streams=[(f'wide_{i}',settings['wide_resolution']) for i in range(1,7)]+[('front',settings['front_resolution']),('handeye',settings['handeye_resolution'])]
        for name,size in streams:self.camera_writers[name]=writer(self.directory/f'{name}.mp4',size,self.fps)

    def capture(self):
        super().capture()
        if self.initializing or not self.camera_writers:return
        e=self.env;s=self.samples[-1];record=dict(time=s['time'],phase=self.phase,phase_progress=self.phase_progress,distance=s['tcp_object_distance'])
        stage=('start' if self.phase_progress<.05 else 'middle' if .48<=self.phase_progress<=.53 else 'end' if self.phase_progress>=.98 else None)
        for name,cfg,size in [('front',self.macro_geometry(),self.settings['front_resolution']),
            ('handeye',handeye_geometry(e,self.settings['handeye']),self.settings['handeye_resolution'])]:
            frame,seg=self.frame(cfg,*size);self.camera_writers[name].send(frame)
            record[name]={**mask_stats(seg,e),'camera':cfg}
            if stage and (self.phase,stage,name) not in self.phase_frames:
                (self.directory/'frames').mkdir(exist_ok=True)
                cv2.imwrite(str(self.directory/'frames'/f'{self.phase}_{stage}_{name}.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                self.phase_frames[self.phase,stage,name]=True
        record['wide']=[]
        for i,cfg in enumerate(CINEMATIC_VIEWS,1):
            frame,seg=self.frame(cfg,*self.settings['wide_resolution']);self.camera_writers[f'wide_{i}'].send(frame)
            ids=seg&((1<<24)-1);pick=ids==e.pick_id;robot=(ids==e.robot_id)|(ids==e.gripper.body)
            costs={}
            for key,c in self.layout_candidates.items():
                object_covered=robot_covered=0
                for _,x,y,w,h in c['rects']:
                    region=(slice(y-3,y+h+4),slice(x-3,x+w+4))
                    object_covered+=int(pick[region].sum());robot_covered+=int(robot[region].sum())
                costs[key]=[object_covered,robot_covered]
            record['wide'].append(dict(object_pixels=int(pick.sum()),candidate_costs=costs))
        self.visibility.append(record)

    def close(self):
        for w in self.camera_writers.values():w.close()
        self.camera_writers={};super().close()


def replay_proof(source,baseline):
    old=json.loads((baseline/'trajectory.json').read_text());new=json.loads((source/'trajectory.json').read_text())
    errors={key:float(np.max(abs(np.array([s[key] for s in old])-np.array([s[key] for s in new])))) for key in ('time','tcp','object','joints')}
    return dict(baseline=str(baseline),errors=errors,pass_exact=max(errors.values())==0.)


def choose_layout(visibility,mode,view,settings):
    first=visibility[0]['wide'][view-1]
    all_candidates=reused_layouts(first['baseline_inset_rect'],settings) if 'baseline_inset_rect' in first else candidates(settings)
    feasible=[]
    for key,c in all_candidates.items():
        if c['mode']!=mode:continue
        costs=[v['wide'][view-1]['candidate_costs'][key] for v in visibility]
        covered=max(t[0] for t in costs)
        if covered==0:feasible.append((-c['fraction'],sum(t[1] for t in costs),key,c))
    if not feasible and 'baseline_inset_rect' in first:
        # Render a clean wide view and relocate both equal-sized panels if
        # their combined footprint reaches the object. Never inpaint the old inset.
        for key,c in candidates(settings).items():
            if c['mode']!=mode:continue
            areas=[]
            for v in visibility:
                wide=v['wide'][view-1]
                boxes=[wide['projected_pick_box']]+([wide['projected_place_box']] if wide['projected_place_box'] is not None else [])
                areas.append(sum(overlap(box,r) for box in boxes for r in c['rects']))
            if max(areas)==0:feasible.append((-c['fraction'],0,'rawfallback:'+key,{**c,'requires_clean_wide':True}))
    if not feasible:raise ValueError(f'No unoccluded stable inset layout for {mode}, view {view}')
    _,_,key,c=min(feasible)
    return {**c,'key':key,'max_object_pixels_covered':0,
        'robot_pixels_covered_sum':sum(v['wide'][view-1]['candidate_costs'].get(key,[0,0])[1] for v in visibility)}


def link(source,target):
    if target.exists():return
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def composite(source,target,view,layout,settings):
    path=target/f'view_{view}.mp4'
    if path.exists():raise FileExistsError(path)
    patched=layout.get('requires_clean_wide',False)
    reused=(source/'source_layout.json').exists() and not patched
    rects=[r for r in layout['rects'] if not reused or r[0]!='front']
    if not rects:link(source/f'wide_{view}.mp4',path);return
    wide_path=source/(f'clean_wide_{view}.mp4' if patched else f'wide_{view}.mp4')
    inputs=['-threads','1','-i',str(wide_path)];input_ids={}
    for name,*_ in rects:
        input_ids[name]=len(input_ids)+1;inputs+=['-threads','1','-i',str(source/f'{name}.mp4')]
    filters=[];previous='0:v'
    for index,(name,x,y,w,h) in enumerate(rects):
        filters.append(f'[{input_ids[name]}:v]scale={w}:{h}:flags=area,pad=iw+6:ih+6:3:3:color=0xdde3e8[p{index}]')
        filters.append(f'[{previous}][p{index}]overlay={x-3}:{y-3}:shortest=1[v{index}]');previous=f'v{index}'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-n','-threads','2',*inputs,
        '-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map',f'[{previous}]',
        '-an','-c:v','libx264','-crf','10','-preset','fast','-tune','zerolatency','-rc-lookahead','0',
        '-bf','0','-refs','1','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    subprocess.run(cmd,check=True)


def compose_scene(source,target,mode,settings):
    if (target/'camera_audit.json').exists():
        if (target/'camera_settings.json').exists() and json.loads((target/'camera_settings.json').read_text())!=settings:
            raise FileExistsError(f'Completed camera edition uses other settings; choose a new name: {target}')
        if json.loads((target/'camera_audit.json').read_text())['pass']:return
        raise RuntimeError(f'An earlier camera audit failed: {target}')
    if (target/'audit.json').exists() and not (target/'layout.json').exists():
        raise FileExistsError(f'An earlier video edition exists here and is protected: {target}')
    target.mkdir(parents=True,exist_ok=True)
    visibility=json.loads((source/'visibility.json').read_text());layouts={}
    for view in range(1,7):
        layout=choose_layout(visibility,mode,view,settings);layouts[str(view)]=layout
        if layout.get('requires_clean_wide') and not (source/f'clean_wide_{view}.mp4').exists():
            import sys
            subprocess.run([sys.executable,'-u','-m','env_eval.patch_wide_camera','--source',str(source),'--view',str(view)],check=True)
        if not (target/f'view_{view}.mp4').exists():composite(source,target,view,layout,settings)
    for name in ('trajectory.json','replay_proof.json','joint_reference.npz'):link(source/name,target/name)
    if mode in ('front','front_handeye'):link(source/'front.mp4',target/'gripper_closeup.mp4')
    if mode in ('handeye','front_handeye'):link(source/'handeye.mp4',target/'handeye.mp4')
    physics=json.loads((source/'audit.json').read_text())
    physics.update(camera_edition=mode,videos=sorted(p.name for p in target.glob('*.mp4')))
    temp_audit=target/'audit.edition.json';dump(temp_audit,physics);temp_audit.replace(target/'audit.json')
    dump(target/'layout.json',layouts)
    dump(target/'camera_settings.json',settings)
    # Auditing is a separate consumer of encoded files and recorded segmentation.
    from env_eval.audit_camera_editions import audit_scene
    result=audit_scene(source,target,mode,settings)
    dump(target/'camera_audit.json',result)
    if not result['pass']:raise RuntimeError(f'Camera audit failed: {target}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-parent',type=Path,default=Path('env_eval/demos'))
    parser.add_argument('--name',default='oracle_feedback_v3_20261015')
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--settings',type=Path,default=SETTINGS)
    parser.add_argument('--profile',type=Path,default=Path('env_eval/demos/oracle_feedback_v3_20261015_01_front_gripper/selected_profiles.json'))
    parser.add_argument('--robots',default='uf850,ur5e');parser.add_argument('--actions',default='touch,pick,move,put')
    parser.add_argument('--configs',default='1,2,3,4');parser.add_argument('--modes',default='front,front_handeye,handeye')
    parser.add_argument('--render-only',action='store_true');parser.add_argument('--compose-only',action='store_true')
    parser.add_argument('--reuse-baseline-wide',action='store_true',help='Render only the new hand-eye; preserve existing front frames and cover the entire old inset when replacing it')
    args=parser.parse_args();settings=json.loads(args.settings.read_text());profiles=json.loads(args.profile.read_text())
    for mode in args.modes.split(','):
        if mode not in settings['modes']:raise ValueError(f'Unknown camera mode: {mode}')
        target_root=args.output_parent/(args.name+settings['folder_suffixes'][mode])
        if args.baseline and target_root.resolve()==args.baseline.resolve():
            raise FileExistsError('The baseline folder is immutable; choose a fresh --name for all three editions')
    cache=args.cache or args.output_parent/(args.name+'_camera_sources');cache.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[1]
    hashes={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in ENGINE_FILES}
    metadata=dict(controller_sources=hashes,camera_settings=settings,profile=profiles,reuse_baseline_wide=args.reuse_baseline_wide)
    if (cache/'run_sources.json').exists():
        if json.loads((cache/'run_sources.json').read_text())!=metadata:raise ValueError('Camera/source settings differ from existing cache; use a new cache')
    else:dump(cache/'run_sources.json',metadata)
    selected=[s for s in scenes() if s.action in args.actions.split(',') and str(s.config) in args.configs.split(',')]
    for robot in args.robots.split(','):
        label='UF850' if robot=='uf850' else 'UR5e_Robotiq2F';profile=MotionProfile(**profiles.get(robot,profiles))
        with FeedbackEnvironment(robot,cache/('assets_'+robot),not args.compose_only) as env:
            env.joint_jerk=profile.joint_jerk
            for scene in selected:
                relative=Path(label)/scene.action/f'config{scene.config}';source=cache/relative
                if not (source/'audit.json').exists():
                    if args.compose_only:raise FileNotFoundError(source/'audit.json')
                    source.mkdir(parents=True,exist_ok=True)
                    if any(source.glob('*.mp4')):raise FileExistsError(f'Partial render exists: {source}')
                    env.reset(scene.task())
                    if args.reuse_baseline_wide:
                        if not args.baseline:raise ValueError('--reuse-baseline-wide requires --baseline')
                        rollout=ReusedWideRollout(env,scene,source,settings=settings,profile=profile,baseline=args.baseline/relative)
                    else:rollout=CameraSourceRollout(env,scene,source,settings=settings,profile=profile)
                    try:result=rollout.run()
                    finally:rollout.close()
                    dump(source/'trajectory.json',rollout.samples);dump(source/'visibility.json',rollout.visibility)
                    commands=env.physics_commands
                    np.savez_compressed(source/'joint_reference.npz',q=np.array([s['q'] for s in commands]),v=np.array([s['v'] for s in commands]),a=np.array([s['a'] for s in commands]),dt=1/env.physics_hz)
                    proof=replay_proof(source,args.baseline/relative) if args.baseline else {'pass_exact':None,'baseline':None,'comparison_requested':False}
                    dump(source/'replay_proof.json',proof)
                    if not result['overall_pass'] or (args.baseline and not proof['pass_exact']):raise RuntimeError(f'Physics/replay failed: {source}')
                    dump(source/'audit.json',result)
                    print('RENDERED',relative,'exact replay',proof['pass_exact'],flush=True)
                if args.render_only:continue
                for mode in args.modes.split(','):
                    target=args.output_parent/(args.name+settings['folder_suffixes'][mode])/relative
                    compose_scene(source,target,mode,settings)
                    print('COMPOSED + AUDITED',mode,relative,flush=True)
    if not args.render_only:
        from env_eval.camera_gallery import finalize_galleries
        finalize_galleries(args.output_parent,args.name,args.modes.split(','),settings)


if __name__=='__main__':main()
