"""Inspect every requested scene's actual EGL objects before finishing HD queue."""
from pathlib import Path
from types import SimpleNamespace
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from VLA.common.artifact_io import atomic_json
from env_eval.oracle_cinematic import CinematicRollout
from utils.camera import CINEMATIC_VIEWS
from .assets import ICCASEnvironment
from .scenes import get_scene
from .render import apply_frame,VIEWS
from .qualify import OUT,ROBOTS


def main():
    result=[];thumbs=[];folder=OUT/'_asset_audit';folder.mkdir(exist_ok=True)
    for robot,label in ROBOTS.items():
        for sid in range(1,11):
            selected=json.loads((OUT/label/f'scene{sid:02d}/PHYSICS.json').read_text())
            source=Path(selected['qualified_directory']);schema=json.loads((source/'REPLAY_SCHEMA.json').read_text())
            states=np.load(source/'frame_states.npz')
            with ICCASEnvironment(robot,OUT/'_derived_assets'/label,True) as env:
                env.reset(get_scene(sid).task());apply_frame(env,schema,states,0)
                camera=SimpleNamespace(renderer=env.demo_renderer);views={};color_samples={}
                for view in VIEWS:
                    rgb,seg=CinematicRollout.frame(camera,CINEMATIC_VIEWS[view-1],1920,1080)
                    ids=seg&((1<<24)-1)
                    views[str(view)]={name:int((ids==b).sum()) for name,b in env.object_ids.items()}
                    for name in ('green chalk_brush','blue cup','brown bottle','red pen'):
                        if name in env.object_ids:
                            pixels=rgb[ids==env.object_ids[name]]
                            if len(pixels):color_samples.setdefault(name,[]).append(pixels.mean(0))
                    if view==3:
                        path=folder/f'{label}_scene{sid:02d}_initial_view3.jpg'
                        cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        picture=Image.fromarray(rgb);picture.thumbnail((640,360));thumbs.append((f'{label} / scene {sid:02d}',picture))
                totals={n:sum(v[n] for v in views.values()) for n in env.object_ids}
                colors={n:np.mean(v,axis=0).tolist() for n,v in color_samples.items()}
                def correct_color(name,rgb):
                    r,g,b=rgb
                    if name=='green chalk_brush':return g>max(r,b)*1.2 and g>30
                    if name=='blue cup':return b>r*1.2 and b>g*1.05
                    if name=='brown bottle':return r>g*1.1 and g>b*1.1
                    if name=='red pen':return r>max(g,b)*1.4
                    return False
                palette_ok=all(correct_color(n,c) for n,c in colors.items())
                row={'robot':robot,'scene':sid,'per_view':views,'per_object_total_pixels':totals,
                    'procedural_mean_RGB':colors,'requested_procedural_colors_passed':palette_ok,
                    'all_requested_objects_actually_rendered':all(v>=20 for v in totals.values())}
                result.append(row)
                atomic_json(OUT/'INITIAL_SCENE_COVERAGE.json',{'checked_cases':len(result),'expected_cases':20,'cases':result,
                    'all_checked_passed':all(r['all_requested_objects_actually_rendered'] and r['requested_procedural_colors_passed'] for r in result)})
                if not row['all_requested_objects_actually_rendered']:raise RuntimeError('An initial requested object is invisible: '+str(row))
                if not palette_ok:raise RuntimeError('A requested procedural color is incorrect: '+str(colors))
                print({'robot':robot,'scene':sid,'all_objects_visible':True,'object_count':len(totals)},flush=True)
    sheet=Image.new('RGB',(4*640,5*385),'white');draw=ImageDraw.Draw(sheet)
    for i,(label,picture) in enumerate(thumbs):
        x=(i%4)*640;y=(i//4)*385;sheet.paste(picture,(x,y+22));draw.text((x+6,y+3),label,fill='black')
    sheet.save(folder/'ALL20_INITIAL_SCENES.jpg',quality=93)


if __name__=='__main__':main()
