"""OFFLINE4-task physical reset audit of the diagram's metric frame/goals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import pybullet as p

from VLA.common.artifact_io import atomic_json
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment,reset_source_episode
from VLA.simstate_scene64 import scene_state
from .data import MapData
from .scene_map import render,geometry,contract


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    data=MapData(args.data,verify_rgb_hashes=False)
    rows=[];sheet=Image.new('RGB',(1152,4*410),'#eeeeee');drawing=ImageDraw.Draw(sheet)
    for j,task in enumerate(('touch','pick','move','pick_place')):
        ei=next(i for i,r in enumerate(data.rows) if r['task']==task and r['source_kind']=='complete_original')
        item=data.rows[ei];source=Path(item['source']).parent
        original=next(r for r in json.loads((source/'index.json').read_text()) if r['file']==Path(item['source']).name)
        env=audit_environment(source,robot='ur5e',image_size=384,control_hz=10,seed=0,required_success_steps=1)
        try:
            obs=reset_source_episode(env,source,original)
            raw=scene_state(env,obs);before=raw.copy();count=env.step_count
            image=render(raw,task)
            _,pad,target,goal,_=geometry(raw,task)
            measured=env.debug_state()
            np.testing.assert_allclose(target,measured['pick_center'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(pad,measured['grasp_center'],rtol=0,atol=1e-6)
            expected=(measured['pick_top'] if task=='touch' else
                env._reward_target()+np.array([0.,0.,(env._pick_top()[2]-env._pick_bottom()[2])/2]))
            np.testing.assert_allclose(goal,expected,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(scene_state(env,obs),before)
            if env.step_count!=count:raise ValueError('Diagram advanced physics')
            for k,rgb in enumerate((obs.front_rgb,obs.wrist_rgb,image)):
                sheet.paste(Image.fromarray(rgb),(384*k,j*410+26))
            drawing.text((3,j*410+3),task+' / actual TRAIN reset / NO policy or success-rate trial',fill='#111111')
            rows.append({'task':task,'source':str(item['source']),
                'map_world_target_pad_and_task_goal_match_actual_physics_at1e6':True,
                'state_or_physics_not_changed_by_diagram':True,
                'map_RGB_sha256':hashlib.sha256(image.tobytes()).hexdigest()})
        finally:env.close()
    sheet.save(args.output/'actual_reset_world_geometry.png')
    atomic_json(args.output/'audit.json',{'passed':True,'scope':'four_actual_TRAIN_resets_OFFLINE_geometry_only_NOT_neural_SR',
        'profile':contract(),'simulator_binary_sha256':hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
        'rows':rows,'teacher_actions_or_extra_physics_steps':0})
    print('All4 actual TRAIN reset map/target/pad/goals match physical geometry.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
