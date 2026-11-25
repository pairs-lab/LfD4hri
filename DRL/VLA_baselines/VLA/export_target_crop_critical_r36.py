"""Export real causal corrective RGB views at handoff/middle/last-label.

Not simulation rerenders or new episodes. Last is BEFORE the final action,
not a fabricated terminal success picture. Explicit fixed selection of all
four task types, not successful-only neural video selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.smolvla.target_crop import target_view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('admission','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    admission=json.loads(a.admission.read_text());root=Path(admission['candidate'])
    if not admission.get('passed') or 'smolvla' not in admission['allowed_models']:raise ValueError('Actual canonical source qualification required')
    index=json.loads((root/'index.json').read_text());allowed={r['file']:r for r in admission['rows']};reports=[]
    for task in ('touch','pick','move','pick_place'):
        row=next(r for r in index if r.get('file') in allowed and r['task']['name']==task)
        path=root/row['file']
        if sha256_file(path)!=allowed[row['file']]['payload_sha256']:raise ValueError('Actual corrective input changed')
        with np.load(path,allow_pickle=False) as z:
            for phase,i in [('handoff',0),('middle',row['steps']//2),('last_BEFORE_action',row['steps']-1)]:
                front,wrist=z['front_rgb'][i],z['wrist_rgb'][i]
                crop,receipt=target_view(front,wrist,z['front_pick_mask'][i],z['wrist_pick_mask'][i])
                name=f'{task}_{phase}_raw_front_wrist_GTcrop.png'
                Image.fromarray(np.concatenate((front,wrist,crop),axis=1)).save(a.output/name)
                raw=z['audit_pre_action_PRIVILEGED_state64'][i]
                reports.append({'task':task,'source':str(path),'source_sha256':allowed[row['file']]['payload_sha256'],
                    'frame':i,'phase':phase,'actual_current_input':True,'file':name,'prompt':str(z['prompt'].item()),
                    'correct_bilateral_hold':bool(raw[17]),'wrong_attached':bool(raw[30]),'any_attached':bool(raw[31]),
                    'pad_target_relative_m':raw[11:14].tolist(),'receipt':receipt})
    atomic_json(a.output/'audit.json',{'scope':'actual_causal_source_RGB_critical_phase_exports_NOT_VLA_SR',
        'rows':reports,'simulation_rerendered':False,'neural_weights_loaded':False,'training_performed':False,
        'last_frame_is_pre_action_not_terminal':True,'admission_sha256':sha256_file(a.admission),
        'implementation_sha256':sha256_file(Path(__file__))})


if __name__=='__main__':main()
