"""Link each selected HD exposure to the last actual recorded RGB query.

Never substitutes a presentation camera for actor input or invents observations
between queries. Complements the five-query Scene13 policy timeline.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .qualitative_figure import (OUTPUT,POLICY,read,write,digest,configure,header,
    file_image,axes_image,save_figure)


def main(args):
    out=args.output.resolve();manifest=read(out/'manifest.json')
    case=next(c for c in manifest['cases'] if c['scene_id']==13)
    method=next(m for m in case['methods'] if m['name']=='XVLA_terminal_release')
    records={r['frame_index_0_based']:r for r in method['panels']}
    metadata=read(POLICY/'evidence/03_gray_block_target_miss/episode.json')
    if metadata['seed']!=case['seed']:raise ValueError('Wrong policy query episode')
    align=[]
    for pick in method['selection']['main']:
        panel=records[pick['index']];time=panel['simulation_time_s']
        query=max((q for q in metadata['queries'] if q['simulation_time_s']<=time),
                  key=lambda q:q['simulation_time_s'])
        if not query['raw_front_wrist_and_segmentation_bit_exact']:
            raise ValueError('Actual RPC verification missing')
        files={}
        for role in ('front','wrist'):
            source=POLICY/query['files'][role]['model224'];path=out/'policy_observations'/source.name
            expected=query['files'][role]['model224_png_sha256']
            if digest(str(source))!=expected or digest(str(path))!=expected:
                raise ValueError('Need original unmodified query image')
            files[role]={'file':str(path.relative_to(out)),'sha256':expected,
                         'original_RPC_sha256':query['original_RPC_sha256']}
        align.append({'scene_id':13,'seed':case['seed'],'HD_frame_index_0_based':pick['index'],
            'HD_simulation_time_s':time,'HD_event_label':pick['label'],
            'query':query['query'],'latest_policy_query_time_s':query['simulation_time_s'],
            'query_age_at_HD_exposure_s':time-query['simulation_time_s'],'actual_RGB224':files,
            'no_new_observation_inferred_between_queries':True})
    configure();width=180.;height=152.;fig=plt.figure(figsize=(width/25.4,height/25.4))
    header(fig,'Scene 13 · HD events versus actual actor observations',4,147,width,height,size=10,weight='bold')
    header(fig,'Each column uses the last verified query preceding the selected HD frame.',4,141,width,height,size=8)
    left=16.;gap=3.;w=(width-left-4-2*gap)/3
    for j,row in enumerate(align):
        x=left+j*(w+gap)
        header(fig,f"HD t = {row['HD_simulation_time_s']:.1f} s",x+w/2,134,width,height,size=8,ha='center',weight='bold')
        header(fig,f"Query q{row['query']}: t = {row['latest_policy_query_time_s']:.1f} s",x+w/2,129,width,height,size=8,ha='center')
        for k,role in enumerate(('front','wrist')):
            axes_image(fig,[x/width,(124-(k+1)*w-k)/height,w/width,w/height],file_image(out,row['actual_RGB224'][role]['file']))
    header(fig,'Front',2,124-w/2,width,height,size=8)
    header(fig,'Wrist',2,123-1.5*w,width,height,size=8)
    header(fig,'Native RGB: 224 × 224. No replacement input is inferred between policy queries.',4,11,width,height,size=8)
    header(fig,'GT map + numeric/proprio inputs also supplied; wrist absence does not prove memory loss.',4,5.5,width,height,size=8)
    result=save_figure(fig,'policy_observations/scene13_HD_query_alignment',600,out)
    write(out/'policy_alignment.json',{'verified':True,'records':align,'figure':result,
        'source_episode_json':str(POLICY/'evidence/03_gray_block_target_miss/episode.json'),
        'source_episode_sha256':digest(str(POLICY/'evidence/03_gray_block_target_miss/episode.json')),
        'script_sha256':digest(str(Path(__file__).resolve()))})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    main(parser.parse_args())
