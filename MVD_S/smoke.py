"""Bounded synthetic-video train/save/reload/inference gate, not an accuracy test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='New software-test output directory')
    args=p.parse_args()
    if args.output.exists():
        p.error('Refusing to overwrite any existing test directory')
    args.output.mkdir(parents=True)
    for name,color in [('train',(20,40,200)),('val',(180,80,20))]:
        video=args.output/(name+'.avi')
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(320,240))
        if not writer.isOpened():raise RuntimeError('MJPG writer unavailable')
        for i in range(32):
            image=np.zeros((240,320,3),np.uint8)
            cv2.rectangle(image,(10+3*i,90),(50+3*i,130),color,-1)
            writer.write(image)
        writer.release()
        (args.output/(name+'.csv')).write_text(f'{video.resolve()} 17\n')
    start=time.perf_counter()
    subprocess.run([sys.executable,'-B','-m','MVD_S.train','--random-init','--software-smoke',
                    '--train-list',str(args.output/'train.csv'),'--val-list',str(args.output/'val.csv'),
                    '--output',str(args.output/'fit'),'--epochs','1','--max-updates','1',
                    '--batch-size','1','--warmup-epochs','0','--threads','4','--device','cpu'],check=True)
    report=json.loads((args.output/'fit/TRAINING.json').read_text())
    if report['updates']!=1 or report['head_before_sha256']==report['head_after_sha256']:
        raise RuntimeError('No verified saved parameter update')
    subprocess.run([sys.executable,'-B','-m','MVD_S','--video',str(args.output/'val.avi'),
                    '--checkpoint',str(args.output/'fit/last.pth'),'--device','cpu','--preset','paper',
                    '--allow-software-smoke','--output',str(args.output/'INFERENCE.json')],check=True)
    prediction=json.loads((args.output/'INFERENCE.json').read_text())
    if prediction['num_views']!=6 or prediction['action'] is not None:
        raise RuntimeError('Inference gate did not preserve six-view/smoke-only semantics')
    summary={'software_only':True,'synthetic_videos':2,'actual_optimizer_updates':report['updates'],
             'saved_head_changed':True,'strict_reload':True,'inference_views':prediction['num_views'],
             'meaningful_action_accuracy_tested':False,'elapsed_seconds':time.perf_counter()-start}
    (args.output/'SMOKE_REPORT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
