import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description='Native MVD-S action recognition, independent of MMAction')
    p.add_argument('--video',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--sha256',default=None)
    p.add_argument('--label-map',type=Path,default=Path(__file__).with_name('labels.txt'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--preset',choices=('paper','fast'),default='paper')
    p.add_argument('--decoder',choices=('decord','opencv'),default='decord')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,help='Optional new JSON receipt; existing files are not overwritten')
    p.add_argument('--allow-software-smoke',action='store_true',help='Test plumbing only, never interpret as learned predictions')
    args=p.parse_args(argv)
    if args.threads<1:
        p.error('--threads must be positive')
    import torch
    torch.set_num_threads(args.threads)
    from .inference import Recognizer
    recognizer=Recognizer(args.checkpoint,device=args.device,label_map=args.label_map,
                          expected_sha256=args.sha256,allow_smoke_checkpoint=args.allow_software_smoke)
    result=recognizer.predict(args.video,preset=args.preset,decoder=args.decoder,batch_size=args.batch_size)
    if result['software_smoke_only']:
        # Still exercises complete inference; suppress semantic claims from fixture weights.
        result['action']=None;result['class_index']=None;result['topk']=[];result.pop('scores',None)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scores'},indent=2))


if __name__=='__main__':
    main()
