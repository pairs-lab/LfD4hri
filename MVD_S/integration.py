"""Optional backend dispatch for all existing demos; default TSM remains TSM."""
from functools import lru_cache
from pathlib import Path
import json
import uuid


def add_action_arguments(parser):
    parser.add_argument('--action_backend',choices=('tsm','mvd_s'),default='tsm',
                        help='Video action backend; tsm preserves the original branch')
    parser.add_argument('--mvd_checkpoint',default=None,help='Fine-tuned MVD-S SSV2 checkpoint, required for mvd_s')
    parser.add_argument('--mvd_sha256',default=None,help='Optional expected MVD checkpoint SHA-256')
    parser.add_argument('--mvd_preset',choices=('paper','fast'),default='paper',help='16x2x3 or fast 16x1x1 inference')
    parser.add_argument('--mvd_batch_size',type=int,default=1,help='Number of MVD views per forward')
    parser.add_argument('--mvd_decoder',choices=('decord','opencv'),default='decord')
    parser.add_argument('--single_video',action='store_true',
                        help='Process --video instead of the historical Dataset/Dataset batch')
    parser.add_argument('--mvd_run_dir',default=None,
                        help='New directory for the MVD full-pipeline outputs; existing paths are refused')


def prepare_demo_outputs(args):
    if getattr(args,'action_backend','tsm')!='mvd_s':
        return
    root=Path(args.mvd_run_dir) if getattr(args,'mvd_run_dir',None) else Path('MVD_S/runs')/uuid.uuid4().hex
    root=root.expanduser().resolve()
    if root.exists():
        raise FileExistsError('MVD outputs require a new run directory: '+str(root))
    root.mkdir(parents=True)
    args.mvd_run_dir=str(root)
    for field in ('motion_saver','cropped_place_dir','cropped_pick_dir','save_dir','image_dir'):
        setattr(args,field,str(root/field))
    args.reduced_frame_video=str(root/'reduced_frame_video.mp4')


def demo_inputs(args):
    if getattr(args,'single_video',False):
        path=Path(args.video).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Video not found: '+str(path))
        return str(path.parent),[path.name]
    folder=Path('Dataset/Dataset')
    names=sorted([p.name for p in folder.iterdir() if p.suffix in ('.MOV','.mp4')],
                 key=lambda x:int(x.split('.')[0].split('_')[1]))
    return str(folder),names


@lru_cache(maxsize=1)
def _mvd(checkpoint,mtime_ns,size,device,label_map,label_mtime,checksum):
    from .inference import Recognizer
    return Recognizer(checkpoint,device=device,label_map=label_map,expected_sha256=checksum)


def action_recognition(args):
    backend=getattr(args,'action_backend','tsm')
    if backend=='mvd_s':
        if not getattr(args,'mvd_checkpoint',None):
            raise ValueError('--action_backend mvd_s requires --mvd_checkpoint (SSV2 fine-tuned, not pretrain-only)')
        checkpoint=Path(args.mvd_checkpoint).expanduser().resolve()
        labels=Path(args.label_map).expanduser().resolve()
        stat=checkpoint.stat()
        model=_mvd(str(checkpoint),stat.st_mtime_ns,stat.st_size,str(args.device),
                   str(labels),labels.stat().st_mtime_ns,getattr(args,'mvd_sha256',None))
        result=model.predict(args.reduced_frame_video,preset=getattr(args,'mvd_preset','paper'),
                             batch_size=getattr(args,'mvd_batch_size',1),decoder=getattr(args,'mvd_decoder','decord'))
        action=result['action']
        args.action_recognition_result=result
        if getattr(args,'mvd_run_dir',None):
            with (Path(args.mvd_run_dir)/'action_predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
    elif backend=='tsm':
        # Lazy import: MVD inference does not require MMAction/MMCV.
        from mmaction2.mmaction.apis import inference_recognizer,init_recognizer
        model=init_recognizer(args.config,args.action_checkpoint,device=args.device)
        result=inference_recognizer(model,args.reduced_frame_video)
        from .inference import load_labels
        labels=load_labels(args.label_map)
        if result.pred_score.numel()!=len(labels):
            raise ValueError('TSM classifier and label map dimensions differ')
        index=int(result.pred_score.argmax().item())
        action=labels[index]
    else:
        raise ValueError('Unknown action backend: '+str(backend))
    print('\n',action)
    return action
