"""Single-device supervised MVD-S fine-tuning; explicit practical adapter recipe.

This is not MVD's self-supervised teacher/distillation pretraining. For exact
paper fine-tuning (RandAugment/mixup/CutMix/distributed settings), use upstream.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .checkpoint import read_checkpoint, initialize_for_training, sha256
from .inference import DEFAULT_LABEL_MAP, load_labels
from .model import build_model
from .preprocess import training_view, decode_samples, make_views


def read_annotations(path, data_root):
    rows=[]
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip():
            continue
        try:
            relative,label=line.rsplit(maxsplit=1)
            label=int(label)
        except (ValueError,TypeError) as exc:
            raise ValueError(f'{path}:{number}: expected video_path integer_class_id') from exc
        video=(Path(data_root)/relative).expanduser().resolve()
        if not video.is_file() or not 0<=label<174:
            raise ValueError(f'{path}:{number}: missing video or class outside 0..173')
        rows.append((video,label))
    if not rows:
        raise ValueError('Annotation list is empty: '+str(path))
    return rows


class Videos(Dataset):
    def __init__(self, rows, *, training, seed=0, decoder='decord'):
        self.rows=rows;self.training=training;self.seed=seed;self.epoch=0;self.decoder=decoder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        video,label=self.rows[index]
        if self.training:
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            tensor=training_view(video,rng=rng,decoder=self.decoder)
        else:
            samples,_=decode_samples(video,clips=1,decoder=self.decoder)
            views,_=make_views(samples,crops=1)
            tensor=views[0]
        return tensor,label


def optimizer_groups(model, *, lr, layer_decay, weight_decay):
    groups={}
    for name,param in model.named_parameters():
        if name.startswith('patch_embed.'):
            layer=0
        elif name.startswith('blocks.'):
            layer=int(name.split('.')[1])+1
        else:
            layer=13
        decay=0.0 if param.ndim==1 or name.endswith('.bias') else weight_decay
        key=(layer,decay)
        if key not in groups:
            groups[key]={'params':[],'lr':lr*layer_decay**(13-layer),
                         'lr_scale':layer_decay**(13-layer),'weight_decay':decay}
        groups[key]['params'].append(param)
    return list(groups.values())


def save_new(path, payload):
    temp=path.with_suffix('.partial')
    torch.save(payload,temp)
    temp.replace(path)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval();correct=total=0
    for images,labels in loader:
        output=model(images.to(device))
        correct+=int((output.argmax(-1).cpu()==labels).sum());total+=len(labels)
    return correct/total


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-list',required=True,type=Path)
    p.add_argument('--val-list',required=True,type=Path)
    p.add_argument('--data-root',type=Path,default=Path('.'))
    p.add_argument('--pretrained',type=Path)
    p.add_argument('--random-init',action='store_true',help='Explicitly train from scratch instead of released MVD weights')
    p.add_argument('--software-smoke',action='store_true',help='Mark all outputs as software tests, not useful action classifiers')
    p.add_argument('--label-map',type=Path,default=DEFAULT_LABEL_MAP)
    p.add_argument('--output',required=True,type=Path,help='New output directory; existing nonempty directories are refused')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accum-steps',type=int,default=1)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=1e-3,help='Base LR scaled by effective batch/256, as upstream')
    p.add_argument('--weight-decay',type=float,default=.05)
    p.add_argument('--layer-decay',type=float,default=.7)
    p.add_argument('--warmup-epochs',type=int,default=5)
    p.add_argument('--max-updates',type=int,default=0,help='Optional software-test budget; 0 means full epochs')
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--decoder',choices=('decord','opencv'),default='decord')
    args=p.parse_args(argv)
    if min(args.epochs,args.batch_size,args.accum_steps,args.threads)<1 or args.workers<0 or args.lr<=0 or not 0<args.layer_decay<=1:
        p.error('Invalid epoch/batch/accumulation/thread/LR configuration')
    if args.warmup_epochs<0 or args.max_updates<0 or args.weight_decay<0 or args.seed<0:
        p.error('Warmup/update budget/decay/seed must be nonnegative')
    if bool(args.pretrained)==bool(args.random_init):
        p.error('Specify exactly one of --pretrained or --random-init')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Refusing to overwrite an existing nonempty output directory')
    load_labels(args.label_map)
    train_rows=read_annotations(args.train_list,args.data_root)
    val_rows=read_annotations(args.val_list,args.data_root)
    if set(p for p,_ in train_rows)&set(p for p,_ in val_rows):
        p.error('Train/validation video paths overlap')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    model=build_model(training=True)
    initialization={'source':'random-explicit'}
    if args.pretrained:
        state,receipt=read_checkpoint(args.pretrained)
        initialization={**receipt,**initialize_for_training(model,state)}
        del state
    model.to(device)
    train=Videos(train_rows,training=True,seed=args.seed,decoder=args.decoder)
    validation=Videos(val_rows,training=False,decoder=args.decoder)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator,
                      num_workers=args.workers,persistent_workers=False)
    val_loader=DataLoader(validation,batch_size=args.batch_size,num_workers=args.workers)
    peak=args.lr*args.batch_size*args.accum_steps/256
    optimizer=torch.optim.AdamW(optimizer_groups(model,lr=peak,layer_decay=args.layer_decay,
                                                weight_decay=args.weight_decay),betas=(.9,.999),eps=1e-8)
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=.1)
    epoch_updates=math.ceil(len(loader)/args.accum_steps)
    total_updates=args.epochs*epoch_updates
    warmup=min(args.warmup_epochs*epoch_updates,max(total_updates-1,0))
    metadata={'architecture':'MVD ViT-S/16','num_classes':174,'num_frames':16,
              'label_map_sha256':sha256(args.label_map),'initialization':initialization,
              'software_smoke_only':args.software_smoke,'recipe':'practical-supervised-adapter-v1',
              'not_full_published_finetune_recipe':True,'actual_peak_lr':peak,
              'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'train_list_sha256':sha256(args.train_list),'val_list_sha256':sha256(args.val_list),
              'torch_version':str(torch.__version__),'validation_preset':'fast-16x1x1',
              'head_before_sha256':hashlib.sha256(model.head.weight.detach().cpu().numpy().tobytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'TRAINING.json').write_text(json.dumps(metadata,indent=2)+'\n')
    updates=0;best=-1.;start=time.perf_counter()
    for epoch in range(args.epochs):
        train.epoch=epoch;model.train();optimizer.zero_grad(set_to_none=True)
        losses=[]
        for batch,(images,labels) in enumerate(loader):
            group_start=(batch//args.accum_steps)*args.accum_steps
            group_size=min(args.accum_steps,len(loader)-group_start)
            loss=criterion(model(images.to(device)),labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite train loss')
            losses.append(float(loss.detach()))
            (loss/group_size).backward()
            if (batch+1)%args.accum_steps and batch+1!=len(loader):
                continue
            if warmup and updates<warmup:
                multiplier=(updates+1)/warmup
            else:
                progress=(updates-warmup)/max(total_updates-warmup-1,1)
                multiplier=.01+.99*.5*(1+math.cos(math.pi*min(max(progress,0),1)))
            for group in optimizer.param_groups:
                group['lr']=peak*group['lr_scale']*multiplier
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient')
            optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            print(json.dumps({'epoch':epoch,'update':updates,'loss':losses[-1]}),flush=True)
            if args.max_updates and updates>=args.max_updates:
                break
        accuracy=validate(model,val_loader,device)
        final={**metadata,'epoch':epoch,'updates':updates,'val_top1_fast':accuracy,
               'head_after_sha256':hashlib.sha256(model.head.weight.detach().cpu().numpy().tobytes()).hexdigest(),
               'elapsed_seconds':time.perf_counter()-start}
        payload={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'mvd_metadata':final}
        save_new(args.output/'last.pth',payload)
        if accuracy>best:
            best=accuracy;save_new(args.output/'best.pth',payload)
        (args.output/'TRAINING.json').write_text(json.dumps(final,indent=2)+'\n')
        if args.max_updates and updates>=args.max_updates:
            break
    print('Saved inference-compatible best.pth/last.pth; this trainer does not save optimizer-resume state.')


if __name__=='__main__':
    main()
