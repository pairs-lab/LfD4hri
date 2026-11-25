"""Bounded, resumable public authentic asset downloads; never unsafe pickle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import time
import requests

from .common import ROOT,OUT,ASSETS,read,atomic_json,utc


def download(name,url,destination,*,size=None,sha256=None):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    receipt=OUT/'assets'/(name+'.json');status=OUT/'assets'/(name+'_status.json')
    if receipt.exists() and destination.exists():
        old=read(receipt)
        if old['size']==destination.stat().st_size and old['sha256']==digest(destination):return old
        raise ValueError('Existing completed asset changed: '+name)
    if destination.exists():raise FileExistsError('Unreceipted final asset, inspect before reuse: '+str(destination))
    partial=destination.with_name(destination.name+'.partial')
    session=requests.Session();started=time.monotonic()
    for attempt in range(4):
        offset=partial.stat().st_size if partial.exists() else 0
        headers={'Range':f'bytes={offset}-'} if offset else {}
        try:
            response=session.get(url,headers=headers,stream=True,timeout=(30,60))
            response.raise_for_status()
            if 'text/html' in response.headers.get('Content-Type','').lower():
                preview=response.text[:2000]
                atomic_json(status,{'UTC':utc(),'status':'HTML_NOT_tensor_payload','url':response.url,
                    'preview':preview,'attempt':attempt,'family_asset':name})
                raise ValueError('Public download returned HTML, not checkpoint bytes: '+name)
            if offset and response.status_code!=206:
                raise ValueError('Server ignored partial range; do not silently append/restart: '+name)
            if response.status_code==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                raise ValueError('Unexpected HTTP byte range: '+name)
            count=offset;last=0
            with partial.open('ab' if offset else 'xb') as stream:
                for block in response.iter_content(1024*1024):
                    if not block:continue
                    stream.write(block);count+=len(block)
                    if size and count>size:raise ValueError('Download exceeds authentic declared size: '+name)
                    if time.monotonic()-last>5:
                        atomic_json(status,{'UTC':utc(),'status':'downloading_authentic_public_asset',
                            'bytes':count,'expected_bytes':size,'attempt':attempt,'elapsed_s':time.monotonic()-started})
                        last=time.monotonic()
                stream.flush();os.fsync(stream.fileno())
            if size is not None and count!=size:raise IOError('Incomplete payload length: '+name)
            actual=digest(partial)
            if sha256 and actual!=sha256:raise ValueError('Authentic source checksum mismatch: '+name)
            os.replace(partial,destination)
            value={'UTC':utc(),'completed':True,'name':name,'source_url':url,'resolved_url':response.url,
                'path':str(destination),'size':count,'sha256':actual,'upstream_sha256':sha256,
                'upstream_checksum_available':sha256 is not None,'elapsed_s':time.monotonic()-started}
            atomic_json(receipt,value);atomic_json(status,{'UTC':utc(),'status':'downloaded_verified',**value})
            return value
        except requests.RequestException as error:
            atomic_json(status,{'UTC':utc(),'status':'transport_retry','error':str(error),'attempt':attempt})
            if attempt==3:raise
            time.sleep(2*(attempt+1))
        finally:
            if 'response' in locals():response.close()
    raise RuntimeError('Bounded public download exhausted')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def groot():
    meta=read(ROOT/'VLA/audits/backup_vlas_r93_20261022/gr00t_n15/MODEL_ASSETS.json')
    base='https://huggingface.co/'+meta['model_id']+'/resolve/'+meta['revision']+'/'
    receipts=[]
    for row in meta['files']:
        receipts.append(download('gr00t_'+row['rfilename'],base+row['rfilename'],ASSETS/'gr00t_n15'/row['rfilename'],
            size=row['size'],sha256=row['lfs']['sha256']))
    for name in ('config.json','model.safetensors.index.json','README.md'):
        receipts.append(download('gr00t_'+name,base+name,ASSETS/'gr00t_n15'/name))
    atomic_json(OUT/'gr00t_n15/WEIGHTS_DOWNLOADED.json',{'passed':True,'receipts':receipts,
        'full_model_loaded_or_trained':False})


def seer():
    fileid='147GmrVGR_oDq9abhCAkoyJil_mVZ_SRh'
    url='https://drive.usercontent.google.com/download?id='+fileid+'&export=download&authuser=0&confirm=t'
    receipt=download('seer_DROID',url,ASSETS/'seer/seer_droid.pth',size=809422808)
    atomic_json(OUT/'seer/WEIGHTS_DOWNLOADED.json',{'passed':True,'receipt':receipt,
        'public_official_folder_provenance':str(ROOT/'VLA/audits/backup_vlas_r93_20261022/seer/CHECKPOINT_ACCESS.json'),
        'strict_tensor_inspection_still_required':True})


def seer_encoders():
    mae=download('seer_MAE','https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        ASSETS/'seer/mae_pretrain_vit_base.pth')
    clip_sha='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    clip=download('seer_CLIP','https://openaipublic.azureedge.net/clip/models/'+clip_sha+'/ViT-B-32.pt',
        ASSETS/'seer/ViT-B-32.pt',sha256=clip_sha)
    atomic_json(OUT/'seer/FROZEN_ENCODERS_DOWNLOADED.json',{'passed':True,'MAE':mae,'CLIP':clip})


def main(family):
    OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    atomic_json(OUT/'ASSET_OWNER.json',{'pid':os.getpid(),'UTC':utc(),'family':family})
    jobs={'seer':seer,'gr00t_n15':groot,'seer_encoders':seer_encoders}
    def safe(name):
        try:jobs[name]();return name,True
        except Exception as error:
            import traceback
            atomic_json(OUT/name/'ASSET_ERROR.json',{'UTC':utc(),'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True});return name,False
    with ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(safe,jobs if family=='all' else [family]))
    atomic_json(OUT/'ASSET_RESULTS.json',{'UTC':utc(),'completed_downloads':results,'not_training_readiness':True})
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('all','seer','seer_encoders','gr00t_n15'),default='all')
    main(p.parse_args().family)
