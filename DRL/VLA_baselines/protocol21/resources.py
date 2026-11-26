"""Local resource preflight, without importing a VLA or downloading weights."""
import argparse
import json
from pathlib import Path
from .specs import SPECS

REPOS={'flower':'flower_vla_pret','xvla':'lerobot','smolvla':'lerobot',
       'vla_adapter':'vla_adapter','pi0':'openpi','vla_jepa':'lerobot_jepa',
       'dita':'dita','vlanext_s':'vlanext','falcon_esm':'falcon_vla','seer':'seer'}


def paths(family,root,foundation=None):
    root=Path(root);base=root/'VLA/checkpoints'
    default={'flower':base/'candidates_r68/flower','xvla':base/'xvla-base',
             'vla_adapter':base/'vla-adapter-spatial-pro','smolvla':base/'smolvla_base',
             'pi0':base/'pi0_robot21','vla_jepa':base/'vla_jepa_pretrain_r65',
             'dita':base/'replacement_r84/dita/official_oxe_policy.pth',
             'vlanext_s':base/'replacement_r84/qwen3.5-0.8b',
             'falcon_esm':base/'falcon_esm_r91/falcon_esm_fc_calvin_abc_wdepth.pt',
             'seer':base/'admission_r100/seer/seer_droid.pth'}
    f=Path(foundation) if foundation is not None else default[family]
    result={'upstream_source':root/'VLA/third_party'/REPOS[family]}
    if family in ('xvla','smolvla','vla_adapter','vla_jepa'):
        result.update(weights=f/'model.safetensors',config=f/'config.json')
    elif family=='flower':
        result.update(weights=f/'360000_model_weights.pt',config=f/'config.yaml',processor=f/'florence_assets_r69')
    elif family=='pi0':result.update(params=f/'params',assets=f/'assets')
    elif family=='vlanext_s':result.update(foundation_config=f/'config.json')
    else:result['weights']=f
    if family=='vla_adapter':
        result.update(head=f/'action_head--checkpoint.pt',projector=f/'proprio_projector--checkpoint.pt')
    if family=='seer':
        result.update(clip=base/'admission_r100/seer/ViT-B-32.pt',mae=base/'admission_r100/seer/mae_pretrain_vit_base.pth')
    return result


def check(family,root,foundation=None):
    required=paths(family,root,foundation)
    missing={k:str(v) for k,v in required.items() if not v.exists()}
    return {'family':family,'local_files_present':not missing,'missing':missing,
            'native_forward_verified':False,'dependency_compatibility_verified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['all',*SPECS],default='all')
    p.add_argument('--resources',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--foundation',type=Path)
    a=p.parse_args()
    if a.model=='all' and a.foundation:p.error('--foundation requires a single model')
    print(json.dumps([check(f,a.resources,a.foundation) for f in (SPECS if a.model=='all' else [a.model])],indent=2))


if __name__=='__main__':main()
