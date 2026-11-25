"""Finite target-domain adaptation budgets, explicitly not exact paper recipes."""
import math

FAMILIES=('seer','gr00t_n15','falcon_esm')
PYTHONS={'seer':'VLA/.venvs/seer_r100/bin/python','gr00t_n15':'VLA/.venvs/gr00t_r100/bin/python',
    'falcon_esm':'VLA/.venvs/falcon_esm_r91/bin/python'}
GATES={'seer':'gate2_attempt02','gr00t_n15':'gate2_attempt01','falcon_esm':'gate2_attempt00'}
HORIZONS={'seer':3,'gr00t_n15':16,'falcon_esm':10}
PORTS={'seer':6011,'gr00t_n15':6012,'falcon_esm':6013}
RECIPES={
    'seer':{'lr':1e-4,'weight_decay':1e-4,'effective_batch':16,'initial_updates':2048,'maximum_updates':4096,
        'native_reference':'scripts/CALVIN_ABC_D/Seer/finetune.sh: 1e-3, global multi-GPU batch512, native image/action losses',
        'disclosed_deviation':'DROID initialization, history7, learned causal67, conservative single-GPU lr/batch; no action/obs token reset'},
    'gr00t_n15':{'lr':1e-4,'weight_decay':1e-5,'effective_batch':16,'initial_updates':2048,'maximum_updates':4096,
        'native_reference':'scripts/gr00t_finetune.py: lr1e-4, wd1e-5, warmup5%, new_embodiment and optional action-head LoRA',
        'disclosed_deviation':'custom31 state80/valid74, horizon16, LoRA8 action expert +native custom projector rows; frozen authentic Eagle'},
    'falcon_esm':{'lr':5e-5,'weight_decay':0.,'effective_batch':16,'initial_updates':2048,'maximum_updates':4096,
        'native_reference':'configs/calvin_finetune/finetune_kosmos_cont-fc-post_esm_wd-0_use-hand_ws-1_act-10.json',
        'disclosed_deviation':'new UR5e output head calibration +causal67 +VLM LoRA8 +native spatial injector; authentic ESM frozen but executed'},
}


def learning_rate(step,recipe):
    # One locked schedule across a continuation/restart; no reset of warmup.
    warmup=64;horizon=recipe['maximum_updates']
    if step<warmup:return recipe['lr']*(step+1)/warmup
    fraction=min(1.,(step-warmup)/max(1,horizon-warmup))
    return recipe['lr']*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
