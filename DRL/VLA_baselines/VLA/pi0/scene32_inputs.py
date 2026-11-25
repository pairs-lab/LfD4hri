"""Direct causal simulator feedback in OpenPI's native32 state slots.

The complete64 scene still appears in the language input. No action/phase,
desired waypoint or success flag is constructed. This is an explicitly
privileged VLA variant, not plain vision+proprioception.
"""
import numpy as np
from VLA.scene64_conditioning import validate_scene
from VLA.scene64_schema import FEATURE_NAMES

SCHEMA='openpi_native32_causal_feedback_plus_scene64_language_v1'
RAW_SLOTS=tuple(range(18))+tuple(range(18,30))+(30,39)
FEATURES=[FEATURE_NAMES[i] for i in RAW_SLOTS]


def state32(raw):
    raw=validate_scene(raw);result=raw[list(RAW_SLOTS)].copy()
    result[:6]=((raw[:6]+np.pi)%(2*np.pi)-np.pi).astype(np.float32)
    if result.shape!=(32,) or not np.isfinite(result).all():raise ValueError('Finite causal native32 required')
    return result


def statistics(raw64):
    values=np.asarray([state32(v) for v in raw64],np.float64)
    if values.ndim!=2 or len(values)<2:raise ValueError('Actual TRAIN timeline required')
    mean=values.mean(0);std=values.std(0);low=np.quantile(values,.01,axis=0);high=np.quantile(values,.99,axis=0)
    empirical={k:v.tolist() for k,v in [('mean',mean.copy()),('std',std.copy()),('q01',low.copy()),('q99',high.copy())]}
    # Keep measured native joints/gripper normalization unchanged. Feedback
    # binaries have known [0,1] support even if TRAIN rarely sees a failed
    # grasp. Continuous near-static feedback never divides by ~zero.
    binary=[i for i,raw in enumerate(RAW_SLOTS) if raw in (10,17,28,29,30)]
    for i in range(7,32):
        if i in binary:mean[i]=.5;std[i]=.5;low[i]=0.;high[i]=1.
        else:
            std[i]=max(std[i],.001)
            if high[i]-low[i]<.002:low[i]=mean[i]-.001;high[i]=mean[i]+.001
    effective={k:v.astype(np.float32).tolist() for k,v in [('mean',mean),('std',std),('q01',low),('q99',high)]}
    return effective,{'schema':SCHEMA,'feature_names':FEATURES,'raw_slots':list(RAW_SLOTS),
        'first7_native_empirical_unchanged':True,'binary_indices':binary,'feedback_std_floor':.001,
        'feedback_quantile_span_floor':.002,'empirical_TRAIN_stats':empirical,'future_action_in_state':False}
