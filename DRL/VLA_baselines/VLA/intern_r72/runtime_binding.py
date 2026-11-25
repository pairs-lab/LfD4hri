from VLA.intern_r72.common import bindings as shared_bindings, PY, SOURCE, ROOT
from VLA.common.provenance import sha256_file


def bindings():
    result=shared_bindings()
    patch=SOURCE/'src/lerobot/policies/internvla_a1_5/transformers_replace/models/qwen3_5/modeling_qwen3_5.py'
    actual=PY.parent.parent/'lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py'
    expected=sha256_file(patch)
    if sha256_file(actual)!=expected:raise ValueError('Actual isolated Qwen kernel differs from official replacement source')
    result[str(actual)]=expected
    sampler=ROOT/'VLA/target_balance_r74/sampler.py'
    result[str(sampler)]=sha256_file(sampler)
    return result
