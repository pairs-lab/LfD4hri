"""Validate FAST coefficients BEFORE the upstream decoder's zero fallback."""
import numpy as np


def require_actual_coefficients(processor,tokens,horizon,dimension):
    values=np.asarray(tokens,np.int64).reshape(-1)
    vocabulary=set(processor.bpe_tokenizer.get_vocab().values())
    if not len(values) or any(int(i) not in vocabulary for i in values):
        raise ValueError('Malformed FAST neural output: missing/invalid action BPE IDs; no zero fallback')
    decoded=processor.bpe_tokenizer.decode(values.tolist())
    expected=int(horizon)*int(dimension)
    if len(decoded)!=expected:
        raise ValueError(f'Malformed FAST neural output: {len(decoded)} DCT coefficients, expected {expected} ({horizon}x{dimension}); no zero fallback')
    coefficients=np.fromiter(map(ord,decoded),dtype=np.int64)+int(processor.min_token)
    return coefficients.reshape(horizon,dimension)
