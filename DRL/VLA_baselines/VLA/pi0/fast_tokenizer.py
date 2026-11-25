"""Lossless extraction of FAST action BPE tokens from PaliGemma output IDs."""

from __future__ import annotations

import logging

import numpy as np

from openpi.models.tokenizer import FASTTokenizer


def _find_subsequence(values: np.ndarray, pattern: np.ndarray) -> int | None:
    if pattern.size == 0 or values.size < pattern.size:
        return None
    for index in range(values.size - pattern.size + 1):
        if np.array_equal(values[index : index + pattern.size], pattern):
            return index
    return None


class UR5FASTTokenizer(FASTTokenizer):
    """Decode action tokens directly, without a lossy text round trip.

    Upstream ``FASTTokenizer.extract_actions`` decodes PaliGemma token IDs to
    text and immediately re-encodes the text.  SentencePiece can merge or drop
    terminal action pieces during that round trip.  On this 7D UR5 contract we
    observed valid, ground-truth 224-coefficient chunks returning only 222
    coefficients, at which point the universal action processor returns an
    all-zero action.  The action pieces already exist losslessly in the model's
    output token IDs, so delimit them by the official ``Action: `` and ``|``
    token sequences and pass them directly to the official FAST decoder.

    Token creation during training is inherited unchanged.  Malformed model
    output still fails closed to a zero chunk; no robot coefficient is guessed
    or interpolated.
    """

    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        codec_scale: float = 10.0,
    ) -> None:
        if not np.isfinite(codec_scale) or codec_scale <= 0:
            raise ValueError("FAST codec scale must be positive and finite")
        super().__init__(max_len, fast_tokenizer_path=fast_tokenizer_path)
        # The official UniversalActionProcessor exposes this quantization
        # parameter.  A robot-specific, explicitly recorded scale can improve
        # joint/EEF fidelity without changing its DCT, BPE or token extraction.
        self._fast_tokenizer.scale = float(codec_scale)

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        values = np.asarray(tokens, dtype=np.int64).reshape(-1)
        marker = np.asarray(self._paligemma_tokenizer.encode("Action: "), dtype=np.int64)
        delimiter = np.asarray(
            self._paligemma_tokenizer.encode("|", add_eos=True), dtype=np.int64
        )
        marker_index = _find_subsequence(values, marker)
        if marker_index is None:
            logging.warning("FAST output did not contain the Action marker; returning a zero chunk")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        start = marker_index + marker.size
        # The first delimiter token is sufficient and cannot collide with FAST
        # action pieces, which are mapped into PaliGemma's reserved loc range.
        delimiter_hits = np.flatnonzero(values[start:] == delimiter[0])
        if delimiter_hits.size == 0:
            logging.warning("FAST output did not contain the action delimiter; returning a zero chunk")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        encoded_pieces = values[start : start + int(delimiter_hits[0])]
        if encoded_pieces.size == 0:
            logging.warning("FAST output contained an empty action; returning a zero chunk")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        action_tokens = self._act_tokens_to_paligemma_tokens(encoded_pieces)
        try:
            decoded = np.asarray(
                self._fast_tokenizer.decode(
                    [action_tokens.tolist()],
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                )[0],
                dtype=np.float32,
            )
        except Exception:
            logging.exception("FAST action decoder rejected model output; returning a zero chunk")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        if decoded.shape != (action_horizon, action_dim) or not np.isfinite(decoded).all():
            logging.warning(
                "FAST decoder produced invalid shape/data %s; returning a zero chunk", decoded.shape
            )
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        return decoded
