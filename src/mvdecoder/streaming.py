"""StreamingDecoder: run the full causal pipeline on a chunk stream.

`process(chunk)` takes one raw chunk [channels, samples] of any length and returns the
decision events finalised during it. The pipeline per chunk is: select subset channels
-> causal notch -> PARRM (gated by the causal stim-on mask) -> causal bandpass ->
trailing windows -> gate/gesture features -> decoder -> decider. All state lives on the
instance and is carried across calls, so the output does not depend on how the stream
is chunked.

The stim source must be calibrated (P locked) before use; PARRM's history takes about
25*P samples to fill, and events before that are flagged `warm=False`.

numpy/scipy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig
from .dsp import design_filters, CausalFilter
from .parrm import StreamPARRM
from .features import FeatureExtractor, AmplitudeHistory
from .decoder import TwoStageDecoder
from .decider import Decider
from .stim_source import StimSource


class StreamingDecoder:
    def __init__(self, decoder: TwoStageDecoder, source: StimSource):
        self.cfg: DecoderConfig = decoder.cfg
        self.decoder = decoder
        self.source = source
        self.feat = FeatureExtractor(self.cfg, decoder.good)
        self.decider = Decider(self.cfg)
        self.keep_abs = decoder.good[self.feat.keep_pos]        # absolute subset channels
        self.nsub = len(self.keep_abs)
        bp_sos, notch_sos = design_filters(self.cfg)
        self.notch = CausalFilter(notch_sos, self.nsub)
        self.bp = CausalFilter(bp_sos, self.nsub)
        self.amp_hist = AmplitudeHistory(self.feat.used_lags)
        self.reset()

    def reset(self):
        """Clear all streaming state. Requires the source's P to be set."""
        if not self.source.P:
            self.source.calibrate()
        self.source.begin_stream()
        self.notch.reset()
        self.bp.reset()
        self.parrm = StreamPARRM(self.nsub, self.source.P, self.cfg.parrm_m, self.cfg.parrm_skip)
        self.amp_hist.reset()
        self.decider.reset()
        self.warmup = self.parrm.need
        self.total_in = 0
        self.bpbuf = np.zeros((self.nsub, 0))
        self.onbuf = np.zeros(0, bool)
        self.buf_start = 0
        self.win_next = 0
        self._first = True

    def process(self, chunk: np.ndarray) -> list:
        chunk = np.asarray(chunk)
        n = chunk.shape[1]
        if n == 0:
            return []
        s0 = self.total_in
        x = chunk[self.keep_abs].astype(float)
        self.source.push(chunk, s0)
        y = self.notch.apply(x)
        on = self.source.on_block(s0, n)
        cl = self.parrm.push(y)
        cl[:, ~on] = y[:, ~on]                      # gate: only remove where stim runs
        bp = self.bp.apply(cl)
        self.bpbuf = np.hstack([self.bpbuf, bp])
        self.onbuf = np.concatenate([self.onbuf, on])
        self.total_in += n
        return self._emit()

    def _emit(self) -> list:
        win, step = self.cfg.win, self.cfg.step
        fs = self.cfg.fs
        events = []
        while self.win_next + win <= self.buf_start + self.bpbuf.shape[1]:
            off = self.win_next - self.buf_start
            seg = self.bpbuf[:, off:off + win]
            stim_flag = bool(self.onbuf[off:off + win].any())
            A, spatial, eg, mf, A_sub = self.feat.window_kernel(seg)
            dA = self.amp_hist.push(A)
            xg = self.feat.gate_vector(A, dA)
            xs = self.feat.gesture_vector(spatial, eg, mf, A_sub)
            gate_p = self.decoder.gate_proba(xg)
            gest_p = self.decoder.gesture_proba(xs)
            end = self.win_next + win - 1
            new_run = self._first
            self._first = False
            dec = self.decider.update(gate_p, gest_p, end / fs * 1000.0, new_run)
            events.append(dict(end=end, stim=stim_flag, warm=end >= self.warmup,
                               gate_proba=gate_p, gest_proba=gest_p,
                               xg=xg, xs=xs, **dec))
            self.win_next += step
        if self.win_next > self.buf_start:              # drop consumed history
            cut = self.win_next - self.buf_start
            self.bpbuf = self.bpbuf[:, cut:]
            self.onbuf = self.onbuf[cut:]
            self.buf_start = self.win_next
        return events
