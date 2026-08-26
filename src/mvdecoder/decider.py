"""Decider: fuse the two heads and drive a move/rest stimulation gate.

Per window it smooths each head's probabilities with an EMA, forms the final label
(rest if the gate says rest, otherwise the gesture argmax), and produces a stim
command from p_move = P(move):

  * START when at rest and p_move >= on_thr
  * STOP  when stimming and p_move < off_thr and the onset latch has elapsed
  * HOLD  otherwise

on_thr >= off_thr gives hysteresis around the boundary; the latch holds a freshly
started train on for at least `latch_ms` so a brief early rest cannot stop it. This is
the control logic from MovementStimController.js, as a pure policy: it returns the
command and does not talk to a stimulator. Time is passed in (`t_ms`), so it works off
sample time offline and wall-clock time live.

numpy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig

START, STOP, HOLD = "START", "STOP", "HOLD"


class Decider:
    def __init__(self, cfg: DecoderConfig):
        self.cfg = cfg
        self.n_move = cfg.rest_idx
        self.enabled = True
        self.reset()

    def reset(self):
        """Clear smoothing and gate state for a new recording/stream."""
        self.gate_ema = np.array([0.5, 0.5])                 # [P(move), P(rest)]
        self.gest_ema = np.full(self.n_move, 1.0 / self.n_move)
        self.stimming = False
        self.stim_started_ms = 0.0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.stimming = False

    def emergency_stop(self):
        """Stop stimulating and stop reacting until re-enabled."""
        self.enabled = False
        self.stimming = False

    def update(self, gate_proba, gest_proba, t_ms, new_run=False) -> dict:
        """Advance the decider by one window.

        gate_proba : [P(move), P(rest)]
        gest_proba : probabilities over move classes 0..n_move-1
        t_ms       : current time in ms (sample time offline, wall-clock live)
        new_run    : True if this window starts a new contiguous run
        """
        a = self.cfg.ema_alpha
        gate_proba = np.asarray(gate_proba, float)
        gest_proba = np.asarray(gest_proba, float)

        if new_run and self.cfg.ema_reset:
            self.gate_ema = gate_proba.copy()
            self.gest_ema = gest_proba.copy()
        else:
            self.gate_ema = a * gate_proba + (1 - a) * self.gate_ema
            self.gest_ema = a * gest_proba + (1 - a) * self.gest_ema

        is_rest = bool(self.gate_ema.argmax() == 1)
        p_move = float(self.gate_ema[0])
        final = self.cfg.rest_idx if is_rest else int(self.gest_ema.argmax())

        cmd = HOLD
        if self.enabled:
            if not self.stimming and p_move >= self.cfg.on_thr:
                self.stimming = True
                self.stim_started_ms = t_ms
                cmd = START
            elif (self.stimming and p_move < self.cfg.off_thr
                  and (t_ms - self.stim_started_ms) >= self.cfg.latch_ms):
                self.stimming = False
                cmd = STOP

        return dict(final=final, label=self.cfg.class_names[final], p_move=p_move,
                    is_rest=is_rest, gate_ema=self.gate_ema.copy(),
                    gest_ema=self.gest_ema.copy(), stim_cmd=cmd, stimming=self.stimming)
