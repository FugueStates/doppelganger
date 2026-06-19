"""
Parameter codec (v3) — converts Operator's 195 raw params <-> the model's grouped,
normalized targets, computes gating masks, and handles the special encodings.

Output head kinds:
  • continuous : K-bin classification, target = Gaussian SOFT label around the true bin
                 (InverSynth/Sound2Synth trick — fixes regression-to-mean)
  • binary     : one logit, BCE, target 0/1
  • categorical: one softmax head each (own cardinality), cross-entropy
  • RATIO      : the per-oscillator Coarse params, treated as CATEGORICAL over integer
                 ratios (ratio = floor(Coarse); freq-map finding). This is the correct
                 representation for FM's primary timbre control and avoids the
                 binned-continuous edge-decode bias that put the carrier an octave off.
  • FROZEN     : out-of-scope params pinned to a fixed value — NOT predicted (no head, no
                 loss), set at decode. Used for params whose binned-continuous decode would
                 be audibly wrong at a range edge yet aren't gated inaudible (Fine detune,
                 oscillator feedback). Keeps the predicted preset clean.

GATING (now LEVEL-based + continuous = perceptual weighting): an oscillator's params matter
in proportion to how much it's heard, i.e. its output/modulation LEVEL. `encode` returns a
per-param WEIGHT = the controlling oscillator's level (0..1); the loss/metrics weight each
param by it. A modulator at level 0 contributes nothing (FM index 0), so its ratio/envelope
get weight 0 (= old hard gate); a half-level modulator gets half weight. This both lets us
control oscillator presence by level alone (the On toggles are frozen on) and focuses
supervision where the sound actually is (the Sound2Synth perceptual-weighting idea, and the
fix for the misleading raw ratio accuracy at low modulation index).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..schema import OperatorSchema, Param

# Coarse params -> categorical integer ratio. ratio = floor(raw Coarse), raw in [0,48].
RATIO_PARAMS = ("A Coarse", "B Coarse", "C Coarse", "D Coarse")
N_RATIO = 49  # classes 0..48

# Out-of-scope params pinned to a fixed value (not predicted), because their
# binned-continuous decode is biased (off a range edge or just imprecise) and the error is
# AUDIBLE — they aren't gated inaudible. Especially the pitch params: the carrier must
# track the played note exactly, so any drift here detunes the whole sound.
#   • Fine detune + oscillator feedback (edge-pinned at 0)
#   • Transpose + per-osc Freq<Vel — global/velocity pitch offsets, pinned to 0 so pitch is
#     exactly the played note (a stray -0.05 semitone Transpose was audibly flattening it)
# Filter stage (matcher-v3 Stage 3: STATIC filter). The model LEARNS the perceptually
# dominant static-filter controls — Filter Type / Freq / Res / Slope stay as heads. The rest
# of the filter section is out of scope for now and frozen:
#   • Filter ENGAGED (On pinned on): presence is handled by cutoff (a wide-open lowpass ≈
#     transparent); making On a gated binary is a later expansion.
#   • circuits / drive / morph / vel-key / LFO routing: secondary coloring, pinned neutral.
#   • the filter ENVELOPE: Fe Amount 0 ⇒ NO sweep (static filter). The sweep is the Stage-4
#     expansion (expected to need a perceptual loss, as the amp envelope did) — unfreeze the
#     Fe ADSR + Fe Amount then. (The remaining inert Fe params are pinned to schema default in
#     ParamCodec.__init__ via FILTER_ENV_FROZEN, since they need the schema to resolve.)
FILTER_FROZEN: dict[str, float] = {
    "Filter On": 1.0,
    "Filter Circuit - LP/HP": 0.0,           # Clean
    "Filter Circuit - BP/NO/Morph": 0.0,     # Clean
    "Filter Drive": 0.0,
    "Filter Morph": 0.0,
    "Filt < Vel": 0.0,
    "Filt < Key": 0.0,
    "Filt < LFO": 0.0,
    "Fe Amount": 0.0,                        # no filter-envelope sweep (Stage 4 unfreezes)
}
FILTER_ENV_FROZEN = (
    "Fe Attack", "Fe Init", "Fe A Slope", "Fe Decay", "Fe Peak", "Fe D Slope",
    "Fe Sustain", "Fe Release", "Fe End", "Fe R Slope", "Fe Mode", "Fe Loop",
    "Fe Retrig", "Fe R < Vel",
)

FROZEN_DEFAULT: dict[str, float] = {
    **{f"{X} Fine": 0.0 for X in "ABCD"},
    **{f"Osc-{X} Feedb": 0.0 for X in "ABCD"},
    "Transpose": 0.0,
    **{f"{X} Freq<Vel": 0.0 for X in "ABCD"},
    # Oscillator On toggles pinned ON: oscillator PRESENCE is controlled by level (level 0
    # = silent), so we don't predict the toggles — one fewer discrete decision, and it makes
    # level the single knob for "how many operators" (decided 2026-06-17).
    **{f"Osc-{X} On": 1.0 for X in "ABCD"},
    **FILTER_FROZEN,
}


def _oscillator_of(name: str) -> str | None:
    for o in "ABCD":
        if name.startswith(f"Osc-{o}") or name.startswith(f"{o} ") or name.startswith(f"{o}e "):
            return o
    return None


def _remap_osc(name: str, src: str, dst: str) -> str:
    """Rename a param of oscillator `src` to the corresponding param of oscillator `dst`."""
    for pre in (f"Osc-{src}", f"{src}e ", f"{src} "):
        if name.startswith(pre):
            return {f"Osc-{src}": f"Osc-{dst}", f"{src}e ": f"{dst}e ",
                    f"{src} ": f"{dst} "}[pre] + name[len(pre):]
    return name


def _gate_of(name: str) -> str | None:
    """The oscillator LEVEL that gates/weights this param's audibility (None if ungated).
    A param of oscillator X is weighted by Osc-X Level; the level itself is never gated."""
    o = _oscillator_of(name)
    if o is None:
        return None
    gate = f"Osc-{o} Level"
    return None if name == gate else gate


@dataclass
class EncodedTargets:
    cont: np.ndarray
    binary: np.ndarray
    cat: np.ndarray        # int64 [n_cat] class indices (incl. ratio params at the end)
    cont_mask: np.ndarray  # float32 weight per param (= controlling osc level, 0..1; 1 ungated)
    binary_mask: np.ndarray
    cat_mask: np.ndarray


class ParamCodec:
    def __init__(self, schema: OperatorSchema, n_bins: int = 32,
                 frozen: dict[str, float] | None = None,
                 ratio_params: tuple[str, ...] = RATIO_PARAMS, n_ratio: int = N_RATIO,
                 symmetric_mods: tuple[str, ...] = ("B", "C", "D")):
        self.schema = schema
        self.n_bins = n_bins
        self.n_ratio = n_ratio
        # The parallel modulators (B/C/D → A) are an UNORDERED set: swapping two of them gives
        # identical audio, so the slot labels are arbitrary. Canonicalize by sorting them
        # (dominant level first) before encoding, so each sound maps to ONE labeling — removes
        # the permutation ambiguity that otherwise feeds the model contradictory targets.
        self.symmetric_mods = symmetric_mods
        use_default = frozen is None
        self.frozen = dict(FROZEN_DEFAULT if use_default else frozen)
        by = {p.name: p for p in schema.params}
        # pin the (inert at Fe Amount 0) filter-envelope params to their schema defaults, so
        # they're neither learned nor audible until Stage 4 unfreezes the sweep.
        if use_default:
            for n in FILTER_ENV_FROZEN:
                if n in by:
                    self.frozen.setdefault(n, float(by[n].default))
        froz = set(self.frozen)
        ratio_set = {n for n in ratio_params if n in by}
        self.ratio_set = ratio_set

        self.cont = [p for p in schema.continuous if p.name not in ratio_set and p.name not in froz]
        self.binary = [p for p in schema.binary if p.name not in froz]
        base_cat = [p for p in schema.categorical if p.name not in froz]
        ratio_ps = [by[n] for n in ratio_params if n in by and n not in froz]
        self.categorical = base_cat + ratio_ps               # ratio params appended
        self.cat_is_ratio = [False] * len(base_cat) + [True] * len(ratio_ps)

        self.cont_gates = [_gate_of(p.name) for p in self.cont]
        self.binary_gates = [_gate_of(p.name) for p in self.binary]
        self.cat_gates = [_gate_of(p.name) for p in self.categorical]

    @property
    def n_cont(self) -> int:
        return len(self.cont)

    @property
    def n_binary(self) -> int:
        return len(self.binary)

    @property
    def cat_cardinalities(self) -> list[int]:
        return [self.n_ratio if r else (p.cardinality or 1)
                for p, r in zip(self.categorical, self.cat_is_ratio)]

    @staticmethod
    def _weight(gates, params) -> np.ndarray:
        """Per-param weight = controlling oscillator's level (0..1), 1.0 if ungated.
        Continuous gate: level 0 -> weight 0 (params unsupervised, osc inaudible); level 1
        -> full supervision. This is the perceptual weighting."""
        return np.array([1.0 if g is None else min(1.0, max(0.0, float(params.get(g, 1.0))))
                         for g in gates], dtype=np.float32)

    def _cat_class(self, p: Param, is_ratio: bool, params: dict) -> int:
        v = params.get(p.name, p.default)
        if is_ratio:
            return int(min(self.n_ratio - 1, max(0, math.floor(v))))   # ratio = floor(Coarse)
        return int(p.normalize(v))                                     # categorical index

    def _canonicalize(self, params: dict) -> dict:
        """Reorder the symmetric modulator slots by (level desc, ratio) so the labeling is
        unique. Audio is unchanged (parallel modulators sum commutatively into the carrier)."""
        mods = [X for X in self.symmetric_mods]
        if len(mods) < 2:
            return params
        order = sorted(mods, key=lambda X: (-float(params.get(f"Osc-{X} Level", 0.0)),
                                            math.floor(float(params.get(f"{X} Coarse", 0.0)))))
        if order == mods:
            return params
        new = dict(params)
        for dst, src in zip(mods, order):           # slot dst receives oscillator src's params
            if dst == src:
                continue
            for name, val in params.items():
                if _oscillator_of(name) == src:
                    new[_remap_osc(name, src, dst)] = val
        return new

    def _filter_label(self, params: dict) -> dict:
        """If the sample was rendered with the filter OFF, its audio is unfiltered — so the
        consistent target for the always-on filter is 'wide open' (transparent): cutoff at
        max, no resonance, lowpass. Without this, mixing filter-off data (the FM stages) with
        filter-on data teaches the cutoff head a wrong default: the filter-off render carries
        a stale mid-cutoff value that never actually colored the audio."""
        if float(params.get("Filter On", 1.0)) >= 0.5:
            return params
        return {**params, "Filter Freq": 1.0, "Filter Res": 0.0, "Filter Type": 0.0}

    def encode(self, params: dict) -> EncodedTargets:
        params = self._canonicalize(params)
        params = self._filter_label(params)
        cont = np.array([p.normalize(params.get(p.name, p.default)) for p in self.cont], dtype=np.float32)
        binary = np.array([p.normalize(params.get(p.name, p.default)) for p in self.binary], dtype=np.float32)
        cat = np.array([self._cat_class(p, r, params)
                        for p, r in zip(self.categorical, self.cat_is_ratio)], dtype=np.int64)
        return EncodedTargets(cont, binary, cat,
                              self._weight(self.cont_gates, params),
                              self._weight(self.binary_gates, params),
                              self._weight(self.cat_gates, params))

    def decode(self, cont, binary, cat_indices) -> dict:
        out: dict[str, float] = {}
        for p, v in zip(self.cont, cont):
            out[p.name] = p.denormalize(float(v))
        for p, v in zip(self.binary, binary):
            out[p.name] = p.denormalize(1.0 if v >= 0.5 else 0.0)
        for p, is_ratio, idx in zip(self.categorical, self.cat_is_ratio, cat_indices):
            out[p.name] = float(int(idx)) if is_ratio else p.denormalize(int(idx))
        for name, val in self.frozen.items():
            out[name] = float(val)                                     # pinned, not predicted
        return out

    # -- binned-classification helpers (continuous params) --------------------
    def soft_bins(self, values01, sigma: float = 1.5):
        import torch
        K = self.n_bins
        centers = (torch.arange(K, device=values01.device) + 0.5) / K
        d = (values01.unsqueeze(-1) - centers) * K
        return torch.softmax(-0.5 * (d / sigma) ** 2, dim=-1)

    def expected_value(self, bin_probs):
        import torch
        K = self.n_bins
        centers = (torch.arange(K, device=bin_probs.device) + 0.5) / K
        return (bin_probs * centers).sum(-1)
