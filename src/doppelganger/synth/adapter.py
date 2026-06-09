"""
Map Operator's raw parameters (schema names) -> DiffOperator control tensors.

Used by calibration (render the clone for known params) and later by the
consistency-loss matcher. v1 reads the params the sine-FM core uses; filter/LFO/
waveform/feedback params are ignored until those are modeled.
"""

from __future__ import annotations

import torch

OSCS = ["A", "B", "C", "D"]
ENV_PREFIX = ["Ae", "Be", "Ce", "De"]


def to_controls(params_list: list[dict], device: str = "cpu") -> dict:
    """params_list: list of {param_name: raw_value}. Returns batched control tensors
    plus the per-sample algorithm index."""
    B = len(params_list)
    coarse = torch.zeros(B, 4)
    level = torch.zeros(B, 4)
    adsr = torch.zeros(B, 4, 4)  # (attack, decay, sustain, release)
    volume = torch.zeros(B)
    algo = []

    for b, p in enumerate(params_list):
        for i, X in enumerate(OSCS):
            on = 1.0 if p.get(f"Osc-{X} On", 1.0) >= 0.5 else 0.0
            coarse[b, i] = float(p.get(f"{X} Coarse", 1.0))
            level[b, i] = float(p.get(f"Osc-{X} Level", 0.0)) * on
            e = ENV_PREFIX[i]
            adsr[b, i, 0] = float(p.get(f"{e} Attack", 0.0))
            adsr[b, i, 1] = float(p.get(f"{e} Decay", 0.0))
            adsr[b, i, 2] = float(p.get(f"{e} Sustain", 1.0))
            adsr[b, i, 3] = float(p.get(f"{e} Release", 0.0))
        volume[b] = float(p.get("Volume", 1.0))
        algo.append(int(round(p.get("Algorithm", 0.0))))

    return {
        "coarse": coarse.to(device),
        "level": level.to(device),
        "adsr": adsr.to(device),
        "volume": volume.to(device),
        "algo": algo,
    }
