"""
One-shot inference (v3): target audio -> predicted Operator preset, single forward pass.

    uv run python -m doppelganger.matcher.match --target "input.wav"
    #   then in Live: right-click -> "doppelganger: Apply Predicted Batch" -> play + export

Pure feed-forward: audio -> log-mel -> CNN -> per-param heads -> decode -> raw preset.
No renderer, no search, no iteration — this is the shape that ships as ONNX on CPU.
(v3 is not pitch-conditioned; it predicts params directly from the input timbre.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..schema import OperatorSchema
from ..training.data import _load_audio
from .model import MatcherConfig, ParamMatcher


def main():
    root = _repo_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--matcher", default=str(root / "models" / "operator" / "matcher_v3.pt"))
    ap.add_argument("--out", default=str(root / "dataset" / "predict" / "manifest.json"))
    ap.add_argument("--no-loop", action="store_true", help="zero-pad short clips instead of tiling")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    ck = torch.load(args.matcher, map_location=device, weights_only=False)
    cfg = MatcherConfig(**ck["cfg"]) if "cfg" in ck else MatcherConfig()
    model = ParamMatcher(schema, cfg).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    wave = torch.from_numpy(_load_audio(Path(args.target), cfg.sample_rate, cfg.n_samples,
                                        loop=not args.no_loop)).unsqueeze(0).to(device)
    with torch.no_grad():
        cont, binary, cat = model.decode(model(wave))
    preset = model.codec.decode(cont[0].cpu().numpy(), binary[0].cpu().numpy(), cat[0].cpu().numpy())

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": "Operator", "target": str(Path(args.target)),
                               "tracks": {"op_0000": preset}}, indent=2), encoding="utf-8")
    wave_name = schema_wave_name(schema, preset)
    print(f"target: {Path(args.target).name}  | predicted Osc-A Wave: {wave_name}  "
          f"Algorithm {int(preset.get('Algorithm', 0)) + 1}")
    print(f"Wrote predicted preset -> {out}")
    print('Now in Live: "doppelganger: Apply Predicted Batch", then play a note and Export.')


def schema_wave_name(schema: OperatorSchema, preset: dict) -> str:
    for p in schema.params:
        if p.name == "Osc-A Wave" and p.value_items:
            return p.value_items[int(min(len(p.value_items) - 1, max(0, round(preset[p.name]))))]
    return "?"


if __name__ == "__main__":
    main()
