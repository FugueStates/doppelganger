"""
One-shot inference: target audio -> predicted Operator preset(s).

    uv run python -m doppelganger.matcher.match --target "dataset/operator/wav/0000000.wav"
    #   then in Live: right-click -> "doppelganger: Apply Predicted Batch" -> export to hear

DIRECT prediction (v2, deterministic): encode the target, then for each of the Algorithm
classifier's top-k algorithms produce ONE predicted preset (a single forward pass each).
Every prediction is confined to the audible dataset manifold, so it's a real, playable
patch — never the silent/insane presets the old audio-driven version emitted. The
candidates are rendered through the FROZEN NEURAL renderer (CPU, no real-synth render) and
RANKED by spectral loss; the best is op_0000. This whole path is what ships in the
extension (matcher.onnx + renderer.onnx, on CPU) — no Ableton render, no iteration.

Pitch: the renderer is pitch-conditioned, so pass the played --note/--velocity (default
C3/100). Automatic f0 detection (YIN) on the input is a Tool-3 follow-up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..schema import OperatorSchema
from ..synth.adapter import denormalize_params
from ..synth.spectral import renderer_loss
from ..synth.train_renderer import _repo_root
from ..training.data import _load_audio
from .model import Matcher
from .train_matcher import load_frozen_renderer


def predict(matcher, renderer, target, note, velocity, device, canon, algo_topk, alpha):
    """Returns (ranked full normalized param matrix [k,195], per-candidate audio loss [k],
    per-candidate algo index [k], classifier top-1 algo) sorted best-first.

    One deterministic preset per top-k algorithm — the renderer-ranking guards against the
    weak classifier's argmax being wrong (FM topology is hard to read from one one-shot)."""
    z = matcher.encode(target[canon])
    probs = matcher.algo_head(z).softmax(-1)[0]
    top1 = int(probs.argmax().item())
    k = max(1, min(algo_topk, matcher.n_algo))
    algos = probs.topk(k).indices                                        # [k] likeliest
    note_t = torch.full((k,), float(note), device=device)
    vel_t = torch.full((k,), float(velocity), device=device)
    pos = matcher.predict(z.expand(k, -1), algos, note_t, vel_t)          # [k, dim]
    full = matcher.assemble(pos, algos)                                  # [k, 195]
    preds = renderer(full, device, note=note_t, velocity=vel_t)          # [k, ...]
    losses = torch.tensor([
        renderer_loss({r: v[i:i + 1] for r, v in preds.items()},
                      {r: v[:1] for r, v in target.items()}, alpha).item()
        for i in range(k)], device=device)
    order = torch.argsort(losses)
    return full[order], losses[order], algos[order], top1


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target wav to match")
    ap.add_argument("--matcher", default=str(root / "models" / "operator" / "matcher.pt"))
    ap.add_argument("--renderer", default=str(root / "models" / "operator" / "renderer.pt"))
    ap.add_argument("--out", default=str(root / "dataset" / "predict" / "manifest.json"))
    ap.add_argument("--note", type=int, default=60, help="played MIDI note of the target")
    ap.add_argument("--velocity", type=int, default=100)
    ap.add_argument("--candidates", type=int, default=5,
                    help="top-k algorithms to predict + rank (1 = trust the classifier argmax)")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    renderer, rcfg, canon = load_frozen_renderer(Path(args.renderer), schema, device)
    ck = torch.load(args.matcher, map_location=device, weights_only=False)
    matcher = Matcher(schema).to(device)
    matcher.load_state_dict(ck["state_dict"]); matcher.eval()  # manifold buffers load too

    wave = torch.from_numpy(_load_audio(Path(args.target), rcfg.sample_rate, rcfg.n_samples))
    target = renderer.target_logmag(wave.unsqueeze(0).to(device))

    with torch.no_grad():
        ranked, losses, algos, top1 = predict(matcher, renderer, target, args.note,
                                              args.velocity, device, canon, args.candidates,
                                              args.alpha)

    n = min(ranked.shape[0], 64)  # rack size
    tracks = {f"op_{i:04d}": denormalize_params(ranked[i], schema) for i in range(n)}
    best_algo = int(algos[0].item())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": "Operator", "target": str(Path(args.target)),
                               "classifier_top1_algorithm": top1,
                               "best_candidate_algorithm": best_algo, "tracks": tracks},
                              indent=2), encoding="utf-8")
    print(f"target: {Path(args.target).name}  | classifier top-1 Algorithm {top1 + 1}  "
          f"| best candidate Algorithm {best_algo + 1}  | note {args.note} vel {args.velocity}")
    print(f"best candidate audio loss = {losses[0].item():.4f}  "
          f"(worst of {ranked.shape[0]} = {losses[-1].item():.4f})")
    print(f"Wrote {n} ranked candidate(s) (op_0000 = best) -> {out}")
    print('Now in Live: "doppelganger: Apply Predicted Batch", then Export to hear them.')


if __name__ == "__main__":
    main()
