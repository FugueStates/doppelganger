"""
One-shot inference: target audio -> predicted Operator preset(s).

    uv run python -m doppelganger.matcher.match --target "dataset/operator/wav/0000000.wav"
    #   then in Live: right-click -> "doppelganger: Apply Predicted Batch" -> export to hear

Encodes the target, classifies the Algorithm, then DDIM-samples a POPULATION of candidate
presets (diffusion is multimodal — the param->sound map is many-to-one, so several distinct
presets can match), renders every candidate through the frozen renderer, and RANKS them by
the magnitude-weighted spectral loss vs the target. The ranked top-K are written to the
predict manifest (op_0000 = best), so Apply Predicted Batch + Export plays them in the REAL
Operator — and this same ranked population is the natural seed set for the Phase-C CMA-ES
polish.

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
from .diffusion import Diffusion
from .model import Matcher
from .train_matcher import load_frozen_renderer


def predict(matcher, diffusion, renderer, target, note, velocity, device, canon,
            candidates, steps, alpha, algo_topk=3):
    """Returns (ranked full normalized param matrix [K,195], per-candidate audio loss [K],
    per-candidate algo index [K], classifier top-1 algo) sorted best-first.

    The Algorithm classifier is weak (FM topology is under-identifiable from one one-shot),
    so instead of trusting its argmax we spread the candidate population across its top-k
    algorithms and let the renderer-ranking decide — robust to a wrong top-1."""
    z = matcher.encode(target[canon])
    probs = matcher.algo_head(z).softmax(-1)[0]
    top1 = int(probs.argmax().item())
    k = max(1, min(algo_topk, matcher.n_algo))
    algos = probs.topk(k).indices                                         # [k] likeliest
    per = max(1, candidates // k)
    algo_t = algos.repeat_interleave(per)                                 # [k*per]
    n = algo_t.shape[0]
    note_t = torch.full((n,), float(note), device=device)
    vel_t = torch.full((n,), float(velocity), device=device)
    cond = matcher.cond(z.expand(n, -1), algo_t, note_t, vel_t)
    x0 = diffusion.ddim_sample(matcher.x0, (n, matcher.dim), cond, device, steps)
    full = matcher.assemble(x0, algo_t)                                   # [n,195]
    preds = renderer(full, device, note=note_t, velocity=vel_t)           # [n,...]
    # per-candidate loss: each rendered candidate vs the single target (renderer_loss
    # reduces over the batch, so score one candidate at a time).
    losses = torch.tensor([
        renderer_loss({k2: v[i:i + 1] for k2, v in preds.items()},
                      {k2: v[:1] for k2, v in target.items()}, alpha).item()
        for i in range(n)], device=device)
    order = torch.argsort(losses)
    return full[order], losses[order], algo_t[order], top1


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target wav to match")
    ap.add_argument("--matcher", default=str(root / "models" / "operator" / "matcher.pt"))
    ap.add_argument("--renderer", default=str(root / "models" / "operator" / "renderer.pt"))
    ap.add_argument("--out", default=str(root / "dataset" / "predict" / "manifest.json"))
    ap.add_argument("--note", type=int, default=60, help="played MIDI note of the target")
    ap.add_argument("--velocity", type=int, default=100)
    ap.add_argument("--candidates", type=int, default=16, help="presets to sample + rank")
    ap.add_argument("--algo-topk", type=int, default=3,
                    help="spread candidates across the classifier's top-k algorithms (1 = trust argmax)")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    renderer, rcfg, canon = load_frozen_renderer(Path(args.renderer), schema, device)
    ck = torch.load(args.matcher, map_location=device, weights_only=False)
    matcher = Matcher(schema).to(device)
    matcher.load_state_dict(ck["state_dict"]); matcher.eval()
    diffusion = Diffusion(ck.get("n_steps", 1000)).to(device)

    wave = torch.from_numpy(_load_audio(Path(args.target), rcfg.sample_rate, rcfg.n_samples))
    target = renderer.target_logmag(wave.unsqueeze(0).to(device))

    with torch.no_grad():
        ranked, losses, algos, top1 = predict(
            matcher, diffusion, renderer, target, args.note, args.velocity, device, canon,
            args.candidates, args.steps, args.alpha, args.algo_topk)

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
    print(f"Wrote {n} ranked candidate(s) (op_0000 = best, across top-{args.algo_topk} "
          f"algorithms) -> {out}")
    print('Now in Live: "doppelganger: Apply Predicted Batch", then Export to hear them.')


if __name__ == "__main__":
    main()
