# Phase B — One-shot matcher design

> **v2 (2026-06-17) — DETERMINISTIC, MANIFOLD-BOUNDED. Supersedes the diffusion design
> below (kept as history).** The matcher is `input audio → Operator params` in a single
> forward pass, trained with the magnitude-weighted spectral loss through the frozen
> renderer. Code: `src/doppelganger/matcher/` (`model.py`, `train_matcher.py`, `match.py`;
> `diffusion.py` removed).

## v2 — what it is and why it changed

**The product constraint (the thing that drove the redesign):** this ships inside an
Ableton extension as `audio → params`, instant, on CPU, with **no synth in the loop** —
the SDK can't render an instrument, so any approach that needs real-Operator renders
(CMA-ES) is a dev tool, not the product. So the matcher must predict *directly*.

**Why deterministic, not diffusion.** The diffusion version (history below) had two fatal
behaviours on this problem: its x0-MSE *rose* instead of falling (the diffusion objective
was vestigial), and — driven by the audio loss — it **railed params to extremes that game
the renderer's slack**: Volume → 0 (the renderer's targets are peak-normalized, so Volume
is invisible to it → no gradient → it rails → −∞ dB, silent), Transpose → +48, envelopes
→ instant. The renderer-space metric looked fine (+31%) because the renderer ignores those
params; the *real* Operator doesn't, so the applied presets were insane. A deterministic
predictor is simpler, is the most direct realization of the product, and removes the
vestigial machinery.

**The fix for railing — confine predictions to the audible data manifold.** The dataset
was collected with `SAMPLING_RULES` that make every training preset audible by
construction (Volume ∈ [0.45,1], Transpose ∈ ±12, device on, carrier audible…). We
compute the per-param observed `[min,max]` over the training set (the **manifold box**)
and the ParamHead's `sigmoid` output is mapped **affinely** into that box. So:
- the predictor can only emit presets inside the audible distribution the renderer was
  trained on — Volume can't go below ~0.45, Transpose can't exceed ±12, forced params
  (Device On, etc.) are pinned automatically because their box has zero width;
- the audio loss still has smooth gradient *everywhere inside* the box, so it matches
  within the manifold — but it physically **cannot escape** to chase the renderer's slack.

This is the structural fix: the railing isn't penalized, it's made impossible.

### v2 architecture (`model.py`)

```
target wav ─► renderer.target_logmag ─► AudioEncoder ─► z ──┬─► AlgoClassifier ─► 11-way logits
                                                            │
                                          note, velocity ───┼─► cond ─► ParamHead (MLP, sigmoid)
                                          (z, algo embed) ──┘            └► affine into manifold box ─► 194 params
```
- **AudioEncoder** — strided 2-D CNN over the canonical log-mag → 256-d `z`.
- **AlgoClassifier** — `z → 11-way softmax` over Algorithm (the one discrete param, kept
  out of the continuous head; same rationale as before).
- **ParamHead** — residual MLP, `sigmoid` output, affine-mapped into the manifold box for
  the other 194 params. Deterministic: one forward pass, no sampling.

### v2 training (`train_matcher.py`)

`total = w_audio · audio  +  w_param · MSE(pos, true position-in-manifold)  +  w_algo · CE`.
The audio loss (through the frozen renderer) is the perceptual driver; the param-anchor
keeps predictions on real presets and breaks many-to-one ties toward plausible ones; the
manifold mapping makes both safe. True algorithm is teacher-forced into the render so a
classifier mistake can't poison the continuous head; EMA weights; eval mirrors inference
(predict for the top-k algorithms, render each through the renderer, keep the best per
sample, vs the mean-params baseline). Smoke-tested: predicted presets come out audible and
in-range (`Volume 0.77`, `Transpose 0.62`, `Device On 1`) instead of silent/railed.

### v2 inference (`match.py`) + deployment

Encode → classify Algorithm → predict ONE preset per top-k algorithm → render all through
the frozen **neural** renderer (CPU, no real render) → rank by spectral loss → `op_0000` =
best. The whole path is what ships: `matcher.onnx` (+ optionally `renderer.onnx` for the
in-extension ranking) running on CPU. CMA-ES against the real Operator stays a separate
dev-only polish, not part of the extension.

---

## (History) v1 — diffusion design (2026-06-15)

> Superseded by v2 above. Kept for the reasoning trail.

Built once the pitch-conditioned renderer cleared its gate (+50.3% canonical loud-bin
gap across the full pitch range). The matcher is `input audio → Operator params`, trained
with the magnitude-weighted spectral loss backpropped **through the frozen renderer**.

## Why this shape (and why regression failed before)

The original Tool-2 attempt — one-shot mel→params **regression** — hit a chance-level
ceiling because (a) the param→sound map is many-to-one (many presets sound identical, so
param-space supervision fights itself) and (b) param-loss ≠ perceptual-loss. The fix the
whole project was built toward: optimize for **sound**, using the renderer as a
differentiable audio loss. The Batch-5 refactor made the renderer's `forward` accept a
param **tensor** and stay differentiable end-to-end, which is the hook this depends on.

## The three pieces (`model.py`)

```
target wav ─► renderer.target_logmag ─► AudioEncoder ─► z ──┬─► AlgoClassifier ─► 11-way logits
                                                            │
                                          note, velocity ───┼─► cond ─► ParamDenoiser (x0-diffusion)
                                                            │            over the 194 non-Algorithm params
                                          (z, algo embed) ──┘
```

- **AudioEncoder** — a compact strided 2-D CNN over the canonical-resolution log-mag
  spectrogram → a 256-D embedding `z`. (The encoder reuses the renderer's own
  `target_logmag`, so train and inference see byte-identical features.)
- **AlgoClassifier** — `z → 11-way softmax` over Algorithm. **This is the decided
  discrete-param strategy: a separate classifier, not the denoiser.** FM topology is
  categorical and changes everything downstream; a cross-entropy head fits it far better
  than asking a continuous denoiser to land on a rounded bin, and it keeps the discrete
  choice out of the diffusion entirely.
- **ParamDenoiser** — a FiLM-conditioned **MLP over the flat 194-D param vector** (every
  param *except* Algorithm). Not a 2-D U-Net: a preset has no spatial structure.
  x0-parameterization (predicts the clean params, not the noise), cosine schedule, DDIM
  sampling. Conditioned on `[z, algo-embedding, note, velocity]`.

The `Matcher` owns the schema bookkeeping that splits the 195-D normalized vector into
{Algorithm column} + {the 194 the denoiser models} and reassembles a full 195-D vector
(Algorithm slot filled from the classifier) to feed the renderer.

## Why diffusion (not a deterministic regressor)

The many-to-one map means the posterior `p(params | audio)` is **multimodal** — several
genuinely different presets match one sound. A deterministic net is forced to average
them (the regression-to-the-mean failure again, in param space). Diffusion samples the
modes, which is also exactly what Phase C wants: a *population* of plausible seeds to hand
to the in-the-loop CMA-ES polish against the real Operator.

## Training (`train_matcher.py`)

Per step, on a real-Operator render from the dataset:

```
target log-mag ─► z ─► algo logits           ── CE vs true Algorithm        (w_algo)
diffusion: x_t = q_sample(true_params₍₋algo₎, t);  x0̂ = denoiser(x_t, t, cond)
                                              ── MSE(x0̂, true)  (diffusion / param nudge, w_diff)
assemble[x0̂ + TRUE algo] ─► FROZEN renderer(note, vel) ─► renderer_loss vs target
                                              ── magnitude-weighted spectral loss (THE DRIVER, w_audio)
```

Key decisions:
- **Audio loss is the driver, diffusion-MSE is the nudge.** The x0-MSE both trains the
  generative model and supplies a small param-space anchor; the spectral loss through the
  renderer is what actually enforces *sounding* right. Gradients flow into the predicted
  params through the renderer (its weights frozen, `requires_grad_(False)`; gradient still
  propagates to the input — verified in the smoke test, and the renderer accumulates none).
- **Teacher-forced algorithm.** During training the *true* Algorithm is fed to both the
  denoiser conditioning and the render, so a classifier mistake can't poison the
  continuous matcher; the classifier learns independently by CE. At inference the
  predicted Algorithm is used throughout (a standard teacher-forcing train/infer gap;
  acceptable while classifier accuracy is reasonable).
- **Eval metric = the honest one.** DDIM-sample params from val audio, render, score vs
  target; compare to rendering the **mean-params** preset (the predict-the-mean baseline,
  analogous to the renderer's gate). `gain%` > 0 means the matcher genuinely matches.
  Also reports Algorithm top-1 accuracy.

Defaults: `--w-audio 1 --w-diff 1 --w-algo 1`, 1000 train timesteps, 25 DDIM eval steps,
**batch 16** (the renderer's residual over 3 STFT resolutions is the VRAM driver). v1 is
single-device (the renderer forward each step makes it heavier than renderer training);
DDP is a follow-up if throughput needs it.

**VRAM note (the gradient-checkpointing gate).** The matcher backprops the audio loss
through the *frozen, eval-mode* renderer to the input params. The renderer's checkpointing
was originally gated on `self.training`, which is False for a frozen renderer → no
checkpointing → the residual's full activation graph over 3 resolutions OOM'd a 24 GB card.
Fixed by gating checkpointing on `torch.is_grad_enabled()` instead (correct anyway —
checkpointing only matters for the backward pass): it now fires during the matcher's
backprop and stays a no-op under `no_grad` (the renderer's own eval / inspect / the
matcher's DDIM sampling). `load_frozen_renderer` also force-sets `grad_checkpoint=True`
so a renderer trained with `--no-grad-ckpt` still fits. If VRAM is still tight, lower
`--batch-size` or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Inference (`match.py`)

`target wav → predicted preset(s)`. Encode → classify Algorithm → DDIM-sample a
**population** of candidates → render all through the frozen renderer → **rank by spectral
loss** → write the ranked top-K to the predict manifest (`op_0000` = best). Then
**"doppelganger: Apply Predicted Batch"** + Export plays them in the *real* Operator.
The ranked population is the seed set for Phase-C CMA-ES.

`denormalize_params` (in `synth/adapter.py`) is the inverse of `normalize_params` and is
what writes raw Live values — NOT `schema.Param.denormalize`, which uses a different
quantized convention and would silently corrupt every multi-option categorical (Algorithm,
waveforms, filter type, envelope modes) in the applied preset. (Found + fixed in the v1
smoke test.)

Pitch: the renderer is pitch-conditioned, so `match.py` takes `--note/--velocity` (default
C3/100). Detecting f0 from the input automatically (YIN) is a Tool-3 follow-up.

## Status / next

- ✅ Built + smoke-tested end-to-end on CPU (gradient-to-params through the frozen renderer
  confirmed; train loop, DDIM eval, ranked-candidate inference + manifest all run).
- **Next:** train on the box against the real `renderer.pt`; read the eval `gain%` vs the
  mean-params baseline and Algorithm accuracy. Then Phase C (CMA-ES polish seeded by the
  ranked candidates) and Tool 3 (the inference extension + YIN pitch detection).
- Possible refinements if quality needs it: classifier-free guidance on the conditioning,
  DDP.

## v1.1 stabilization (2026-06-16, from the first full run)

The first 60-epoch run on the box revealed three things; the continuous matcher worked
(val match **+33%** vs the mean-params baseline, improving) but:

1. **The diffusion x0-MSE rose (0.57 → 3-4) instead of falling.** Diagnosis: the data
   lives in [-1,1] but the denoiser output was unbounded, and the audio loss only sees the
   *clamped* x0 (via `assemble`), so for a many-to-one synth the audio gradient drove the
   raw prediction far past the rails to chase a sound match — exploding the x0-MSE and
   feeding the DDIM sampler out-of-distribution x0 estimates (unstable, bouncy val).
   **Fixes:** (a) **tanh-bound the denoiser output** to [-1,1] by construction; (b)
   **SNR-weight the audio loss** by `alpha_bar[t]` — at high noise the denoiser can't
   predict a clean x0, so supervising its *sound* there only fights the diffusion
   objective; near-clean steps get full audio supervision, pure-noise steps ~none
   (`renderer_loss_per_sample` in `spectral.py` makes the per-timestep weighting possible).
   In smoke tests the x0-MSE now stays flat instead of diverging.
2. **The Algorithm classifier sat at chance** (CE ≈ ln 11, acc ≈ 0.10). FM topology is
   largely under-identifiable from one random-preset one-shot (a modulator at ~0 level or
   an oscillator off hides the routing). **Fix:** don't trust the argmax — `match.py`
   `--algo-topk` (default 3) spreads the candidate population across the classifier's
   top-k algorithms and lets the renderer-ranking pick. (Smoke test: classifier top-1 was
   Algorithm 1, but the ranking chose an Algorithm 6 candidate.)
3. **Bouncy val** (no EMA). **Fix:** EMA of the matcher weights (decay 0.999), evaluated +
   deployed like the renderer's; checkpoint carries `state_dict` (EMA) + `raw_state_dict`.
