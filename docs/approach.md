# doppelganger — Approach (v2: audio-objective)

**Current direction:** match sounds by their **audio output**, not their parameters.
Solve each target with **in-the-loop search against the real Operator** (gradient-free,
parameter-agnostic), then **amortize that solver into a one-shot model**. This supersedes
the parameter-supervised approach below.

---

## ⛔ What we tried first and abandoned — DO NOT REPEAT

**Approach:** generate random Operator presets → render → train a CNN/ResNet to predict the
195 parameters from the audio (binned-classification + BCE + CE, gating masks, etc.).

**Why it failed (with evidence):**
1. **Many-to-one is fatal for parameter loss.** Many different parameter sets produce the
   same/similar sound, so supervising on the *one* ground-truth parameter vector penalizes
   acoustically-correct answers. Parameter-space loss is misaligned with perception.
2. **Validation never generalized**, across *every* variant tried (small CNN, big CNN,
   ResNet, binned, gating-masked): val parameter accuracy stayed at ~random while train
   slowly memorized. Per-parameter check: only the forced carrier's waveform learned a
   weak signal (~4.5× random); modulators/algorithm/envelope modes were ~random.
3. **The audio metric exposed it.** Re-rendering predictions: matched-vs-shuffled log-mel
   ratio ≈ **0.86 (≈ random)** on a properly-trained model — "diverse but wrong" presets
   that sound nothing like the target. (An earlier "0.72" was just hedging to the dataset
   mean, not real matching.)
4. **The neural proxy failed** (params→audio-embedding surrogate): val L1 **worse than
   predicting the mean** (ratio 1.09) — even the *forward* map won't generalize from ~20k.
5. **A from-scratch differentiable clone calibrated poorly.** After fixing a real bug
   (Ableton MIDI 60 = **261.6 Hz**, not 130.8) and adding exponential envelopes, sweep
   fidelity ratios were still ~1–1.9 (a pure sine didn't even match) — the gap is
   Operator's **undocumented DSP** (exact envelope curvature/slopes, level→FM-index map,
   filter circuits, anti-aliasing). Not worth chasing.
6. **Data/stimulus limits compounded it:** ~20k samples is ~10–50× too few for 195 params
   × 11 algorithms learned jointly; a single sustained C3 makes envelope-release,
   key-tracking, and velocity params unidentifiable in principle.

**One-line lesson:** *Don't predict parameters with a parameter loss for an FM synth.
Optimize the audio, and let the parameters be whatever reproduces it.*

(Full research: `docs/research-architecture-2026-06.md`. The param-supervised code under
`src/doppelganger/training/` is kept for reference but is not the path forward.)

---

## ✅ The v2 approach: audio-objective, parameter-agnostic

**Principle:** only the rendered output matters; parameters are just the search variables.

**Hard constraints that shape it:**
- Real Operator is **non-differentiable** → no backprop through the synth → use a
  **gradient-free** optimizer (CMA-ES / evolution).
- Real renders are **slow** (~64 presets per ~12 s export) → the synth is in the loop only
  for the *solver* and for *offline data generation*, never per training step of the net.

This is the proven approach for FM (Sound2Synth re-ranks by audio; "Instrumental" shows
CMA-ES beats one-shot neural inference ~138× on a non-convex synth) and is how Synplant 2's
**Genopatch** (this project's inspiration) actually works.

---

## Plan (phases)

**Phase A — Rank parameters by audio variability.** Perturb each parameter one-at-a-time
from a base preset, render, measure spectral change → rank. Start the search on the **top-K
most audio-impactful** parameters; expand the subset later by diversity contribution.
(`src/doppelganger/synth/sensitivity.py`.)

**Phase B — In-the-loop CMA-ES solver.** For a target sound, optimize the subset's
parameters to minimize a multi-scale-STFT / mel distance to the target, evaluated by
**rendering candidates on the real Operator**. The rack maps perfectly: a CMA-ES population
of 64 = one export. ~10–30 min/match. (`src/doppelganger/synth/search.py` + an SDK
per-generation handshake.)

**Phase C — Amortization dataset.** Run the solver on many diverse target sounds →
`(target audio → audio-matched parameters [+ convergence trajectory])`. These labels are
audio-optimal by construction, sidestepping many-to-one. (Render-heavy but offline.)

**Phase D — One-shot model (the end goal).** Train a model to reproduce the solver in one
shot: **audio → parameters**.
- **Baseline:** direct regression on the matched params (fast to stand up, sanity check).
- **Strongest fit — conditional generative model:** the audio→params map is multimodal, so
  model the *distribution* p(params | audio-embedding), not a point. **Conditional
  diffusion** is the SOTA choice and matches the "learn the convergence" idea — diffusion's
  denoising *is* learned iterative refinement, sampleable in a few steps for fast inference.
  Caveat on the U-Net idea: a 2-D U-Net is for image/spectrogram diffusion; since we
  generate a **flat parameter vector**, the denoiser should be an **MLP/Transformer
  conditioned on the audio embedding** (a "1-D" diffusion), not a 2-D U-Net. (Normalizing
  flows — Flow Synthesizer — are a lighter proven alternative for the same multimodality.)
- **Optional inference polish:** seed the model's prediction into a few CMA-ES steps on the
  real Operator to close any residual gap (full Genopatch = learned seed + short search).

---

## Reusable infrastructure (all already built)
Rack builder + `applyParams` (set any preset on 64 tracks), the pywinauto **export**
automation, the **file-handshake** loop (Auto Collect / Run Sweeps), `spectral.py`
(multi-scale STFT loss), `schema.py`, and the ~20k real renders (now usable as **targets**).

## Open questions to revisit
- Search dimensionality vs convergence time (subset size; seeding the CMA-ES mean).
- Categorical handling in CMA-ES (per-algorithm search vs relaxed-and-rounded).
- Amortization target sources (Operator-rendered vs augmented/external sounds).
- Diffusion vs flow vs regression for Phase D — decide after Phase C data exists.
