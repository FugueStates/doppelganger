# Matcher v3 — architecture rationale & build log

> Living document. **Why** we rebuilt the matcher this way, **what the constraints are**,
> the **validated recipe**, and a running **build log** of what works / what broke / how we
> fixed it as we add complexity. (History of the failed v1/v2 approaches is in
> [matcher-design.md](matcher-design.md); this doc is the current path.)

## Why this architecture (the journey to v3)

The matcher's job: **input audio → Operator preset, one shot.** Three approaches were tried:

1. **Direct parameter regression (Tool 2 v1).** Hit a chance-level ceiling — the param→sound
   map is many-to-one and parameter-MSE ≠ perception, so the net regressed to the mean.
2. **Audio loss through a neural renderer (v2, diffusion then deterministic).** We trained a
   differentiable "renderer" surrogate and backpropped a spectral loss through it. It
   *looked* like it worked in renderer-space (+30–57%) but produced **noise in the real
   Operator** — the renderer only faithfully models a thin slice (sine-FM), so the matcher
   learned to game its slack (predicting silent/extreme presets that scored well through the
   smoothing surrogate but sound nothing alike on the real device).
3. **v3 (current).** A deep-research pass over 20 sources (see [matcher-design.md] + memory)
   found the decisive lesson: **it's the loss/formulation, not the encoder** ("AST beats CNN"
   was refuted), and the SoTA methods (InverSynth, Sound2Synth) use a **perceptually-weighted
   *parameter* loss with classification heads — no synth in the loop at all.** That's
   deployable and sidesteps every failure above.

## Hard constraints (these are permanent)

- **One-shot feed-forward.** No in-the-loop search / no rendering candidates against the real
  synth — an Ableton extension **cannot render an instrument track** (the SDK only renders
  AudioTracks). Genopatch-style iteration is impossible in the product; don't propose it.
- **Deployable as ONNX on CPU** inside the extension: `audio → params` in a single pass.
- So the renderer is *not* in the matcher at all (it was the thing that kept breaking).

## The validated recipe

```
audio ─► LogMel (in-graph) ─► CNN encoder ─► pool to freq×TIME grid ─► per-param heads
                                                                       ├ continuous → K-bin classification + Gaussian soft labels
                                                                       ├ binary      → BCE
                                                                       └ categorical → softmax cross-entropy
                              (oscillator GATING masks inaudible params from the loss)
```

- **Encoder:** a modest CNN (research: encoder isn't the lever). **Pools to a freq × time
  grid, never global** — see Build-log finding #1.
- **Heads (the lever):** continuous params as **binned classification with Gaussian soft
  labels** (InverSynth/Sound2Synth — beats regression, no collapse-to-mean); binary as BCE;
  categorical as softmax CE.
- **Gating:** an off oscillator's params are masked out of the loss (inaudible → supervising
  them is noise + a source of many-to-one).
- **Loss:** masked soft-CE + BCE + CE. **Per-param perceptual weighting** (Sound2Synth-style,
  from `sensitivity.py`) is deferred until the param space is large enough to need it.
- **No renderer, no differentiable synth.** Honest held-out param accuracy is the metric.

Code: `src/doppelganger/matcher/` — `features.py` (LogMel), `codec.py` (encode/decode,
binning, gating), `model.py` (encoder + heads), `data.py`, `train_matcher.py`, `match.py`.

## Scope — the "core FM v1" target (decided 2026-06-17)

Deliberately narrow the synthesis space to what's learnable + expressive, expand later.

**In scope (the model predicts these):**
- **Routing: the fixed linear chain D→C→B→A only** (the 11 algorithms are dropped → the
  weak Algorithm classifier is removed). **Complexity is controlled by how many oscillators
  are enabled** (1 = pure carrier, 2 = B→A FM, 3 = C→B→A, 4 = full chain).
- Per oscillator: **On/Off, Level** (carrier level / FM modulation index), **ADSR envelope**,
  **Waveform** (kept — sniff-validated, cheap, expressive).
- **Modulator frequency ratio (B/C/D Coarse): IN ✅ (decided 2026-06-17).** Implemented as a
  **categorical integer-ratio head** (ratio = `floor(Coarse)`). It's FM's primary timbre
  control — the harmonic vs inharmonic/bell/metallic character — so it's essential to sound
  diversity and stays in the final version. Categorical is the correct representation and
  also fixes the Stage-0 decode bug. (Carrier A's ratio remains pinned = the note pitch.)

**Out of scope for v1 (pinned to neutral, not predicted):**
- Algorithm selection (fixed to the linear chain).
- **Fine detune (all oscillators)** — subtle, dropped.
- **Carrier frequency (Osc-A Coarse/Fine)** — tracks the played note; never predicted.
- Oscillator **feedback** (major harshness source), **waveshaper**, **filter**, **LFO**,
  **pitch envelope**.
- Velocity/key scaling, pan/spread/glide, exotic envelope modes (standard ADSR only).

**Later stages, add back highest-value first:** filter (most impactful) → feedback → then
optionally parallel/branching algorithms, LFO, pitch envelope.

## Methodology: grow gradually, fix flaws as they surface

We expand the synthesis space in **stages** (sniff → +osc/FM → +waveforms → +filter → …).
Each stage is a controlled dataset where we know what *should* be learnable, so when a new
param class breaks something we catch it in isolation and fix it before adding more. The
sniff test below is stage 0.

## Build log

### Stage 0 — Sniff test ✅ PASSED (2026-06-17)
**Data:** ~556 samples, single Osc-A, waveform ∈ {Sine, Saw 64, Square 64, Triangle},
random ADSR, everything else neutral, C3. **Purpose:** can the formulation learn a basic
waveform + envelope at all? If not, nothing else matters.

- **Waveform: perfect** — `WAVE_ACC = 1.000`, held-out 8/8 exact. The binned-classification /
  categorical formulation is validated for spectral params.
- **Finding #1 — global time-pooling killed the envelope.** First run: ADSR_MAE stuck at
  ~0.20. Cause: the encoder ended with `AdaptiveAvgPool2d(1)`, averaging over **time**, which
  is exactly where the ADSR envelope lives. **Fix:** pool to a freq×time grid
  (`AdaptiveAvgPool2d((4,16))`). ADSR_MAE **0.20 → 0.084**, waveform still perfect.
  *Lesson: any temporal param (envelope, LFO) needs the time axis preserved — never
  global-time-pool.*
- **Residual ADSR error is partly intrinsic:** samples with very long attack+decay never
  reach sustain within the 1.5 s note / 3 s window, so sustain is genuinely unobservable.
  *Lesson: don't ask the model to predict what the stimulus doesn't expose (identifiability).*
- **Finding #2 — pitch/ratio params mis-decode (Coarse, Fine).** By ear the match was close
  but played the wrong note: Coarse decoded 1→2.2 (octave up) and Fine 0→~38, **identically
  on every sample** (so: systematic *decode* bias, not model error). Two causes: (a)
  expected-value decoding of a binned soft distribution is **biased upward at a range edge**,
  and Coarse=1/Fine=0 sit at the bottom edge; (b) **Coarse is effectively categorical** —
  ratio = `floor(Coarse)` (freq-map finding), so generic [0,48] binning can't land on integer
  ratios, and ratio 1↔2 is an octave. **Planned fix (Stage 1, where Coarse varies):** a
  categorical head over integer ratios for Coarse; finer resolution (or argmax decode) for
  Fine. *Lesson: pitch/ratio params need categorical / fine treatment, and edge-sitting
  values reveal expected-value decode bias.*

**Verdict:** the foundation is sound — the formulation learns spectral (waveform) and
temporal (envelope) params with honest metrics. Cleared to expand.

### Stage 1 — (next) 2-operator FM (B→A)
Enable Osc-B as a modulator of Osc-A (the start of the linear chain). Carrier A pinned to
the note; Fine dropped everywhere. Modulator ratio (B Coarse) is IN as a **categorical
integer-ratio head** — restores FM's core timbre control and fixes the Coarse decode bug
from Stage 0. Everything from the Out-of-scope list stays neutral. Watch for the next flaw. (Then: 3- and 4-op chains → filter → feedback → …, adding
the perceptual per-param loss weighting once the space is large.)

## Open levers (when needed)
- Per-param perceptual loss weighting (Sound2Synth, via `sensitivity.py`).
- `--bins 64` / larger `pool_t` for finer continuous / temporal resolution.
- Identifiability: constrain stimulus so asked-for params are observable.
