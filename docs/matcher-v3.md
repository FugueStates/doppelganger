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
- Per oscillator: **Level** (carrier level / FM modulation index — also controls oscillator
  *presence*: level 0 = silent), **ADSR envelope**, **Waveform** (kept — sniff-validated).
  The **On/Off toggles are pinned ON and not predicted** (decided Stage 2) — presence is set
  by level, so there's no separate discrete on/off to get wrong.
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

### Stage 1 — 2-operator FM (B→A) 🔵 BUILT, awaiting data (2026-06-17)
Enable Osc-B as a sine modulator of carrier Osc-A (algorithm 0 with C/D off = B→A).
**Data** (`COLLECTION_STAGE=1` in config.ts): carrier A varies waveform + amp ADSR (pitch
pinned to the note); modulator B varies **ratio (B Coarse, integer 1–16)** and **level
(= modulation index)** with a steady envelope; everything else neutral. **Model/codec:**
- **Categorical integer-ratio head.** All Coarse params (A/B/C/D) moved from binned-continuous
  to categorical over integer ratios (ratio = `floor(Coarse)`). Confirmed: A Coarse now
  decodes to **exactly 1.0** (carrier on-pitch) vs the old ~2.2 octave error; B's ratio is a
  learnable class. Restores FM's core timbre lever *and* fixes the Stage-0 decode bug.
- **Frozen params.** Fine (all) + oscillator Feedback (all) pinned to 0 and *not* predicted
  (no head/loss, set at decode) — they sit at a range edge where binned decode is biased and
  aren't gated inaudible. Confirmed Fine/Feedb decode to 0.
- **New eval probes:** `RATIO_ACC` (B Coarse class accuracy) + `MODIDX_MAE` (Osc-B Level),
  alongside `WAVE_ACC` / `ADSR_MAE`.

Gate: on Stage-1 data, learn the FM ratio (RATIO_ACC → high) + modulation index
(MODIDX_MAE → low) while keeping waveform + envelope; then hear it. Code smoke-tested on the
Stage-0 set (runs; waveform stays 1.000; A Coarse exact; Fine frozen) — RATIO/MODIDX become
meaningful only once Stage-1 data (Osc-B on) is collected.

**Stage 1 RESULT ✅ PASSED (2026-06-17, 568 samples).** Best ckpt (epoch 25): WAVE_ACC 0.95,
ADSR_MAE 0.14, MODIDX_MAE 0.11, RATIO_ACC 0.69. Carrier pitch is now correct (A Coarse exact).
- **Finding #3 — FM ratio observability ∝ modulation index.** Raw RATIO_ACC 0.62 is misleading:
  split by modulator level, accuracy is **0.25 (idx<0.2) / 0.61 / 0.89 (idx>0.5)**. When the
  FM is audible the integer ratio is identified ~89%; the misses are low-index samples where
  the ratio is physically unobservable (and perceptually irrelevant) — the same identifiability
  lesson as ADSR sustain. *This is the concrete motivation for the deferred perceptual per-param
  weighting: weight the ratio loss/metric by the modulation index (how much it matters).*
- **Overfitting** on ~500 samples (val ratio peaks ~epoch 25 then declines as train→0) — more
  data + early stop. Best-ckpt saving already keeps the peak.
- Waveform/ADSR degraded slightly vs Stage 0 (sidebands obscure the carrier) — acceptable.

**Stage 1 — pitch fixes + final (2026-06-17): SOUNDS "extremely close" by ear ✅.**
- **Finding #4 — out-of-scope *pitch* params must be frozen, not predicted.** A param diff on
  a match showed Transpose decoding to −0.054 (~5 cents flat) — audible. Transpose + per-osc
  `Freq<Vel` are pinned out-of-scope pitch params; added them to the codec `FROZEN_DEFAULT`
  (now: Fine, Feedback, Transpose, Freq<Vel) so carrier pitch is *exactly* the played note.
  (Also confirmed the FM ratio was already correct — an apparent ratio error was a stale
  manifest, not the model.)
- Bumped bins **32 → 64** (config + train `--bins` default): cont_mae 0.022 → **0.014**, finer
  envelopes, less edge bias. Retrained: pitch clean, sounds extremely close.

### Stage 2 — 3-operator FM (C→B→A) 🔵 BUILT, awaiting data (2026-06-17)
Add Osc-C as a modulator of B (chain C→B→A; A carrier, B+C sine modulators, D off). Data
(`COLLECTION_STAGE=2`): all oscillators ON; carrier A varies waveform + amp ADSR; B and C
vary **ratio + level (= modulation index)**, steady envelopes; D's level = 0. Because C's
level spans 0→1, the set naturally mixes 2-op and 3-op patches.

- **Finding #5 — control presence by LEVEL, gate/weight by LEVEL (the perceptual-weighting
  unlock).** A modulator at level 0 contributes nothing, so the On toggles are redundant:
  pinned ON + frozen (one fewer discrete prediction). Gating switched from "Osc-X On < 0.5"
  to a **continuous weight = Osc-X Level** (0..1): each oscillator's params (ratio, envelope,
  waveform) are supervised — and the eval probes weighted — *in proportion to how much the
  oscillator is heard*. This is exactly the deferred Sound2Synth per-param perceptual
  weighting, and it makes RATIO_ACC honest (counts ratios where the modulator is audible,
  not the low-index don't-cares from Finding #3). Smoke-tested: gate(B Coarse)=Osc-B Level,
  On frozen to 1, probes RATIO_B/C/D + IDX_B/C/D.
- **More data** still wanted to close the ~500-sample overfitting.

Gate: on Stage-2 data, learn B *and* C ratios (index-weighted RATIO_B / RATIO_C → high) +
both indices while keeping waveform/envelope; then hear a 3-op patch.

**Stage 2 RESULT ⚠ REGRESSION — deep serial FM is hard to invert (2026-06-17, 694 samples).**
WAVE_ACC 0.84, ADSR 0.13, IDX_B 0.07 (fine), but **RATIO_B fell 0.66 → ~0.35** and
**RATIO_C ~0.13, IDX_C ~0.24 (poor)**. Diagnostic by audibility (unlike Stage 1, this is NOT
a metric artifact): RATIO_B *when B is loud* = 0.43 (was ~0.89 in 2-op); RATIO_C *when chain
weight C_lvl·B_lvl > 0.3* = 0.19 (no better than low-chain) — i.e. C's ratio is genuinely
unreadable even when audible.
- **Finding #6 — chain depth breaks FM inversion.** (a) A deeper modulator obscures the
  shallower one: C FM-ing B means B is no longer a clean sine, so B's own ratio is muddied
  (B regressed). (b) FM-of-FM is an ill-posed inverse — a doubly-nested spectrum doesn't
  factor into separate (ratio, level) pairs (many-to-one worsens with depth). This is the
  documented "deep DX7-class FM is the hard case," quantified.
- **Implications / options:** 2-op is the clean sweet spot. (1) **Hear it** — FM is forgiving;
  the sound may still approximate acceptably even with C's ratio often wrong. (2) **Topology
  rethink: parallel modulators** (B→A *and* C→A, e.g. algorithm 7) instead of a serial chain
  — sidebands land directly on the carrier → far more separable/identifiable, and own-level
  weighting becomes exactly correct. "Complexity = # operators" as more *parallel* modulators,
  not a deeper chain. (3) chain-aware weighting (C by C_lvl·B_lvl) + more data — correctness
  improvements but won't fix the core difficulty alone. *Decision pending: serial vs parallel,
  after hearing.*

**Stage 2 PIVOT → PARALLEL modulators (2026-06-17).** Heard the 3-op serial match (0000050):
audibly looser than 2-op, as the curve predicted. Switched the topology: extra operators now
modulate the carrier **in parallel** (B→A *and* C→A, algorithm 7) instead of a serial chain.
Only the collector changed (Algorithm 0 → 6); codec/model/weighting unchanged because each
parallel modulator's effect depends only on its own level (the own-level weighting is already
exact). Rationale: parallel puts each operator's sidebands directly on the carrier → far more
separable than nested serial FM. "Complexity = # operators" now means more *parallel*
modulators on the carrier, not a deeper chain. Serial chains may return later as an explicit
hard mode. NEXT: collect fresh parallel 3-op data, retrain, compare RATIO_B/RATIO_C vs the
serial run, hear it.

**Finding #7 — parallel modulators are a permutation-symmetric SET; canonicalize the labels
(2026-06-17).** With B→A, C→A, D→A, swapping two modulators gives *identical* audio (FM
modulation sums commutatively into the carrier), but the per-slot supervised loss compares
against the arbitrary ground-truth slot assignment — so the model was fed **contradictory
targets** (same spectrogram, different "correct" labels across samples) and learned a hedged,
blurry predictor. Diagnostic on the parallel model: per-slot ratio acc 0.33 vs
permutation-invariant (set-matched) 0.38 — permutation directly explains ~5 pts, but the
deeper damage is to *training*. **Fix:** `codec._canonicalize` sorts the symmetric modulators
(B/C/D) by level (dominant first, ratio tiebreak) before encoding, so each sound maps to ONE
labeling. Audio unchanged; loss well-posed; pairs with level-gating so the dominant modulator
lands in the high-weight slot. Reuses existing data (re-encoding only). RATIO_B = the dominant
modulator's ratio (should rise); judge the real gain by whether permutation-invariant set
accuracy climbs above 0.38 after retrain. (Also explains "sounds close despite low ratio acc":
the model nails the big perceptual factors + the dominant modulator; secondary ratios are
finer/often interchangeable.)

**Finding #8 — 10k run: data lifts everything EXCEPT the secondary-modulator ratio (the
fundamental ceiling) (2026-06-17).** Scaled 3-op parallel to ~10.8k samples (2×3090 box,
one GPU). Best ckpt (epoch ~20): WAVE_ACC **0.94** (was 0.86), ADSR_MAE **0.09** (0.13),
RATIO_B/dominant **0.75** (0.61), IDX_B **0.05**, IDX_C 0.11 — all clearly improved toward
their ceilings. But **RATIO_C/secondary stayed ~0.24** (was 0.26) — 14× the data didn't move
it. → The secondary modulator's ratio is a **fundamental identifiability ceiling** (masked by
the dominant, ambiguous from one note: the many-to-one wall), NOT overfitting. Everything
perceptually dominant scales with data; the secondary is rough but perceptually minor (FM-
forgiving; user heard "quite close"). **Conclusion: the FM core is done and good** — carrier
(waveform/envelope) + dominant modulator reliably matched, secondary rough. Mild overfitting
remains (best ~ep20, slight decline after; best-ckpt saves the peak) → early-stop ~ep30.

### Later stages
filter (next — biggest remaining timbral subsystem) → feedback → 4-op → …

**Finding #9 — envelope param-loss ≠ perceptual-loss; added an envelope-shape loss
(2026-06-17).** Sample 0008262 sounded wrong: the model's Ae ADSR was close (Attack/Release
exact, Sustain near) but **Decay was 0.63 vs 0.92** — and that flipped a slow *swell* (decay
so slow it never reaches the low sustain within the 1.5 s note) into a *pluck* (faster decay
reaches the low sustain → crashes to ~0.08). A moderate param error, a huge perceptual one —
because the binned param-loss penalizes Decay uniformly with no idea this region is
perceptually explosive. (Two layers: identifiability — a non-completing slow decay is
genuinely ambiguous from 1.5 s; and param≠perception.) **Fix (built):** an auxiliary
**envelope-shape loss** in `train_matcher` — render each oscillator's predicted (soft-decoded)
vs true ADSR amplitude CURVE via `diff_operator.adsr` and L1 them, level-weighted. This
weights the envelope error by its actual *curve* impact (swell↔pluck → large gradient;
inaudible decay error → ~none), cheap (1-D, no FM render, nothing to game), differentiable
through the bins' expected value. New eval probe `ENV_MAE`; `--w-env` weight; folded into the
gate score. This is the first instance of targeted perceptual weighting (below).

## Perceptual-weighting candidates (params where param-distance ≠ perceptual-distance)

Running list of params that most need perceptual weighting (Sound2Synth per-param MFCCD
weighting / PNP) or a targeted perceptual loss — because a small parameter error there causes
a large change in the *sound*. Add to this as stages surface them.

| param(s) | why sensitive | status |
|---|---|---|
| **Envelope Decay** (then Attack/Release) | decay rate flips swell↔pluck near a low sustain; nonlinear time mapping | ✅ addressed by the envelope-shape loss (Finding #9) |
| **Filter cutoff (Filter Freq)** | shifts the whole spectral balance; small move = big timbre change | ⏳ pending the filter stage — *expect to need it* |
| **Filter resonance** | near self-oscillation a tiny change is drastic | ⏳ pending the filter stage |
| **Modulator ratio (Coarse)** | wrong integer ratio = wrong harmonic character | handled via categorical head + index weighting |
| **Modulation index (Osc Level, modulators)** | sets FM brightness | handled via level-gating (the weighting) |
| **Feedback** | chaotic; small change → harsh/inharmonic shift | ⏳ when added (currently frozen off) |
| **Waveshaper drive** | nonlinear; small change → big harmonic change | ⏳ when added |
| **Sustain** (if dB-mapped) | perceptually log, not linear | watch via ENV_MAE |

General mechanism to reach for: render the relevant differentiable sub-signal (envelope
curve, or eventually a cheap spectral feature) and L1 it — a *targeted* perceptual loss that
needs no full synth, rather than a uniform param loss.

## Open levers (when needed)
- Per-param perceptual loss weighting (Sound2Synth, via `sensitivity.py`).
- `--bins 64` / larger `pool_t` for finer continuous / temporal resolution.
- Identifiability: constrain stimulus so asked-for params are observable (longer render for
  slow envelopes).
