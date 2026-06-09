# Architecture & Formulation Research — June 2026

Two focused literature passes (input representation; architecture/problem-formulation),
triggered by repeated failure of one-shot audio→parameter regression on Operator. Both
passes converged on the same conclusion, which changes our strategy.

## Headline

**We've been using the one approach the literature says does NOT work for FM:
one-shot, parameter-supervised regression over all algorithms at once.** No published
system solves unconstrained DX7/Operator-class FM that way. *Every* FM success uses at
least one of: (1) **condition on / fix the FM algorithm** + per-operator structured
output, (2) an **audio-space objective or re-ranking**, or (3) **iterative / search-based
inference against the synth** (CMA-ES, RL, or Synplant 2's Genopatch). Our symptoms
(train memorizes, val at chance, "diverse but wrong" presets, the proxy also failing) are
the textbook signature of this.

## Your two hypotheses — both confirmed

**1. Input representation.** 128-bin log-mel is the resolution-poorest common choice, and
*no top FM system relies on mel alone*:
- InverSynth II (DX7) uses **257-bin linear STFT**.
- Sound2Synth uses **STFT + Mel + CQT + MFCC + stats** and a **Prime-Dilated Conv** that
  aligns filters to integer harmonics — explicit acknowledgement that resolving FM
  ratios/sidebands needs **harmonic-aligned, higher frequency resolution**. Mel warps/
  compresses exactly where FM sidebands proliferate.
- **Best data-efficient front-end (Neural Proxies 2025):** a *frozen pretrained encoder*
  (EfficientAT / PaSST) **concatenated with mel** — most relevant to our small-data case.
- **Self-inflicted flaw:** our neural proxy used a **time-pooled** mel embedding (mean/std
  over time) → destroys all envelope/transient/dynamics info → a large fraction of params
  become unidentifiable. (The matcher uses the full spectrogram, so this hit the proxy,
  not the matcher — but it's why the proxy "couldn't beat the mean.")

**2. FM nonlinearity / many-to-one.** Confirmed as fundamental. The param→audio map is
chaotic and many-to-one (different params → same sound; tiny ratio change → wildly
different spectrum). Parameter-space loss is **misaligned with perception** (PNP loss
papers, DDSP line, even the AST paper names param-only loss as its limitation). For FM
specifically, **pure parameter-supervised one-shot regression is fundamentally limited.**

## The specific causes of our chance-level val (ranked)

1. **Learning 11 algorithms jointly with shared heads.** The 11 Operator algorithms are
   *different signal-flow topologies*. One shared regressor is asked to fit 11 different
   functions → it averages incompatible mappings → chance val + "diverse but wrong."
   Sound2Synth only worked after **fixing the algorithm and training per-algorithm**.
2. **Data scale.** 20k is ~10× too thin. Sound2Synth's 30k was *split per-algorithm* (32
   models, ≈hundreds–1k each) **plus** audio re-ranking. AST used **1,000,000** for just
   16 params. We have 20k for 195 params × 11 algorithms, jointly, with no audio crutch.
3. **Single-note stimulus.** One sustained C3 makes key-tracking, velocity-sensitivity,
   and (without note-off) release params **unidentifiable in principle** → label noise.
4. **Param loss ≠ sound** (above). Even a "correct-sounding" prediction scores as wrong.

## What actually works for FM (the proven stack)

- **Sound2Synth (IJCAI 2022)** — first usable FM matcher. Decisive: **per-algorithm models
  + per-operator structured heads + multi-rep input + render-candidates-and-pick-best-by-
  audio (MFCCD).** 5.36 MFCCD vs prior SOTA 14.70 (human threshold ~10–15). It is *not*
  pure one-shot regression — it re-ranks by audio.
- **SPINVAE** — single Transformer that takes the **algorithm as an input token**, decodes
  the preset as a sequence. Avoids 32 separate models.
- **Retrieval (SynthScribe 2023)** — CLAP-embed a large rendered preset bank, do audio→
  preset **nearest-neighbor retrieval**. Proven, low-risk way to get a strong *seed*.
- **In-the-loop search beats one-shot, dramatically.** "Instrumental" (2026): on a
  non-convex synth, **CMA-ES loss 2.09 vs a neural encoder 322 — ~138× worse for the
  one-shot net.** SerumRNN: iterative refinement holds accuracy where one-shot collapses.
  **Synplant 2's Genopatch** (our literal inspiration) is **iterative search against its
  own engine, seeded by a NN's "educated guess."**
- **RL (SynthRL 2025)** — encoder/decoder over param queries with an **audio-similarity
  reward** (no param labels needed); reportedly beats SOTA (margins unverified).

## Recommended pivot: the Genopatch pattern

Stop asking the network to one-shot the exact 195 params. Instead:

> **Neural net / retrieval produces a good *seed* preset → in-the-loop search (CMA-ES)
> against the REAL Operator, minimizing an audio distance, refines it to match.**

Why this fits us perfectly:
- It **sidesteps** the many-to-one problem, the param-loss misalignment, AND the
  non-differentiable-synth problem — search optimizes *audio* against the real engine, no
  gradients or proxy needed.
- **We already have the render pipeline** (set params via the SDK, export via pywinauto) —
  exactly what an in-the-loop search needs.
- It's **how Synplant's Genopatch works** — the tool that inspired this project.
- The NN's job shrinks to "good initialization / narrow the search," which is *achievable*
  with modest data — unlike one-shot exact prediction, which isn't.

Cost: search renders many candidates per match (minutes/match, like Genopatch's visible
iteration). Acceptable for an offline "match this sample" tool.

## Ranked action plan

**Tier 1 — proven, improves the NN path (do regardless):**
1. **Condition on the algorithm**: predict algorithm first (11-way, learnable), predict the
   rest *conditioned on it*; **per-operator structured heads with weights shared across the
   4 operators**. Likely the biggest single fix for chance-level val.
2. **Richer stimulus**: release capture (started) **+ multiple pitches + velocities** so
   envelope/key-track/velocity params become identifiable.
3. **Higher-res / harmonic input**: add **CQT** and/or **257-bin linear STFT**; consider a
   frozen **EfficientAT/PaSST** embedding concatenated with mel (best for small data).
4. **Much more data** (10⁵+), and **audio re-ranking** of top-k candidates at inference.

**Tier 2 — the real unlock for FM matching:**
5. **In-the-loop CMA-ES search against the real Operator** (audio-embedding/multi-res-STFT
   loss), seeded by the Tier-1 net or by **retrieval** over a rendered preset bank. This is
   the Genopatch pattern and the most likely route to genuinely matching timbres.

**Avoid:** betting on pure one-shot regression with more data/bigger backbone (the path
we're on — diminishing returns); diffusion-over-params (no precedent); the time-pooled
proxy (information-destroying).

## Key sources
Sound2Synth (arXiv:2205.03043) · InverSynth / InverSynth II (1812.06349; ISMIR 2023) ·
SPINVAE / preset-gen-vae (2210.16984) · Flow Synthesizer (1907.00971) · AST matching
(2407.16643) · Neural Proxies (2509.07635) · PNP / inverse problems (2301.02886, 2311.14213)
· SerumRNN (2104.03876) · SynthRL (IJCAI 2025) · Instrumental / CMA-ES (2603.15905) ·
SynthScribe CLAP retrieval (2312.04690) · Synplant 2 Genopatch (soniccharge.com/synplant).
