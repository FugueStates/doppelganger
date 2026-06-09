# doppelganger — Project Plan

> Synth sound-matching for Ableton Live, inspired by Synplant 2's *Genopatch*: feed in an audio sample (≤ ~5 s) and get the closest-matching preset for a selected built-in instrument. **v1 target: Operator.** Architecture must generalize to other instruments later.

Status legend: 🔴 blocked/needs decision · 🟡 design pending validation · 🟢 ready to build · ✅ done · 🔵 in progress

---

# ★ CURRENT PLAN (v3, 2026-06) — supersedes §0–§10 below

> The original plan (one-shot **mel→params regression**, §4) was proven to **fail for FM** — many-to-one param→sound + param-loss ≠ perception → chance-level validation (see [[training-findings]], `docs/approach.md`). The project pivoted to an **audio-objective** approach. The sections below (§0–§10) are kept as **history**.

## The approach (Genopatch-style, audio-objective)

```
            ┌────────────────────────────────────────────────────────────────┐
            │ A. NEURAL RENDERER (the differentiable Operator surrogate)        │
real        │   params (+pitch/vel)  →  HybridRenderer  →  predicted log-mag    │
Operator →  │     gray-box: DiffOperator physics core + FiLM neural residual    │  ← we are here
renders     │   trained on (params, real-Operator audio); frozen when faithful  │
(dataset)   └───────────────────────────────┬────────────────────────────────┘
                                             │ provides a differentiable AUDIO loss
            ┌────────────────────────────────▼───────────────────────────────┐
            │ B. ONE-SHOT MATCHER (conditional diffusion over params)          │
            │   input audio → params, trained with audio loss THROUGH the      │
            │   frozen renderer (param loss = nudge, audio loss = driver)      │
            └────────────────────────────────┬────────────────────────────────┘
                                             │ seed preset
            ┌────────────────────────────────▼───────────────────────────────┐
            │ C. INFERENCE POLISH (optional): short CMA-ES against the REAL    │
            │   Operator to refine the seed (search.py pipeline exists)        │
            └─────────────────────────────────────────────────────────────────┘
```
Why this works where regression failed: the **forward** map (params→sound) is a well-posed function and **is** learnable; we optimize for **sound**, not parameter values; the real Operator stays the final authority (renderer is a training aid; CMA-ES polish + the applied preset use real Operator).

## Phases & status

**Phase A — Neural renderer (IN PROGRESS 🔵).** Validate fidelity by **energy-weighted / loud-bin log-mag** vs a predict-the-mean baseline (`inspect_renderer.py`); the loud bins are what the matcher's gradient rides on. SoTA built in attributable batches:
- **Batch 1 ✅ VALIDATED** — multi-resolution STFT output (512/1024/2048) + spectral-convergence + **frequency-transport (1-D Wasserstein) loss** (places sidebands at the right bin) + **EMA**. Loud-bin gap-closed vs mean **+28.9% → +42.0%**.
- **Batch 2 ✅ BUILT (awaiting fresh train run)** — physics core: **anti-aliased** oversampled rendering + **bounded learnable calibration** (fm depth, envelope times, per-segment curvature) so the prior self-calibrates via the spectral loss. Watch the `phys` columns drop.
- **Batch 3 🟢** — preset-tokenizer + small **Transformer** conditioning encoder (best on Dexed FM in the Neural-Proxies work) replacing the MLP.
- **Batch 4 🟢** — capacity / `alpha` sweep + optional **spectrogram adversarial** polish if outputs over-smooth.
- **Batch 5 🟡 — PITCH/VELOCITY CONDITIONING** (see below). Gate: pitch-conditioned renderer must match single-pitch quality, then generalize.

**Phase B — One-shot diffusion matcher (🟡 NEXT after renderer is faithful).** Conditional diffusion / flat-vector denoiser (MLP/Transformer, NOT 2-D U-Net) over the normalized param vector, conditioned on the input audio embedding; trained with the **magnitude-weighted audio loss backpropped through the frozen renderer** + a small param-supervision nudge. Clamp modulation index (I_max≈2) for the matcher's gradient stability only.

**Phase C — Sim-to-real polish (🟡 LATER).** Short in-the-loop **CMA-ES against the real Operator** (`search.py`, population 64 = one export) to refine the matcher's seed; optional online renderer fine-tuning on fresh real renders.

## Pitch & velocity (decided 2026-06-09)
Operator timbre is **not** pitch-invariant (Fixed-freq operators + the absolute-Hz filter change with the note; velocity modulates level/FM index). Plan: **condition the model on pitch** (and velocity) rather than fight it.
- **Data (✅ collector updated):** `extensions/data-collection` now **randomizes the MIDI note (36–84) and velocity (70–127) per sample** and records them through the manifest into each sample JSON (`note`, `velocity`). One random pitch per preset = full coverage at the **same dataset size** (no N× blow-up); it also adds preset diversity. Existing 47k C3 samples stay valid as the `note=60` examples.
- **Existing dataset:** **no metadata migration needed** — the loader will default a missing `note` to 60 (C3) / `velocity` 100. New data carries the fields.
- **Model (Batch 5, not yet built):** thread `f0` (and velocity) through `DiffOperator` (drop the hardcoded 261.63; the physics already scales every frequency by f0 → cheap), `RendererConfig`, the dataset loader, and the conditioning encoder. At inference, detect input f0 (YIN/autocorrelation on monophonic synth sounds) and render/match at that pitch.
- **Recommended additional samples:** ~**50k** random-pitch renders (≈ doubles the set; ~1k/semitone over 36–84), targeting ~100k total. ~25k is a usable minimum; up to ~100k new if we want C3 to be a clear minority. Move this to the 3090 box together at Batch 5.

## Infra
2×3090 + NVLink box for training. `train_renderer.py` does **DDP auto-spawn** (NCCL on Linux, gloo on Windows); torch pinned **cu124** (matches the box's CUDA 12.4 driver; also runs on the 4070). `inspect_renderer.py` is the renderer go/no-go. Source of truth = `origin/dev` (user handles all git).

---

## 0. TL;DR of where this stands  *(history below — see CURRENT PLAN above)*

Two research passes are done: (a) the **Ableton Extensions SDK** is mapped and the API verified against source; (b) the **synth sound-matching ML literature** is summarized. Two things dominate the design:

1. **The hard constraint (🔴):** The SDK can only render **AudioTracks** to WAV (`renderPreFxAudio`). Instruments live on **MIDI tracks**, and there is **no freeze/flatten/bounce/resample/record/routing/transport API**. So "render Operator to audio inside the extension" is *not* a documented capability. **Everything about Tool 1 hinges on resolving this** → see Phase 0.
2. **The ML path is well-trodden (🟢):** offline-render a dataset of `(random params → audio)` pairs once, then train a **Mel-spectrogram → parameter** network on GPU. The synth is never in the training loop. This is exactly InverSynth / Spiegelib / Sound2Synth / the 2024 Audio-Spectrogram-Transformer (NI) paper. The real bottleneck is **dataset generation**, not training.

---

## 1. Architecture overview

```
┌─────────────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────────┐
│ TOOL 1: Data Collection     │     │ TOOL 2: Training         │     │ TOOL 3: Inference Extension │
│ (Ableton Extension, TS)     │ →   │ (Python + uv)            │ →   │ (Ableton Extension, TS)     │
│                             │ wav │                          │ .onnx│                             │
│ random params → render →    │ +   │ audio → mel-spec → CNN/  │ +    │ input sample → mel-spec →   │
│ save WAV + params JSON       │json │ transformer → params     │ meta │ model → set Operator params │
└─────────────────────────────┘     └──────────────────────────┘     └─────────────────────────────┘
        dataset/                          models/operator/                  runs in Live
```

**Shared contracts (the glue between tools):**
- **Parameter schema** (`schemas/operator.json`): the ordered list of Operator parameters we model, each with `{name, kind: continuous|categorical, min, max, valueItems?}`. Produced by Tool 1 (read live from the device), consumed by Tools 2 & 3. This is what makes the system generalize to new instruments: a new instrument = a new schema + a new model.
- **Audio format contract:** fixed sample rate (e.g. 16 kHz mono — InverSynth used 16384 Hz; NI used higher), fixed analysis window (≤ 5 s, zero-padded), identical mel-spectrogram params at train and inference. Define once in a shared config, mirror it exactly in TS and Python (a common bug source).
- **Model bundle** (`models/operator/`): `model.onnx` + `meta.json` (schema, audio config, normalization stats). Tool 3 discovers instruments by scanning this folder.

---

## 1b. PIVOT (2026-06-04): rendering reality & the chosen data path

**Phase 0 verdict (confirmed in Live 12.4.5b3):** `renderPreFxAudio` **rejects MIDI tracks** (throws), renders **AudioTracks only**. Broader research confirmed **no Ableton automation API renders an instrument offline** (Extensions SDK, AbletonOSC, LOM, Max for Live all lack render/export/freeze-trigger). Realtime recording is the only API capture path (~1×, days-to-weeks). Operator is **not a VST/AU**, so DawDreamer/pedalboard can't host it.

**User decision:** must use the REAL Operator — **no Python clone** ("it will not be the same").

**Chosen data-generation path:** Ableton's GUI **Export Audio/Video renders offline, faster-than-realtime** (the one offline path; GUI-only). So:
- Keep one Live set open with **many MIDI tracks**, each = Operator + a MIDI note clip (named `op_0000…` so files are identifiable).
- Programmatically **randomize all tracks' params** each round (Extensions SDK or AbletonOSC), log track→params.
- **Export Audio with "Rendered Track: All Individual Tracks"** → one WAV per track per export (N samples/export, each soloed, no bleed).
- The export *trigger* is GUI-only → automate with **pywinauto** (Python) or **AutoHotkey** (NOT Playwright — web only).
- Split/rename per-track WAVs, pair with logged params → dataset. Re-randomize same tracks, re-export, loop.
- Throughput ≈ (total audio secs ÷ offline speedup ~2–10×) + dialog overhead → ~100k in ~a day, full dataset in ~2–4 days unattended. *Offline speedup is machine-specific — measure with a manual export test first.*

**Operator output spec (from the live `schemas/operator.json` dump): 195 params = 53 categorical + 142 continuous.** Multi-class heads: Algorithm(11), 4×Wave(23), 7×EnvMode(5), Pe/LFO Dst(24), LFO Type(7), LFO Range(3), Filter Type(5), Filter Circuit(5), Shaper Type(5); 34 binary toggles; 142 continuous (per-param min/max normalization — ranges vary widely).

**Open: orchestration substrate** — (A) AbletonOSC + Python single master (recommended; no UI conflict; +1 install) vs (B) Extensions SDK sets params + Python exports (reuses working SDK; possible UI contention). Decide after the manual export test.

## 2. Phase 0 — Validate the rendering path 🟢 DONE (verdict above)
*(historical — kept for context)*

Nothing else in Tool 1 matters until we know how we can render Operator. Build a throwaway 30-line extension that:
1. Creates a MIDI track, `insertDevice("Operator", 0)`.
2. Creates a MIDI clip with one note (C3, a couple beats).
3. Calls `renderPreFxAudio(midiTrack, 0, N)` and logs whether it returns a WAV with non-silent audio.

**Outcomes:**
- **(A) It works on MIDI tracks** → 🎉 cleanest path; Tool 1 is a pure extension. Proceed with the design in §3.
- **(B) It rejects MIDI tracks** → fall back to one of:
  - **(B1) External Live automation:** drive Live via **AbletonOSC** (open-source remote script exposing the Live Object Model over OSC) or **Max for Live** to set params + **Freeze/Flatten** + locate the rendered audio. Heavier, but uses the *real* Operator. Likely a Python orchestrator.
  - **(B2) Reimplement Operator's FM engine in Python:** fast, batchable, even differentiable; but fidelity risk — predicted params must sound right in the *real* Operator, so any divergence between our clone and Live's Operator hurts. Contradicts the "use Ableton's instrument" goal; keep as last resort.

> **Decision needed (Q1):** preferred fallback if Phase 0 outcome is (B). Until then the plan assumes (A)/(B1) — i.e. render the real Operator.

Also worth a quick test in Phase 0: **render speed** (wall-clock per render). At ~10⁵–10⁶ samples this determines whether the dataset is generated in hours, days, or is infeasible and forces parallel Live instances / sampling fewer params.

---

## 3. Tool 1 — Data Collection 🟢 MVP BUILT (manual-export loop; pywinauto next)

**Built & verified (2026-06-04):**
- `extensions/data-collection/` (TS): commands "Build Operator Rack" (creates NUM_TRACKS `op_####` tracks w/ Operator + C3 note) and "Randomize Batch" (randomizes all 195 params in one transaction, writes `dataset/pending/<batchId>/manifest.json`). Builds & packages clean.
- `src/doppelganger/` (Python, uv): `schema.py` (loads `schemas/operator.json`; normalize/denormalize verified), `audio.py` (soundfile WAV reader), `datagen/split_export.py` (`inspect` + `process` — pairs exported WAVs with manifest, filters silent renders). End-to-end `process` tested on real export data ✅.

**Verified in Live (2026-06-04):** rack of 64 builds & names correctly; randomize+export ≈ **12 s for 64 samples (~5/sec)**; pitch constraints confirmed applied (Transpose ±12, A Coarse ~1–4, Fixed off); audible **yield ~73%** (RMS≥0.01). Sampler now constrains audibility-gating + carrier-pitch params (`config.ts SAMPLING_RULES`), Python filters remaining quiet renders by **RMS**, and `process` **auto-detects the export prefix** (robust to manifest/folder mismatch).

**Current loop (semi-manual):** Build Rack → Randomize Batch → (manually) Export "All Individual Tracks" w/ the dialog's prefix into the batch dir → `uv run python -m doppelganger.datagen.split_export process <batchDir>`.
**Export automation WORKS (2026-06-04):** `export_ableton.py` drives Live's Export via keyboard + clipboard — Live's Export dialog is custom-drawn (not pywinauto-introspectable), so: focus Live → `Ctrl+Shift+R` → `Enter` (default Export button) → native Save dialog (filename focused) → `Ctrl+A` + paste full path via clipboard → `Enter` → poll for files. Verified end-to-end (67 files → 48 examples).

**Unattended loop BUILT:** file handshake in `dataset/pending/<batchId>/`. SDK "Auto Collect" command loops: randomize → manifest + `READY` → wait for `DONE` (no modal, so Live stays free) → repeat (MAX_BATCHES, or `dataset/STOP`). Python `datagen/run_loop.py` watches: `READY` → `export_batch` → `process_pending` → `DONE`. ONE unverified assumption: that Live accepts the watcher's export keystrokes while the SDK Auto-Collect command is mid-loop (no modal shown). If that fails, fallback = AbletonOSC sets params from Python (rack already built by SDK), making Python the single master with no keystroke contention.

### (original Tool 1 design notes below)
## 3a. Tool 1 design notes 🟡

**Goal:** generate `N` examples of `(Operator parameter setting → rendered audio)` and save them in an organized dataset.

**Flow per sample:**
1. Once at start: create MIDI track + Operator; read `device.parameters` to build/emit the **parameter schema**.
2. For each of N iterations:
   - Sample a random value for each modeled parameter (continuous: uniform in `[min,max]` or smarter sampling; categorical: random `valueItems` index). `await param.setValue(...)`.
   - Ensure a MIDI note clip exists for the render window (fixed pitch/velocity/duration to start; can randomize later).
   - `renderPreFxAudio(track, 0, windowBeats)` → WAV path → copy/rename into `dataset/operator/wav/000123.wav`.
   - Write `dataset/operator/params/000123.json` (raw values + normalized vector).
   - Update progress dialog; honor abort.

**Design decisions baked in:**
- **Parameter scope (Q2, 🔴):** start with a **curated subset** (recommended) or the **full Operator param set**. Full Operator with 4 oscillators is ~150+ params (à la Dexed's 155 in Sound2Synth) — learnable but data-hungry and FM-pitch is the documented hard case. A curated v1 (e.g. 1–2 oscillators + envelopes + filter + global, ~30–50 params) de-risks the whole pipeline and proves the concept faster.
- **Sampling strategy:** pure-random under-covers *musical* sounds (Sound2Synth & Neural-Proxies both report this). v1 = uniform random; v1.5 = seed from factory presets + perturb. (Preset loading isn't in the API, so "seeding" means hand-encoding a few presets' param vectors, or harvesting from `.adv` files offline.)
- **Note content:** fixed single note (C3, vel 100, sustained then released within the window) like the literature. Randomizing pitch later helps generalization to pitched input.
- **Determinism/resumability:** store the RNG seed and an index; allow resuming a partial run.

**Code shape (clean, chunked):** `src/extension.ts` (command/UI wiring) · `src/operator/params.ts` (read schema, sample, set) · `src/dataset/writer.ts` (paths, WAV copy, JSON) · `src/render.ts` (render+window helpers) · `src/config.ts` (shared audio/render constants). Heavy explanatory comments (user is new to TS/Ableton).

**Outputs:** `dataset/operator/{wav,params}/` + `schemas/operator.json` + a `dataset/operator/manifest.json` (counts, audio config, seed).

---

## 4. Tool 2 — Training Pipeline (Python + uv) 🟢 BUILT & ITERATING

**Stack:** `uv` project, **PyTorch (CUDA `cu128`)**. Train on GPU, deploy ONNX on CPU (independent). Hardware: RTX 4070 (dev) + 2×3090 (big runs / DDP later). Modules under `src/doppelganger/training/`: `config.py`, `features.py`, `codec.py`, `data.py`, `model.py`, `train.py`, `eval.py`, `diagnose.py`, `hear_it.py`.

**Input:** raw audio → **log-mel computed *inside the model*** (`features.LogMel`: torch.stft + hand-built mel filterbank) so ONNX export needs no JS mel reimplementation. 16 kHz mono, 2.0 s fixed window, n_fft 1024 / hop 256 / 128 mels. Audio is **peak-normalized** (match timbre, not loudness) and the spectrogram is **per-sample standardized** in-graph.

**Model (`model.SoundMatcher`):** **ResNet-18-style backbone** (stem + 4 stages 64→512, 2×2 adaptive pool, 768-dim embedding + dropout) → three head groups (`codec.ParamCodec`):
- **Continuous (142):** **binned classification, K=64 bins**, cross-entropy vs **Gaussian soft labels** (σ≈1.5 bins) — the literature's key trick (InverSynth, Sound2Synth); fixes the regression-to-mean we measured. Decode = argmax bin → value.
- **Binary (34):** BCE-with-logits.
- **Categorical (19):** softmax CE + light label smoothing (0.05).

**Loss research (done; `docs/` + memory):** parameter-space only (Operator non-differentiable). Adopted #1–#2 below; #3–#5 are the roadmap, ranked by expected impact:
1. ✅ binned + Gaussian soft labels for continuous.  2. ✅ CE+smoothing categorical / BCE binary.
3. **Audio-metric model selection** — `hear_it.py` re-renders predictions in real Operator, log-mel L1 vs shuffled baseline (param accuracy ≠ perceptual; *measure by sound*).  4. **Gating-mask** inaudible params (e.g. Osc-B params when Osc-B off) — cheap, high precision.  5. **Neural proxy** (Combes 2025) — differentiable surrogate trained on our renders to add a perceptual loss without a clone; best at 50k+ samples.

**Findings so far:** see [[training-findings]]. Param accuracy is data/signal-limited (bigger model didn't move val), but the **audio metric is positive**: predicted-renders score **ratio 0.72** (matched/shuffled log-mel) at ~5k samples + plain CNN — the model captures real timbre. Now scaling: **20k samples + ResNet + binned loss** (training as of 2026-06-05); measure the new ratio.

**Monitoring:** TensorBoard (`uv run tensorboard --logdir runs`) — logs train+val cont_mae/bin_acc/cat_acc/loss/LR per run.

**Known gap:** trains on fixed-C3 2 s one-shots; real inference audio is arbitrary pitch/content (out-of-distribution). Fix = **richer stimuli** (vary pitch/velocity, capture release) in a future data run.

**Still to build:** `export_onnx.py` (raw-audio-in ONNX, opset ≥17 for stft) + `meta.json`; gating-mask; DDP for 2×3090.

---

## 5. Tool 3 — Inference Extension (TS) 🟡

**Goal:** user selects an audio sample + target instrument → predict params → apply them to an Operator instance.

**Getting the input audio:** the sample is on an **AudioTrack/clip**, so `renderPreFxAudio` (or reading the sample file) *does* work here — the §2 limitation only blocks rendering *instruments*, not reading user audio. Decode with `audio-decode`.

**Running the model on CPU (Q3, 🔴):**
- **(Recommended) `onnxruntime-node`** inside the extension → fully self-contained, no Python required by end users → best for distribution. Mel-spectrogram computed in TS (must byte-match the Python preprocessing).
- **(Alt) Python sidecar** the extension shells out to → reuses Tool 2 code exactly, but end users need Python. Worse for distribution.

**Apply predicted params:** denormalize each output head → `param.setValue(...)` on the live Operator (continuous: scale to `[min,max]`; categorical: argmax → `valueItems` index). Wrap in `withinTransaction` for one undo step.

**Instrument discovery:** scan `models/` for subfolders containing `model.onnx` + `meta.json`; populate a dropdown. Adding an instrument later = drop in a new model bundle, no code change.

**UI (modal HTML webview):** v1 barebones (button + "apply"). Nice-to-haves (feasible in the webview, which runs arbitrary JS): waveform view + mel-spectrogram view of the input, instrument dropdown, maybe top-k preset suggestions.

---

## 6. Repository layout (proposed)

```
doppelganger/
  docs/                       PLAN.md (this), research notes, learning guides
  .claude/skills/ableton-extensions-sdk/SKILL.md   (SDK reference — already created)
  schemas/                    operator.json (param schema, shared contract)
  extensions/
    data-collection/          Tool 1 (TS)
    inference/                Tool 3 (TS)
  training/                   Tool 2 (Python, uv)
  models/operator/            model.onnx + meta.json (built by Tool 2)
  dataset/operator/           wav/ + params/ (built by Tool 1; gitignored)
```

---

## 7. Build sequence

1. **Phase 0** — rendering validation spike (§2). 🔴 gate.
2. **Tool 1** — data collection; produce a small dataset (e.g. 1–5k) + the param schema. Verify audio sounds like Operator and params round-trip.
3. **Tool 2** — training on the small set first (overfit-a-batch sanity → small real run) → ONNX export → MFCC-distance eval.
4. **Tool 3** — inference extension; close the loop on a few held-out renders (predict → apply → re-render → compare).
5. **Scale up** dataset & model; add preset-seeding, perceptual/proxy loss, more params, more instruments.

---

## 8. Decisions (locked 2026-06-04)

- **Q1 — Rendering fallback:** ✅ **Test first (Phase 0), then AbletonOSC** if MIDI tracks can't be rendered. (Python Operator reimplementation is off the table for now.)
- **Q2 — v1 parameter scope:** ✅ **Full Operator (~150+ params)** — all 4 oscillators, LFO, filter, pitch env, routing. Implication: data-hungry (plan ~100k–1M renders), categorical heads for waveforms/algorithm, expect FM pitch to be the hard case. Dataset generation throughput (measured in Phase 0) becomes critical.
- **Q3 — Inference runtime:** ✅ **`onnxruntime-node` in-extension** — fully self-contained, no Python for end users.

### Phase 0 status: 🟢 built, awaiting an in-Live run
The validation spike lives at `extensions/phase0-render-test/`. It compiles and bundles cleanly against the real SDK. It must be **run inside Ableton** (cannot be automated here) to determine outcome (A) vs (B) in §2.

---

## 9. Key risks & honest expectations

- **Rendering instruments may require leaving the SDK** (the #1 risk). Mitigation: Phase 0.
- **Arbitrary 5 s real audio is out-of-domain.** Published systems shine on single-note, in-domain tests; real samples (polyphonic, pitched-moving) are harder, and **FM pitch is the documented hard case** (DiffMoog, NI-AST). Set expectations accordingly; consider restricting v1 demos to one-shot/tonal samples.
- **Synplant's Genopatch iterates against its *own* fast synth at runtime** — a luxury we lack with Operator. Our path is offline dataset + feed-forward prediction; invest in dataset coverage + (later) a neural proxy rather than runtime search.
- **Param-loss ≠ sound match.** Mitigate with binned outputs + label smoothing, and the v2 neural-proxy perceptual loss.

---

## 10. References (from research pass)

InverSynth (Barkan 2018, arXiv:1812.06349) · Spiegelib (Shier, AES 2020, github.com/spiegelib/spiegelib) · Sound2Synth (Chen 2022, arXiv:2205.03043) · Audio Spectrogram Transformers for sound matching (Bruford/NI, DAFx 2024, arXiv:2407.16643) · Flow Synthesizer (Esling 2019/2020) · DDSP (Engel, ICLR 2020) · DiffMoog (Uzrad 2024, arXiv:2401.12570) · Neural Proxies for Sound Synthesizers (Combes 2024, arXiv:2509.07635) · rendering tools if Operator-as-VST were possible: pedalboard / DawDreamer / RenderMan (note: Operator is Ableton-only, likely not loadable in these).
