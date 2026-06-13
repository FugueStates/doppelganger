# Batch-5 Refactor (2026-06-12) — what changed and why

This documents the architecture-review-driven refactor of the neural-renderer stack
(`src/doppelganger/synth/`). Every change is listed as **old → new, with the reasoning**.
The review found the locked Batch-3 architecture (physics prior + FiLM residual +
preset-tokenizer Transformer) sound — nothing here changes the model's *shape*; it fixes
the things that would have blocked Phase B (the matcher), silently corrupted the Batch-5
training run, or wasted GPU time.

> ⚠ **Checkpoint compatibility:** the new default config (`condition_pitch=True`) adds
> pitch/velocity encoder parameters, so the next training run starts FROM SCRATCH (the
> established per-batch precedent). Pre-refactor checkpoints still load everywhere:
> `inspect_renderer` injects `condition_pitch=False` for checkpoints that predate the
> field, and all new constant buffers are registered `persistent=False` so they never
> appear in (or demand from) a state dict.

---

## 1. The renderer is now differentiable w.r.t. its INPUT (the matcher blocker)

**Old:** `HybridRenderer.forward(list[dict])` → `to_controls()` / `_param_vector()` built
control tensors with Python loops over `float(p.get(...))`.

**New:** `forward` accepts the **normalized param matrix `[B, 195]` as a tensor**;
`adapter.ControlMap` derives the physics controls (coarse / level / ADSR / transpose)
with differentiable slicing + affine denormalization. The conditioning encoder consumes
the same tensor. The dict path still exists for offline tools (search, smoke tests).

**Why:** Phase B trains a matcher by backpropagating the spectral loss **through the
frozen renderer into a predicted param tensor**. `float()` severs autograd — the old
interface made the entire Phase-B plan impossible without a rewrite, and the rewrite
touches exactly the code the f0 refactor touches, so they were done as one change.
Verified in the smoke test: `d(loss)/d(params)` is non-zero on the physics columns
(coarse/level/ADSR/transpose) as well as the encoder path.

**Still discrete (matcher must handle separately):** `Algorithm` (hard routing
selection) and `Osc-X On` (hard gate; gradient flows through the *level* instead).
Options when the matcher is built: classify algorithm separately, condition per
algorithm, or a straight-through estimator.

## 2. Pitch + velocity conditioning (Batch 5 model work)

**Old:** physics rendered everything at a hardcoded `f0 = 261.63` (C3); the encoder saw
only the param vector; the dataset loader read `note` but never used it.

**New:**
- `forward(..., note, velocity)` — per-sample tensors. Physics f0 =
  `midi_to_hz(note) * 2^(Transpose/12)`; `DiffOperator.render` takes a per-sample `[B]`
  f0 (and per-sample algorithm — see §4).
- `PresetTransformerConditioner` gains **two extra tokens** (note, velocity) using the
  same Fourier value-lift plus their own ID embeddings — the exact slot the tokenizer
  design reserved. `MlpConditioner` concatenates the two values. Both are gated by
  `RendererConfig.condition_pitch` (default **True**).
- Velocity reaches **only the encoder/residual**, not the physics: Operator's
  velocity→level/FM-index mapping is undocumented secret sauce, i.e. residual territory.

**Why:** Operator timbre is not pitch-invariant (decided 2026-06-09), the ~50k
pitch-diverse samples are being collected now, and a C3-only renderer can only ever
match C3 input.

## 3. The pitch guard + the stable val split (data-safety before the Batch-5 merge)

- **Pitch guard.** Old: nothing stopped a training run on the mixed (C3 + random-pitch)
  dataset with a fixed-f0 model — ~20% of targets would have been at the wrong pitch:
  pure label noise, silently. New: `RendererDataset.restrict_to_note(60)` runs whenever
  the model is not pitch-conditioned (`--no-condition-pitch` ablations, old configs).
- **Hash split.** Old: `random_split(seed=0)` over the *current dataset length* — adding
  Batch-5 data would have re-dealt train/val membership entirely (old val samples
  becoming train), quietly invalidating cross-version fidelity comparisons. New:
  `crc32(sample_id) % 10 == 0 → val`. An id's membership never changes as data grows.
  **One-time discontinuity:** numbers from before/after the split change aren't strictly
  comparable; everything after is stable forever.

## 4. Physics core (`diff_operator.py`)

- **Vectorized across algorithms.** Old: the batch was grouped per algorithm in Python
  (`_render_physics`), so a batch of 16 with ~10 distinct algorithms became ~10 sub-renders
  of 1–2 samples — hundreds of tiny kernel launches at terrible GPU occupancy. New: every
  edge in all 11 routings goes (higher op index → lower), so **D→C→B→A is a valid
  topological order for every algorithm** (asserted at import). One batched pass with
  per-sample adjacency masks (`adj[algo]`, `carriers[algo]`) renders mixed algorithms
  together. This also removes the data-dependent Python control flow that broke
  `torch.compile`, and pays again in Phase B (the matcher renders *changing* params every
  step, so the freeze-calib waveform precompute can't help there).
- **Transpose mapped.** Old: ignored — yet data collection randomizes it ±12 semitones,
  so the prior's fundamental was up to 2× off for most samples. New: one line in the
  control map (`f0_eff = f0 * 2^(semitones/12)`). (Batch 2's lesson — better prior ≠
  better hybrid at fixed C3 — noted; but with f0 now *variable*, prior frequency
  correctness is part of the conditioning signal, and it's free.)
- **fp64 phase wrap.** Old: `sin(2π·f0·coarse·t)` in fp32 — at the 3 s tail with ratio 48
  the phase is ~2.4e5 rad where fp32 resolution is ~0.03 rad → a broadband noise floor
  baked into the prior. New: phase *cycles* computed in fp64, wrapped with
  `remainder(·, 1)`, cast to fp32 (elementwise fp64 is bandwidth-bound; cost ~0).
- **Strided decimation.** Old: full-rate anti-alias conv then `[::os]` slicing. New:
  `conv1d(..., stride=os)` — identical output, ~os× less work.
- **`volume` removed.** It was applied *before* peak normalization, which cancels a
  global scale exactly: zero effect on the output, identically-zero gradient. A dead knob
  that wasted a control slot and implied a sensitivity that didn't exist.
- **Fine / Fix On / Quantize still unmodeled — deliberately.** Their raw-value→Hz
  semantics are unverified guesses; the freq-map tool (§8) measures them so they can be
  added as *facts*. (If `Fine` turns out to be `ratio += fine/1000`, that's a one-liner
  in `ControlMap`.)

## 5. Trainer (`train_renderer.py`)

- **Tensor batches end-to-end.** Old: the collate produced `list[dict]` and the model
  re-ran the B×195 Python normalization loop *every step*. New: the dataset yields the
  normalized fp32 vector + fp16 waveform + note + velocity; batches go straight to GPU.
- **fp16 on the wire.** Waveforms stay fp16 through pinned-memory H2D transfer and are
  cast to fp32 on the GPU — half the transfer volume, same numerics (the cache is fp16
  anyway).
- **`mean_spec` reuse.** Old: the predict-the-mean baseline was recomputed over the FULL
  train set at every launch, even on `--resume` (it was already saved in the checkpoint).
  New: `--resume` loads it from the checkpoint.
- **Eval renders physics once.** Old: `evaluate()` called `forward` twice (hybrid +
  physics-only) → two physics renders per batch. New: `forward(..., return_physics=True)`
  returns both prediction sets from one render.
- **Snake activation algebra.** `x + sin²(ax)/a` → `x + (1−cos(2ax))/(2a)` — same
  function, same parameter, one upcast trig intermediate instead of two. This activation's
  fp32 upcasts under bf16 autocast are what forced gradient checkpointing at batch 16;
  this shaves the peak. (Same params ⇒ old EMA weights still load.)

## 6. Cache v2 (`precompute.py`)

**Old:** `meta.json` held the raw param dicts — at the 100k target that's ~400+ MB of
JSON, parsed and held as Python dicts by EVERY process (2 DDP ranks × (main + workers)
under Windows spawn ≈ 10 copies, gigabytes of RAM), then re-normalized per step.

**New:** `audio.npy` (fp16 memmap, unchanged) + `params.npy` (`[N,195]` fp32 normalized
— the shared encoding for encoder/ControlMap/matcher) + `notes.npy` + `vels.npy`, and a
small `meta.json` (ids + shapes + `version: 2`). ~45 MB at 58k samples, instant load.
The staleness check now includes the **version**, so the old cache format triggers the
live-fallback warning instead of a crash. **Rerun `synth.precompute` after adding data`**
(unchanged rule).

> **OOM fix (2026-06-13).** All four cache arrays are now opened as memmaps **lazily,
> per process**, and stripped from the Dataset pickle via `__getstate__`. The first
> Batch-5 run on the 2×3090 box SIGKILL'd at startup: under DDP's spawn start method the
> DataLoader pickles the Dataset to each worker, and an **open `np.memmap` held as an
> attribute serializes its ENTIRE file into the worker** — at ~100k samples that's a
> ~10 GB private copy per worker per rank, blowing past system RAM (the OOM killer, not
> CUDA — CUDA OOM is a Python exception). The same code survived at 47k samples; doubling
> the dataset crossed the RAM ceiling. Lazy-open + `__getstate__` means each worker mmaps
> the files itself and shares pages through the OS page cache (~0 extra RAM), which is
> what the "one copy shared across ranks" claim always intended.

## 7. `inspect_renderer.py`

- Reports overall / energy-weighted / loud-bin metrics **at all three STFT resolutions**,
  not just the canonical 1024: sideband *placement* (what the freq-transport loss exists
  for) is sharpest at 2048; envelope/transient shape at 512. The canonical loud-bin gap
  stays the headline go/no-go number.
- Loads pre-refactor checkpoints via `_LEGACY_CFG_DEFAULTS` (missing config fields
  default to the OLD architecture, not the new defaults).
- Uses the same hash split + pitch guard as training.

## 8. Frequency-mapping verification tool (new)

The physics prior assumes raw `Coarse` *is* the FM ratio and ignores `Fine`/`Fix`/
`Quantize`. Nobody ever measured whether that's true (the old `coarse_ratio` calibration
sweep scoring a poor 1.37 suggests it isn't, exactly). New tooling measures the real
device's raw-value→frequency laws end-to-end:

- **Python:** `synth/freq_map.py` —
  `gen` writes seven 64-track sweep manifests (`dataset/sweeps/fmap_*`) and watches/
  auto-exports (same handshake as `run_sweeps`):
  | experiment | question it answers |
  |---|---|
  | `fmap_coarse`, `fmap_coarse_n72` | is raw Coarse the literal ratio? quantized? note-proportional? |
  | `fmap_fine_c1`, `fmap_fine_c2` | Fine: additive (`ratio += fine/1000`) or multiplicative? |
  | `fmap_fixed`, `fmap_fixed_n72` | Fix Freq × Mul → Hz law; truly note-independent? |
  | `fmap_quant` | what Quantize snaps Coarse to |

  `analyze` measures each WAV's dominant frequency at **native sample rate** (steady
  0.3–1.3 s window, Hann, 8× zero-padded rFFT, parabolic peak interp ≈ sub-0.1 Hz),
  fits candidate models (identity vs round vs snap-set; linear vs exponential; octave
  and note-independence checks in cents), prints verdict tables and writes
  `dataset/freq_maps_report.json`.
- **Extension:** new command **"doppelganger: Collect Freq Maps"** — like Run Sweeps,
  but first **pins every rack clip to the manifest's declared note/velocity**
  (`operator.ts pinNotes`). This matters: after Auto Collect the clips hold *random*
  pitches, which would make every measurement meaningless. "Run Sweeps" now skips
  `kind: "freq_map"` manifests (and `calibrate.py` ignores them), so the two flows
  can't contaminate each other.

Workflow: `uv run python -m doppelganger.synth.freq_map gen` → in Live:
*Collect Freq Maps* → `uv run python -m doppelganger.synth.freq_map analyze`.

### 8b. MEASURED RESULTS (2026-06-12) — now encoded in the prior

The first collection ran successfully (after fixing `find_live()`: the UIA window scan
could take minutes with many windows open and looked like a hang before any keystroke;
it now uses instant win32 enumeration). The laws, all with ~zero residual:

| param | measured law | old assumption |
|---|---|---|
| Coarse | **ratio = max(floor(raw), 0.5)** — floor, NOT round (raw 3.81 → 3); 0.0 mean cents at both octaves; note-proportional (octave check 0.0 cents) | raw fed in as the literal ratio (up to a whole ratio unit off) |
| Fine | **multiplicative**: freq ×= (1 + fine/1000) — slope exactly doubled at Coarse 2 | ignored |
| Fixed | **Hz = 10^(step(mul) − 3) × 200^fixfreq**, note-independent (0.0 cents C3 vs C4), exact exponential (residual 0.0000 decades) | ignored (rendered as ratio osc at f0×coarse — wholesale wrong) |
| Quantize | **no effect** on coarse→ratio (identical floor mapping) | unknown |

These are implemented in `ControlMap` (with straight-through floors so the matcher keeps
gradients) and `DiffOperator.render(freq_hz=...)` (per-op absolute Hz; fixed ops bypass
f0 entirely). Notes: (a) Mul's quantizer is **floor, confirmed** by the follow-up
`fmap_fixedmul` sweep — step transitions measured at raw ≈ 4.0 and 5.0 (0.024 from the
integers vs 0.476 from half-integers), consistent with Coarse. (b) The C4 coarse
sweep's top points alias above ~20 kHz in the 44.1k render (ratio 45 @ C4 folded to
exactly 44100−f) — the analyzer excludes those, and it's a live demonstration of why
the physics core's anti-aliasing matters once pitch varies.

---

## Not changed (and why)

- **Loss functions / metrics / EMA / warmup+clip / DDP structure** — all validated by
  the batch studies; no reason to touch.
- **`weight_norm` deprecation warning** — migrating to `parametrizations.weight_norm`
  renames state-dict keys and would break every existing checkpoint for zero functional
  gain. Revisit only when checkpoints are being retired anyway.
- **Caching target STFTs** — ~580 KB/sample × 100k ≈ 58 GB; GPU STFT is cheap. Not worth it.
- **Physics-waveform precompute (freeze-calib)** — still on the table for *renderer*
  training once calibration has converged, but it cannot help the matcher (changing
  params), which is why the vectorization in §4 was prioritized instead.

## Suggested next steps

1. Run the freq-map collection (one Live session, ~10 min) and `analyze`; fold verified
   `Fine`/`Fix`/`Coarse` mappings into `ControlMap`.
2. Move the new pitch-diverse data + `synth.precompute` to the 3090 box; retrain from
   scratch (`--epochs 80`); gate on `inspect_renderer` — pitch-conditioned must match the
   C3-only +50% loud-bin gap on the note-60 slice AND hold up across pitch.
3. Then Phase B: the matcher now has the differentiable tensor interface it needs; decide
   the discrete-algorithm strategy before building it.
