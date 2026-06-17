# doppelganger

ML synth sound-matching for Ableton Live's **Operator**: feed in an audio sample and get
the closest-matching Operator preset. Inspired by Synplant 2's Genopatch, but it programs
the **real** Ableton Operator (no clone).

It has three parts:
1. **Data collection** — an Ableton Extension builds many Operator tracks with random
   params (and now random note + velocity per sample); Ableton's offline multitrack
   **Export Audio** renders them; Python splits the export into `(audio, params)` pairs.
2. **Training (audio-objective)** — the current approach optimizes for **sound**, not
   parameter values (direct param regression was proven to fail for FM). A **neural
   renderer** (`synth/`) learns `params → spectrogram` from real-Operator audio; once
   faithful it provides a differentiable **audio loss** to train a one-shot **diffusion
   matcher** (`input audio → params`). See `docs/PLAN.md` → *CURRENT PLAN (v3)*.
3. **Inference** *(in progress)* — an Ableton Extension runs the matcher and applies the
   predicted preset to a selected Operator, optionally refined by an in-the-loop
   CMA-ES search against the real Operator.

> **Note:** the original `training/` param-regression pipeline (log-mel → ResNet → param
> heads) is **deprecated** — kept for reference only. The live work is in `synth/`.

See [docs/PLAN.md](docs/PLAN.md) for the full design and rationale, and
[.claude/skills/ableton-extensions-sdk/SKILL.md](.claude/skills/ableton-extensions-sdk/SKILL.md)
for the SDK reference.

---

## Prerequisites

- **Ableton Live 12 Beta** with **Extensions** support, and **Developer Mode** enabled
  (Preferences → Extensions) for the dev/hot-load workflow.
- **Node.js ≥ 22** (for the extensions).
- **uv** (Python package manager) — Python 3.12 is pinned automatically.
- **Windows** — the export automation uses `pywinauto` (Windows-only).
- **NVIDIA GPU** (optional, for training). Training uses CUDA; inference is CPU-only.

The Ableton Extensions SDK tarballs live at `../abletonExtensionsSDK/` (referenced by the
extensions' `package.json`).

---

## Repository layout

```
doppelganger/
  docs/PLAN.md                         design + decisions
  schemas/operator.json                the 195-param schema (shared contract)
  extensions/
    phase0-render-test/                spike: render test + "Dump Operator Schema"
    data-collection/                   Tool 1 extension (build rack, randomize, auto-collect)
  src/doppelganger/                    Python package (uv)
    schema.py  audio.py
    datagen/   split_export.py  export_ableton.py  run_loop.py
    synth/     renderer.py train_renderer.py inspect_renderer.py   ← Tool 2 (neural renderer)
               diff_operator.py spectral.py adapter.py precompute.py
               freq_map.py sensitivity.py search.py sweeps.py run_sweeps.py calibrate.py
    matcher/   model.py diffusion.py train_matcher.py match.py     ← Tool 2.5 (one-shot matcher)
    training/  (DEPRECATED param-regression: config/features/codec/data/model/train/eval/…)
  dataset/operator/{wav,params}/       generated dataset (gitignored; params JSON now also
                                       records `note` + `velocity` per sample)
  models/operator/renderer.pt          trained neural-renderer checkpoint
  runs/                                TensorBoard logs
```

---

## One-time setup

**Python (uv):**
```powershell
cd D:\AbletonExtensions\doppelganger
uv sync                      # base: numpy, soundfile
uv sync --extra datagen      # + pywinauto (Windows export automation)
uv sync --extra train        # + PyTorch (CUDA) + scipy + tensorboard
```

**Extensions** (per extension folder):
```powershell
cd extensions\data-collection
npm install
```
Each extension has a `.env` with `EXTENSION_HOST_PATH` pointing at Live's Extension Host
module (already set for this machine).

---

## Loading an extension into Live

Two ways:

- **Dev hot-load** (fastest iteration): with Live running + Developer Mode on:
  ```powershell
  cd extensions\data-collection
  npm start            # builds, then loads into Live's Extension Host
  ```
- **Packaged install**: build a `.ablx` and drag it into Live → Settings → Extensions:
  ```powershell
  npm run package      # writes dist\<name>.ablx
  ```

Commands appear in Live's **right-click menu on a MIDI track**.

> Debugging: extension `console.*` output + stack traces go to
> `%APPDATA%\Ableton\Live 12.x\Preferences\ExtensionHost.txt`.

---

## Tool 1 — Data collection

### Schema (already generated)
`schemas/operator.json` is produced by the **phase0-render-test** extension's
**"doppelganger: Dump Operator Schema"** command (creates an Operator, dumps all 195
params). Re-run only if Operator's parameters change.

### Manual loop (good for testing)
In Live (data-collection extension loaded), right-click a MIDI track:
1. **"doppelganger: Build Operator Rack"** — once; creates `NUM_TRACKS` (64) `op_0000…`
   tracks, each with Operator + a note clip.
2. **"doppelganger: Randomize Batch"** — randomizes all params **and the note + velocity
   per track** (pitch 36–84, velocity 70–127; see `config.ts`), writes
   `dataset/pending/<batchId>/manifest.json` (params + a `notes` map), and tells you the
   export prefix/folder. The chosen note/velocity land in each sample's params JSON.
3. **Export Audio** (`Ctrl+Shift+R`) → **Rendered Track: All Individual Tracks**, ~3 s,
   name prefix = the batch id, save into that batch folder. *(Set this once; Live
   remembers it.)*
4. Pair the WAVs with their params:
   ```powershell
   uv run python -m doppelganger.datagen.split_export process "dataset\pending\<batchId>"
   ```

Inspect an export folder without processing:
```powershell
uv run python -m doppelganger.datagen.split_export inspect "<folder>" --prefix <id> --all
```

### Automated loop (for real datasets)
Fully unattended: the extension randomizes + signals, a Python watcher exports + processes.
```powershell
# terminal 1 — extension (hot-load)
cd extensions\data-collection ; npm start
# terminal 2 — watcher
cd D:\AbletonExtensions\doppelganger
uv run python -m doppelganger.datagen.run_loop
```
Then in Live: right-click → **"doppelganger: Auto Collect"**. Keep Live frontmost and hands
off — it drives the Export dialog each batch. Runs up to `MAX_BATCHES` (1000 ≈ ~48k usable
samples, ~3–4 h). Stop early by creating `dataset\STOP`.

Standalone export automation (one folder):
```powershell
uv run python -m doppelganger.datagen.export_ableton export "<batch_folder>" --expected 64
uv run python -m doppelganger.datagen.export_ableton inspect    # dialog-control diagnostic
```

Tunables live in `extensions/data-collection/src/config.ts` (`NUM_TRACKS`, note, sampling
rules, `MAX_BATCHES`) — rebuild with `npm run package` / re-run `npm start` after edits.

---

## Tool 2 — Neural renderer (current, audio-objective)

Trains the differentiable Operator surrogate `params → multi-resolution log-mag spectrogram`
on the `(params, real-Operator audio)` dataset. Its held-out **loud-bin fidelity** is the
go/no-go for building the diffusion matcher on top.

```powershell
cd D:\AbletonExtensions\doppelganger
uv sync --extra train        # PyTorch (cu124) + scipy + tensorboard

# (recommended) pack the v2 training cache: fp16 audio memmap + normalized param matrix
# + notes/velocities — skips per-epoch WAV/JSON reads AND the per-step Python param
# loops; one copy shared across DDP ranks. RERUN after adding data (training falls back
# to live loading + warns if the cache is stale).
uv run python -m doppelganger.synth.precompute

# train; checkpoint -> models/operator/renderer.pt (EMA weights). AUTO-USES ALL GPUs (DDP):
#   2x3090 box -> one process per GPU over NVLink automatically, no extra flags.
# Pitch/velocity-conditioned by default (Batch 5): the physics renders at each sample's
# recorded note (+Transpose) and the encoder gets note+velocity tokens.
uv run python -m doppelganger.synth.train_renderer --epochs 80

#   single GPU / CPU is the same command (falls through to one process).
#   useful flags: --batch-size N (PER-GPU) --alpha 10 (loud-bin emphasis) --resume
#                 --gpus 1 (force single) --limit 80 (tiny dev run) --out path.pt
#                 --no-condition-pitch (legacy C3-only ablation; auto-filters to note=60)
```

Train/val membership is a stable per-id hash (`crc32(id) % 10`), so adding data never
reshuffles which samples are held out. The renderer's `forward` also accepts the
normalized param matrix as a **tensor** (differentiable end-to-end into the physics) —
this is the interface the matcher trains through. See
[docs/batch5-refactor.md](docs/batch5-refactor.md) for the full change log + reasoning.

**Validate fidelity** (the decisive metric — energy-weighted / top-10%-loud-bin log-mag L1
vs a predict-the-mean baseline; runs single-process, fine on CPU):
```powershell
uv run python -m doppelganger.synth.inspect_renderer --n 512 --device cpu
# look at "loud-bin gap closed vs mean" — higher = the renderer captures real sideband
# structure (the gradient the matcher will ride on), not just the average sound.
# Now reported per STFT resolution (2048 = sideband placement, 512 = envelope shape).
```

**Frequency-map verification** (measures how Operator's RAW param values map to real
frequencies — Coarse ratio law, Fine semantics, Fixed-freq law, Quantize — so the
physics prior encodes facts, not guesses):
```powershell
uv run python -m doppelganger.synth.freq_map gen        # writes manifests + watches/exports
# then in Live: right-click -> "doppelganger: Collect Freq Maps"
#   (pins every clip to each sweep's declared note first — required for measurement)
uv run python -m doppelganger.synth.freq_map analyze    # tables + fits + verdicts
```

Renderer/loss settings live in `synth/renderer.py` (`RendererConfig`: multi-res n_ffts,
oversample, f0) and `synth/spectral.py` (energy-weighted + spectral-convergence +
frequency-transport losses). The FM physics core is `synth/diff_operator.py`.

---

## Tool 2.5 — One-shot matcher (direct, audio → params)

Trains `input audio → Operator params` as a **deterministic one-shot predictor**: an audio
encoder feeds an Algorithm classifier (the one discrete param) and a parameter head that
emits the other 194 params in a single forward pass, **confined to the audible dataset
manifold**, trained with the magnitude-weighted spectral loss through the frozen renderer +
a param anchor + algorithm cross-entropy. The manifold bounding is what keeps predictions
audible — an earlier audio-loss-driven diffusion version railed params to silent/extreme
values that gamed the renderer's slack. Designed to ship in the extension as ONNX on CPU
(no synth in the loop). See [docs/matcher-design.md](docs/matcher-design.md).

```powershell
# train against the frozen renderer (needs models/operator/renderer.pt + the dataset cache)
uv run python -m doppelganger.matcher.train_matcher --epochs 60
#   eval prints val match audio loss vs a predict-the-mean-params baseline (gain% > 0 =
#   real matching) + Algorithm top-1 accuracy. Best checkpoint -> models/operator/matcher.pt
#   flags: --w-audio/--w-param/--w-algo (loss weights) --batch-size N --algo-topk N

# inference: match a target sound -> ranked candidate presets -> predict manifest
uv run python -m doppelganger.matcher.match --target "dataset\operator\wav\0000000.wav" `
    --note 60 --velocity 100 --candidates 5
# then in Live: right-click -> "doppelganger: Apply Predicted Batch" -> Export to hear them
# (op_0000 = best-ranked of the top-k-algorithm predictions)
```

The matcher predicts one preset per top-k Algorithm (`--candidates`), renders them through
the frozen **neural** renderer (CPU, no real render), and ranks by spectral loss — robust
to the weak Algorithm classifier. Pass the played `--note/--velocity` (the renderer is
pitch-conditioned); automatic f0 detection is a Tool-3 follow-up.

### In-the-loop CMA-ES search (real Operator, optional inference polish)
Refines a preset directly against the real Operator (population 64 = one export):
```powershell
uv run python -m doppelganger.synth.search --target "dataset\operator\wav\0000000.wav" --gens 40
# then in Live: right-click -> "doppelganger: Run Search"
```

---

## (Deprecated) Tool 2 v1 — param-regression

The original log-mel → ResNet → param-head pipeline under `training/` is **kept for
reference but no longer the path** (it hit a chance-level validation ceiling — many params
sound alike, and param-loss ≠ perception; see `docs/approach.md` / `docs/PLAN.md`). Its
commands (`training.train`, `training.eval`, `training.diagnose`, `training.hear_it`) still
run but are not part of the current workflow.

---

## Status

- ✅ Tool 1 (data collection) — automated; ~100k samples across notes 36–84 (C3 + pitch-diverse).
- ✅ Tool 2 (neural renderer) — **FAITHFUL, gate passed.** Trained on 2×3090 (DDP, bf16). After a
  batch-by-batch study on the loud-bin fidelity metric:
  - Batch 1 (multi-res STFT + frequency-transport loss + EMA): **+42%** loud-bin gap ✅
  - Batch 2 (anti-aliased + self-calibrating physics core): wash at C3 (residual isn't
    prior-limited; anti-aliasing is latent insurance for high-pitch data)
  - Batch 3 (**preset-tokenizer Transformer encoder**): **+50.1%** — the winning lever
    (conditioning, not capacity: a matched MLP+capacity run was a wash at +41%) ✅ **locked**
  - Efficiency: bf16 + TF32 + grad-checkpoint/LR-warmup/grad-clip + an fp16 memmap data cache
    (`synth.precompute`).
  - **Batch 5 refactor ✅ BUILT (2026-06-12,** see `docs/batch5-refactor.md`**):**
    tensor-in differentiable forward (the matcher interface), pitch/velocity conditioning
    (physics f0 from note+Transpose; encoder note/vel tokens), vectorized physics (one pass
    for mixed algorithms), fp64 phase, cache v2 (normalized param matrix), stable hash
    train/val split, pitch guard, per-resolution fidelity report. Smoke-tested end-to-end.
  - **Freq-map sweeps ✅ MEASURED (2026-06-12)** — the real Operator's frequency laws,
    now encoded in the physics prior: Coarse ratio = `max(floor(raw), 0.5)` (floor, not
    round!), Fine is multiplicative `×(1 + fine/1000)`, Fixed mode is note-independent
    `Hz = 10^(floor(mul)−3)·200^fixfreq` (exact exponential; floor confirmed by the
    boundary sweep), Quantize is a no-op for the ratio. All ~0-cent residual — the
    prior's frequency placement now matches the real device exactly.
  - **GATE PASSED (2026-06-15):** pitch-conditioned renderer trained on the full diverse
    dataset (best_val 1.19 @ ep70) — canonical loud-bin gap **+50.3%** scored across notes
    36–84, *matching* the old C3-only +50.1% while generalizing across pitch (512 → +57%,
    2048 → +47%; energy-wt win 100%). Frozen → it now provides the matcher's audio loss.
- 🔵 Tool 2.5 (one-shot matcher) — **DETERMINISTIC direct predictor, BUILT + smoke-tested**
  (2026-06-17, see `docs/matcher-design.md`). `audio → params` in one forward pass:
  AudioEncoder → Algorithm classifier + ParamHead, the 194 non-Algorithm params confined
  to the **audible dataset manifold** so predictions stay playable; trained with the
  spectral loss through the frozen renderer + param anchor + algorithm CE; inference
  predicts per top-k algorithm and ranks via the neural renderer (CPU, no real render).
  - *History:* a first audio-loss-driven **diffusion** version trained but railed params to
    silent/extreme values that gamed the renderer's slack (Volume→−∞, Transpose→+48);
    replaced by the manifold-bounded deterministic predictor above, which is also the
    simplest thing to ship as ONNX in the extension.
  - **Next:** train on the box vs the real renderer; read eval `gain%` + Algorithm acc, then
    hear `op_0000` via Apply Predicted Batch.
- ⬜ Tool 3 (inference extension) — `matcher.onnx` (+ optional `renderer.onnx` ranker) on
  CPU in-extension, + YIN pitch detection on the input. Not yet built.
- ⬜ (Dev-only, optional) CMA-ES polish against the real Operator (`synth/search.py`) — a
  power-user refinement, not part of the shipped extension (needs real renders).

See `docs/PLAN.md` → **CURRENT PLAN (v3)** for the full phased roadmap.

See [docs/PLAN.md](docs/PLAN.md) → **CURRENT PLAN (v3)** for the full roadmap (renderer
Batches 1–5, the diffusion matcher, sim-to-real polish, pitch/velocity conditioning).
