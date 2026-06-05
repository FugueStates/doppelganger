# doppelganger

ML synth sound-matching for Ableton Live's **Operator**: feed in an audio sample and get
the closest-matching Operator preset. Inspired by Synplant 2's Genopatch, but it programs
the **real** Ableton Operator (no clone).

It has three parts:
1. **Data collection** — an Ableton Extension builds many Operator tracks with random
   params; Ableton's offline multitrack **Export Audio** renders them; Python splits the
   export into `(audio, params)` training pairs.
2. **Training** — PyTorch model: log-mel spectrogram → ResNet → 195 Operator parameter
   heads (continuous params via binned classification, binary/categorical via BCE/CE).
3. **Inference** *(in progress)* — an Ableton Extension runs the model (onnxruntime-node)
   and applies the predicted preset to a selected Operator.

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
    training/  config.py features.py codec.py data.py model.py
               train.py eval.py diagnose.py hear_it.py
  dataset/operator/{wav,params}/       generated dataset (gitignored)
  models/operator/model.pt             trained checkpoint
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
   tracks, each with Operator + a C3 note.
2. **"doppelganger: Randomize Batch"** — randomizes all params, writes
   `dataset/pending/<batchId>/manifest.json`, and tells you the export prefix/folder.
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

## Tool 2 — Training

```powershell
cd D:\AbletonExtensions\doppelganger

# train on dataset/operator (GPU if available); checkpoint -> models/operator/model.pt
uv run python -m doppelganger.training.train --epochs 60 --batch-size 64 --lr 1e-3 --run-name resnet_binned_v1

# sanity: should memorize a tiny subset (cont_mae -> ~0)
uv run python -m doppelganger.training.train --overfit 64 --epochs 300

# monitor (train vs val curves)
uv run tensorboard --logdir runs        # http://localhost:6006

# evaluate a checkpoint (train vs val, with the predict-the-mean baseline)
uv run python -m doppelganger.training.eval

# per-parameter accuracy (which params are learnable)
uv run python -m doppelganger.training.diagnose
```

Audio/feature settings are in `training/config.py`; the model in `training/model.py`.

---

## The "hear it" evaluation (the metric that matters)

Parameter accuracy is misleading (many settings sound alike). This re-renders the model's
**predictions** in the real Operator and compares **audio**.

```powershell
# 1) predict params for K held-out samples -> dataset/predict/{manifest.json,targets/}
uv run python -m doppelganger.training.hear_it predict --k 64
```
2. In Live: right-click → **"doppelganger: Apply Predicted Batch"** (sets the predicted
   presets on the rack).
3. Render them:
   ```powershell
   uv run python -m doppelganger.datagen.export_ableton export "dataset\predict\render" --expected 64
   ```
4. Compare (log-mel L1, matched vs shuffled baseline; ratio < 1 = capturing the sound):
   ```powershell
   uv run python -m doppelganger.training.hear_it compare
   ```
Then listen to `dataset/predict/targets/` vs `dataset/predict/render/`.

---

## Status

- ✅ Tool 1 (data collection) — working, automated; ~20k samples collected.
- 🔄 Tool 2 (training) — ResNet + binned-classification model; iterating on data + loss.
  Current audio baseline: hear-it ratio **0.72** (lower is better).
- ⬜ Tool 3 (inference extension) — ONNX export + in-Live inference, not yet built.

See [docs/PLAN.md](docs/PLAN.md) for what's next (gating-mask loss, neural-proxy perceptual
loss, richer stimuli, ONNX export, AST backbone, multi-GPU).
