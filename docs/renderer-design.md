# Neural Renderer Design (the differentiable Operator surrogate)

**Goal:** a frozen, differentiable `params → spectrogram` model that faithfully mimics
real Operator, so it can provide an **audio-reconstruction loss** to train the one-shot
sound-matcher (the diffusion model). Trained on unlimited offline `(params, real audio)`
pairs. This is the linchpin of the v2 audio-objective approach (`docs/approach.md`).

Grounded in two 2025 research passes (DDX7, NAS-FM, DiffMoog, DDSP, BigVGAN, Vocos,
Neural Proxies JAES 2025, the "spectral distance is bad at pitch" results).

## Chosen architecture: HYBRID (gray-box)

```
params (195) ──┬─► adapter ─► DiffOperator (physics FM core, 11 algos) ─► waveform ─► STFT ─► physics log-mag spec
               │                                                                                    │  (+)
               └─► param encoder (norm vector / preset-tokenizer) ─► FiLM ─► Residual CNN ──────────┘
                                                                              (corrects the spec)
                                                                                    │
                                                                          predicted log-mag spectrogram
```
- **Physics core** gives the correct harmonic skeleton (FM ratios → sideband locations) cheaply and data-efficiently. Even a crude core (our `DiffOperator`) is a strong prior.
- **Neural residual** (FiLM-conditioned 2-D CNN over the spectrogram) learns the un-modeled remainder — the exact envelope curvature, anti-aliasing coloration, filter circuits, saturation — i.e. the "secret sauce" the pure clone couldn't fit.
- Whole thing is differentiable → freeze → its spectral distance is the matcher's audio loss.

## Recipe (SoTA-grounded)
- **Output domain:** multi-resolution **linear**-STFT log-magnitude (n_fft ∈ {512,1024,2048}); per-bin normalization. (Not mel — preserves FM sidebands.)
- **Conditioning:** start with a normalized-param-vector MLP → FiLM at each residual block; upgrade to a **preset-tokenizer + small Transformer** encoder (best on Dexed FM in Neural Proxies) if needed.
- **Loss:** multi-scale STFT (spectral-convergence + log-mag L1) **+ Wasserstein/optimal-transport spectral term** (hardens FM frequency gradients) + optional frozen-audio-embedding L1. Adversarial (MPD/MRD + feature-matching) only as a final polish if outputs are over-smoothed.
- **Tricks:** Snake periodic activations (FM = sums of sinusoids), oversample→low-pass→decimate anti-aliasing in the physics core, weight-norm, EMA (≈0.999), frequency positional encodings, learnable **Bézier/LPF-smoothed envelope curves** in the physics core.
- **Curriculum:** warm-start the physics core (fixed ratios per algorithm) → unfreeze residual → spectral fine-tune. (FM won't converge under spectral loss cold — DiffMoog.)
- **FM gotchas:** phase-modulation form (have it); spectral loss is blind to frequency (fix ratios / Wasserstein); modulation index > ~1.83 → non-monotonic Bessel sidebands → only clamp index for the *matcher's* gradient stability later, NOT in the renderer (it needs full range for fidelity).

## Data (step 1 — regenerate)
`dataset/operator` was cleared. Regenerate a large set via Auto Collect, and **mix the
sampling distribution**: the existing rich-but-uniform random sampler PLUS musically-
plausible presets (so the renderer is accurate where the matcher will actually operate).
Target ~100k+; richer stimulus (note-off/release captured, fixed C3 for now).

## Build / validate plan (step 2)
1. `synth/renderer.py` — the hybrid module (physics core + FiLM residual → spectrogram) + multi-res STFT loss. *(forward pass built + smoke-tested; full SoTA trims added iteratively.)*
2. `training/train_renderer.py` — train on `(params, audio)`, EMA, curriculum.
3. **Validate fidelity** on held-out params: predicted vs real log-mag distance, and vs a baseline — the go/no-go. If faithful → it becomes the matcher's frozen audio-loss provider.

## Risks / fallbacks
- FM chaos / high-index regions are hard (DiffMoog "refused to converge"); mitigations above.
- Renderer fidelity is the make-or-break; we validate before building the diffusion matcher on top.
- Fallback if the spectrogram renderer underperforms: a **params → pretrained-audio-embedding** proxy (Neural Proxies, JAES 2025) — simpler, perceptually-aligned gradient, no audible output. Or diffusion-on-param-data + real-Operator audio re-ranking at inference.
