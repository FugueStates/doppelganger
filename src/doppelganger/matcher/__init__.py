"""
Phase B — the one-shot matcher (input audio -> Operator params).

A conditional diffusion model over the normalized parameter vector, conditioned on an
embedding of the target audio, trained with the magnitude-weighted spectral loss
backpropped THROUGH the frozen neural renderer (the audio loss is the driver; a diffusion
x0 / param term is the nudge). The discrete Algorithm is handled by a SEPARATE classifier
head (decided 2026-06-15); the diffusion denoiser models only the remaining continuous +
binary params. See docs/matcher-design.md.
"""
