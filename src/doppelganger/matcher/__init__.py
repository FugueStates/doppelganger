"""
Phase B — the one-shot matcher (input audio -> Operator params).

A DETERMINISTIC predictor: an audio encoder feeds an Algorithm classifier (the one
discrete param) and a parameter head that emits, in a single forward pass, the other 194
params confined to the audible dataset manifold. Trained with the magnitude-weighted
spectral loss backpropped THROUGH the frozen neural renderer (perceptual driver) + a
param-anchor MSE + algorithm cross-entropy. Designed to ship in the extension as ONNX on
CPU — audio -> params in one shot, no synth in the loop. See docs/matcher-design.md.
"""
