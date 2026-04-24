# Masked Demo Data

This folder contains a tiny synthetic demo dataset used for smoke tests and example commands.

It is intentionally not a real field dataset:

- frame images are generated masked placeholders
- audio clips are synthetic tones
- GNSS values are synthetic numeric sequences
- labels only preserve the expected CSV schema

Files:

- `train.csv`, `val.csv`, `test.csv`
- `frames/`
- `audio/`
- `gnss_normalization.json`
- `class_weights.json`

Use this data only to verify code paths, not to measure model quality.
