# Seeded Pair Design

## Overview

For each **master seed** (e.g., 101), two models are built: a **heterogeneous** model with diverse membrane time constants and a **homogeneous anchor** model with a single shared time constant. Every other aspect of the two models is identical, isolating heterogeneity in τ_m as the only causal variable.

## The Design

### 1. Same random state → same weight init

Both models call `set_seed(master_seed)` before construction, so they start from the **same random state**. Weight initialization is therefore matched.

### 2. `sync_non_tau_parameters` guarantees identical weights

After building both, the function explicitly copies **all non-tau parameters** from the heterogeneous model to the homogeneous model. This covers every weight matrix — input→hidden, hidden→hidden, hidden→output. The assertion `linear_sync_verified` checks they're bitwise identical.

### 3. The ONLY difference: τ_m (membrane time constants)

| Model | Hidden τ_m | Hidden τ_syn |
|-------|-----------|-------------|
| **Heterogeneous** | 32 unique values, log-normal sampled per seed | All identical (10 ms) |
| **Homogeneous (anchor)** | All 32 = geometric mean of the hetero samples | All identical (10 ms) |

Output layer τ values are identical for both models.

### 4. What changes across seeds (101 → 202 → 210...)

- **Different weight initializations** (but always matched within the pair)
- **Different τ_m samples** drawn from the same log-normal distribution fitted to empirical SNN data (`Tau from 15 epoch run.json`)

## Summary Diagram

```
Seed 101:
  ├── Hetero: weights=A, τ_m = [18.2, 22.1, 15.7, ...]  (32 unique)
  └── Homo:   weights=A, τ_m = [19.4, 19.4, 19.4, ...]  (all = geom mean)

Seed 202:
  ├── Hetero: weights=B, τ_m = [21.3, 17.9, 20.1, ...]  (32 unique)
  └── Homo:   weights=B, τ_m = [20.0, 20.0, 20.0, ...]  (all = geom mean)
```

This isolates **heterogeneity in τ_m** as the only causal variable — any accuracy gap between the two models in a pair is purely due to diverse vs. uniform membrane time constants.

## Key Functions

| Function | Purpose |
|----------|---------|
| `build_seeded_pair()` | Creates matched hetero + homo pair from one master seed |
| `build_heterogeneous_model()` | Builds model with `het_ab=1` (heterogeneous α/β enabled) |
| `build_homogeneous_model()` | Builds model with `het_ab=0` (no heterogeneity) |
| `sync_non_tau_parameters()` | Copies all weight params from hetero → homo |
| `set_layer_taus_ms()` | Assigns τ_syn and τ_m values to a layer |
| `sample_mem_tau_ms()` | Draws τ_m samples from log-normal distribution |
| `geometric_mean_ms()` | Computes geometric mean for the anchor τ_m |
| `compare_non_tau_parameters()` | Verifies non-tau params are identical across the pair |
