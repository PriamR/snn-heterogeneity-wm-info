# Spiking Neural Network Information-Theoretic Analysis
## Cross-Task Comparison: Local Homogeneous vs Fitted Heterogeneous Architectures

**Dataset:** Spiking Heidelberg Digits (SHD) — 700 input channels, 1000 ms trials  
**Tasks:** All-class 20-way classification · 2-class parity · 4-class parity-language  
**Date:** May 2026

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Hyperparameter Comparison: Repo vs Local Networks](#2-hyperparameter-comparison-repo-vs-local-networks)
3. [Task Performance Summary](#3-task-performance-summary)
4. [Observed Information Curves (M and W)](#4-observed-information-curves-m-and-w)
5. [PCA Variance Decay Analysis](#5-pca-variance-decay-analysis)
6. [Null Z-Scoring: Methodology & Results](#6-null-z-scoring-methodology--results)
7. [Membrane vs Spike Representation Analysis](#7-membrane-vs-spike-representation-analysis)
8. [Average M-Information vs Task Performance](#8-average-m-information-vs-task-performance)
9. [Z-Score Percentile Cross-Task Comparison](#9-z-score-percentile-cross-task-comparison)
10. [Convergence Audit](#10-convergence-audit)

---

## 1. Architecture Overview

Nine networks are trained and compared across three tasks. Each network belongs to one of three architecture families:

### Architecture Family Comparison

| Property | **Local-Hom** | **FittedHet-LogNorm** | **FittedHet-LogUniform** | **Repo-Learned-Het** (reference) |
|---|---|---|---|---|
| **Tau heterogeneity** | None — shared τ across all hidden neurons | Per-neuron τ sampled from empirically fitted distributions | Per-neuron τ sampled from fitted log-uniform bounds | Per-neuron τ learned end-to-end by gradient descent |
| **τ_syn distribution** | Homogeneous (one shared value) | Gamma fit to observed learned taus (shape ≈ 1.03, scale ≈ 22.74 ms) | Gamma fit (same as LogNorm variant) | a priori Gamma(k=3, mean=τ_syn_default) |
| **τ_mem distribution** | Homogeneous (one shared value) | Log-Normal fit (shape ≈ 0.99, scale ≈ 25.50 ms) | Log-Uniform over empirical [min, max] bounds | a priori Gamma(k=3, mean=τ_mem_default) — same family as τ_syn |
| **Tau bounds clipping** | N/A | Clipped to observed empirical [min, max] | Clipped to observed empirical [min, max] | No clipping |
| **Tau source** | N/A | Learned taus from a 15-epoch freely-trained run (`Tau from 15 epoch run.json`) | Same 15-epoch run artifact | Fixed a priori scale parameter |
| **Tau trainable after init?** | τ is trained but shared (all neurons coupled) | No — frozen with `requires_grad_(False)` | No — frozen with `requires_grad_(False)` | Yes — trained per-neuron throughout |
| **`het_ab` flag** | 0 | 1 | 1 | 1 |
| **`train_ab` flag** | 0 | 0 | 0 | 1 |
| **`train_hom_ab` flag** | 1 | 0 | 0 | 0 |
| **Optimization stability** | Highest — shared tau avoids τ-learning instability | High — frozen taus remove τ from gradient graph | High — same as LogNorm | Lower — per-neuron τ gradients can be unstable |
| **Expressivity** | Lowest — no temporal diversity | Medium — diversity fixed at init from real data | Medium — wider uniform coverage of timing regimes | Highest — diversity fully task-adaptive |
| **Synergy source** | Emergent from shared recurrent dynamics | Injected via distribution sampling at network construction | Injected via broader distribution sampling | Co-adapted with recurrent weights |

---

## 2. Hyperparameter Comparison: Repo vs Local Networks

Values extracted from `neural_heterogeneity/SuGD_code/main.py` (paper repo defaults) and saved checkpoint `prms` dictionaries (local networks).

### Key Hyperparameter Table

| Hyperparameter | **Paper Repo (128-neuron)** | **ALL-RepoHet** | **ALL-LH** | **ALL-LN** | **ALL-LU** | Notes |
|---|---|---|---|---|---|---|
| `nb_inputs` | 700 | 700 | 700 | 700 | 700 | SHD has 700 afferent channels |
| `nb_recurrent` | **128** | **32** | 32 | 32 | 32 | Paper repo default is 4× wider; local networks use 32 |
| `nb_outputs` | 20 | 20 | 20 | 20 | 20 | All 20 SHD classes |
| `batch_size` | **64** | **256** | 256 | 256 | 256 | Paper repo uses smaller batches |
| `time_step` (ms) | **0.5** | **1** | 1 | 1 | 1 | Paper repo uses 0.5 ms; local networks use 1 ms |
| `nb_steps` | **2000** | **1000** | 1000 | 1000 | 1000 | 2000 steps × 0.5 ms = 1 s; 1000 steps × 1 ms = 1 s (same trial duration) |
| `nb_epochs` | **150** | **25** | 25 | 25 | 25 | Paper repo trains for 6× more epochs |
| `lr` | **1e-3** | **4e-3** | 4e-3 | 4e-3 | 4e-3 | |
| `lr_ab` | **1e-3** | **4e-3** | 4e-3 | 4e-3 | 4e-3 | Separate learning rate for α/β (tau) parameters |
| `betas` | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) | (0.9, 0.999) | Adam optimizer momentum terms |
| `weight_decay` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | No weight decay in any network |
| `clip` | 1 | 1 | 1 | 1 | 1 | Gradient clipping enabled (norm=1) in all networks |
| `tau_syn` (ms) | 10 | 10 | 10 | 10* | 10* | Default init value; overwritten for FittedHet |
| `tau_mem` (ms) | 20 | 20 | 20 | 20* | 20* | Default init value; overwritten for FittedHet |
| `time_scale` | [0.5, 1.0] | [0.5, 1.0] | [0.5, 1.0] | [0.5, 1.0] | [0.5, 1.0] | |
| `seed` | 1000 | 1000 | 1000 | 1000 | 1000 | |
| `dist` | `gamma` | `gamma` | `gamma` | `gamma` | `gamma` | Distribution family; FittedHet overrides α/β directly |
| `dist_prms` | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | Shape param k=3 for Gamma; used by repo at runtime, ignored by FittedHet |
| `het_ab` | 0 | **1** | **0** | **1** | **1** | Whether per-neuron α/β heterogeneity is active |
| `train_ab` | 0 | **1** | **0** | **0** | **0** | Whether α/β are updated by gradient descent |
| `train_hom_ab` | 0 | **0** | **1** | **0** | **0** | Whether shared (homogeneous) α/β are trained |

*FittedHet networks store the default `tau_syn`/`tau_mem` in `prms` as initialisation references, but the actual per-neuron α/β values are overwritten from fitted distributions before training and frozen via `requires_grad_(False)`.

### Repo vs Local Study: Key Differences

| Difference | Paper Repo (128-neuron) | Local study networks |
|---|---|---|
| Hidden layer size | **128** neurons | **32** neurons |
| Batch size | **64** | **256** |
| Time resolution | **0.5 ms**, 2000 steps | **1 ms**, 1000 steps (same 1 s duration) |
| Training epochs | **150** | **25** |
| Learning rate | **1e-3** | **4e-3** |

### Flag Combinations

| Flag combination | Architecture | Effect |
|---|---|---|
| `het_ab=0, train_hom_ab=0, train_ab=0` | Paper Repo default | No tau heterogeneity; taus fixed at default values |
| `het_ab=0, train_hom_ab=1, train_ab=0` | Local-Hom | One shared τ trained per layer; all neurons identical |
| `het_ab=1, train_hom_ab=0, train_ab=0` | FittedHet-LogNorm/LU | Per-neuron τ set at construction, frozen for training |
| `het_ab=1, train_hom_ab=0, train_ab=1` | Repo-Learned-Het | Per-neuron τ initialised from Gamma, then learned end-to-end |

---

## 3. Task Performance Summary

### All-Class (20-way SHD classification)

![Accuracy bar chart](Report%20Figures/fig01_accuracy_bar.png)

| Network | Architecture | Final Test Acc | Best Test Acc | Avg M (bits) | M@k=32 (bits) | M z-score @k=32 |
|---|---|---|---|---|---|---|
| ALL-LH | Local-Hom | 65.24% | 65.24% | 0.5616 | 2.3486 | 27,307.6 |
| ALL-LN | FittedHet-LogNorm | 69.04% | 71.29% | 0.3663 | 1.5209 | 17,677.9 |
| ALL-LU | FittedHet-LogUniform | 60.07% | 60.07% | 0.3237 | 1.3442 | 15,621.6 |
| ALL-RepoHet | Repo-Learned-Het | 64.93% | 65.90% | 0.4122 | 1.5540 | 18,062.7 |

### 2-Class Parity Task

Two classes selected from SHD (binary classification). Networks trained from scratch with same architecture families.

| Network | Architecture | Final Test Acc | Best Test Acc | Avg M (bits) | M@k=32 (bits) | M z-score @k=32 |
|---|---|---|---|---|---|---|
| 2C-LH | Local-Hom | 81.49% | 82.91% | 0.6958 | 2.2371 | 16,991.5 |
| 2C-LN | FittedHet-LogNorm | **87.23%** | **88.21%** | **1.1843** | **4.2318** | 18,112.3 |
| 2C-LU | FittedHet-LogUniform | 83.04% | 83.08% | 1.1455 | 3.5510 | 15,195.3 |

### 4-Class Parity-Language Task

Four classes spanning both English and German digits from SHD (cross-language classification).

| Network | Architecture | Final Test Acc | Best Test Acc | Avg M (bits) | M@k=32 (bits) | M z-score @k=32 |
|---|---|---|---|---|---|---|
| 4C-LH | Local-Hom | 51.37% | 54.24% | 0.1404 | 0.4971 | 3,762.9 |
| 4C-LN | FittedHet-LogNorm | **62.46%** | **64.40%** | **1.2421** | **4.4662** | 7,918.5 |
| 4C-LU | FittedHet-LogUniform | 59.36% | 59.98% | 1.2633 | 4.4241 | **18,936.3** |

### Subset-Wise Observed M (bits)

| Network | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| ALL-LH | −0.0027 | 0.0030 | 0.0825 | 0.3767 | 2.3486 |
| ALL-LN | 0.0037 | 0.0070 | 0.0774 | 0.2225 | 1.5209 |
| ALL-LU | −0.0054 | 0.0031 | 0.0560 | 0.2207 | 1.3442 |
| ALL-RepoHet | −0.0032 | 0.0114 | 0.0300 | 0.4689 | 1.5540 |
| 2C-LH | 0.0024 | 0.0480 | 0.2421 | 0.9496 | 2.2371 |
| 2C-LN | −0.0032 | 0.0045 | 0.5170 | 1.1714 | 4.2318 |
| 2C-LU | −0.0017 | 0.1221 | 0.4050 | 1.6512 | 3.5510 |
| 4C-LH | −0.0028 | 0.0149 | 0.0192 | 0.1734 | 0.4971 |
| 4C-LN | 0.0011 | 0.0871 | 0.4126 | 1.2433 | 4.4662 |
| 4C-LU | −0.0029 | 0.0338 | 0.2004 | 1.6609 | 4.4241 |

---

## 4. Observed Information Curves (M and W)

![M and W information curves](Report%20Figures/fig02_mw_curves.png)

**Figure:** 3×2 panel plot. Rows = tasks; columns = M curves (left) and W curves (right). Each curve traces a single network across subset sizes k = {2, 4, 8, 16, 32}.

### What M and W Measure

| Quantity | Formula | Interpretation |
|---|---|---|
| **W (redundancy)** | Sum of pairwise mutual informations − total information | How much information is redundantly encoded across neuron pairs |
| **M (synergy)** | Collective information − W − individual information | How much additional information emerges only when all k neurons are considered together |

The `wimfo` library (`W_M_calculator`) computes Gaussian M-information using a Gaussian copula normalisation of the raw hidden neuron activity. Two solver modes are attempted in order: (1) Adam optimiser on Gaussian-copula-normalised data (`option='data'`); (2) Newton solver on pre-computed regularised lagged covariance (`option='distr'`) as fallback.

A lag of τ=1 time step is used, capturing the one-step temporal predictive structure of the hidden representation.

---

## 5. PCA Variance Decay Analysis

![PCA variance decay](Report%20Figures/fig_pca_variance.png)

**Figure:** 2-panel plot. Left: cumulative explained variance; Right: per-component variance on log scale. Both panels show all 9 networks simultaneously, colour-coded by architecture and with line style indicating task.

### Methodology

PCA is applied to the membrane potential hidden state matrix extracted from the test set:

1. **Data collection:** For each network, up to 2 batches of test data are passed through the frozen model. The membrane potential tensor (shape: `[batch × time × nb_recurrent]`) is extracted from the first recurrent layer.
2. **Downsampling:** A temporal stride of 4 is applied (every 4th timestep), reducing memory requirements while preserving the variance structure.
3. **Subset selection:** 32 neurons are selected (or all neurons if fewer) using uniform linear spacing across the hidden layer index.
4. **Centering:** The neuron × sample matrix `X` (shape: `[32 × T]`) is mean-centred across samples.
5. **SVD:** `np.linalg.svd(X, full_matrices=False)` is applied. Explained variance for component `i` is `σ_i² / (n_samples - 1)`. The explained variance ratio (EVR) is `σ_i² / Σ σ_j²`.
6. **Cumulative EVR** is plotted, with reference lines at 80% and 90% thresholds.

### Summary Statistics

| Network | Task | Architecture | cum_pc1 | cum_pc2 | cum_pc4 | n80 | n90 | n95 |
|---|---|---|---|---|---|---|---|---|
| ALL-LH | all-class | Local-Hom | 0.467 | 0.788 | 0.913 | 3 | 4 | 6 |
| ALL-LN | all-class | FittedHet-LogNorm | 0.506 | 0.718 | 0.848 | 3 | 6 | 9 |
| ALL-LU | all-class | FittedHet-LogUniform | 0.575 | 0.761 | 0.944 | 3 | 4 | 5 |
| 2C-LH | 2-class | Local-Hom | 0.795 | 0.881 | 0.950 | 2 | 3 | 4 |
| 2C-LN | 2-class | FittedHet-LogNorm | 0.725 | 0.827 | 0.942 | 2 | 4 | 5 |
| 2C-LU | 2-class | FittedHet-LogUniform | 0.498 | 0.674 | 0.882 | 4 | 5 | 7 |
| 4C-LH | 4-class | Local-Hom | 0.863 | 0.937 | 0.963 | 1 | 2 | 3 |
| 4C-LN | 4-class | FittedHet-LogNorm | 0.787 | 0.880 | 0.959 | 2 | 3 | 4 |
| 4C-LU | 4-class | FittedHet-LogUniform | 0.768 | 0.865 | 0.941 | 2 | 3 | 5 |

*cum_pc1/2/4: cumulative EVR at component 1/2/4. n80/90/95: minimum PCs to reach that % explained variance.*

---

## 6. Null Z-Scoring: Methodology & Results

### 6.1 Methodology: Building the Null Distribution

For each network and each subset size k, a Zero-M Gaussian null distribution is constructed:

1. **Marginal statistics extraction:** The mean and variance of each neuron's activity in the real data are computed (the univariate statistics).
2. **Synthetic null samples:** 200 synthetic datasets are generated, each with the same univariate marginals but with **no collective synergy** (i.e., the multivariate structure is destroyed so that M = 0 by construction). This is the "Zero-M Gaussian" null model — it has the same pairwise correlation baseline but no high-order structure.
3. **Null M distribution:** The wimfo library computes M-information for each of the 200 synthetic datasets, producing a distribution `{M_null_1, ..., M_null_200}`.
4. **Z-score:** The observed M-information from the real network is z-scored against this null:
   ```
   z = (M_observed - mean(M_null)) / std(M_null)
   ```
5. **P-value:** The one-sided upper-tail p-value `P(M_null > M_observed)` is computed from the empirical null distribution. Because z-scores reach thousands, all p-values saturate to p ≈ 0 and are reported as such.

**Cached results** are stored as JSON files in `Project Files/Null Caches/` to avoid recomputation. Each file contains the null distribution parameters (mean, std) and the final z-score per subset size.

### 6.2 Why Z-score, Not Raw M

| Issue | Raw M | Z-scored M |
|---|---|---|
| Pairwise correlation baseline | Contaminated — Local-Hom has high pairwise sync, inflating M | Removed — null already has the same pairwise baseline |
| Subset-size comparability | M grows with k regardless of architecture | Z-score compares against a same-k null |
| Architecture comparisons | Biased toward high-correlation (homogeneous) networks | Unbiased — asks "how much MORE than expected?" |

### 6.3 Z-Score Heatmap (Membrane)

![Z-score heatmaps](Report%20Figures/fig04_zscore_heatmaps.png)

**Figure:** Two-panel heatmap. Left panel: M z-score heatmap (coolwarm colourmap). Right panel: −log₁₀(p-value) heatmap (magma colourmap).

Rows: subset sizes k ∈ {2, 4, 8, 16, 32}. Columns: all 9 networks in task order (ALL-LH → ... → 4C-LU).

### 6.4 Z-Score Line Plots (by Task)

![Z-score line plots](Report%20Figures/fig05_zscore_lines.png)

**Figure:** Three-panel line plot. One panel per task. Each curve traces one network's M z-score as k increases from 2 to 32.

### 6.5 Full Z-Score Tables

All values below are M-information z-scores (z = (M_observed − null_mean) / null_std, 200-sample Zero-M Gaussian null). Negative z at k=2 indicates that pairs of neurons are below the pairwise null baseline — expected for small k.

#### Membrane M z-scores — all 9 networks

| Network | Task | Architecture | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---:|---:|---:|---:|---:|
| ALL-LH | all-class | Local-Hom | −886 | 367 | 4,171 | 9,224 | 27,308 |
| ALL-LN | all-class | FittedHet-LogNorm | 1,203 | 860 | 3,916 | 5,444 | 17,678 |
| ALL-LU | all-class | FittedHet-LogUniform | −1,761 | 377 | 2,832 | 5,400 | 15,622 |
| 2C-LH | 2-class | Local-Hom | 527 | 3,623 | 9,002 | 16,960 | 16,992 |
| 2C-LN | 2-class | FittedHet-LogNorm | −705 | 341 | 19,228 | 20,925 | 18,112 |
| 2C-LU | 2-class | FittedHet-LogUniform | −368 | 9,220 | 15,060 | 29,499 | 15,195 |
| 4C-LH | 4-class | Local-Hom | −606 | 1,126 | 709 | 3,089 | 3,763 |
| 4C-LN | 4-class | FittedHet-LogNorm | 248 | 6,575 | 15,342 | 22,209 | 7,919 |
| 4C-LU | 4-class | FittedHet-LogUniform | −638 | 2,554 | 7,451 | 13,753 | 18,936 |

#### Membrane observed M (bits) — all 9 networks

| Network | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| ALL-LH | −0.00273 | 0.00298 | 0.08247 | 0.37673 | 2.34863 |
| ALL-LN | 0.00371 | 0.00698 | 0.07743 | 0.22250 | 1.52093 |
| ALL-LU | −0.00543 | 0.00307 | 0.05603 | 0.22072 | 1.34418 |
| 2C-LH | 0.00243 | 0.04798 | 0.24214 | 0.94956 | 2.23714 |
| 2C-LN | −0.00324 | 0.00454 | 0.51705 | 1.17138 | 4.23179 |
| 2C-LU | −0.00169 | 0.12208 | 0.40500 | 1.65116 | 3.55096 |
| 4C-LH | −0.00279 | 0.01493 | 0.01917 | 0.17337 | 0.49712 |
| 4C-LN | 0.00115 | 0.08707 | 0.41259 | 1.24326 | 4.46624 |
| 4C-LU | −0.00293 | 0.03384 | 0.20042 | 1.66093 | 4.42412 |

#### Spike M z-scores — parity networks (2-class and 4-class only)

| Network | Task | Architecture | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---:|---:|---:|---:|---:|
| 2C-LH | 2-class | Local-Hom | −323 | 3,848 | 2,221 | 8,933 | 14,977 |
| 2C-LN | 2-class | FittedHet-LogNorm | −466 | 3,577 | 8,882 | 14,162 | 17,402 |
| 2C-LU | 2-class | FittedHet-LogUniform | 7,918 | 16,128 | 17,798 | 6,325 | 17,261 |
| 4C-LH | 4-class | Local-Hom | −131 | 3,942 | 2,042 | 2,958 | 4,297 |
| 4C-LN | 4-class | FittedHet-LogNorm | −204 | 11,256 | 15,105 | 28,580 | 11,400 |
| 4C-LU | 4-class | FittedHet-LogUniform | 692 | 1,034 | 16,507 | 26,633 | 19,018 |

#### Spike observed M (bits) — parity networks

| Network | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---:|---:|
| 2C-LH | −0.00179 | 0.04616 | 0.06207 | 0.57643 | 2.01848 |
| 2C-LN | −0.00260 | 0.04290 | 0.24783 | 1.79212 | 4.32922 |
| 2C-LU | 0.04413 | 0.19338 | 0.49648 | 1.50826 | 4.29417 |
| 4C-LH | −0.00073 | 0.04729 | 0.05707 | 0.19120 | 0.58060 |
| 4C-LN | −0.00113 | 0.13496 | 0.42137 | 3.61568 | 5.71655 |
| 4C-LU | 0.00386 | 0.01243 | 0.46047 | 1.71745 | 4.73083 |

> *Z-scores are rounded to the nearest integer; observed M to 5 decimal places. Spike z-scores are not available for all-class networks (ALL-LH/LN/LU) as spike scoring was only run for parity tasks.*

---

## 7. Membrane vs Spike Representation Analysis

![Mem vs spike delta M](Report%20Figures/fig06_mem_vs_spk.png)

**Figure:** Two-panel plot. Left: delta M by subset size. Right: average M scatter (mem x-axis, spk y-axis).

### What is Compared

For each network, M-information is computed from two representations:
- **Membrane** (`mem`): The hidden neuron membrane potential time series — a continuous-valued, graded signal reflecting subthreshold integration.
- **Spike** (`spk`): The binary spike output — a discrete, event-based signal reflecting threshold crossings.

`delta_M = M_spk - M_mem` at each subset size k.

### Average Mem→Spk Change Table

> Full table printed in Cell 10 under "Average mem→spk change table". Key columns: `avg_M_mem`, `avg_M_spk`, `delta_M_spk_minus_mem`.

---

## 8. Average M-Information vs Task Performance

![M vs accuracy scatter](Report%20Figures/fig03_m_vs_accuracy.png)

**Figure:** Scatter plot. X-axis: average M-information across all subset sizes (membrane). Y-axis: final test accuracy. Each point is one of the 9 networks, coloured by architecture and marker-shaped by task.

---

## 9. Z-Score Percentile Cross-Task Comparison

### 9.1 Percentile Definition

Z-scores range from approximately 300 to 30,000+. A rank-based percentile is used to compare across tasks without distributional assumptions:

$$\text{percentile} = \frac{\text{rank} - 1}{n - 1} \times 100\%$$

This maps the highest z-score → 100% and lowest z-score → 0% regardless of absolute scale.

Two variants are computed:
- **Global percentile:** Rank among all 9 networks at each subset size k.
- **Within-task percentile:** Rank among the 3 networks sharing the same task and subset size k.

### 9.2 Global Percentile Heatmap

![Percentile heatmap](Report%20Figures/fig07_percentile_heatmap.png)

**Figure:** Heatmap of global percentile ranks across all 9 networks × 5 subset sizes. Annotated with % values; white vertical lines separate task groups.

| subset k | ALL-LH | ALL-LN | ALL-LU | 2C-LH | 2C-LN | 2C-LU | 4C-LH | 4C-LN | 4C-LU |
|---|---|---|---|---|---|---|---|---|---|
| **k=2**  | 12.5%  | 100.0% | 0.0%   | 87.5% | 25.0% | 62.5% | 50.0% | 75.0% | 37.5% |
| **k=4**  | 12.5%  | 37.5%  | 25.0%  | 75.0% | 0.0%  | 100.0%| 50.0% | 87.5% | 62.5% |
| **k=8**  | 37.5%  | 25.0%  | 12.5%  | 62.5% | 100.0%| 75.0% | 0.0%  | 87.5% | 50.0% |
| **k=16** | 37.5%  | 25.0%  | 12.5%  | 62.5% | 75.0% | 100.0%| 0.0%  | 87.5% | 50.0% |
| **k=32** | 100.0% | 62.5%  | 37.5%  | 50.0% | 75.0% | 25.0% | 0.0%  | 12.5% | 87.5% |

### 9.3 Global Percentile Curves (Per Task)

![Global percentile curves](Report%20Figures/fig08_percentile_curves_global.png)

**Figure:** Three-panel line plot with endpoint annotations. The dashed line at 50% marks the global median.

### 9.4 Within-Task Percentile (Architecture Winner per Task)

![Within-task percentile](Report%20Figures/fig09_percentile_within_task.png)

**Figure:** Three-panel line plot showing which architecture wins within its own task.

### 9.5 Average Global Percentile by Architecture and Task

![Average global percentile bar chart](Report%20Figures/fig10_avg_percentile_bar.png)

**Figure:** Grouped bar chart (3 architecture groups × 3 task bars). Dashed line at 50% = global median.

| Architecture | all-class | 2-class | 4-class |
|---|---|---|---|
| **Local-Hom** | 40% | 68% | 20% |
| **FittedHet-LogNorm** | 50% | 55% | 70% |
| **FittedHet-LogUniform** | 18% | 72% | 58% |

---

## 10. Convergence Audit

### Training Curves

![Training curves](Report%20Figures/fig12_training_curves.png)

### Convergence Diagnostics

![Convergence plots](Report%20Figures/fig11_convergence.png)

**Figures:** Training accuracy per epoch (top) and three convergence diagnostic panels (bottom).

### Convergence Criteria

| Criterion | Threshold | Rationale |
|---|---|---|
| **Tail flatness** | `|delta_last5| ≤ 0.010` AND `std_last5 ≤ 0.015` | Last 5 epochs are stable — no meaningful trend |
| **Best-to-final drop** | `best_minus_final ≤ 0.015` | Peak accuracy not substantially regressed at end of training |
| **Late peak** | `best_epoch ≥ nb_epochs - 2` | Best epoch was near the end (still climbing allowed if tail is stable) |

Status labels: `"plateaued / converged"`, `"near plateau"`, `"still improving"`, `"peaked then regressed"`, `"mixed"`.

### Diagnostic Panel Descriptions

- **Peak-to-final test drop (left panel):** Bar chart of `best_test_acc - final_test_acc`. The dashed line at 0.015 is the convergence threshold. Networks above the line peaked and then regressed.
- **Last-5-epoch test trend (middle panel):** Bar chart of `test_acc[-1] - test_acc[-5]`. Networks near zero are flat; positive values are still improving; negative values regressed at the end.
- **Generalization gap scatter (right panel):** `final_train_acc - final_test_acc` vs `final_test_acc`. Points above zero have a train-test gap. The dashed line at zero is the ideal.

---

## Appendix: Network Naming Convention

| Short ID | Task | Architecture | Description |
|---|---|---|---|
| ALL-LH | All-class (20-way SHD) | Local-Hom | Homogeneous τ, trained on all 20 classes |
| ALL-LN | All-class | FittedHet-LogNorm | Per-neuron τ from Gamma (syn) + LogNormal (mem), frozen |
| ALL-LU | All-class | FittedHet-LogUniform | Per-neuron τ from Gamma (syn) + LogUniform (mem), frozen |
| 2C-LH | 2-class parity | Local-Hom | Homogeneous τ, binary SHD parity task |
| 2C-LN | 2-class parity | FittedHet-LogNorm | LogNormal membrane τ, binary task |
| 2C-LU | 2-class parity | FittedHet-LogUniform | LogUniform membrane τ, binary task |
| 4C-LH | 4-class parity-language | Local-Hom | Homogeneous τ, cross-language 4-class task |
| 4C-LN | 4-class parity-language | FittedHet-LogNorm | LogNormal membrane τ, 4-class task |
| 4C-LU | 4-class parity-language | FittedHet-LogUniform | LogUniform membrane τ, 4-class task |
| ALL-RepoHet | All-class | Repo-Learned-Het | Reference: per-neuron τ learned end-to-end (Perez-Nieves et al.) |

## Appendix: File Locations

| Artifact | Path |
|---|---|
| Main analysis notebook | `Project Files/Initial visualizations notebook.ipynb` |
| Parity classification notebook | `Project Files/7 - Parity Classification.ipynb` |
| FittedHet training notebook | `Project Files/5 - FittedHet.ipynb` |
| Null z-score caches | `Project Files/Null Caches/*.json` |
| Spike sweep caches | `Project Files/initial_viz_*_spk_*.json` and `parity_*_spk_*.json` |
| Model checkpoints | `Project Files/Checkpoints/` and `Project Files/Checkpoints/Parity/` |
| Tau distribution artifact | `Tau from 15 epoch run.json` (project root) |
| Paper repository | `neural_heterogeneity/SuGD_code/` |
| wimfo library | `wimfo/W_M_Info.py` |
