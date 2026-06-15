# Methodology Reference Document
## Linking Thesis Files to the Null Distribution Mechanism and PCA Analysis

This document maps specific sections and concepts from the papers in the `Thesis Files/` folder to the two analytical methodologies used in this study: the **zero-M Gaussian null distribution and z-scoring** procedure, and the **PCA variance decay analysis** of membrane potential dynamics.

---

## Paper Index

| Ref | File | Citation |
|-----|------|----------|
| [S] | `Scalabal Estimator.pdf` | Liardi, A., Blackburne, G., Rajpal, H., Rosas, F.E., & Mediano, P.A.M. "A scalable estimator of higher-order information in complex dynamical systems." *Preprint.* |
| [N] | `Null Info.pdf` | Liardi, A., Rosas, F.E., Carhart-Harris, R.L., Blackburne, G., Bor, D., & Mediano, P.A.M. "Null models for comparing information decomposition across complex systems." *Preprint.* |
| [H] | `Neural 1 - High Dimension.pdf` | Sussillo, D. & Barak, O. (2013). "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks." *Neural Computation, 25*(3), 626–649. |
| [F] | `Neural 2 - Flexible.pdf` | Driscoll, L.N., Shenoy, K., & Sussillo, D. (2024). "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs." *Nature Neuroscience, 27*, 1349–1363. |
| [G] | `Neural 3 - Population Geometry.pdf` | Wakhloo, A.J., Slatton, W., & Chung, S.Y. (2026). "Neural population geometry and optimal coding of tasks with shared latent structure." *Nature Neuroscience, 29*, 682–692. |
| [C] | `Collective Dynamics.pdf` | Ah-Weng, R. & Rajpal, H. "Collective Dynamics in Spiking Neural Networks Beyond Dale's Principle." *Preprint.* |

---

---

# Part 1 — Null Distribution Mechanism

## Summary of the Local Implementation

For each of the 9 networks and both signal types (membrane potential, spike trains), the `wimfo` library (`W_M_Info.W_M_calculator`) computes M-information across random subsets of size $k \in \{2, 4, 8, 16, 32\}$ neurons. Significance is established by z-scoring against a **zero-M Gaussian null**:

1. The observed neural data is normalised via **Gaussian copula** before computing M.
2. A null distribution of **200 samples** is drawn from a zero-mean Gaussian with the same covariance structure as the data, but with M-information equal to zero by construction (only pairwise marginals are matched, higher-order structure is destroyed).
3. The z-score is computed as:

$$z = \frac{M_{\text{observed}} - \bar{M}_{\text{null}}}{\sigma_{\text{null}}}$$

Results are cached to `*_zero_m_zscore_sweep.json` files.

---

## [S] — Scalable Estimator: Primary Source for M-Information and the Zero-M Null

### What M-information is

**Section II.A–B (pp. 2–3):** M-information is defined as the difference between total mutual information $I(X;Y)$ and the W-information (the lower-order, pairwise-explainable component):

$$M(X;Y) = I(X;Y) - W(X;Y)$$

W-information is computed by minimising over all distributions $Q$ that share the same pairwise marginals as $P$, so M captures strictly the higher-order, beyond-pairwise coordination that cannot be explained by any pair of neurons acting alone.

**Section II.C (p. 3–4):** The minimisation for W is solved via **convex optimisation** when $X, Y$ are jointly Gaussian. This is the algorithm implemented by `wimfo`. The Cholesky parametrisation of the covariance matrix provides the unconstrained search space. This is what makes M-information tractable on 32-neuron subsets.

### Why Gaussian approximation is valid for non-Gaussian neural data

**Section III, Experiment 4 (p. 7):** The authors explicitly validate M-information on the **Wilson-Cowan model** — a non-linear neural population with gain functions — by fitting a **Gaussian copula** to the steady-state distributions before running the convex optimisation. They show M-information correctly identifies the critical regime of sustained oscillations even for non-Gaussian data. This directly justifies the Gaussian copula normalisation step in `wimfo` applied to SNN membrane potential and spike data.

> *"We estimate the M-information by fitting a Gaussian copula to the simulated steady-state distributions of neuronal activity, and employ the optimisation technique presented in Sec. IIC across a range of... weights..."* — [S], p.7

### The zero-mean null hypothesis

**Figures 3 and 4 captions (pp. 8–9):** Both empirical applications (macaque ECoG and mouse Neuropixel data) report significance of M-information using a **one-sample t-test against the zero-mean null hypothesis**. This is the direct precedent for our null z-scoring approach: the baseline hypothesis is that M = 0 (no higher-order structure beyond pairwise).

> *"All M-information values above are normalised w.r.t. the mutual information. (P-values calculated with a one-sample t-test against the zero-mean null hypothesis.)"* — [S], Fig. 3 & 4 captions

The local study adopts this same null (M = 0) but operationalises it as a sampled Gaussian null distribution (200 realisations) rather than a parametric t-test, giving a z-score distribution that is more robust for small subset sizes.

### Normalisation by mutual information

The same figures show M-information normalised by total MI before reporting. In the local study, raw M-bits are reported alongside z-scores; the z-score is the primary significance indicator.

---

## [N] — NuMIT: Primary Source for the Null Distribution Framework

### The algorithmic template

**Section III.A (pp. 4–5):** The NuMIT paper introduces the **four-step null model normalisation procedure** that directly templates our null z-scoring:

> *"1. Given the specific system under examination p(S, T), calculate its TMI and perform the PID.*
> *2. Sample a null model qᵢ(S, T) that has the same TMI as p(S, T) but is otherwise random, and compute its PID.*
> *3. Repeat the previous step N times for many sampled qᵢ, obtaining a null distribution of each PID atom.*
> *4. The relative amount of synergistic, unique, or redundant information of p can be quantified by taking the quantile of the PID atoms of p w.r.t. the null models {qᵢ}ᵢ₌₁ᴺ."*
> — [N], p.5

The local implementation follows steps 1–3 identically (replacing PID atoms with M-information), and replaces step 4's quantile with a z-score — a continuous analogue appropriate for parametric Gaussian nulls.

### The Gaussian null construction

**Section III.B "The Gaussian Case" (pp. 4–5):** For Gaussian systems, the null is constructed by matching the total mutual information (TMI) of the null samples to that of the observed data while randomising the higher-order structure. In the zero-M Gaussian null used here, the matching condition is set to M = 0 specifically: the null samples are drawn from a multivariate Gaussian whose covariance structure is constrained to have zero M-information (i.e. the joint covariance is the product of pairwise marginals). This is the most conservative null: any positive z-score means M exceeds what could be produced by pairwise correlations alone.

### Why raw M-values are not directly comparable across networks

**Section II.B (p. 2):** A key motivation for null normalisation is that raw PID/information atoms vary substantially with the total mutual information (TMI) of the system, independent of the information structure. The paper demonstrates that even systems with identical source-target relationships can give opposite raw PID interpretations depending on noise level. The z-score against the null distribution removes this dependence, making M-values **comparable across the 9 networks** despite their different firing rates and signal scales — which is the main reason for using the null approach in this study.

### Neural null models (contextual background)

**Section V.B "Neural null models" (p. 10):** The paper contextualises null models within the broader neuroscience tradition — comparing observed network metrics against randomised controls that preserve certain structural properties. The local null (zero-M Gaussian) is the information-theoretic analogue: it preserves the pairwise covariance structure but destroys all higher-order coordination.

---

## [C] — Collective Dynamics: Methodological Precedent from Prior Work

This paper (Ah-Weng & Rajpal) is prior work from the same research group applying information-theoretic measures to SNN spike trains. While it uses discrete estimators (JIDT package) rather than M-information, it establishes the methodological convention of:

- Computing **mutual information, O-information, and S-information** across random neuron triplets from SNN spike trains (Section "Methods", p. 3–4).
- Sweeping across parameter regimes to reveal phase transitions in information signatures.
- Using information-theoretic signatures as the primary characterisation of SNN dynamics, not just accuracy.

The local study extends this approach by (a) using M-information instead of O/S-information, (b) applying it to trained RSNN models on a classification task, and (c) introducing the Gaussian null z-score to control for signal amplitude differences across network types.

---

---

# Part 2 — PCA Analysis

## Summary of the Local Implementation

For each of the 9 networks, membrane potential hidden states are extracted across 2 test batches of SHD trials. The matrix is **mean-centred** and **stride-4 downsampled** along the time axis. A **32-neuron random subset** is selected. **SVD** is applied to obtain the singular values $\sigma_i$. The **explained variance ratio (EVR)** per component is:

$$\text{EVR}_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

From this, the **cumulative EVR** is plotted, and the threshold PCs needed to explain 80%, 90%, and 95% of variance ($n_{80}$, $n_{90}$, $n_{95}$) are extracted. The first four cumulative EVR values (`cum_pc1`, `cum_pc2`, `cum_pc4`) summarise the steepness of the variance decay.

---

## [H] — Sussillo & Barak: Establishing PCA as the Standard Tool for RNN State Analysis

### The core methodological precedent

**Pages 8 and 10:** This landmark paper establishes the convention of projecting high-dimensional RNN hidden state trajectories onto the **first few principal components of the network activations** to reveal their computational structure:

> *"The network state x(t) is plotted in the basis of the first three principal components of the network activations."* — [H], p.8

> *"A state-space portrait showing the results of the fixed-point and linear stability analyses is shown in Figure 4B. The network state (activation variable x) is plotted on the basis of the first three principal components of the network activations."* — [H], p.10

This establishes that **membrane potential trajectories of RNNs inherently lie near a low-dimensional manifold**, and PCA of the hidden state matrix is the appropriate tool to reveal that manifold's dimensionality.

**Page 22:** In the methods section, PCA is applied to **slow points** of the unperturbed system to visualise approximate plane attractors — directly paralleling the use of PCA on SHD inference trajectories in the local study.

### Dimensionality depends on the task

**Page 18:** The paper explicitly notes that *"the task and input definition explicitly influence the dimensionality of the network state."* This directly motivates the comparison across the three task partitions (ALL-class, 2-parity, 4-parity) in the local study: networks trained on simpler tasks are expected to require lower-dimensional state representations, and this prediction is confirmed by the PCA results (4C networks reach $n_{90} = 2$–$3$; ALL-class networks require $n_{90} = 4$–$6$).

---

## [F] — Driscoll et al.: Fraction of Variance Explained as a Quantitative Summary

### Variance explained as the primary PCA metric

**Page 3 (Fig. 2b caption):** This paper introduces the use of **fraction of variance explained in the top PCs** as the standard quantitative summary of RNN state dimensionality across task periods:

> *"Fraction variance explained in each task period by top two PCs of neural state trajectories for 1,024 stimulus conditions from every other task period... Right: top 11 PCs of neural state trajectories for 1,024 stimulus conditions from each task period."* — [F], p.3

**Pages 3–5:** The fraction of variance explained is tracked for the **top 10–11 PCs** — exactly the range plotted in the local cumulative EVR figure. The local `cum_pc1`, `cum_pc2`, `cum_pc4` statistics are direct analogues of reading off the EVR spectrum at the first few PC indices.

### Cross-condition variance analysis

**Pages 4–5:** The paper applies PCA defined in one task period to measure how much variance is explained in another task period. The analogous operation in the local study is applying PCA defined on membrane potentials during SHD inference to quantify whether the representational geometry is intrinsic to the network (task-invariant) or specific to the input set.

### Dynamical motifs revealed by PCA dimensionality

**Pages 7–8:** PCA is used to test whether two tasks share a common low-dimensional subspace ("similar dynamical motifs") or occupy non-overlapping subspaces. The local study uses the same logic when comparing LH (heterogeneous) vs. LN (no-heterogeneity) vs. LU (homogeneous) variants: if LH and LU have similar EVR curves, their state geometry is aligned regardless of heterogeneity.

---

## [G] — Wakhloo et al.: Theoretical Justification for PCA Dimensionality as a Performance Predictor

### The four geometric statistics

**Pages 3–4 (Eq. 1 and surrounding discussion):** This paper derives analytically that the generalisation error of a neural population on shared-latent-structure tasks is controlled by four geometric statistics, one of which is the **Neural Dimension**:

$$\text{PR}(\Psi) = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}$$

where $\lambda_i$ are the eigenvalues of the neural covariance matrix. This is the **Participation Ratio** — a smooth summary of the same variance decay spectrum measured by the local PCA ($n_{80}$, $n_{90}$, $n_{95}$ are threshold-based versions of the same concept).

> *"Neural dimension: PR(Ψ). The participation ratio of..."* — [G], p.3

The paper shows that **lower-dimensional representations generalise better early in learning** (small $p$), while higher-dimensional representations are needed for the many-sample regime. For the local 32-neuron SHD networks (trained on 8,156 samples, 25 epochs — the low-data regime), lower-dimensional state geometry is therefore predicted to correlate with better generalisation, which the 4C-LH results confirm.

### Principal components of neural activity map onto task structure

**Page 7:** The paper proves analytically that for optimal representations, the principal components of the latent task variables **map directly onto the principal components of the neural activity**:

> *"We show that the principal components of the latent variables directly map onto the (mutually orthogonal) principal components of the neural activity."* — [G], p.7

This is the key theoretical result justifying our PCA approach: a steeply decaying eigenspectrum (most variance in PC1–PC2) indicates that the network has formed a **compressed, disentangled representation** where the few dominant PCs capture the task-relevant latent structure. Networks with slow decay (e.g., ALL-LN with $\text{cum\_pc1} = 0.506$) use more distributed representations.

### Dimensionality–correlation trade-off

**Pages 4–5:** The paper identifies a systematic **trade-off between neural dimension and neural-latent correlation**: transformations that increase dimension often reduce the single-unit correlation with task variables, and vice versa. This is observed in the local results: LH networks consistently show higher variance concentration in PC1 (higher `cum_pc1`) than LN/LU variants at the same task complexity, consistent with heterogeneity producing more correlated, lower-dimensional representations aligned with task variables.

### Whitened vs. unwhitened representations

**Page 3 (Signal-Signal Factorization, SSF):** The SSF term measures whether different task variables are represented along uncorrelated PCA directions with **equal variance** (whitened). The local EVR analysis directly tests this: a flat top-4 EVR (equal variance across PCs 1–4) indicates a whitened representation, while steep decay indicates that one dominant mode captures most task-relevant variance.

---

## Summary Table: Paper → Methodology Mapping

### Null Distribution

| Concept in local study | Source paper | Specific section / quote |
|---|---|---|
| M-information definition | [S] §II.A–B | Union information framework, W = lower-order, M = higher-order residual |
| Gaussian convex optimisation for W/M | [S] §II.C | Cholesky parametrisation, Theorem 1 |
| Gaussian copula for non-Gaussian data | [S] §III Experiment 4 | Wilson-Cowan validation |
| Zero-M null hypothesis | [S] Figs 3 & 4 | "one-sample t-test against zero-mean null hypothesis" |
| 4-step null sampling algorithm | [N] §III.A | NuMIT procedure, steps 1–4 |
| Gaussian null construction (M=0 by design) | [N] §III.B | "The Gaussian Case" |
| Comparability across networks via normalisation | [N] §II.B | Raw PID atoms incomparable across noise levels |
| Neural null model framing | [N] §V.B | Network-science null models applied to information theory |
| Precedent for IT measures on SNN spikes | [C] §Methods | MI, O-info, S-info on random neuron triplets |

### PCA Analysis

| Concept in local study | Source paper | Specific section / quote |
|---|---|---|
| PCA on RNN hidden state matrix | [H] pp. 8, 10 | "network state x(t) plotted in basis of first 3 PCs of network activations" |
| Low-dimensional manifold in RNN dynamics | [H] p. 18 | "task and input definition explicitly influence dimensionality" |
| PCA on slow-point trajectories | [H] p. 22 | "principal component analysis on slow points with q ≤ 1e-4" |
| EVR in top N PCs as quantitative metric | [F] pp. 3–5 | "fraction variance explained in top 10 PCs" per task period |
| Per-component EVR spectrum (log decay plot) | [F] pp. 3, 5 | "top 11 PCs of neural state trajectories" figure panels |
| Cross-condition subspace comparison | [F] pp. 7–8 | PCA of one task period explained onto another |
| Participation Ratio as dimensionality summary | [G] Eq. 1, p. 3 | PR(Ψ) = (Σλᵢ)²/(Σλᵢ²), one of four key geometric statistics |
| PC structure predicts task generalisation | [G] p. 4 | Lower dimension → better generalisation in low-data regime |
| PC₁ of neural activity = PC₁ of task latents | [G] p. 7 | "principal components of latent variables directly map onto PCs of neural activity" |
| Dimensionality–correlation trade-off | [G] pp. 4–5 | SSF and neural-latent correlation tension |

---

*Document created: May 2026. File locations: all PDFs are in `Thesis Files/`. Analysis results are in `Project Files/*.json` and `Project Files/SNN_Analysis_Report.md`.*
