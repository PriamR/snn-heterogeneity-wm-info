> **Superseded.** The final code, trained networks and analyses for the MSc thesis *Neural heterogeneity promotes robust learning by increasing representational separability* (Imperial College London, 2026) are at **https://github.com/PriamR/neural-heterogeneity-separability**. This repository is an early working snapshot (July 2026), kept read-only for reference; its results predate the final thesis.

# Heterogeneous Time Constants in Recurrent Spiking Neural Networks

MSc Artificial Intelligence thesis — Imperial College London, 2026 (in progress).

**The question:** do biologically-inspired *neuron time-constant diversity* — giving
each neuron its own membrane/synaptic timescale, the way real cortical neurons
vary — make a recurrent spiking network a better temporal model, and if so, *why*?

**The task:** Spiking Heidelberg Digits (SHD) — spoken-digit audio encoded as
spikes over 700 cochlear channels, classified from the spike stream alone.

## Result

Neuron-level timescale heterogeneity lifts accuracy, with the largest gain on the
hardest task. Best test accuracy under a controlled reduced-compute budget:

| Task | Local-Hom (shared τ) | FittedHet-LogNorm (diverse τ) | Gain |
|---|---|---|---|
| 20-class | 65.2% | 71.3% | **+6.0 pp** |
| 4-class parity-language | 54.2% | 64.4% | **+10.2 pp** |
| 2-class parity | — | — | +5.7 pp |

> These runs use a deliberately shortened training budget (25 epochs vs. the
> 150 in the reference paper) so that many controlled variants could be swept and
> compared cleanly. The absolute ceilings are therefore lower than a fully-converged
> run — the *comparison between architectures*, not the absolute number, is the result.

## What makes it a controlled study, not just a benchmark

- **Matched-pair, multi-seed design (10+ seeds):** each homogeneous/heterogeneous
  pair is trained with every network weight held bitwise-identical at init, varying
  *only* the tau distribution — so any accuracy gap is attributable to heterogeneity
  itself, not luck of initialisation.
- **Principled tau distributions:** heterogeneous taus are sampled from distributions
  *fit to the taus a network learns when left free* (Log-Normal membrane, Gamma
  synaptic, clipped to observed bounds), rather than arbitrary priors, then frozen.

## Why it helps — the information-theoretic + geometry analysis

The novel part of this repo is *explaining* the gain rather than only reporting it:

- **W-/M-information decomposition** (redundancy vs. synergy) across representation
  subset sizes, with a **NuMIT / Zero-M null model** and z-scoring so effects are
  tested against a proper significance baseline rather than eyeballed.
- **PCA representational-geometry analysis:** heterogeneous networks learn
  higher-dimensional, more distributed representations (slower PCA variance decay).
- **Membrane vs. spike** representation comparison, and an M-information-vs-accuracy
  scatter showing heterogeneous networks reach the high-synergy / high-accuracy region.

Full write-up with figures: [`Project Files/SNN_Analysis_Report.md`](Project%20Files/SNN_Analysis_Report.md)
(and `SNN_Analysis_Report_Data_Only.md` for the numbers-only version). Statistical
basis for the paired tests is documented under `Project Files/Explanation files/`.

## Stack

PyTorch, surrogate-gradient training on 700-channel cochlear-encoded spike streams;
MATLAB information-theory toolboxes for the W/M decomposition; NumPy/pandas/matplotlib
analysis notebooks. Built on the Perez-Nieves et al. heterogeneity codebase as the
homogeneous/learned-heterogeneous reference point.

## Layout

```
Project Files/
  1–7 …ipynb                 the training + analysis pipeline, in order
  SNN_Analysis_Report.md     the full cross-task information-theory writeup
  Explanation files/         statistical-method notes (paired t, Wilcoxon, nulls)
  Seeded <N> Class/          matched-pair multi-seed runs per task
  *Tau Distributions.json    fitted tau distributions used to build heterogeneity
```

Author: Priyam Rai — MSc AI (Imperial College London) · [github.com/PriamR](https://github.com/PriamR)
