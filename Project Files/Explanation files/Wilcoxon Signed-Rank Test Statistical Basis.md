# Wilcoxon Signed-Rank Test — Mathematical Basis & Usage in This Project

## Simple Explanation (The Judge Analogy)

Imagine 10 judges taste-test two recipes — old and new — and each gives a score. The **paired t-test** asks: *"Is the average score difference big enough?"* But what if one judge gives the new recipe a wildly high score? That single judge could swing the average, making the t-test unreliable.

The **Wilcoxon signed-rank test** takes a different approach. It doesn't care about the **size** of the differences — it asks: *"How consistently does the new recipe beat the old one?"*

It works like this:
1. For each judge, note who won (new or old) and by how much
2. **Rank** all the differences by their absolute size (smallest difference = rank 1, largest = rank 10)
3. Sum the ranks where the new recipe won
4. If this sum is large enough, the new recipe is consistently better

This makes it **robust to outliers** — one extreme score can only contribute its rank (at most 10), not an unlimited value.

---

## How It's Used in This Project

### Why Both Tests?

The paired t-test and Wilcoxon signed-rank test answer **complementary** questions:

| Test | Question | Sensitive to |
|------|----------|--------------|
| **Paired t-test** | Is the *average* difference reliably non-zero? | Magnitude of differences |
| **Wilcoxon signed-rank** | Is hetero *consistently* better than homo across seeds? | Direction and consistency |

If both tests agree ($p < 0.05$), we have **converging evidence** from parametric and non-parametric perspectives.

### The Setup (Same as t-Test)

For $n = 10$ seeds, we have paired differences:

$$D_i = A_{het}^{(i)} - A_{hom}^{(i)}, \quad i = 1, \ldots, n$$

---

## Mathematical Basis

### Step 1: Compute Absolute Differences & Rank Them

Let $|D_i|$ be the absolute difference for seed $i$. Sort all $|D_i|$ values and assign ranks $R_i$ from 1 (smallest) to $n$ (largest).

- Ties receive the **average** of the ranks they span
- Zero differences are typically discarded (reducing $n$)

### Step 2: Compute the Signed Rank Sum

$$W = \sum_{i: D_i > 0} R_i$$

That is, sum the ranks only for seeds where hetero **beat** homo.

Under the null hypothesis $H_0$: "hetero and homo are equally good," the differences are symmetric around zero, so positive and negative differences should have roughly equal rank sums. The expected value is:

$$E[W] = \frac{n(n+1)}{4}$$

### Step 3: The Test Statistic

For small $n$: The exact distribution of $W$ is used (tabled).

For larger $n$ (our case, $n = 10$): A normal approximation is used:

$$z = \frac{W - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}$$

### Step 4: The p-value

For a one-sided test ($H_A$: hetero > homo):

$$p = P(Z \ge z)$$

where $Z \sim \mathcal{N}(0, 1)$.

If $p < 0.05$, we reject the null — hetero is consistently better than homo.

---

## Why the Wilcoxon Test Adds Rigor

### Robustness to Distribution Shape

The t-test assumes **normality** of differences. With $n = 10$, this can't be reliably verified. The Wilcoxon test makes **no distributional assumptions** — it only assumes:

1. **Paired observations** (same as t-test)
2. **Independence** across pairs
3. **Symmetry** of the difference distribution (weaker than normality)

### Protection Against Outliers

Consider these hypothetical differences across 10 seeds:

$$D = [ +2, +3, +1, +2, -1, +2, +1, +3, +2, \mathbf{+40} ]$$

| Test | Result | Interpretation |
|------|--------|----------------|
| t-test | $p = 0.04$ ★ | Significant, but driven entirely by the +40 outlier |
| Wilcoxon | $p = 0.12$ — | Not significant — only 9/10 seeds favour hetero, which could happen by chance |

The Wilcoxon test correctly identifies that 9/10 wins isn't compelling enough, while the t-test is fooled by the single extreme value.

### Complementary Evidence

When **both** tests give $p < 0.05$, we can be confident the result isn't an artifact of:

- **Distribution shape** (Wilcoxon guards against non-normality)
- **Outlier sensitivity** (Wilcoxon guards against extreme values)
- **Small sample noise** (t-test guards against low-rank information loss)

---

## Comparing the Two Tests

| Property | Paired t-Test | Wilcoxon Signed-Rank |
|----------|--------------|---------------------|
| **Type** | Parametric | Non-parametric |
| **Null hypothesis** | $\mu_D = 0$ | Distribution of $D$ is symmetric around 0 |
| **Uses magnitude?** | Yes — full difference values | Partially — ranks preserve ordering |
| **Sensitive to outliers?** | Yes | No (bounded by $n$) |
| **Statistical power** | Higher if assumptions met | Slightly lower (uses less information) |
| **Reports effect size?** | Yes — Cohen's d | No direct equivalent |
| **Minimum n** | 3 (but 10+ for CLT) | 5 (for meaningful significance) |

---

## Implementation in This Project

```python
from scipy import stats

# homo_accs, hetero_accs = arrays of shape (n_seeds,)
w_stat, p_value = stats.wilcoxon(hetero_accs, homo_accs, alternative='greater')

# w_stat is the smaller of the two rank sums (positive and negative)
# p_value is the one-sided p-value for "hetero > homo"
```

### Where It Appears in the Code

- `Seeded 4 Class/Seeded run 1 4class 64neurons WMinfo Local.ipynb` — Cell 5 (Section 2, Results Summary), reported alongside t-test
- `Seeded 4 Class/Seeded run 1 4class 64neurons ZeroM Zscore.ipynb` — accuracy summary cells
- All seeded run analysis notebooks that report statistical significance

### Typical Output

```
  Hetero > Homo: 9/10
  Mean Δacc:     +7.15 pp
  SD Δacc:       6.12 pp
  95% CI:        [+2.77, +11.53] pp
  Cohen's d:     1.168
  t-test p:      0.002423  ★ significant
  Wilcoxon p:    0.003906  ★ significant
```

Both tests converging on $p < 0.01$ gives **strong evidence** that heterogeneity improves performance.

---

## When Tests Disagree

If the t-test is significant but Wilcoxon is not:

> The average improvement is real, but it's not consistent across seeds — a few seeds may be driving the effect. Check for outliers or bimodal distributions.

If Wilcoxon is significant but the t-test is not:

> Hetero consistently beats homo, but the margins are small. This is a robust but modest effect — worth reporting with the Wilcoxon p-value.

In our 64-neuron 4-class runs, both tests agree strongly ($p < 0.01$ for both), indicating a **reliable, consistent advantage** for heterogeneity.

---

## Historical Context

The test was developed by **Frank Wilcoxon** in 1945 as a "quick and dirty" alternative to the t-test that didn't require normality assumptions or computational tables. It was one of the first non-parametric tests and remains among the most widely used in scientific research. Wilcoxon originally described it in a single paragraph of a paper about insecticides — a reminder that the most enduring statistical methods often come from practical, applied problems.
