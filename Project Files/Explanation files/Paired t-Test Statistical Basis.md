# Paired t-Test — Mathematical Basis & Usage in This Project

## Simple Explanation (The Ice Cream Shop Analogy)

Imagine you own 10 ice cream shops. You try a **new recipe** in all 10 shops and want to know if sales improved. You can't just compare "all old-recipe shops" vs "all new-recipe shops" because weather, location, and day-of-week would confuse the result.

Instead, you pair each shop **against itself**: record sales with the old recipe, then with the new recipe, and take the difference at **each shop**. If the average difference is positive and large relative to how much the differences vary, you have evidence the new recipe works.

That's exactly what a **paired t-test** does — it compares two conditions measured on the **same subjects** (in our case, the same seed).

---

## How It's Used in This Project

### The Setup

We train **10 seeded pairs** of SNN models. For each seed $s$:

- **Homogeneous anchor** — all neurons share the same membrane time constant $\tau_m$
- **Heterogeneous sampled** — each neuron gets its own $\tau_m$ drawn from a lognormal distribution

Both models in a pair are trained on the **same data split** (same random seed), so they form a natural matched pair.

We record the final test accuracy for each model:

$$(A_{hom}^{(1)}, A_{het}^{(1)}), (A_{hom}^{(2)}, A_{het}^{(2)}), \ldots, (A_{hom}^{(n)}, A_{het}^{(n)})$$

where $n = 10$ seeds.

### The Question

> Does heterogeneity produce a statistically significant improvement in test accuracy?

### The Difference Scores

For each seed $i$, compute the paired difference:

$$D_i = A_{het}^{(i)} - A_{hom}^{(i)}$$

If heterogeneity helps, $D_i$ should be positive on average.

---

## Mathematical Basis

### Assumptions

1. **Paired observations** — each pair shares a seed (same data split, same initialisation)
2. **Independence** — differences $D_i$ are independent across seeds
3. **Normality** — the differences are approximately normally distributed (can be relaxed for $n \ge 10$)

### The Test Statistic

The paired t-statistic is:

$$t = \frac{\bar{D}}{s_D / \sqrt{n}}$$

where:
- $\bar{D} = \frac{1}{n}\sum_{i=1}^{n} D_i$ — mean difference
- $s_D = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(D_i - \bar{D})^2}$ — sample standard deviation of differences
- $n$ — number of pairs (seeds)

### What the t-statistic Measures

$$t = \frac{\text{signal}}{\text{noise}} = \frac{\text{mean difference}}{\text{standard error of the mean}}$$

- **Large positive t**: Hetero consistently outperforms homo (differences are positive and consistent)
- **t near zero**: No systematic difference
- **Large negative t**: Homo consistently outperforms hetero

### The p-value

Under the null hypothesis $H_0: \mu_D = 0$ (no true difference), the t-statistic follows a **Student's t-distribution** with $df = n-1$ degrees of freedom.

The **one-sided p-value** (testing $H_A: \mu_D > 0$, i.e., "hetero is better") is:

$$p = P(T_{n-1} \ge t_{obs})$$

If $p < 0.05$, we reject the null — the improvement is statistically significant.

---

## Why the Paired Design Matters

### Paired vs. Unpaired

| | Unpaired t-test | Paired t-test |
|---|---|---|
| **What it compares** | Two independent groups | Matched pairs |
| **Variance source** | Between-subject + within-subject | Within-pair only |
| **Power** | Lower (more noise) | Higher (controls for seed effects) |
| **df for n=10 seeds** | 18 | 9 |

By pairing on seed, we **eliminate the seed-to-seed variability** from the comparison. Two models trained on different seeds might differ by 5% just due to the random split — the paired design subtracts this out.

### Concrete Example

| Seed | Homo Acc | Hetero Acc | Difference $D_i$ |
|------|----------|------------|-------------------|
| 101 | 58.2% | 66.2% | +8.0 |
| 202 | 57.0% | 69.4% | +12.4 |
| 210 | 50.9% | 71.6% | +20.7 |
| ... | ... | ... | ... |

The paired test asks: *"Is the mean of the rightmost column reliably above zero?"*

If we'd used an unpaired test, it would compare the raw homo column vs the raw hetero column, losing the fact that seed 210 was just a "hard" seed for both models.

---

## Effect Size: Cohen's d

Statistical significance alone isn't enough — with enough data, even tiny effects become "significant." We also report **practical significance** via Cohen's d:

$$d = \frac{\bar{D}}{s_D}$$

| d | Interpretation |
|---|---------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |
| > 1.0 | Very large effect |

For our 64-neuron runs: $d \approx 1.0-1.5$, indicating a **large to very large** effect of heterogeneity.

---

## 95% Confidence Interval

We also report the 95% CI for the true mean difference:

$$\bar{D} \pm t_{0.025, n-1} \cdot \frac{s_D}{\sqrt{n}}$$

This gives a range of plausible values for the true heterogeneity advantage. If the CI excludes zero, we can be 95% confident that heterogeneity genuinely improves performance.

---

## Implementation in This Project

```python
from scipy import stats

# diffs = array of paired differences (hetero - homo)
t_stat, p_value = stats.ttest_rel(hetero_accs, homo_accs, alternative='greater')

# Cohen's d
d = np.mean(diffs) / np.std(diffs, ddof=1)

# 95% CI
ci = stats.t.interval(0.95, df=n-1, loc=np.mean(diffs),
                      scale=np.std(diffs, ddof=1)/np.sqrt(n))
```

### Where It Appears in the Code

- `Seeded 4 Class/Seeded run 1 4class 64neurons WMinfo Local.ipynb` — Cell 5 (Section 2, Results Summary)
- `Seeded 4 Class/Seeded run 1 4class 64neurons ZeroM Zscore.ipynb` — results loading cells
- All seeded run analysis notebooks

---

## When the t-Test Is (and Isn't) Appropriate

| ✅ Appropriate when | ❌ Use Wilcoxon instead when |
|---|---|
| Differences are roughly symmetric | Differences are heavily skewed |
| $n \ge 10$ (CLT helps) | $n < 5$ |
| No extreme outliers | Outliers dominate the mean |
| You want to estimate the effect size | You only care about consistency of direction |

In this project, we report **both** the paired t-test and the Wilcoxon signed-rank test. If they agree, we have strong evidence. If they disagree, we investigate further.
