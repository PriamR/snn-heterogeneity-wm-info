# Time-Shuffle Null Z-Scoring for W/M-Information

## Motivation

Raw W- and M-information values are **not directly comparable** across networks because they scale with the total dual mutual information (TDMI), which is strongly influenced by network properties like firing rates, noise levels, and dimensionality. Two networks with identical higher-order structure can show wildly different raw M values simply because one has higher overall correlation.

Z-scoring against a null distribution normalizes for this, making comparisons meaningful.

## Three Approaches Compared

| Approach | Null Construction | Complexity | Assumptions |
|----------|-------------------|------------|-------------|
| **Time-Shuffle** (recommended) | Independently permute each neuron's time series | Low | None — non-parametric |
| Zero-M Gaussian | Sample from multivariate Gaussian with M=0 by construction | Medium | Assumes Gaussianity |
| NuMIT TDMI-matched | Random logic gates + bit-flip noise tuned to match observed TDMI | High | Requires careful TDMI fitting |

## Time-Shuffle Method

### Algorithm

For each k-neuron subset:

1. **Extract** the time series data matrix $X \in \mathbb{R}^{k \times T}$ (k neurons, T timesteps)

2. **Compute observed W/M**: Run the Wimfo estimator on the original data to get $W_{obs}$ and $M_{obs}$

3. **Generate N null samples**: For each null sample $i = 1 \ldots N$:
   - For each neuron $j = 1 \ldots k$, independently permute its time indices: $\tilde{X}_j = \text{shuffle}(X_j)$
   - This preserves each neuron's marginal distribution (mean, variance, firing rate)
   - This **destroys all temporal correlations** between neurons
   - Run Wimfo on the shuffled data to get $W_{null}^{(i)}$, $M_{null}^{(i)}$

4. **Compute Z-score**:
   $$z_W = \frac{W_{obs} - \bar{W}_{null}}{\sigma_{W,null}} \quad\quad z_M = \frac{M_{obs} - \bar{M}_{null}}{\sigma_{M,null}}$$

   where $\bar{W}_{null} = \frac{1}{N}\sum_i W_{null}^{(i)}$ and $\sigma_{W,null} = \sqrt{\frac{1}{N-1}\sum_i (W_{null}^{(i)} - \bar{W}_{null})^2}$

### Why This Works

- **Destroys causal structure**: Shuffling breaks the temporal ordering that carries predictive information between neurons. Any W/M remaining in the shuffled data is artifactual — it comes from marginal distributions, not genuine coordination.

- **Non-parametric**: No assumptions about Gaussianity, distribution shape, or data type. Works identically on membrane potentials, spike trains, or any other representation.

- **Preserves marginals**: Each neuron's firing statistics remain intact, so the null captures the "background" information level from independent activity.

- **Corrects for dimensionality**: Networks with more neurons or higher baseline correlations will have higher null W/M values, so Z-scoring automatically normalizes for network size and activity level.

### Comparison to NuMIT

| | Time-Shuffle | NuMIT (Discrete) |
|---|---|---|
| Null generation time | ~O(k·T) per sample (fast) | ~O(k²) per sample (slow) |
| TDMI matching | Approximate (marginals preserved) | Exact (iterative root-finding) |
| Interpretation of z > 0 | "More W/M than expected from independent neurons" | "More W/M than expected from random logic gates with same TDMI" |
| Works on continuous data? | ✅ Yes | ❌ Discrete only |

### When to Use Which

- **Time-Shuffle**: Quick, robust, works on any data type. Use for exploratory analysis and when comparing many networks/subset sizes.

- **Zero-M Gaussian**: Mathematically principled for Gaussian-copula-normalized membrane potentials. Use when the Gaussian copula step is already applied.

- **NuMIT**: Most rigorous null — preserves exact TDMI. Use for final publication figures on spike data when maximum statistical rigor is needed.

## Implementation

```python
def compute_shuffle_null(traces, k, rng, n_null=200, delay_t=1):
    """
    Compute W/M Z-scores using time-shuffle null.
    
    Args:
        traces: [n_neurons, n_timesteps] array (membrane potential or spikes)
        k: subset size
        rng: numpy random generator
        n_null: number of null samples
        delay_t: time lag for covariance
    
    Returns:
        w_obs, m_obs, z_w, z_m
    """
    from wimfo import W_M_calculator
    from wimfo.utils.utils_gauss import get_cov
    
    # Select k random neurons
    idx = rng.choice(traces.shape[0], k, replace=False)
    data = traces[idx, :]  # [k, T]
    
    # Observed W/M
    cov_obs = get_cov(data, t=delay_t)
    result_obs = W_M_calculator(cov_obs, option="distr", type="gaussian", optimiser="Adam")
    w_obs = result_obs.get("W", 0)
    m_obs = result_obs.get("M", 0)
    
    # Null: shuffle each neuron independently
    w_null, m_null = [], []
    for _ in range(n_null):
        shuffled = data.copy()
        for j in range(k):
            rng.shuffle(shuffled[j, :])
        cov_null = get_cov(shuffled, t=delay_t)
        result_null = W_M_calculator(cov_null, option="distr", type="gaussian", optimiser="Adam")
        w_null.append(result_null.get("W", 0))
        m_null.append(result_null.get("M", 0))
    
    w_null = np.array(w_null)
    m_null = np.array(m_null)
    
    z_w = (w_obs - w_null.mean()) / w_null.std() if w_null.std() > 0 else 0
    z_m = (m_obs - m_null.mean()) / m_null.std() if m_null.std() > 0 else 0
    
    return w_obs, m_obs, z_w, z_m
```

## Interpretation

- **$z_W > 1.96$**: Statistically significant unique information beyond independent activity (p < 0.05, two-tailed)
- **$z_M > 1.96$**: Statistically significant higher-order information beyond independent activity
- **Comparing networks**: Higher $z_W$ for heterogeneous vs homogeneous at the same k supports the hypothesis that τ_m diversity increases information-theoretic capacity
- **Comparing subset sizes**: $z$ should increase with k if the network has genuine collective dynamics — more neurons should capture more complementary information
