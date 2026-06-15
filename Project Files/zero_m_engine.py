"""Zero-M (NuMIT) W/M estimation engine — shared between notebooks."""
import hashlib
import io
import itertools
import math
import numpy as np
from contextlib import redirect_stdout
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import root_scalar
from scipy.stats import norm, rankdata, wishart


def _stable_seed(*parts):
    token = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def _gc_normalize(data):
    transformed = np.zeros_like(data, dtype=np.float64)
    for idx, row in enumerate(data):
        if np.allclose(row, row[0]):
            continue
        ranks = rankdata(row, method="average")
        uniform = np.clip((ranks - 0.5) / len(row), 1e-6, 1.0 - 1e-6)
        transformed[idx] = norm.ppf(uniform)
    return transformed


def _inject_jitter(data, eps=1e-6):
    row_std = np.nanstd(data, axis=1)
    deg_idx = np.flatnonzero(row_std <= 1e-12)
    if deg_idx.size == 0:
        return data
    data = data.copy()
    base = np.linspace(-1.0, 1.0, data.shape[1], dtype=np.float64)
    for offset, ridx in enumerate(deg_idx, start=1):
        data[ridx] = (eps * offset) * base
    return data


def _regularize_cov(cov, ridge=1e-9):
    cov = np.asarray(cov, dtype=np.float64)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    scale = np.trace(cov) / max(cov.shape[0], 1)
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 1.0
    return cov + float(ridge) * scale * np.eye(cov.shape[0], dtype=np.float64)


def _logdet_pd(mat):
    sign, val = np.linalg.slogdet(mat)
    if sign <= 0 or (not np.isfinite(val)):
        raise ValueError("Matrix not positive definite for logdet.")
    return float(val)


def _spectral_radius(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def _sigmoid(x, M=1.0):
    x = np.clip(float(x), -60.0, 60.0)
    return float(M) / (1.0 + np.exp(-x))


def _mi_var1_bits(A, V):
    S = solve_discrete_lyapunov(A, V)
    S = _regularize_cov(S, ridge=1e-12)
    return 0.5 * (_logdet_pd(S) - _logdet_pd(V)) / np.log(2.0)


def _lagged_cov_from_var1(A, V):
    S = solve_discrete_lyapunov(A, V)
    S = _regularize_cov(S, ridge=1e-12)
    cpf = S @ A.T
    return np.block([[S, cpf], [cpf.T, S]])


def _numit_var1_from_mi(mi_target_bits, nvar, rng, g_cap=0.9995, tol_bits=1e-3):
    A0 = rng.normal(0.0, 1.0, size=(int(nvar), int(nvar)))
    sr = _spectral_radius(A0)
    if (not np.isfinite(sr)) or sr <= 1e-12:
        return None, None, True, np.nan
    A_unit = A0 / sr

    V = wishart.rvs(df=int(nvar) + 1, scale=np.eye(int(nvar)), random_state=rng)
    V = _regularize_cov(V, ridge=1e-9)

    def f_x(x):
        g = float(g_cap) * _sigmoid(x, 1.0)
        A = g * A_unit
        try:
            mi_val = _mi_var1_bits(A, V)
        except Exception:
            return np.nan
        return float(mi_val - float(mi_target_bits))

    xs = np.array([-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12], dtype=np.float64)
    fs = np.asarray([f_x(x) for x in xs], dtype=np.float64)

    x_star = None
    near_idx = np.where(np.isfinite(fs) & (np.abs(fs) <= float(tol_bits)))[0]
    if near_idx.size > 0:
        x_star = float(xs[int(near_idx[0])])
    else:
        for idx in range(len(xs) - 1):
            f1, f2 = fs[idx], fs[idx + 1]
            if not (np.isfinite(f1) and np.isfinite(f2)):
                continue
            if f1 * f2 > 0:
                continue
            try:
                sol = root_scalar(f_x, bracket=[float(xs[idx]), float(xs[idx + 1])],
                                  method="brentq", xtol=1e-6, rtol=1e-6, maxiter=100)
            except Exception:
                sol = None
            if sol is not None and sol.converged and np.isfinite(sol.root):
                x_star = float(sol.root)
                break

    if x_star is None:
        return None, None, True, np.nan

    g_star = float(g_cap) * _sigmoid(x_star, 1.0)
    A = g_star * A_unit
    try:
        mi_check = _mi_var1_bits(A, V)
    except Exception:
        return None, None, True, np.nan
    if not np.isfinite(mi_check):
        return None, None, True, np.nan
    allowed_err = max(float(tol_bits), 0.02 * max(float(mi_target_bits), 1e-8))
    if abs(float(mi_check) - float(mi_target_bits)) > allowed_err:
        return None, None, True, float(mi_check)
    return A, V, False, float(mi_check)


def _wm_from_lagged_cov(lagged_cov, ridge=1e-2):
    from wimfo.gaussian.double_union_gauss import double_union
    from wimfo.utils.utils_gauss import tdmi_from_cov

    cov = _regularize_cov(lagged_cov, ridge=float(ridge))
    nvar = cov.shape[0] // 2

    if nvar > 15:
        solver_candidates = [
            ("Adam", {"atol": 1e-3, "rtol": 1e-3, "max_iter": 400, "window_size": 21}),
            ("Newton", {"max_iter": 40}),
        ]
    else:
        solver_candidates = [
            ("Newton", {"max_iter": 75}),
            ("Adam", {"atol": 1e-3, "rtol": 1e-3, "max_iter": 800, "window_size": 21}),
        ]

    for optimiser, options in solver_candidates:
        with io.StringIO() as buf, redirect_stdout(buf):
            w_nats = double_union(cov, nx=nvar, optimiser=optimiser,
                                  options=options, verbose=False, switch_opt=False)
        if np.isfinite(w_nats):
            tdmi_bits = float(tdmi_from_cov(cov, xdim=nvar))
            w_bits = float(w_nats / np.log(2.0))
            m_bits = float(tdmi_bits - w_bits)
            return w_bits, m_bits
    raise RuntimeError(f"Null W solver failed for nvar={nvar}")


def compute_wm_from_spike_matrix(hidden_data, lag=1, ridge=1e-2):
    from wimfo.W_M_Info import W_M_calculator
    from wimfo.utils.utils_gauss import get_cov

    gdata = _gc_normalize(hidden_data.astype(np.float64).copy())
    gdata = np.nan_to_num(gdata, nan=0.0, posinf=0.0, neginf=0.0)
    gdata = _inject_jitter(gdata)

    opt_order = [
        ("Adam", {"atol": 1e-3, "rtol": 1e-3, "max_iter": 30000}),
        ("Adam", {"atol": 5e-3, "rtol": 5e-3, "max_iter": 60000}),
        ("Newton", None),
    ]
    lag0 = max(int(lag), 1)
    lag_candidates = sorted({lag0, lag0 * 2, lag0 * 4})
    stride_candidates = [1, 2, 4]
    ridge_candidates = sorted({float(ridge), 5.0 * float(ridge), 1e-1, 2e-1, 5e-1, 1.0})

    for stride in stride_candidates:
        dv = gdata[:, ::stride]
        if dv.shape[1] <= max(8, 2 * dv.shape[0] + 2):
            continue
        for lag_try in lag_candidates:
            for optimiser, options in opt_order:
                try:
                    with io.StringIO() as buf, redirect_stdout(buf):
                        w_bits, m_bits = W_M_calculator(
                            dv, t=lag_try, option="data", type="gaussian",
                            unit="bits", verbose=False,
                            optimiser=optimiser, options=options)
                except Exception:
                    continue
                if np.isfinite(w_bits) and np.isfinite(m_bits):
                    return float(w_bits), float(m_bits), int(dv.shape[1])

    last_err = None
    for stride in stride_candidates:
        dv = gdata[:, ::stride]
        if dv.shape[1] <= max(8, 2 * dv.shape[0] + 2):
            continue
        for lag_try in lag_candidates:
            try:
                cov = np.asarray(get_cov(dv, t=lag_try), dtype=np.float64)
            except Exception as exc:
                last_err = exc
                continue
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov = 0.5 * (cov + cov.T)
            scale = np.trace(cov) / max(cov.shape[0], 1)
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            eye = np.eye(cov.shape[0], dtype=np.float64)
            for ridge_try in ridge_candidates:
                lc = cov + ridge_try * scale * eye
                try:
                    evals, evecs = np.linalg.eigh(lc)
                    evals = np.clip(evals, max(scale * 1e-8, 1e-10), None)
                    lc = (evecs * evals[np.newaxis, :]) @ evecs.T
                    lc = 0.5 * (lc + lc.T)
                except np.linalg.LinAlgError as exc:
                    last_err = exc
                    continue
                for optimiser, options in opt_order:
                    try:
                        with io.StringIO() as buf, redirect_stdout(buf):
                            w_bits, m_bits = W_M_calculator(
                                lc, option="distr", type="gaussian",
                                unit="bits", verbose=False,
                                optimiser=optimiser, options=options)
                    except Exception as exc:
                        last_err = exc
                        continue
                    if np.isfinite(w_bits) and np.isfinite(m_bits):
                        return float(w_bits), float(m_bits), int(dv.shape[1])

    raise RuntimeError(f"All W/M estimation paths failed. Last error: {last_err}")


def build_numit_null_distribution(nvar, tdmi_target_bits, n_null=20, seed=12345,
                                   ridge=1e-2, max_attempts=900):
    rng = np.random.default_rng(int(seed))
    null_m, null_w, model_mi_vals, wm_tdmi_vals = [], [], [], []
    attempts = 0
    while len(null_m) < int(n_null) and attempts < int(max_attempts):
        attempts += 1
        A, V, failed, mi_model = _numit_var1_from_mi(tdmi_target_bits, nvar, rng)
        if failed:
            continue
        try:
            lag_cov = _lagged_cov_from_var1(A, V)
            w_bits, m_bits = _wm_from_lagged_cov(lag_cov, ridge=ridge)
        except Exception:
            continue
        if not (np.isfinite(w_bits) and np.isfinite(m_bits)):
            continue
        null_w.append(float(w_bits))
        null_m.append(float(m_bits))
        model_mi_vals.append(float(mi_model))
        wm_tdmi_vals.append(float(w_bits + m_bits))
    return {
        "null_M_values": null_m, "null_W_values": null_w,
        "model_mi_bits": model_mi_vals, "wm_tdmi_bits": wm_tdmi_vals,
        "n_null_valid": int(len(null_m)), "n_attempts": int(attempts),
    }


def score_observed_vs_null_m(observed_m, null_m_values):
    vals = np.asarray(null_m_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"null_mean": np.nan, "null_std": np.nan, "z": np.nan,
                "p_upper": np.nan, "p_lower": np.nan, "p_two_sided": np.nan}
    mu = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
    z = float((float(observed_m) - mu) / sd) if (np.isfinite(sd) and sd > 1e-12) else np.nan
    ge = np.sum(vals >= float(observed_m))
    le = np.sum(vals <= float(observed_m))
    p_upper = float((1.0 + ge) / (vals.size + 1.0))
    p_lower = float((1.0 + le) / (vals.size + 1.0))
    p_two = float(min(1.0, 2.0 * min(p_upper, p_lower)))
    return {"null_mean": mu, "null_std": sd, "z": z,
            "p_upper": p_upper, "p_lower": p_lower, "p_two_sided": p_two}


def sample_random_subsets(n_neurons, subset_size, sample_size, seed):
    total_possible = math.comb(int(n_neurons), int(subset_size))
    target_size = min(int(sample_size), int(total_possible))
    if target_size == total_possible:
        all_subsets = itertools.combinations(range(int(n_neurons)), int(subset_size))
        return [tuple(idx) for idx in all_subsets], total_possible
    rng = np.random.default_rng(int(seed))
    sampled = set()
    while len(sampled) < target_size:
        subset = tuple(sorted(rng.choice(int(n_neurons), size=int(subset_size),
                                         replace=False).tolist()))
        sampled.add(subset)
    return sorted(sampled), total_possible
