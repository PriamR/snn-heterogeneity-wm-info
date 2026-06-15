import argparse
import hashlib
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import root_scalar
from scipy.stats import norm, rankdata, wishart


NB_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = NB_DIR / "checkpoints"
SPK_TENSOR_DIR = NB_DIR / "spk_tensors"
OUTPUT_DIR = NB_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LOCAL_WIMFO_ROOT = NB_DIR / "wimfo"
if LOCAL_WIMFO_ROOT.exists() and str(LOCAL_WIMFO_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_WIMFO_ROOT))

try:
    from wimfo.W_M_Info import W_M_calculator
    from wimfo.gaussian.double_union_gauss import double_union
    from wimfo.utils.utils_gauss import get_cov, tdmi_from_cov
except Exception:
    import subprocess

    print("Installing wimfo...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wimfo"])
    from wimfo.W_M_Info import W_M_calculator
    from wimfo.gaussian.double_union_gauss import double_union
    from wimfo.utils.utils_gauss import get_cov, tdmi_from_cov


TARGETS = {
    "2class_local_hom": {
        "checkpoint": CHECKPOINT_DIR / "2class_local_hom.pt",
        "spike_tensor": SPK_TENSOR_DIR / "2class_local_hom_spk_tensor.npy",
    },
    "2class_fittedhet_ln": {
        "checkpoint": CHECKPOINT_DIR / "2class_fittedhet_lognorm.pt",
        "spike_tensor": SPK_TENSOR_DIR / "2class_fittedhet_ln_spk_tensor.npy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone Z-scoring diagnostics from packaged spike tensors.")
    parser.add_argument("--subset-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--subset-sample-size", type=int, default=500)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=900)
    return parser.parse_args()


def load_spike_tensor(path):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D [neurons x time] tensor, got {arr.shape} for {path.name}")
    arr = arr.astype(np.float64, copy=False)
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError(f"Spike tensor is not strictly binary for {path.name}")
    return arr


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


def compute_wm_from_spike_matrix(hidden_data, lag=1, ridge=1e-2):
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
                            dv,
                            t=lag_try,
                            option="data",
                            type="gaussian",
                            unit="bits",
                            verbose=False,
                            optimiser=optimiser,
                            options=options,
                        )
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
                                lc,
                                option="distr",
                                type="gaussian",
                                unit="bits",
                                verbose=False,
                                optimiser=optimiser,
                                options=options,
                            )
                    except Exception as exc:
                        last_err = exc
                        continue
                    if np.isfinite(w_bits) and np.isfinite(m_bits):
                        return float(w_bits), float(m_bits), int(dv.shape[1])

    raise RuntimeError(f"All W/M estimation paths failed. Last error: {last_err}")


def _sigmoid(x, M=1.0):
    x = np.clip(float(x), -60.0, 60.0)
    return float(M) / (1.0 + np.exp(-x))


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
    eig = np.linalg.eigvals(A)
    return float(np.max(np.abs(eig)))


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
                sol = root_scalar(
                    f_x,
                    bracket=[float(xs[idx]), float(xs[idx + 1])],
                    method="brentq",
                    xtol=1e-6,
                    rtol=1e-6,
                    maxiter=100,
                )
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
            w_nats = double_union(
                cov,
                nx=nvar,
                optimiser=optimiser,
                options=options,
                verbose=False,
                switch_opt=False,
            )
        if np.isfinite(w_nats):
            tdmi_bits = float(tdmi_from_cov(cov, xdim=nvar))
            w_bits = float(w_nats / np.log(2.0))
            m_bits = float(tdmi_bits - w_bits)
            return w_bits, m_bits

    raise RuntimeError(f"Null W solver failed for nvar={nvar}")


def _stable_seed(net_key, tensor, subset_size):
    token = f"{net_key}|{tensor}|{int(subset_size)}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def build_numit_null_distribution(nvar, tdmi_target_bits, n_null=20, seed=12345, ridge=1e-2, max_attempts=900):
    rng = np.random.default_rng(int(seed))
    null_m = []
    null_w = []
    model_mi_vals = []
    wm_tdmi_vals = []

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
        "null_M_values": null_m,
        "null_W_values": null_w,
        "model_mi_bits": model_mi_vals,
        "wm_tdmi_bits": wm_tdmi_vals,
        "n_null_valid": int(len(null_m)),
        "n_attempts": int(attempts),
    }


def score_observed_vs_null_m(observed_m, null_m_values):
    vals = np.asarray(null_m_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "null_mean": np.nan,
            "null_std": np.nan,
            "z": np.nan,
            "p_upper": np.nan,
            "p_lower": np.nan,
            "p_two_sided": np.nan,
        }

    mu = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else np.nan
    z = float((float(observed_m) - mu) / sd) if (np.isfinite(sd) and sd > 1e-12) else np.nan

    ge = np.sum(vals >= float(observed_m))
    le = np.sum(vals <= float(observed_m))
    p_upper = float((1.0 + ge) / (vals.size + 1.0))
    p_lower = float((1.0 + le) / (vals.size + 1.0))
    p_two = float(min(1.0, 2.0 * min(p_upper, p_lower)))

    return {
        "null_mean": mu,
        "null_std": sd,
        "z": z,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "p_two_sided": p_two,
    }


def sample_random_subsets(n_neurons, subset_size, sample_size, seed):
    total_possible = math.comb(int(n_neurons), int(subset_size))
    target_size = min(int(sample_size), int(total_possible))
    rng = np.random.default_rng(int(seed))

    sampled = set()
    while len(sampled) < target_size:
        subset = tuple(sorted(rng.choice(n_neurons, size=subset_size, replace=False).tolist()))
        sampled.add(subset)

    return sorted(sampled), total_possible


def main():
    args = parse_args()
    subset_sizes = [int(k) for k in args.subset_sizes]

    print(f"Package dir      : {NB_DIR}")
    print(f"Checkpoint dir   : {CHECKPOINT_DIR}")
    print(f"Spike tensor dir : {SPK_TENSOR_DIR}")
    print(f"Output dir       : {OUTPUT_DIR}")
    print(f"Local wimfo dir  : {LOCAL_WIMFO_ROOT}")
    print(f"Subset sizes     : {subset_sizes}")
    print(f"Subset sample sz : {args.subset_sample_size}")
    print(f"N_NULL           : {args.n_null}")
    print(f"Lag              : {args.lag}")
    print(f"Ridge            : {args.ridge}")
    print(f"Max attempts     : {args.max_attempts}")

    missing = []
    for net_key, meta in TARGETS.items():
        ckpt_path = meta["checkpoint"]
        spk_path = meta["spike_tensor"]
        print(f"{net_key}: checkpoint={ckpt_path.exists()} | spike_tensor={spk_path.exists()}")
        if not ckpt_path.exists():
            missing.append(str(ckpt_path))
        if not spk_path.exists():
            missing.append(str(spk_path))
    if missing:
        raise FileNotFoundError("Missing required package files:\n" + "\n".join(missing))

    spike_tensors = {}
    for net_key, meta in TARGETS.items():
        arr = load_spike_tensor(meta["spike_tensor"])
        spike_tensors[net_key] = arr
        print(f"{net_key:22s} shape={arr.shape} mean_rate={arr.mean():.4f}")

    all_sweeps = {}
    observed_rows = []
    for net_key, spk in spike_tensors.items():
        nb_hidden = int(spk.shape[0])
        rows = []
        print(f"\n{net_key}")
        for subset_size in subset_sizes:
            sampled_subsets, total_possible = sample_random_subsets(
                n_neurons=nb_hidden,
                subset_size=subset_size,
                sample_size=args.subset_sample_size,
                seed=_stable_seed(net_key, "subset_sampling", subset_size),
            )

            ok_count = 0
            w_values = []
            m_values = []

            for subset_draw, subset_idx in enumerate(sampled_subsets, start=1):
                try:
                    w_bits, m_bits, n_samp = compute_wm_from_spike_matrix(
                        spk[list(subset_idx), :],
                        lag=args.lag,
                        ridge=args.ridge,
                    )
                    status = "ok"
                    ok_count += 1
                    w_values.append(float(w_bits))
                    m_values.append(float(m_bits))
                except Exception as exc:
                    w_bits, m_bits, n_samp = np.nan, np.nan, 0
                    status = f"error: {exc}"

                row = {
                    "network": net_key,
                    "tensor": "spk",
                    "subset_size": int(subset_size),
                    "subset_draw": int(subset_draw),
                    "subset_indices": ",".join(map(str, subset_idx)),
                    "total_possible_subsets": int(total_possible),
                    "observed_W_bits": float(w_bits) if np.isfinite(w_bits) else np.nan,
                    "observed_M_bits": float(m_bits) if np.isfinite(m_bits) else np.nan,
                    "samples": int(n_samp),
                    "status": status,
                }
                rows.append(row)
                observed_rows.append(row.copy())

            print(
                f"  k={subset_size:2d} | sampled {len(sampled_subsets):3d} / {total_possible} subsets | "
                f"ok={ok_count:3d} | "
                f"mean W={np.nanmean(w_values):9.5f} | "
                f"mean M={np.nanmean(m_values):9.5f}"
            )
        all_sweeps[(net_key, "spk")] = rows

    observed_df = pd.DataFrame.from_records(observed_rows)
    observed_df = observed_df.sort_values(["network", "subset_size", "subset_draw"]).reset_index(drop=True)
    observed_summary = (
        observed_df.groupby(["network", "subset_size"], as_index=False)
        .agg(
            sampled_subsets=("subset_draw", "count"),
            total_possible_subsets=("total_possible_subsets", "first"),
            observed_W_mean=("observed_W_bits", "mean"),
            observed_M_mean=("observed_M_bits", "mean"),
            observed_W_median=("observed_W_bits", "median"),
            observed_M_median=("observed_M_bits", "median"),
        )
    )
    observed_csv = OUTPUT_DIR / "z_scoring_diagnostics_observed_wm.csv"
    observed_df.to_csv(observed_csv, index=False)
    print("\nObserved W/M sweep complete.")
    print(observed_summary)
    print(f"Saved observed W/M CSV: {observed_csv}")

    records = []
    for (net_key, tensor), rows in sorted(all_sweeps.items(), key=lambda x: (x[0][0], x[0][1])):
        print(f"\nScoring {net_key} / {tensor}")
        for subset_size in sorted({int(row["subset_size"]) for row in rows}):
            subset_rows = [row for row in rows if int(row["subset_size"]) == subset_size]
            subset_records = []

            for row in sorted(subset_rows, key=lambda item: int(item.get("subset_draw", 0))):
                obs_m = float(row["observed_M_bits"])
                obs_w = float(row["observed_W_bits"])

                if not (np.isfinite(obs_m) and np.isfinite(obs_w)):
                    continue

                tdmi_target = float(obs_m + obs_w)
                null_dist = build_numit_null_distribution(
                    nvar=subset_size,
                    tdmi_target_bits=tdmi_target,
                    n_null=args.n_null,
                    seed=_stable_seed(net_key, tensor, f"{subset_size}_{row['subset_draw']}"),
                    ridge=args.ridge,
                    max_attempts=args.max_attempts,
                )
                score = score_observed_vs_null_m(obs_m, null_dist["null_M_values"])
                model_mi = np.asarray(null_dist["model_mi_bits"], dtype=np.float64)
                wm_tdmi = np.asarray(null_dist["wm_tdmi_bits"], dtype=np.float64)

                record = {
                    "network": net_key,
                    "tensor": tensor,
                    "subset_size": subset_size,
                    "subset_draw": int(row["subset_draw"]),
                    "subset_indices": row["subset_indices"],
                    "total_possible_subsets": int(row["total_possible_subsets"]),
                    "observed_M_bits": obs_m,
                    "observed_W_bits": obs_w,
                    "observed_TDMI_bits": tdmi_target,
                    "null_M_mean": score["null_mean"],
                    "null_M_std": score["null_std"],
                    "M_z_tdmi_null": score["z"],
                    "M_p_upper_tdmi_null": score["p_upper"],
                    "M_p_lower_tdmi_null": score["p_lower"],
                    "M_p_two_sided_tdmi_null": score["p_two_sided"],
                    "n_null_valid": int(null_dist["n_null_valid"]),
                    "n_attempts": int(null_dist["n_attempts"]),
                    "null_model_mi_mean": float(np.nanmean(model_mi)) if model_mi.size else np.nan,
                    "null_solver_tdmi_mean": float(np.nanmean(wm_tdmi)) if wm_tdmi.size else np.nan,
                    "null_M_values": null_dist["null_M_values"],
                    "null_W_values": null_dist["null_W_values"],
                    "model_mi_bits": null_dist["model_mi_bits"],
                    "wm_tdmi_bits": null_dist["wm_tdmi_bits"],
                }
                records.append(record)
                subset_records.append(record)

            subset_df = pd.DataFrame.from_records(subset_records)
            print(
                f"  k={subset_size:2d} | scored={len(subset_df):3d} | "
                f"mean z={subset_df['M_z_tdmi_null'].mean():8.3f} | "
                f"mean p_upper={subset_df['M_p_upper_tdmi_null'].mean():.4f} | "
                f"mean p_lower={subset_df['M_p_lower_tdmi_null'].mean():.4f}"
            )

    scores_df = pd.DataFrame.from_records(records)
    scores_df = scores_df.sort_values(["network", "subset_size", "subset_draw"]).reset_index(drop=True)
    score_summary = (
        scores_df.groupby(["network", "subset_size"], as_index=False)
        .agg(
            scored_subsets=("subset_draw", "count"),
            total_possible_subsets=("total_possible_subsets", "first"),
            observed_M_mean=("observed_M_bits", "mean"),
            observed_TDMI_mean=("observed_TDMI_bits", "mean"),
            null_M_mean=("null_M_mean", "mean"),
            null_M_std_mean=("null_M_std", "mean"),
            M_z_mean=("M_z_tdmi_null", "mean"),
            M_z_median=("M_z_tdmi_null", "median"),
            M_p_upper_mean=("M_p_upper_tdmi_null", "mean"),
            M_p_lower_mean=("M_p_lower_tdmi_null", "mean"),
            M_p_two_sided_mean=("M_p_two_sided_tdmi_null", "mean"),
            n_null_valid_mean=("n_null_valid", "mean"),
        )
    )
    scores_csv = OUTPUT_DIR / "z_scoring_diagnostics_scores.csv"
    scores_df.drop(columns=["null_M_values", "null_W_values", "model_mi_bits", "wm_tdmi_bits"]).to_csv(scores_csv, index=False)
    print("\nScoring complete.")
    print(score_summary)
    print(f"Saved scores CSV: {scores_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, net_key in zip(axes, ["2class_local_hom", "2class_fittedhet_ln"]):
        sub = scores_df[scores_df["network"] == net_key].sort_values(["subset_size", "subset_draw"])
        subset_values = sorted(sub["subset_size"].unique())
        box_data = [
            sub.loc[sub["subset_size"] == subset_size, "M_z_tdmi_null"].dropna().to_numpy()
            for subset_size in subset_values
        ]
        axis.boxplot(box_data, tick_labels=[str(k) for k in subset_values], showfliers=False)
        axis.axhline(0.0, color="gray", linewidth=1, linestyle="--")
        axis.set_title(net_key + " (spk)")
        axis.set_xlabel("subset size")
        axis.set_ylabel("M z score")
        axis.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "z_scoring_diagnostics_zscores.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()