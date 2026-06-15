from __future__ import annotations

import hashlib
import io
import itertools
import math
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.optimize import root_scalar

import seeded_runs_common as seeded_runs_common
from wimfo.W_M_Info import W_M_calculator


DEFAULT_ROLE_ORDER = ["homogeneous_anchor", "heterogeneous_sampled"]
DEFAULT_ROLE_LABELS = {
    "homogeneous_anchor": "Homogeneous anchor",
    "heterogeneous_sampled": "Heterogeneous sampled",
}

MIRROR_OPTIONS = {
    "steps": 200,
    "lr": 1e-1,
    "proj_max_iters": 500,
    "atol": 1e-4,
    "rtol": 1e-4,
    "window_size": 10,
}

NULL_GATES = ("XOR", "XOR2", "XOR3", "OR", "OR2", "OR3", "OR4")
GATE_ZERO_STATES = {
    "XOR": {(0, 0), (1, 1)},
    "XOR2": {(0, 0), (0, 1)},
    "XOR3": {(0, 0), (1, 0)},
    "OR": {(0, 0)},
    "OR2": {(0, 1)},
    "OR3": {(1, 0)},
    "OR4": {(1, 1)},
}


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _is_cache(obj) -> bool:
    return hasattr(obj, "units") and hasattr(obj, "times") and hasattr(obj, "labels")


def _stable_seed(*parts) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _nanmedian_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _nanmax_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max())


def _nanmin_or_nan(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.min())


def sample_random_subsets(n_neurons: int, subset_size: int, sample_size: int, seed: int):
    total_possible = math.comb(int(n_neurons), int(subset_size))
    target_size = min(int(sample_size), int(total_possible))

    if target_size == total_possible:
        all_subsets = itertools.combinations(range(int(n_neurons)), int(subset_size))
        return [tuple(idx) for idx in all_subsets], total_possible

    rng = np.random.default_rng(int(seed))
    sampled = set()
    while len(sampled) < target_size:
        subset = tuple(sorted(rng.choice(int(n_neurons), size=int(subset_size), replace=False).tolist()))
        sampled.add(subset)

    return sorted(sampled), total_possible


def sample_random_pairs(subset_idx, sample_size: int, seed: int):
    subset_idx = tuple(int(i) for i in subset_idx)
    total_possible = math.comb(len(subset_idx), 2)
    target_size = min(int(sample_size), int(total_possible))

    if target_size == total_possible:
        all_pairs = itertools.combinations(subset_idx, 2)
        return [tuple(pair) for pair in all_pairs], total_possible

    rng = np.random.default_rng(int(seed))
    sampled = set()
    while len(sampled) < target_size:
        pair_pos = sorted(rng.choice(len(subset_idx), size=2, replace=False).tolist())
        sampled.add((subset_idx[pair_pos[0]], subset_idx[pair_pos[1]]))

    return sorted(sampled), total_possible


def _encode_states(bits: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(bits.shape[0], dtype=np.int64))[:, None]
    return np.sum(bits.astype(np.int64) * weights, axis=0)


def _decode_state(idx: int, k: int) -> np.ndarray:
    return np.array([(idx >> bit) & 1 for bit in range(int(k))], dtype=np.int8)


def _mi_from_joint_table(pxy, eps: float = 1e-12) -> float:
    pxy = np.asarray(pxy, dtype=np.float64)
    pxy = np.clip(pxy, 0.0, None)
    total = pxy.sum()
    if total <= 0.0:
        return float("nan")
    pxy = pxy / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log2((pxy[mask] + eps) / (denom[mask] + eps))))


def calculate_discrete_mi(binary_matrix, delay_steps: int, stride: int = 1):
    binary_matrix = np.asarray(binary_matrix, dtype=np.int8)
    stride = max(int(stride), 1)
    binary_matrix = binary_matrix[:, ::stride]
    if binary_matrix.ndim != 2:
        raise ValueError("binary_matrix must be [k x time].")
    k, total_steps = binary_matrix.shape
    if total_steps <= int(delay_steps):
        raise ValueError("Not enough timepoints for requested delay after downsampling.")

    past = binary_matrix[:, :-int(delay_steps)]
    future = binary_matrix[:, int(delay_steps):]
    x_state = _encode_states(past)
    y_state = _encode_states(future)
    n_states = 2 ** int(k)

    counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(counts, (x_state, y_state), 1.0)
    pxy = counts / counts.sum()
    return float(_mi_from_joint_table(pxy)), pxy


def sample_dirichlet_nulls(rng, n_states: int):
    if int(n_states) != 4:
        raise ValueError("The NuMIT-style discrete null is defined for 2 source bits.")
    alpha_scale = float(rng.uniform(0.1, 3.0))
    alpha = np.full(int(n_states), alpha_scale, dtype=np.float64)
    return rng.dirichlet(alpha), alpha_scale


def _logic_proxy(bits: np.ndarray, gate: str) -> int:
    state = (int(bits[0]), int(bits[1]))
    return 0 if state in GATE_ZERO_STATES[gate] else 1


def _build_joint_from_px_logic(px, p_eps: float, gate: str, k: int = 2):
    if int(k) != 2:
        raise ValueError("Current discrete null is constrained to k=2.")

    n_states = 2 ** int(k)
    pxy = np.zeros((n_states, n_states), dtype=np.float64)

    for x_idx in range(n_states):
        x_bits = _decode_state(x_idx, k)
        y_det = np.array([_logic_proxy(x_bits, gate), x_bits[0]], dtype=np.int8)

        for y_idx in range(n_states):
            y_bits = _decode_state(y_idx, k)
            hamming = np.count_nonzero(y_bits != y_det)
            p_cond = (float(p_eps) ** hamming) * ((1.0 - float(p_eps)) ** (int(k) - hamming))
            pxy[x_idx, y_idx] = float(px[x_idx]) * p_cond

    pxy = pxy / pxy.sum()

    pmf4 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for x_idx in range(n_states):
        xb = _decode_state(x_idx, k)
        for y_idx in range(n_states):
            yb = _decode_state(y_idx, k)
            pmf4[xb[0], xb[1], yb[0], yb[1]] = pxy[x_idx, y_idx]

    return pxy, pmf4


def _numit_discrete_from_mi(tdmi_target_bits: float, rng, k: int = 2, tol_bits: float = 0.03):
    if int(k) != 2:
        raise ValueError("Discrete null fitting is constrained to k=2.")

    px, alpha_scale = sample_dirichlet_nulls(rng, n_states=2 ** int(k))
    gate = str(rng.choice(NULL_GATES))

    pxy_zero, _ = _build_joint_from_px_logic(px, p_eps=0.0, gate=gate, k=k)
    max_tdmi_bits = _mi_from_joint_table(pxy_zero)
    if (not np.isfinite(max_tdmi_bits)) or (float(max_tdmi_bits) + float(tol_bits) < float(tdmi_target_bits)):
        return None, None, True, float(max_tdmi_bits), float("nan"), gate, alpha_scale

    def target_fn(p_eps):
        pxy, _ = _build_joint_from_px_logic(px, p_eps=float(p_eps), gate=gate, k=k)
        return _mi_from_joint_table(pxy) - float(tdmi_target_bits)

    grid = np.linspace(0.0, 0.499, 100, dtype=np.float64)
    vals = np.asarray([target_fn(p) for p in grid], dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None, None, True, float("nan"), float("nan"), gate, alpha_scale

    grid = grid[finite]
    vals = vals[finite]
    nearest = int(np.argmin(np.abs(vals)))

    if abs(vals[nearest]) <= float(tol_bits):
        p_star = float(grid[nearest])
    else:
        p_star = None
        for idx in range(len(grid) - 1):
            left = float(vals[idx])
            right = float(vals[idx + 1])
            if left == 0.0:
                p_star = float(grid[idx])
                break
            if left * right > 0.0:
                continue
            try:
                sol = root_scalar(
                    target_fn,
                    bracket=[float(grid[idx]), float(grid[idx + 1])],
                    method="brentq",
                    xtol=1e-6,
                    rtol=1e-6,
                    maxiter=100,
                )
            except Exception:
                sol = None
            if sol is not None and sol.converged and np.isfinite(sol.root):
                p_star = float(sol.root)
                break

    if p_star is None:
        return None, None, True, float("nan"), float("nan"), gate, alpha_scale

    pxy, pmf4 = _build_joint_from_px_logic(px, p_eps=p_star, gate=gate, k=k)
    mi_check = _mi_from_joint_table(pxy)
    if not np.isfinite(mi_check):
        return None, None, True, float("nan"), float("nan"), gate, alpha_scale

    return pxy, pmf4, False, float(mi_check), float(p_star), gate, alpha_scale


def _wm_from_discrete_pmf(joint_pmf_4d, alphabet_size: int = 2):
    with io.StringIO() as buf, redirect_stdout(buf):
        w_bits, m_bits = W_M_calculator(
            joint_pmf_4d,
            option="distr",
            type="discrete",
            alphabet_size=int(alphabet_size),
            unit="bits",
            verbose=False,
            optimiser="Mirror",
            options=MIRROR_OPTIONS,
        )
    return float(w_bits), float(m_bits)


def compute_observed_wm_discrete(binary_matrix, delay_steps: int, stride: int = 1):
    binary_matrix = np.asarray(binary_matrix, dtype=np.int8)
    stride = max(int(stride), 1)
    binary_matrix = binary_matrix[:, ::stride]
    if binary_matrix.shape[0] != 2:
        raise ValueError("Fully discrete wimfo scoring expects exactly 2 neurons per projection.")
    if binary_matrix.shape[1] <= int(delay_steps):
        raise ValueError("Not enough samples after downsampling for the requested delay.")

    x1 = binary_matrix[0, :-int(delay_steps)].astype(np.int64)
    x2 = binary_matrix[1, :-int(delay_steps)].astype(np.int64)
    y1 = binary_matrix[0, int(delay_steps):].astype(np.int64)
    y2 = binary_matrix[1, int(delay_steps):].astype(np.int64)
    data4 = np.vstack([x1, x2, y1, y2])

    with io.StringIO() as buf, redirect_stdout(buf):
        w_bits, m_bits = W_M_calculator(
            data4,
            option="data",
            type="discrete",
            alphabet_size=2,
            unit="bits",
            verbose=False,
            optimiser="Mirror",
            options=MIRROR_OPTIONS,
        )

    tdmi_bits, _ = calculate_discrete_mi(binary_matrix, delay_steps=delay_steps, stride=1)
    return float(w_bits), float(m_bits), float(tdmi_bits)


def build_numit_discrete_null_distribution(nvar: int, tdmi_target_bits: float, n_null: int = 20, seed: int = 12345, max_attempts: int = 1200):
    if int(nvar) != 2:
        raise ValueError("The discrete null generator is constrained to nvar=2.")

    rng = np.random.default_rng(int(seed))
    null_m = []
    null_w = []
    model_mi_vals = []
    model_p_eps = []
    model_gate = []
    model_alpha = []
    attempts = 0

    while len(null_m) < int(n_null) and attempts < int(max_attempts):
        attempts += 1
        pxy, pmf4, failed, mi_model, p_eps, gate, alpha_scale = _numit_discrete_from_mi(
            tdmi_target_bits=tdmi_target_bits,
            rng=rng,
            k=int(nvar),
            tol_bits=0.03,
        )
        if failed:
            continue

        try:
            w_bits, m_bits = _wm_from_discrete_pmf(pmf4, alphabet_size=2)
        except Exception:
            continue

        if not (np.isfinite(w_bits) and np.isfinite(m_bits)):
            continue

        null_w.append(float(w_bits))
        null_m.append(float(m_bits))
        model_mi_vals.append(float(mi_model))
        model_p_eps.append(float(p_eps))
        model_gate.append(str(gate))
        model_alpha.append(float(alpha_scale))

    return {
        "null_M_values": null_m,
        "null_W_values": null_w,
        "model_mi_bits": model_mi_vals,
        "model_p_eps": model_p_eps,
        "model_gate": model_gate,
        "model_alpha": model_alpha,
        "n_null_valid": int(len(null_m)),
        "n_attempts": int(attempts),
    }


def score_observed_vs_null_m(observed_m: float, null_m_values):
    vals = np.asarray(null_m_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "z": float("nan"),
            "p_upper": float("nan"),
            "p_lower": float("nan"),
            "p_two_sided": float("nan"),
        }

    mu = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
    z = float((float(observed_m) - mu) / sd) if (np.isfinite(sd) and sd > 1e-12) else float("nan")
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


def load_seeded_model_from_checkpoint(checkpoint_path):
    payload = _safe_torch_load(Path(checkpoint_path))
    pair = seeded_runs_common.build_seeded_pair(
        master_seed=int(payload["pair_seed"]),
        task_key=payload["task_key"],
        mem_distribution_family=payload["mem_distribution_family"],
    )

    if payload["pair_role"] == "homogeneous_anchor":
        model = pair["hom_model"]
        prms = dict(pair["hom_prms"])
    elif payload["pair_role"] == "heterogeneous_sampled":
        model = pair["hetero_model"]
        prms = dict(pair["hetero_prms"])
    else:
        raise ValueError(f"Unexpected pair role: {payload['pair_role']}")

    prms.update(payload.get("prms", {}))
    prms["dtype"] = torch.float
    prms["device"] = seeded_runs_common.DEVICE
    prms["cuda"] = seeded_runs_common.DEVICE.type == "cuda"
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(seeded_runs_common.DEVICE)
    model.eval()
    return model, prms, payload


@torch.no_grad()
def collect_all_hidden_spikes(model, prms, data_source, max_batches: int = 2, downsample_stride: int = 4):
    stride = max(int(downsample_stride), 1)
    nb_hidden = int(model.network[0].output_size)
    chunks = []
    ctx = seeded_runs_common.shd_open_cached if _is_cache(data_source) else seeded_runs_common.shd_open

    with ctx(data_source) as (units, times, labels):
        for batch_idx, (x, _) in enumerate(
            seeded_runs_common.shd_generator(
                units,
                times,
                labels,
                prms,
                shuffle=False,
                epoch=0,
                drop_last=False,
            )
        ):
            if batch_idx >= int(max_batches):
                break
            x = x.to(seeded_runs_common.DEVICE)
            layer_recs = model(0, 0, x)
            spikes = layer_recs[0][2][:, ::stride, :].detach().cpu().numpy()
            if not np.all(np.isin(spikes, [0.0, 1.0])):
                raise ValueError("Collected spike tensor is not strictly binary.")
            spikes = np.transpose(spikes, (2, 0, 1)).reshape(nb_hidden, -1)
            chunks.append(spikes.astype(np.int8, copy=False))

    if not chunks:
        raise RuntimeError("No batches were collected for discrete spike analysis.")
    return np.concatenate(chunks, axis=1).astype(np.int8, copy=False)


def compute_delay_metadata(prms: dict, downsample_stride: int = 4, delay_target_ms: float = 15.0):
    raw_step_ms = float(prms["time_step"]) * 1e3
    effective_step_ms = float(raw_step_ms) * max(int(downsample_stride), 1)
    delay_steps = max(1, int(round(float(delay_target_ms) / float(effective_step_ms))))
    actual_delay_ms = float(delay_steps) * float(effective_step_ms)
    return {
        "raw_step_ms": float(raw_step_ms),
        "effective_step_ms": float(effective_step_ms),
        "delay_steps": int(delay_steps),
        "actual_delay_ms": float(actual_delay_ms),
    }


def run_discrete_projection_scan(
    checkpoint_table,
    data_source,
    subset_sizes,
    subset_sample_size: int,
    pair_sample_size: int,
    n_null: int,
    max_batches: int = 2,
    downsample_stride: int = 4,
    delay_target_ms: float = 15.0,
    max_attempts: int = 1200,
    role_order=None,
    role_labels=None,
    label: str = "run",
):
    role_order = list(role_order or DEFAULT_ROLE_ORDER)
    role_labels = dict(DEFAULT_ROLE_LABELS | (role_labels or {}))

    pair_records = []
    subset_records = []
    ordered = checkpoint_table.sort_values(["pair_seed", "pair_role"]).reset_index(drop=True)

    for _, record in ordered.iterrows():
        record_dict = record.to_dict()
        record_dict["pair_seed"] = int(record_dict["pair_seed"])
        record_dict["pair_role_label"] = role_labels.get(record_dict.get("pair_role"), record_dict.get("pair_role"))
        print(f"Running discrete projection scan for seed {record_dict['pair_seed']} :: {record_dict['pair_role']}")

        model, prms, payload = load_seeded_model_from_checkpoint(record_dict["checkpoint"])
        spike_matrix = collect_all_hidden_spikes(
            model,
            prms,
            data_source,
            max_batches=max_batches,
            downsample_stride=downsample_stride,
        )
        delay_meta = compute_delay_metadata(
            prms,
            downsample_stride=downsample_stride,
            delay_target_ms=delay_target_ms,
        )
        n_neurons = int(spike_matrix.shape[0])

        for subset_size in [int(value) for value in subset_sizes]:
            sampled_subsets, total_possible_subsets = sample_random_subsets(
                n_neurons=n_neurons,
                subset_size=subset_size,
                sample_size=int(subset_sample_size),
                seed=_stable_seed(
                    label,
                    record_dict["run_label"],
                    record_dict["pair_seed"],
                    record_dict["pair_role"],
                    subset_size,
                    "subset_sampling",
                ),
            )

            subset_z_means = []
            for subset_draw, subset_idx in enumerate(sampled_subsets, start=1):
                sampled_pairs, total_possible_pairs = sample_random_pairs(
                    subset_idx=subset_idx,
                    sample_size=int(pair_sample_size),
                    seed=_stable_seed(
                        label,
                        record_dict["run_label"],
                        record_dict["pair_seed"],
                        record_dict["pair_role"],
                        subset_size,
                        subset_draw,
                        "pair_sampling",
                    ),
                )

                subset_pair_rows = []
                for pair_draw, (pair_i, pair_j) in enumerate(sampled_pairs, start=1):
                    try:
                        obs_w, obs_m, obs_tdmi_counts = compute_observed_wm_discrete(
                            spike_matrix[[int(pair_i), int(pair_j)], :],
                            delay_steps=delay_meta["delay_steps"],
                            stride=1,
                        )
                        observed_tdmi = float(obs_w + obs_m)
                        null_dist = build_numit_discrete_null_distribution(
                            nvar=2,
                            tdmi_target_bits=observed_tdmi,
                            n_null=int(n_null),
                            seed=_stable_seed(
                                label,
                                record_dict["run_label"],
                                record_dict["pair_seed"],
                                record_dict["pair_role"],
                                subset_size,
                                subset_draw,
                                pair_i,
                                pair_j,
                            ),
                            max_attempts=int(max_attempts),
                        )
                        score = score_observed_vs_null_m(obs_m, null_dist["null_M_values"])
                        model_mi = np.asarray(null_dist["model_mi_bits"], dtype=np.float64)
                        model_p_eps = np.asarray(null_dist["model_p_eps"], dtype=np.float64)
                        model_alpha = np.asarray(null_dist["model_alpha"], dtype=np.float64)
                        status = "ok"
                    except Exception as exc:
                        obs_w = float("nan")
                        obs_m = float("nan")
                        obs_tdmi_counts = float("nan")
                        observed_tdmi = float("nan")
                        null_dist = {
                            "n_null_valid": 0,
                            "n_attempts": 0,
                            "model_gate": [],
                        }
                        score = score_observed_vs_null_m(float("nan"), [])
                        model_mi = np.asarray([], dtype=np.float64)
                        model_p_eps = np.asarray([], dtype=np.float64)
                        model_alpha = np.asarray([], dtype=np.float64)
                        status = f"error: {exc}"

                    pair_record = dict(record_dict)
                    pair_record.update(
                        {
                            "label": label,
                            "subset_size": int(subset_size),
                            "subset_draw": int(subset_draw),
                            "subset_indices": ",".join(map(str, subset_idx)),
                            "total_possible_subsets": int(total_possible_subsets),
                            "sampled_pairs_in_subset": int(len(sampled_pairs)),
                            "total_possible_pairs_in_subset": int(total_possible_pairs),
                            "pair_draw": int(pair_draw),
                            "pair_i": int(pair_i),
                            "pair_j": int(pair_j),
                            "delay_steps": int(delay_meta["delay_steps"]),
                            "delay_ms": float(delay_meta["actual_delay_ms"]),
                            "raw_step_ms": float(delay_meta["raw_step_ms"]),
                            "effective_step_ms": float(delay_meta["effective_step_ms"]),
                            "observed_W_bits": float(obs_w) if np.isfinite(obs_w) else float("nan"),
                            "observed_M_bits": float(obs_m) if np.isfinite(obs_m) else float("nan"),
                            "observed_TDMI_bits": float(observed_tdmi) if np.isfinite(observed_tdmi) else float("nan"),
                            "observed_TDMI_counts_bits": float(obs_tdmi_counts) if np.isfinite(obs_tdmi_counts) else float("nan"),
                            "null_M_mean": score["null_mean"],
                            "null_M_std": score["null_std"],
                            "M_z_tdmi_null": score["z"],
                            "M_p_upper_tdmi_null": score["p_upper"],
                            "M_p_lower_tdmi_null": score["p_lower"],
                            "M_p_two_sided_tdmi_null": score["p_two_sided"],
                            "n_null_valid": int(null_dist["n_null_valid"]),
                            "n_attempts": int(null_dist["n_attempts"]),
                            "null_model_mi_mean": _nanmean_or_nan(model_mi),
                            "null_model_p_eps_mean": _nanmean_or_nan(model_p_eps),
                            "null_model_alpha_mean": _nanmean_or_nan(model_alpha),
                            "null_gate_modes": ",".join(sorted(set(null_dist["model_gate"]))),
                            "status": status,
                        }
                    )
                    pair_records.append(pair_record)
                    subset_pair_rows.append(pair_record)

                subset_pair_df = pd.DataFrame.from_records(subset_pair_rows)
                subset_z = subset_pair_df["M_z_tdmi_null"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_upper = subset_pair_df["M_p_upper_tdmi_null"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_lower = subset_pair_df["M_p_lower_tdmi_null"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_pair_m = subset_pair_df["observed_M_bits"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_pair_w = subset_pair_df["observed_W_bits"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_pair_tdmi = subset_pair_df["observed_TDMI_bits"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)

                subset_record = dict(record_dict)
                subset_record.update(
                    {
                        "label": label,
                        "subset_size": int(subset_size),
                        "subset_draw": int(subset_draw),
                        "subset_indices": ",".join(map(str, subset_idx)),
                        "total_possible_subsets": int(total_possible_subsets),
                        "sampled_pairs": int(len(sampled_pairs)),
                        "total_possible_pairs": int(total_possible_pairs),
                        "delay_steps": int(delay_meta["delay_steps"]),
                        "delay_ms": float(delay_meta["actual_delay_ms"]),
                        "subset_pair_z_mean": _nanmean_or_nan(subset_z),
                        "subset_pair_z_median": _nanmedian_or_nan(subset_z),
                        "subset_pair_z_max": _nanmax_or_nan(subset_z),
                        "subset_pair_z_min": _nanmin_or_nan(subset_z),
                        "subset_pair_p_upper_mean": _nanmean_or_nan(subset_upper),
                        "subset_pair_p_lower_mean": _nanmean_or_nan(subset_lower),
                        "subset_pair_M_mean": _nanmean_or_nan(subset_pair_m),
                        "subset_pair_W_mean": _nanmean_or_nan(subset_pair_w),
                        "subset_pair_TDMI_mean": _nanmean_or_nan(subset_pair_tdmi),
                        "subset_positive_frac": float(np.mean(subset_z > 0.0)) if subset_z.size else float("nan"),
                    }
                )
                subset_records.append(subset_record)
                subset_z_means.append(subset_record["subset_pair_z_mean"])

            finite_subset_z = [value for value in subset_z_means if np.isfinite(value)]
            mean_subset_z = float(np.mean(finite_subset_z)) if finite_subset_z else float("nan")
            print(
                f"  k={subset_size:2d} | sampled {len(sampled_subsets):3d}/{total_possible_subsets} subsets | "
                f"mean subset Z={mean_subset_z:8.3f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_df = pd.DataFrame.from_records(pair_records)
    subset_df = pd.DataFrame.from_records(subset_records)
    if pair_df.empty or subset_df.empty:
        raise RuntimeError("The discrete projection scan did not produce any rows.")

    pair_df = pair_df.sort_values(["pair_seed", "pair_role", "subset_size", "subset_draw", "pair_draw"]).reset_index(drop=True)
    subset_df = subset_df.sort_values(["pair_seed", "pair_role", "subset_size", "subset_draw"]).reset_index(drop=True)

    group_cols = [
        "run_label",
        "pair_seed",
        "pair_role",
        "pair_role_label",
        "task_key",
        "task_name",
        "mem_distribution_family",
        "checkpoint",
        "checkpoint_name",
        "hidden_tau_syn_ms",
        "hidden_tau_mem_geom_mean_ms",
        "final_test_acc",
        "final_test_loss",
        "label",
        "subset_size",
    ]
    model_df = (
        subset_df.groupby(group_cols, as_index=False)
        .agg(
            sampled_subsets=("subset_draw", "count"),
            sampled_pairs_mean=("sampled_pairs", "mean"),
            subset_pair_z_mean=("subset_pair_z_mean", "mean"),
            subset_pair_z_std=("subset_pair_z_mean", "std"),
            subset_pair_z_median=("subset_pair_z_median", "mean"),
            subset_pair_z_max=("subset_pair_z_max", "max"),
            subset_pair_z_min=("subset_pair_z_min", "min"),
            subset_pair_M_mean=("subset_pair_M_mean", "mean"),
            subset_pair_W_mean=("subset_pair_W_mean", "mean"),
            subset_pair_TDMI_mean=("subset_pair_TDMI_mean", "mean"),
            subset_positive_frac=("subset_positive_frac", "mean"),
            subset_pair_p_upper_mean=("subset_pair_p_upper_mean", "mean"),
            subset_pair_p_lower_mean=("subset_pair_p_lower_mean", "mean"),
            delay_steps=("delay_steps", "mean"),
            delay_ms=("delay_ms", "mean"),
        )
        .sort_values(["pair_seed", "pair_role", "subset_size"])
        .reset_index(drop=True)
    )
    model_df["delay_steps"] = model_df["delay_steps"].round().astype(int)
    return pair_df, subset_df, model_df


def summarise_pair_deltas(model_df):
    pivot_values = [
        column
        for column in ["subset_pair_z_mean", "subset_pair_M_mean", "subset_pair_W_mean", "subset_positive_frac"]
        if column in model_df.columns
    ]
    if not pivot_values:
        raise KeyError("No supported paired summary columns were found in the provided model summary table.")

    paired = model_df.pivot_table(
        index=["pair_seed", "subset_size"],
        columns="pair_role",
        values=pivot_values,
    )
    paired.columns = [f"{metric}_{role}" for metric, role in paired.columns]
    paired = paired.reset_index().sort_values(["pair_seed", "subset_size"]).reset_index(drop=True)

    if {"subset_pair_z_mean_homogeneous_anchor", "subset_pair_z_mean_heterogeneous_sampled"}.issubset(paired.columns):
        paired["hetero_minus_hom_subset_pair_z_mean"] = (
            paired["subset_pair_z_mean_heterogeneous_sampled"] - paired["subset_pair_z_mean_homogeneous_anchor"]
        )
    if {"subset_pair_M_mean_homogeneous_anchor", "subset_pair_M_mean_heterogeneous_sampled"}.issubset(paired.columns):
        paired["hetero_minus_hom_subset_pair_M_mean"] = (
            paired["subset_pair_M_mean_heterogeneous_sampled"] - paired["subset_pair_M_mean_homogeneous_anchor"]
        )
    if {"subset_pair_W_mean_homogeneous_anchor", "subset_pair_W_mean_heterogeneous_sampled"}.issubset(paired.columns):
        paired["hetero_minus_hom_subset_pair_W_mean"] = (
            paired["subset_pair_W_mean_heterogeneous_sampled"] - paired["subset_pair_W_mean_homogeneous_anchor"]
        )
    return paired


def save_analysis_artifacts(pair_df, subset_df, model_df, output_dir, role_order=None, role_labels=None):
    role_order = list(role_order or DEFAULT_ROLE_ORDER)
    role_labels = dict(DEFAULT_ROLE_LABELS | (role_labels or {}))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_summary_df = (
        model_df.groupby(["pair_role", "pair_role_label", "subset_size"], as_index=False)
        .agg(
            subset_pair_z_mean_mean=("subset_pair_z_mean", "mean"),
            subset_pair_z_mean_std=("subset_pair_z_mean", "std"),
            subset_pair_M_mean_mean=("subset_pair_M_mean", "mean"),
            subset_pair_M_mean_std=("subset_pair_M_mean", "std"),
            subset_pair_W_mean_mean=("subset_pair_W_mean", "mean"),
            subset_pair_W_mean_std=("subset_pair_W_mean", "std"),
            final_test_acc_mean=("final_test_acc", "mean"),
            n_models=("checkpoint", "nunique"),
        )
        .sort_values(["pair_role", "subset_size"])
        .reset_index(drop=True)
    )
    delta_df = summarise_pair_deltas(model_df)

    paths = {
        "pair_csv": output_dir / "seeded_runs_discrete_pair_records.csv",
        "subset_csv": output_dir / "seeded_runs_discrete_subset_records.csv",
        "model_csv": output_dir / "seeded_runs_discrete_model_summary.csv",
        "role_summary_csv": output_dir / "seeded_runs_discrete_role_summary.csv",
        "pair_delta_csv": output_dir / "seeded_runs_discrete_pair_delta.csv",
        "subset_plot": output_dir / "seeded_runs_discrete_z_by_subset.png",
        "scatter_plot": output_dir / "seeded_runs_discrete_z_vs_accuracy.png",
    }

    pair_df.to_csv(paths["pair_csv"], index=False)
    subset_df.to_csv(paths["subset_csv"], index=False)
    model_df.to_csv(paths["model_csv"], index=False)
    role_summary_df.to_csv(paths["role_summary_csv"], index=False)
    delta_df.to_csv(paths["pair_delta_csv"], index=False)

    plot_df = model_df.dropna(subset=["subset_pair_z_mean"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    if not plot_df.empty:
        sns.lineplot(
            data=plot_df,
            x="subset_size",
            y="subset_pair_z_mean",
            hue="pair_role_label",
            style="pair_role_label",
            estimator="mean",
            errorbar="sd",
            marker="o",
            dashes=False,
            ax=axes[0],
        )
    axes[0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_title("Discrete pair-projection Z by subset size")
    axes[0].set_xlabel("Subset size")
    axes[0].set_ylabel("Subset-mean pair Z score")

    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    if "hetero_minus_hom_subset_pair_z_mean" in delta_df.columns:
        for pair_seed, seed_slice in delta_df.groupby("pair_seed"):
            axes[1].plot(
                seed_slice["subset_size"],
                seed_slice["hetero_minus_hom_subset_pair_z_mean"],
                color="0.85",
                linewidth=1,
                zorder=1,
            )
        sns.lineplot(
            data=delta_df,
            x="subset_size",
            y="hetero_minus_hom_subset_pair_z_mean",
            marker="o",
            estimator="mean",
            errorbar="sd",
            color="black",
            ax=axes[1],
        )
    axes[1].set_title("Paired discrete Z delta")
    axes[1].set_xlabel("Subset size")
    axes[1].set_ylabel("Heterogeneous - homogeneous")
    fig.savefig(paths["subset_plot"], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter_df = plot_df.copy()
    scatter_df["subset_label"] = scatter_df["subset_size"].astype(str)
    if not scatter_df.empty:
        sns.scatterplot(
            data=scatter_df,
            x="final_test_acc",
            y="subset_pair_z_mean",
            hue="pair_role_label",
            style="subset_label",
            s=110,
            ax=ax,
        )
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    ax.set_title("Accuracy vs discrete pair-projection Z")
    ax.set_xlabel("Final test accuracy")
    ax.set_ylabel("Subset-mean pair Z score")
    fig.savefig(paths["scatter_plot"], bbox_inches="tight")
    plt.close(fig)
    return role_summary_df, delta_df, paths


def build_visual_suite(model_df, delta_df, accuracy_df, output_dir, role_order=None, role_labels=None, subset_focus=None):
    role_order = list(role_order or DEFAULT_ROLE_ORDER)
    role_labels = dict(DEFAULT_ROLE_LABELS | (role_labels or {}))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subset_focus is None:
        subset_focus = int(model_df["subset_size"].max())

    role_palette = {
        role_labels["homogeneous_anchor"]: "#4C72B0",
        role_labels["heterogeneous_sampled"]: "#DD8452",
    }

    plot_df = model_df.dropna(subset=["subset_pair_z_mean", "subset_pair_M_mean", "final_test_acc"]).copy()
    plot_df["subset_size"] = plot_df["subset_size"].astype(int)
    pair_meta_df = (
        plot_df[["pair_seed", "pair_role", "pair_role_label", "final_test_acc"]]
        .drop_duplicates()
        .sort_values(["pair_seed", "pair_role"])
        .reset_index(drop=True)
    )
    subset_focus_df = (
        plot_df[plot_df["subset_size"] == int(subset_focus)]
        .sort_values(["pair_seed", "pair_role"])
        .reset_index(drop=True)
    )
    if subset_focus_df.empty:
        raise RuntimeError(f"No discrete rows are available for subset size {subset_focus}.")

    if "hetero_minus_hom_acc" not in accuracy_df.columns and set(role_order).issubset(accuracy_df.columns):
        accuracy_df = accuracy_df.copy()
        accuracy_df["hetero_minus_hom_acc"] = accuracy_df["heterogeneous_sampled"] - accuracy_df["homogeneous_anchor"]

    delta_plot_df = delta_df.sort_values(["pair_seed", "subset_size"]).reset_index(drop=True).copy()
    delta_relationship_df = delta_plot_df.merge(
        accuracy_df[["pair_seed", "hetero_minus_hom_acc"]],
        on="pair_seed",
        how="left",
    )
    delta_focus_df = delta_relationship_df[delta_relationship_df["subset_size"] == int(subset_focus)].copy()

    pair_seeds = sorted(pair_meta_df["pair_seed"].unique())
    seed_positions = {seed: idx for idx, seed in enumerate(pair_seeds)}
    ordered_subset_sizes = sorted(plot_df["subset_size"].unique())
    paths = {}

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, constrained_layout=True)
    for pair_seed in pair_seeds:
        y_pos = seed_positions[pair_seed]

        acc_slice = (
            pair_meta_df[pair_meta_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if acc_slice["final_test_acc"].notna().all():
            axes[0].plot(acc_slice["final_test_acc"], [y_pos, y_pos], color="0.8", linewidth=1.25, zorder=1)
        for role, row in acc_slice.iterrows():
            if pd.notna(row["final_test_acc"]):
                axes[0].scatter(
                    row["final_test_acc"],
                    y_pos,
                    color=role_palette[row["pair_role_label"]],
                    s=80,
                    zorder=2,
                )

        z_slice = (
            subset_focus_df[subset_focus_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if z_slice["subset_pair_z_mean"].notna().all():
            axes[1].plot(z_slice["subset_pair_z_mean"], [y_pos, y_pos], color="0.8", linewidth=1.25, zorder=1)
        for role, row in z_slice.iterrows():
            if pd.notna(row["subset_pair_z_mean"]):
                axes[1].scatter(
                    row["subset_pair_z_mean"],
                    y_pos,
                    color=role_palette[row["pair_role_label"]],
                    s=80,
                    zorder=2,
                )

    for ax in axes:
        ax.set_yticks(range(len(pair_seeds)))
        ax.set_yticklabels([str(seed) for seed in pair_seeds])
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_title("Paired task accuracy by seeded pair")
    axes[0].set_xlabel("Final test accuracy")
    axes[0].set_ylabel("Pair seed")
    axes[1].set_title(f"Paired discrete Z at subset size {subset_focus}")
    axes[1].set_xlabel("Subset-mean pair Z score")
    for label, color in role_palette.items():
        axes[1].scatter([], [], color=color, label=label, s=80)
    axes[1].legend(title="", loc="lower right")
    paths["pairwise_compare_plot"] = output_dir / "seeded_runs_discrete_pairwise_accuracy_z.png"
    fig.savefig(paths["pairwise_compare_plot"], bbox_inches="tight")
    plt.close(fig)

    hom_heatmap = (
        plot_df[plot_df["pair_role"] == "homogeneous_anchor"]
        .pivot(index="pair_seed", columns="subset_size", values="subset_pair_z_mean")
        .reindex(index=pair_seeds, columns=ordered_subset_sizes)
    )
    hetero_heatmap = (
        plot_df[plot_df["pair_role"] == "heterogeneous_sampled"]
        .pivot(index="pair_seed", columns="subset_size", values="subset_pair_z_mean")
        .reindex(index=pair_seeds, columns=ordered_subset_sizes)
    )
    delta_heatmap = (
        delta_plot_df.pivot(index="pair_seed", columns="subset_size", values="hetero_minus_hom_subset_pair_z_mean")
        .reindex(index=pair_seeds, columns=ordered_subset_sizes)
    )
    value_floor = float(min(hom_heatmap.min().min(), hetero_heatmap.min().min()))
    value_ceiling = float(max(hom_heatmap.max().max(), hetero_heatmap.max().max()))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    sns.heatmap(
        hom_heatmap,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=value_floor,
        vmax=value_ceiling,
        cbar_kws={"label": "Subset-mean pair Z"},
        ax=axes[0],
    )
    axes[0].set_title("Homogeneous discrete Z heatmap")
    axes[0].set_xlabel("Subset size")
    axes[0].set_ylabel("Pair seed")

    sns.heatmap(
        hetero_heatmap,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        vmin=value_floor,
        vmax=value_ceiling,
        cbar_kws={"label": "Subset-mean pair Z"},
        ax=axes[1],
    )
    axes[1].set_title("Heterogeneous discrete Z heatmap")
    axes[1].set_xlabel("Subset size")
    axes[1].set_ylabel("")

    sns.heatmap(
        delta_heatmap,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "Heterogeneous - homogeneous"},
        ax=axes[2],
    )
    axes[2].set_title("Paired discrete Z delta heatmap")
    axes[2].set_xlabel("Subset size")
    axes[2].set_ylabel("")
    paths["heatmap_plot"] = output_dir / "seeded_runs_discrete_pairwise_z_heatmaps.png"
    fig.savefig(paths["heatmap_plot"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for pair_seed in pair_seeds:
        pair_slice = (
            subset_focus_df[subset_focus_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if pair_slice[["final_test_acc", "subset_pair_z_mean"]].notna().all().all():
            axes[0].plot(pair_slice["final_test_acc"], pair_slice["subset_pair_z_mean"], color="0.85", linewidth=1.1, zorder=1)

    sns.scatterplot(
        data=subset_focus_df,
        x="final_test_acc",
        y="subset_pair_z_mean",
        hue="pair_role_label",
        palette=role_palette,
        s=120,
        ax=axes[0],
    )
    for row in subset_focus_df.itertuples():
        axes[0].text(row.final_test_acc + 0.0015, row.subset_pair_z_mean + 0.015, str(int(row.pair_seed)), fontsize=8, alpha=0.75)
    axes[0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_title(f"Accuracy vs discrete Z at subset size {subset_focus}")
    axes[0].set_xlabel("Final test accuracy")
    axes[0].set_ylabel("Subset-mean pair Z score")

    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1].axvline(0.0, color="0.5", linestyle="--", linewidth=1)
    sns.scatterplot(
        data=delta_relationship_df,
        x="hetero_minus_hom_acc",
        y="hetero_minus_hom_subset_pair_z_mean",
        hue="subset_size",
        palette="viridis",
        s=100,
        ax=axes[1],
    )
    for row in delta_focus_df.itertuples():
        axes[1].text(
            row.hetero_minus_hom_acc + 0.0015,
            row.hetero_minus_hom_subset_pair_z_mean + 0.015,
            str(int(row.pair_seed)),
            fontsize=8,
            alpha=0.75,
        )
    axes[1].set_title("Paired accuracy delta vs discrete Z delta")
    axes[1].set_xlabel("Heterogeneous - homogeneous accuracy")
    axes[1].set_ylabel("Heterogeneous - homogeneous subset Z")
    axes[1].legend(title="Subset size")
    paths["relationship_plot"] = output_dir / "seeded_runs_discrete_accuracy_z_relationships.png"
    fig.savefig(paths["relationship_plot"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    x_positions = {role: idx for idx, role in enumerate(role_order)}
    for pair_seed in pair_seeds:
        acc_slice = (
            pair_meta_df[pair_meta_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if acc_slice["final_test_acc"].notna().all():
            axes[0, 0].plot(
                [x_positions[role_order[0]], x_positions[role_order[1]]],
                acc_slice["final_test_acc"],
                color="0.8",
                linewidth=1,
                zorder=1,
            )
        for role, row in acc_slice.iterrows():
            if pd.notna(row["final_test_acc"]):
                axes[0, 0].scatter(
                    x_positions[role],
                    row["final_test_acc"],
                    color=role_palette[row["pair_role_label"]],
                    s=75,
                    zorder=2,
                )
    axes[0, 0].set_xticks(range(len(role_order)))
    axes[0, 0].set_xticklabels([role_labels[role] for role in role_order], rotation=12, ha="right")
    axes[0, 0].set_title("Paired accuracy summary")
    axes[0, 0].set_ylabel("Final test accuracy")

    sns.lineplot(
        data=plot_df,
        x="subset_size",
        y="subset_pair_z_mean",
        hue="pair_role_label",
        palette=role_palette,
        estimator="mean",
        errorbar="sd",
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Mean discrete Z by subset size")
    axes[0, 1].set_xlabel("Subset size")
    axes[0, 1].set_ylabel("Subset-mean pair Z score")
    axes[0, 1].legend(title="")

    sns.lineplot(
        data=plot_df,
        x="subset_size",
        y="subset_pair_M_mean",
        hue="pair_role_label",
        palette=role_palette,
        estimator="mean",
        errorbar="sd",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Mean observed discrete M by subset size")
    axes[1, 0].set_xlabel("Subset size")
    axes[1, 0].set_ylabel("Observed pair M (bits)")
    if axes[1, 0].legend_ is not None:
        axes[1, 0].legend_.remove()

    axes[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].axvline(0.0, color="0.5", linestyle="--", linewidth=1)
    sns.scatterplot(
        data=delta_focus_df,
        x="hetero_minus_hom_acc",
        y="hetero_minus_hom_subset_pair_z_mean",
        s=120,
        color="#2A9D8F",
        ax=axes[1, 1],
    )
    for row in delta_focus_df.itertuples():
        axes[1, 1].text(
            row.hetero_minus_hom_acc + 0.0015,
            row.hetero_minus_hom_subset_pair_z_mean + 0.015,
            str(int(row.pair_seed)),
            fontsize=8,
            alpha=0.75,
        )
    axes[1, 1].set_title(f"Unified discrete delta view at subset size {subset_focus}")
    axes[1, 1].set_xlabel("Heterogeneous - homogeneous accuracy")
    axes[1, 1].set_ylabel("Heterogeneous - homogeneous subset Z")
    paths["dashboard_plot"] = output_dir / "seeded_runs_discrete_dashboard.png"
    fig.savefig(paths["dashboard_plot"], bbox_inches="tight")
    plt.close(fig)
    return paths


def run_discrete_observed_scan(
    checkpoint_table,
    data_source,
    subset_sizes,
    subset_sample_size: int,
    pair_sample_size: int,
    max_batches: int = 2,
    downsample_stride: int = 4,
    delay_target_ms: float = 15.0,
    role_order=None,
    role_labels=None,
    label: str = "observed_run",
):
    role_order = list(role_order or DEFAULT_ROLE_ORDER)
    role_labels = dict(DEFAULT_ROLE_LABELS | (role_labels or {}))

    pair_records = []
    subset_records = []
    ordered = checkpoint_table.sort_values(["pair_seed", "pair_role"]).reset_index(drop=True)

    for _, record in ordered.iterrows():
        record_dict = record.to_dict()
        record_dict["pair_seed"] = int(record_dict["pair_seed"])
        record_dict["pair_role_label"] = role_labels.get(record_dict.get("pair_role"), record_dict.get("pair_role"))
        print(f"Running observed discrete scan for seed {record_dict['pair_seed']} :: {record_dict['pair_role']}")

        model, prms, _ = load_seeded_model_from_checkpoint(record_dict["checkpoint"])
        spike_matrix = collect_all_hidden_spikes(
            model,
            prms,
            data_source,
            max_batches=max_batches,
            downsample_stride=downsample_stride,
        )
        delay_meta = compute_delay_metadata(
            prms,
            downsample_stride=downsample_stride,
            delay_target_ms=delay_target_ms,
        )
        n_neurons = int(spike_matrix.shape[0])

        for subset_size in [int(value) for value in subset_sizes]:
            sampled_subsets, total_possible_subsets = sample_random_subsets(
                n_neurons=n_neurons,
                subset_size=subset_size,
                sample_size=int(subset_sample_size),
                seed=_stable_seed(
                    label,
                    record_dict["run_label"],
                    record_dict["pair_seed"],
                    record_dict["pair_role"],
                    subset_size,
                    "subset_sampling",
                ),
            )

            subset_m_means = []
            subset_w_means = []
            for subset_draw, subset_idx in enumerate(sampled_subsets, start=1):
                sampled_pairs, total_possible_pairs = sample_random_pairs(
                    subset_idx=subset_idx,
                    sample_size=int(pair_sample_size),
                    seed=_stable_seed(
                        label,
                        record_dict["run_label"],
                        record_dict["pair_seed"],
                        record_dict["pair_role"],
                        subset_size,
                        subset_draw,
                        "pair_sampling",
                    ),
                )

                subset_pair_rows = []
                for pair_draw, (pair_i, pair_j) in enumerate(sampled_pairs, start=1):
                    try:
                        obs_w, obs_m, obs_tdmi_counts = compute_observed_wm_discrete(
                            spike_matrix[[int(pair_i), int(pair_j)], :],
                            delay_steps=delay_meta["delay_steps"],
                            stride=1,
                        )
                        observed_tdmi = float(obs_w + obs_m)
                        status = "ok"
                    except Exception as exc:
                        obs_w = float("nan")
                        obs_m = float("nan")
                        obs_tdmi_counts = float("nan")
                        observed_tdmi = float("nan")
                        status = f"error: {exc}"

                    pair_record = dict(record_dict)
                    pair_record.update(
                        {
                            "label": label,
                            "subset_size": int(subset_size),
                            "subset_draw": int(subset_draw),
                            "subset_indices": ",".join(map(str, subset_idx)),
                            "total_possible_subsets": int(total_possible_subsets),
                            "sampled_pairs_in_subset": int(len(sampled_pairs)),
                            "total_possible_pairs_in_subset": int(total_possible_pairs),
                            "pair_draw": int(pair_draw),
                            "pair_i": int(pair_i),
                            "pair_j": int(pair_j),
                            "delay_steps": int(delay_meta["delay_steps"]),
                            "delay_ms": float(delay_meta["actual_delay_ms"]),
                            "raw_step_ms": float(delay_meta["raw_step_ms"]),
                            "effective_step_ms": float(delay_meta["effective_step_ms"]),
                            "observed_W_bits": float(obs_w) if np.isfinite(obs_w) else float("nan"),
                            "observed_M_bits": float(obs_m) if np.isfinite(obs_m) else float("nan"),
                            "observed_TDMI_bits": float(observed_tdmi) if np.isfinite(observed_tdmi) else float("nan"),
                            "observed_TDMI_counts_bits": float(obs_tdmi_counts) if np.isfinite(obs_tdmi_counts) else float("nan"),
                            "status": status,
                        }
                    )
                    pair_records.append(pair_record)
                    subset_pair_rows.append(pair_record)

                subset_pair_df = pd.DataFrame.from_records(subset_pair_rows)
                subset_pair_m = subset_pair_df["observed_M_bits"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_pair_w = subset_pair_df["observed_W_bits"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)
                subset_pair_tdmi = subset_pair_df["observed_TDMI_bits"].dropna().to_numpy() if not subset_pair_df.empty else np.asarray([], dtype=np.float64)

                subset_record = dict(record_dict)
                subset_record.update(
                    {
                        "label": label,
                        "subset_size": int(subset_size),
                        "subset_draw": int(subset_draw),
                        "subset_indices": ",".join(map(str, subset_idx)),
                        "total_possible_subsets": int(total_possible_subsets),
                        "sampled_pairs": int(len(sampled_pairs)),
                        "total_possible_pairs": int(total_possible_pairs),
                        "delay_steps": int(delay_meta["delay_steps"]),
                        "delay_ms": float(delay_meta["actual_delay_ms"]),
                        "subset_pair_M_mean": _nanmean_or_nan(subset_pair_m),
                        "subset_pair_M_std": float(np.nanstd(subset_pair_m, ddof=1)) if subset_pair_m.size > 1 else float("nan"),
                        "subset_pair_W_mean": _nanmean_or_nan(subset_pair_w),
                        "subset_pair_W_std": float(np.nanstd(subset_pair_w, ddof=1)) if subset_pair_w.size > 1 else float("nan"),
                        "subset_pair_TDMI_mean": _nanmean_or_nan(subset_pair_tdmi),
                        "valid_pair_count": int(np.isfinite(subset_pair_df.get("observed_M_bits", pd.Series(dtype=float))).sum()),
                        "error_pair_count": int((subset_pair_df.get("status", pd.Series(dtype=str)) != "ok").sum()) if not subset_pair_df.empty else 0,
                    }
                )
                subset_records.append(subset_record)
                subset_m_means.append(subset_record["subset_pair_M_mean"])
                subset_w_means.append(subset_record["subset_pair_W_mean"])

            mean_subset_m = _nanmean_or_nan(subset_m_means)
            mean_subset_w = _nanmean_or_nan(subset_w_means)
            print(
                f"  k={subset_size:2d} | sampled {len(sampled_subsets):3d}/{total_possible_subsets} subsets | "
                f"mean subset M={mean_subset_m:8.4f} W={mean_subset_w:8.4f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_df = pd.DataFrame.from_records(pair_records)
    subset_df = pd.DataFrame.from_records(subset_records)
    if pair_df.empty or subset_df.empty:
        raise RuntimeError("The observed discrete scan did not produce any rows.")

    pair_df = pair_df.sort_values(["pair_seed", "pair_role", "subset_size", "subset_draw", "pair_draw"]).reset_index(drop=True)
    subset_df = subset_df.sort_values(["pair_seed", "pair_role", "subset_size", "subset_draw"]).reset_index(drop=True)

    group_cols = [
        "run_label",
        "pair_seed",
        "pair_role",
        "pair_role_label",
        "task_key",
        "task_name",
        "mem_distribution_family",
        "checkpoint",
        "checkpoint_name",
        "hidden_tau_syn_ms",
        "hidden_tau_mem_geom_mean_ms",
        "final_test_acc",
        "final_test_loss",
        "label",
        "subset_size",
    ]
    model_df = (
        subset_df.groupby(group_cols, as_index=False)
        .agg(
            sampled_subsets=("subset_draw", "count"),
            sampled_pairs_mean=("sampled_pairs", "mean"),
            subset_pair_M_mean=("subset_pair_M_mean", "mean"),
            subset_pair_M_std=("subset_pair_M_mean", "std"),
            subset_pair_W_mean=("subset_pair_W_mean", "mean"),
            subset_pair_W_std=("subset_pair_W_mean", "std"),
            subset_pair_TDMI_mean=("subset_pair_TDMI_mean", "mean"),
            valid_pair_count=("valid_pair_count", "mean"),
            error_pair_count=("error_pair_count", "sum"),
            delay_steps=("delay_steps", "mean"),
            delay_ms=("delay_ms", "mean"),
        )
        .sort_values(["pair_seed", "pair_role", "subset_size"])
        .reset_index(drop=True)
    )
    model_df["delay_steps"] = model_df["delay_steps"].round().astype(int)
    model_df["valid_pair_count"] = model_df["valid_pair_count"].round().astype(int)
    model_df["error_pair_count"] = model_df["error_pair_count"].round().astype(int)
    return pair_df, subset_df, model_df


def save_observed_analysis_artifacts(pair_df, subset_df, model_df, output_dir, role_order=None, role_labels=None):
    role_order = list(role_order or DEFAULT_ROLE_ORDER)
    role_labels = dict(DEFAULT_ROLE_LABELS | (role_labels or {}))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_summary_df = (
        model_df.groupby(["pair_role", "pair_role_label", "subset_size"], as_index=False)
        .agg(
            subset_pair_M_mean_mean=("subset_pair_M_mean", "mean"),
            subset_pair_M_mean_std=("subset_pair_M_mean", "std"),
            subset_pair_W_mean_mean=("subset_pair_W_mean", "mean"),
            subset_pair_W_mean_std=("subset_pair_W_mean", "std"),
            subset_pair_TDMI_mean_mean=("subset_pair_TDMI_mean", "mean"),
            final_test_acc_mean=("final_test_acc", "mean"),
            n_models=("checkpoint", "nunique"),
        )
        .sort_values(["pair_role", "subset_size"])
        .reset_index(drop=True)
    )
    delta_df = summarise_pair_deltas(model_df)

    paths = {
        "pair_csv": output_dir / "seeded_runs_discrete_observed_pair_records.csv",
        "subset_csv": output_dir / "seeded_runs_discrete_observed_subset_records.csv",
        "model_csv": output_dir / "seeded_runs_discrete_observed_model_summary.csv",
        "role_summary_csv": output_dir / "seeded_runs_discrete_observed_role_summary.csv",
        "pair_delta_csv": output_dir / "seeded_runs_discrete_observed_pair_delta.csv",
        "accuracy_plot": output_dir / "seeded_runs_discrete_observed_accuracy_pairs.png",
        "minfo_plot": output_dir / "seeded_runs_discrete_observed_m_by_subset.png",
        "winfo_plot": output_dir / "seeded_runs_discrete_observed_w_by_subset.png",
        "scatter_plot": output_dir / "seeded_runs_discrete_observed_mw_vs_accuracy.png",
    }

    pair_df.to_csv(paths["pair_csv"], index=False)
    subset_df.to_csv(paths["subset_csv"], index=False)
    model_df.to_csv(paths["model_csv"], index=False)
    role_summary_df.to_csv(paths["role_summary_csv"], index=False)
    delta_df.to_csv(paths["pair_delta_csv"], index=False)

    checkpoint_meta_df = (
        model_df[["pair_seed", "pair_role", "pair_role_label", "final_test_acc"]]
        .drop_duplicates()
        .sort_values(["pair_seed", "pair_role"])
        .reset_index(drop=True)
    )
    accuracy_df = (
        checkpoint_meta_df.pivot(index="pair_seed", columns="pair_role", values="final_test_acc")
        .reset_index()
        .sort_values("pair_seed")
        .reset_index(drop=True)
    )
    if set(role_order).issubset(accuracy_df.columns):
        accuracy_df["hetero_minus_hom_acc"] = accuracy_df["heterogeneous_sampled"] - accuracy_df["homogeneous_anchor"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    x_positions = {role: idx for idx, role in enumerate(role_order)}
    for pair_seed in sorted(checkpoint_meta_df["pair_seed"].unique()):
        seed_slice = checkpoint_meta_df[checkpoint_meta_df["pair_seed"] == pair_seed].set_index("pair_role")
        if set(role_order).issubset(seed_slice.index):
            axes[0].plot(
                [x_positions[role_order[0]], x_positions[role_order[1]]],
                [seed_slice.loc[role_order[0], "final_test_acc"], seed_slice.loc[role_order[1], "final_test_acc"]],
                color="0.8",
                linewidth=1,
                zorder=1,
            )
    sns.stripplot(
        data=checkpoint_meta_df,
        x="pair_role",
        y="final_test_acc",
        order=role_order,
        ax=axes[0],
        size=7,
        jitter=0.08,
    )
    axes[0].set_xticks(range(len(role_order)))
    axes[0].set_xticklabels([role_labels[role] for role in role_order], rotation=12, ha="right")
    axes[0].set_title("Final test accuracy by model type")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Final test accuracy")

    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    if "hetero_minus_hom_acc" in accuracy_df.columns:
        axes[1].plot(
            accuracy_df["pair_seed"],
            accuracy_df["hetero_minus_hom_acc"],
            marker="o",
            linewidth=1.6,
            color="tab:orange",
        )
    axes[1].set_title("Paired accuracy difference")
    axes[1].set_xlabel("Pair seed")
    axes[1].set_ylabel("Heterogeneous - homogeneous")
    fig.savefig(paths["accuracy_plot"], bbox_inches="tight")
    plt.close(fig)

    plot_df = model_df.dropna(subset=["subset_pair_M_mean", "subset_pair_W_mean"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    if not plot_df.empty:
        sns.lineplot(
            data=plot_df,
            x="subset_size",
            y="subset_pair_M_mean",
            hue="pair_role_label",
            style="pair_role_label",
            marker="o",
            dashes=False,
            estimator="mean",
            errorbar="sd",
            ax=axes[0],
        )
    axes[0].set_title("Observed discrete M-information by subset size")
    axes[0].set_xlabel("Subset size")
    axes[0].set_ylabel("Mean projected M (bits)")

    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    if "hetero_minus_hom_subset_pair_M_mean" in delta_df.columns:
        for pair_seed, seed_slice in delta_df.groupby("pair_seed"):
            axes[1].plot(
                seed_slice["subset_size"],
                seed_slice["hetero_minus_hom_subset_pair_M_mean"],
                color="0.85",
                linewidth=1,
                zorder=1,
            )
        sns.lineplot(
            data=delta_df,
            x="subset_size",
            y="hetero_minus_hom_subset_pair_M_mean",
            marker="o",
            estimator="mean",
            errorbar="sd",
            color="black",
            ax=axes[1],
        )
    axes[1].set_title("Paired observed discrete M delta")
    axes[1].set_xlabel("Subset size")
    axes[1].set_ylabel("Heterogeneous - homogeneous (bits)")
    fig.savefig(paths["minfo_plot"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    if not plot_df.empty:
        sns.lineplot(
            data=plot_df,
            x="subset_size",
            y="subset_pair_W_mean",
            hue="pair_role_label",
            style="pair_role_label",
            marker="o",
            dashes=False,
            estimator="mean",
            errorbar="sd",
            ax=axes[0],
        )
    axes[0].set_title("Observed discrete W-information by subset size")
    axes[0].set_xlabel("Subset size")
    axes[0].set_ylabel("Mean projected W (bits)")

    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    if "hetero_minus_hom_subset_pair_W_mean" in delta_df.columns:
        for pair_seed, seed_slice in delta_df.groupby("pair_seed"):
            axes[1].plot(
                seed_slice["subset_size"],
                seed_slice["hetero_minus_hom_subset_pair_W_mean"],
                color="0.85",
                linewidth=1,
                zorder=1,
            )
        sns.lineplot(
            data=delta_df,
            x="subset_size",
            y="hetero_minus_hom_subset_pair_W_mean",
            marker="o",
            estimator="mean",
            errorbar="sd",
            color="black",
            ax=axes[1],
        )
    axes[1].set_title("Paired observed discrete W delta")
    axes[1].set_xlabel("Subset size")
    axes[1].set_ylabel("Heterogeneous - homogeneous (bits)")
    fig.savefig(paths["winfo_plot"], bbox_inches="tight")
    plt.close(fig)

    scatter_df = plot_df.copy()
    scatter_df["subset_label"] = scatter_df["subset_size"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    if not scatter_df.empty:
        sns.scatterplot(
            data=scatter_df,
            x="final_test_acc",
            y="subset_pair_M_mean",
            hue="pair_role_label",
            style="subset_label",
            s=110,
            ax=axes[0],
        )
        sns.scatterplot(
            data=scatter_df,
            x="final_test_acc",
            y="subset_pair_W_mean",
            hue="pair_role_label",
            style="subset_label",
            s=110,
            ax=axes[1],
        )
    axes[0].set_title("Accuracy vs observed discrete M")
    axes[0].set_xlabel("Final test accuracy")
    axes[0].set_ylabel("Mean projected M (bits)")
    axes[1].set_title("Accuracy vs observed discrete W")
    axes[1].set_xlabel("Final test accuracy")
    axes[1].set_ylabel("Mean projected W (bits)")
    fig.savefig(paths["scatter_plot"], bbox_inches="tight")
    plt.close(fig)
    return role_summary_df, delta_df, accuracy_df, paths


def build_observed_visual_suite(model_df, delta_df, accuracy_df, output_dir, role_order=None, role_labels=None, subset_focus=None):
    role_order = list(role_order or DEFAULT_ROLE_ORDER)
    role_labels = dict(DEFAULT_ROLE_LABELS | (role_labels or {}))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if subset_focus is None:
        subset_focus = int(model_df["subset_size"].max())

    role_palette = {
        role_labels["homogeneous_anchor"]: "#4C72B0",
        role_labels["heterogeneous_sampled"]: "#DD8452",
    }

    plot_df = model_df.dropna(subset=["subset_pair_M_mean", "subset_pair_W_mean", "final_test_acc"]).copy()
    plot_df["subset_size"] = plot_df["subset_size"].astype(int)
    pair_meta_df = (
        plot_df[["pair_seed", "pair_role", "pair_role_label", "final_test_acc"]]
        .drop_duplicates()
        .sort_values(["pair_seed", "pair_role"])
        .reset_index(drop=True)
    )
    subset_focus_df = (
        plot_df[plot_df["subset_size"] == int(subset_focus)]
        .sort_values(["pair_seed", "pair_role"])
        .reset_index(drop=True)
    )
    if subset_focus_df.empty:
        raise RuntimeError(f"No observed discrete rows are available for subset size {subset_focus}.")

    if "hetero_minus_hom_acc" not in accuracy_df.columns and set(role_order).issubset(accuracy_df.columns):
        accuracy_df = accuracy_df.copy()
        accuracy_df["hetero_minus_hom_acc"] = accuracy_df["heterogeneous_sampled"] - accuracy_df["homogeneous_anchor"]

    delta_plot_df = delta_df.sort_values(["pair_seed", "subset_size"]).reset_index(drop=True).copy()
    delta_relationship_df = delta_plot_df.merge(
        accuracy_df[["pair_seed", "hetero_minus_hom_acc"]],
        on="pair_seed",
        how="left",
    )
    delta_focus_df = delta_relationship_df[delta_relationship_df["subset_size"] == int(subset_focus)].copy()

    pair_seeds = sorted(pair_meta_df["pair_seed"].unique())
    seed_positions = {seed: idx for idx, seed in enumerate(pair_seeds)}
    ordered_subset_sizes = sorted(plot_df["subset_size"].unique())
    paths = {}

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, constrained_layout=True)
    for pair_seed in pair_seeds:
        y_pos = seed_positions[pair_seed]

        acc_slice = (
            pair_meta_df[pair_meta_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if acc_slice["final_test_acc"].notna().all():
            axes[0].plot(acc_slice["final_test_acc"], [y_pos, y_pos], color="0.8", linewidth=1.25, zorder=1)
        for role, row in acc_slice.iterrows():
            if pd.notna(row["final_test_acc"]):
                axes[0].scatter(
                    row["final_test_acc"],
                    y_pos,
                    color=role_palette[row["pair_role_label"]],
                    s=80,
                    zorder=2,
                )

        m_slice = (
            subset_focus_df[subset_focus_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if m_slice["subset_pair_M_mean"].notna().all():
            axes[1].plot(m_slice["subset_pair_M_mean"], [y_pos, y_pos], color="0.8", linewidth=1.25, zorder=1)
        for role, row in m_slice.iterrows():
            if pd.notna(row["subset_pair_M_mean"]):
                axes[1].scatter(
                    row["subset_pair_M_mean"],
                    y_pos,
                    color=role_palette[row["pair_role_label"]],
                    s=80,
                    zorder=2,
                )

    for ax in axes:
        ax.set_yticks(range(len(pair_seeds)))
        ax.set_yticklabels([str(seed) for seed in pair_seeds])
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_title("Paired task accuracy by seeded pair")
    axes[0].set_xlabel("Final test accuracy")
    axes[0].set_ylabel("Pair seed")
    axes[1].set_title(f"Paired observed discrete M at subset size {subset_focus}")
    axes[1].set_xlabel("Mean projected M (bits)")
    for label, color in role_palette.items():
        axes[1].scatter([], [], color=color, label=label, s=80)
    axes[1].legend(title="", loc="lower right")
    paths["pairwise_compare_plot"] = output_dir / "seeded_runs_discrete_observed_pairwise_accuracy_m.png"
    fig.savefig(paths["pairwise_compare_plot"], bbox_inches="tight")
    plt.close(fig)

    hom_heatmap = (
        plot_df[plot_df["pair_role"] == "homogeneous_anchor"]
        .pivot(index="pair_seed", columns="subset_size", values="subset_pair_M_mean")
        .reindex(index=pair_seeds, columns=ordered_subset_sizes)
    )
    hetero_heatmap = (
        plot_df[plot_df["pair_role"] == "heterogeneous_sampled"]
        .pivot(index="pair_seed", columns="subset_size", values="subset_pair_M_mean")
        .reindex(index=pair_seeds, columns=ordered_subset_sizes)
    )
    delta_heatmap = (
        delta_plot_df.pivot(index="pair_seed", columns="subset_size", values="hetero_minus_hom_subset_pair_M_mean")
        .reindex(index=pair_seeds, columns=ordered_subset_sizes)
    )
    value_floor = float(min(hom_heatmap.min().min(), hetero_heatmap.min().min()))
    value_ceiling = float(max(hom_heatmap.max().max(), hetero_heatmap.max().max()))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    sns.heatmap(
        hom_heatmap,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=value_floor,
        vmax=value_ceiling,
        cbar_kws={"label": "Mean projected M (bits)"},
        ax=axes[0],
    )
    axes[0].set_title("Homogeneous observed discrete M heatmap")
    axes[0].set_xlabel("Subset size")
    axes[0].set_ylabel("Pair seed")

    sns.heatmap(
        hetero_heatmap,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        vmin=value_floor,
        vmax=value_ceiling,
        cbar_kws={"label": "Mean projected M (bits)"},
        ax=axes[1],
    )
    axes[1].set_title("Heterogeneous observed discrete M heatmap")
    axes[1].set_xlabel("Subset size")
    axes[1].set_ylabel("")

    sns.heatmap(
        delta_heatmap,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "Heterogeneous - homogeneous (bits)"},
        ax=axes[2],
    )
    axes[2].set_title("Paired observed discrete M delta heatmap")
    axes[2].set_xlabel("Subset size")
    axes[2].set_ylabel("")
    paths["heatmap_plot"] = output_dir / "seeded_runs_discrete_observed_pairwise_m_heatmaps.png"
    fig.savefig(paths["heatmap_plot"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for pair_seed in pair_seeds:
        pair_slice = (
            subset_focus_df[subset_focus_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if pair_slice[["final_test_acc", "subset_pair_M_mean"]].notna().all().all():
            axes[0].plot(pair_slice["final_test_acc"], pair_slice["subset_pair_M_mean"], color="0.85", linewidth=1.1, zorder=1)

    sns.scatterplot(
        data=subset_focus_df,
        x="final_test_acc",
        y="subset_pair_M_mean",
        hue="pair_role_label",
        palette=role_palette,
        s=120,
        ax=axes[0],
    )
    for row in subset_focus_df.itertuples():
        axes[0].text(row.final_test_acc + 0.0015, row.subset_pair_M_mean + 0.015, str(int(row.pair_seed)), fontsize=8, alpha=0.75)
    axes[0].set_title(f"Accuracy vs observed discrete M at subset size {subset_focus}")
    axes[0].set_xlabel("Final test accuracy")
    axes[0].set_ylabel("Mean projected M (bits)")

    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1].axvline(0.0, color="0.5", linestyle="--", linewidth=1)
    sns.scatterplot(
        data=delta_relationship_df,
        x="hetero_minus_hom_acc",
        y="hetero_minus_hom_subset_pair_M_mean",
        hue="subset_size",
        palette="viridis",
        s=100,
        ax=axes[1],
    )
    for row in delta_focus_df.itertuples():
        axes[1].text(
            row.hetero_minus_hom_acc + 0.0015,
            row.hetero_minus_hom_subset_pair_M_mean + 0.015,
            str(int(row.pair_seed)),
            fontsize=8,
            alpha=0.75,
        )
    axes[1].set_title("Paired accuracy delta vs observed discrete M delta")
    axes[1].set_xlabel("Heterogeneous - homogeneous accuracy")
    axes[1].set_ylabel("Heterogeneous - homogeneous mean projected M")
    axes[1].legend(title="Subset size")
    paths["relationship_plot"] = output_dir / "seeded_runs_discrete_observed_accuracy_m_relationships.png"
    fig.savefig(paths["relationship_plot"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    x_positions = {role: idx for idx, role in enumerate(role_order)}
    for pair_seed in pair_seeds:
        acc_slice = (
            pair_meta_df[pair_meta_df["pair_seed"] == pair_seed]
            .set_index("pair_role")
            .reindex(role_order)
        )
        if acc_slice["final_test_acc"].notna().all():
            axes[0, 0].plot(
                [x_positions[role_order[0]], x_positions[role_order[1]]],
                acc_slice["final_test_acc"],
                color="0.8",
                linewidth=1,
                zorder=1,
            )
        for role, row in acc_slice.iterrows():
            if pd.notna(row["final_test_acc"]):
                axes[0, 0].scatter(
                    x_positions[role],
                    row["final_test_acc"],
                    color=role_palette[row["pair_role_label"]],
                    s=75,
                    zorder=2,
                )
    axes[0, 0].set_xticks(range(len(role_order)))
    axes[0, 0].set_xticklabels([role_labels[role] for role in role_order], rotation=12, ha="right")
    axes[0, 0].set_title("Paired accuracy summary")
    axes[0, 0].set_ylabel("Final test accuracy")

    sns.lineplot(
        data=plot_df,
        x="subset_size",
        y="subset_pair_M_mean",
        hue="pair_role_label",
        palette=role_palette,
        estimator="mean",
        errorbar="sd",
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Mean observed discrete M by subset size")
    axes[0, 1].set_xlabel("Subset size")
    axes[0, 1].set_ylabel("Mean projected M (bits)")
    axes[0, 1].legend(title="")

    sns.lineplot(
        data=plot_df,
        x="subset_size",
        y="subset_pair_W_mean",
        hue="pair_role_label",
        palette=role_palette,
        estimator="mean",
        errorbar="sd",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Mean observed discrete W by subset size")
    axes[1, 0].set_xlabel("Subset size")
    axes[1, 0].set_ylabel("Mean projected W (bits)")
    if axes[1, 0].legend_ is not None:
        axes[1, 0].legend_.remove()

    axes[1, 1].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].axvline(0.0, color="0.5", linestyle="--", linewidth=1)
    sns.scatterplot(
        data=delta_focus_df,
        x="hetero_minus_hom_acc",
        y="hetero_minus_hom_subset_pair_M_mean",
        s=120,
        color="#2A9D8F",
        ax=axes[1, 1],
    )
    for row in delta_focus_df.itertuples():
        axes[1, 1].text(
            row.hetero_minus_hom_acc + 0.0015,
            row.hetero_minus_hom_subset_pair_M_mean + 0.015,
            str(int(row.pair_seed)),
            fontsize=8,
            alpha=0.75,
        )
    axes[1, 1].set_title(f"Unified observed discrete delta view at subset size {subset_focus}")
    axes[1, 1].set_xlabel("Heterogeneous - homogeneous accuracy")
    axes[1, 1].set_ylabel("Heterogeneous - homogeneous mean projected M")
    paths["dashboard_plot"] = output_dir / "seeded_runs_discrete_observed_dashboard.png"
    fig.savefig(paths["dashboard_plot"], bbox_inches="tight")
    plt.close(fig)
    return paths