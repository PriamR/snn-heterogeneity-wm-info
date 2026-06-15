from __future__ import annotations

import csv
import importlib
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.stats import lognorm

PROJECT_ROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
PROJECT_FILES = PROJECT_ROOT / "Project Files"
WIMFO_ROOT = PROJECT_ROOT / "wimfo"
PAPER_ROOT = PROJECT_ROOT / "neural_heterogeneity" / "SuGD_code"
CHECKPOINT_ROOT = PROJECT_FILES / "Checkpoints" / "SeededRuns"
TAU_ARTIFACT_PATH = PROJECT_ROOT / "Tau from 15 epoch run.json"
SHD_TRAIN = PROJECT_ROOT / "data" / "shd" / "shd_train.h5"
SHD_TEST = PROJECT_ROOT / "data" / "shd" / "shd_test.h5"

for extra_path in [WIMFO_ROOT, PAPER_ROOT, PROJECT_FILES]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from clipper import ZeroOneClipper
from data_gen import open_file, sparse_data_generator
from reg_loss import loss as repo_loss

RSNN = importlib.import_module("model").RSNN
clipper = ZeroOneClipper()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PRMS = {
    "seed": 1000,
    "dtype": torch.float,
    "device": DEVICE,
    "cuda": DEVICE.type == "cuda",
    "nb_inputs": 700,
    "nb_hidden": [],
    "nb_recurrent": 32,
    "nb_outputs": 2,
    "batch_size": 256,
    "time_step": 1.0e-3,
    "nb_steps": 1000,
    "tau_syn": 10e-3,
    "tau_mem": 20e-3,
    "threshold": 1.0,
    "tref": 0.0,
    "dist": "gamma",
    "dist_prms": 3.0,
    "lr": 4e-3,
    "lr_ab": 4e-3,
    "betas": (0.9, 0.999),
    "weight_decay": 0.0,
    "nb_epochs": 25,
    "drop_last": True,
    "sl": 0.0,
    "thetal": 0.0,
    "su": 0.0,
    "thetau": 0.0,
    "rate": 0.0,
    "p_del": 0.0,
    "train_th": 0,
    "het_th": 0,
    "train_reset": 0,
    "het_reset": 0,
    "train_rest": 0,
    "het_rest": 0,
    "sparse_data_generator": "sparse_data_generator",
    "time_scale": [0.5, 1.0],
    "model": "RSNN",
    "savestep": 10,
    "clip": 1,
    "plot_step": 50,
    "class_list": list(range(20)),
    "task_label_map": None,
    "het_ab": 0,
    "train_ab": 0,
    "train_hom_ab": 0,
}

DEFAULT_HIDDEN_TAU_SYN_MS = float(BASE_PRMS["tau_syn"] * 1e3)
DEFAULT_OUTPUT_TAU_SYN_MS = float(BASE_PRMS["tau_syn"] * 1e3)
DEFAULT_OUTPUT_TAU_MEM_MS = float(BASE_PRMS["tau_mem"] * 1e3)

_TAU_FIT_CACHE = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SHDCache:
    def __init__(self, path: Path):
        raw_u, raw_t, raw_l = open_file(str(path))
        self.units = list(raw_u[:])
        self.times = list(raw_t[:])
        self.labels = np.array(raw_l[:])
        raw_u._v_file.close()
        print(f"  SHDCache: {len(self.labels)} samples loaded from {Path(str(path)).name}")


@contextmanager
def shd_open(path: Path):
    units, times, labels = open_file(str(path))
    try:
        yield units, times, labels
    finally:
        units._v_file.close()


@contextmanager
def shd_open_cached(cache: SHDCache):
    yield cache.units, cache.times, cache.labels


def _is_cache(obj) -> bool:
    return hasattr(obj, "units") and hasattr(obj, "times") and hasattr(obj, "labels")


def tau_ms_to_decay(tau_ms, time_step: float):
    tau_s = np.asarray(tau_ms, dtype=np.float64) * 1e-3
    tau_s = np.clip(tau_s, time_step * 1.001, None)
    return np.exp(-time_step / tau_s)


def decay_to_tau_ms(decay, time_step: float):
    decay = np.clip(np.asarray(decay, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return (-time_step / np.log(decay)) * 1e3


def geometric_mean_ms(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 1e-9, None)
    return float(np.exp(np.mean(np.log(arr))))


def _load_tau_fit_cache():
    global _TAU_FIT_CACHE
    if _TAU_FIT_CACHE is not None:
        return _TAU_FIT_CACHE

    with open(TAU_ARTIFACT_PATH, "r", encoding="utf-8") as fh:
        tau_artifact = json.load(fh)

    artifact_mem_ms = np.asarray(tau_artifact["heterogeneous"]["tau_mem_ms"], dtype=np.float64)
    mem_lognorm_shape, _, mem_lognorm_scale = lognorm.fit(artifact_mem_ms, floc=0.0)
    artifact_mem_bounds = (float(artifact_mem_ms.min()), float(artifact_mem_ms.max()))

    _TAU_FIT_CACHE = {
        "artifact_mem_ms": artifact_mem_ms,
        "mem_lognorm_shape": float(mem_lognorm_shape),
        "mem_lognorm_scale": float(mem_lognorm_scale),
        "artifact_mem_bounds": artifact_mem_bounds,
    }
    return _TAU_FIT_CACHE


def sample_mem_tau_ms(size: int, master_seed: int, family: str = "lognormal") -> np.ndarray:
    fit = _load_tau_fit_cache()
    rng = np.random.default_rng(master_seed)
    low, high = fit["artifact_mem_bounds"]
    family_key = family.strip().lower()

    if family_key in {"lognormal", "lognorm", "ln"}:
        vals = lognorm.rvs(
            fit["mem_lognorm_shape"],
            loc=0.0,
            scale=fit["mem_lognorm_scale"],
            size=size,
            random_state=rng,
        )
        return np.clip(vals, low, high)
    if family_key in {"loguniform", "log-uniform", "lu"}:
        return np.exp(rng.uniform(np.log(low), np.log(high), size=size))

    raise ValueError(f"Unsupported mem family: {family}")


def make_2class_task() -> dict:
    return {
        "nb_outputs": 2,
        "task_label_map": {i: i % 2 for i in range(20)},
        "class_list": list(range(20)),
        "task_name": "binary_parity",
    }


def make_4class_task() -> dict:
    label_map = {}
    for i in range(20):
        is_german = int(i >= 10)
        is_odd = int(i % 2 == 1)
        label_map[i] = is_german * 2 + is_odd
    return {
        "nb_outputs": 4,
        "task_label_map": label_map,
        "class_list": list(range(20)),
        "task_name": "4class_parity_language",
    }


def make_allclass_task() -> dict:
    return {
        "nb_outputs": 20,
        "task_label_map": {i: i for i in range(20)},
        "class_list": list(range(20)),
        "task_name": "all_class_shd",
    }


TASKS = {
    "2class": make_2class_task(),
    "4class": make_4class_task(),
    "allclass": make_allclass_task(),
}


def expected_2class_odd_even_map() -> dict:
    return {i: i % 2 for i in range(20)}


def _base_prms_for_task(task_key: str, seed: int) -> dict:
    task_override = dict(TASKS[task_key])
    task_override.pop("task_name", None)
    prms = dict(BASE_PRMS)
    prms.update(task_override)
    prms.update({
        "seed": seed,
        "dtype": torch.float,
        "device": DEVICE,
        "cuda": DEVICE.type == "cuda",
    })
    prms["alpha"] = float(np.exp(-prms["time_step"] / prms["tau_syn"]))
    prms["beta"] = float(np.exp(-prms["time_step"] / prms["tau_mem"]))
    return prms


def build_homogeneous_model(task_key: str, seed: int):
    prms = _base_prms_for_task(task_key, seed)
    prms.update({"het_ab": 0, "train_ab": 0, "train_hom_ab": 1,     # shared tau learnable
                 "weight_decay": 0.0})                                 # no WD — matches paper
    set_seed(seed)
    model = RSNN(prms, rec=True).to(DEVICE)
    return model, prms


def build_heterogeneous_model(task_key: str, seed: int):
    prms = _base_prms_for_task(task_key, seed)
    prms.update({"het_ab": 1, "train_ab": 1, "train_hom_ab": 0,     # per-neuron tau learnable
                 "weight_decay": 0.0})                                 # no WD — matches paper
    set_seed(seed)
    model = RSNN(prms, rec=True).to(DEVICE)
    return model, prms


def _copy_param_value(parameter: torch.nn.Parameter, values) -> None:
    array = np.asarray(values, dtype=np.float64)
    if parameter.numel() == 1:
        array = np.asarray([float(array.reshape(-1)[0])], dtype=np.float64)
    else:
        array = array.reshape(parameter.shape)
    tensor = torch.tensor(array, device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad():
        parameter.copy_(tensor)


def set_layer_taus_ms(layer, tau_syn_ms, tau_mem_ms, time_step: float) -> None:
    alpha_decay = tau_ms_to_decay(tau_syn_ms, time_step)
    beta_decay = tau_ms_to_decay(tau_mem_ms, time_step)
    _copy_param_value(layer.alpha, alpha_decay)
    _copy_param_value(layer.beta, beta_decay)


def sync_non_tau_parameters(source_model, target_model) -> list[str]:
    target_named = dict(target_model.named_parameters())
    copied = []
    for name, parameter in source_model.named_parameters():
        if "alpha" in name or "beta" in name:
            continue
        target_parameter = target_named.get(name)
        if target_parameter is None or target_parameter.shape != parameter.shape:
            continue
        with torch.no_grad():
            target_parameter.copy_(parameter.detach())
        copied.append(name)
    return copied


def compare_non_tau_parameters(model_a, model_b) -> dict:
    b_named = dict(model_b.named_parameters())
    mismatched = []
    checked = []
    for name, param_a in model_a.named_parameters():
        if "alpha" in name or "beta" in name:
            continue
        param_b = b_named.get(name)
        if param_b is None or param_a.shape != param_b.shape:
            continue
        checked.append(name)
        if not torch.allclose(param_a.detach().cpu(), param_b.detach().cpu()):
            mismatched.append(name)
    return {
        "matched": len(mismatched) == 0,
        "checked": checked,
        "mismatched": mismatched,
    }


def build_seeded_pair(
    master_seed: int,
    task_key: str = "2class",
    mem_distribution_family: str = "lognormal",
    hidden_tau_syn_ms: float = DEFAULT_HIDDEN_TAU_SYN_MS,
    output_tau_syn_ms: float = DEFAULT_OUTPUT_TAU_SYN_MS,
    output_tau_mem_ms: float = DEFAULT_OUTPUT_TAU_MEM_MS,
):
    hetero_model, hetero_prms = build_heterogeneous_model(task_key, master_seed)
    hom_model, hom_prms = build_homogeneous_model(task_key, master_seed)

    sync_non_tau_parameters(hetero_model, hom_model)

    hidden_size = int(hetero_model.network[0].output_size)
    sampled_tau_mem_ms = sample_mem_tau_ms(hidden_size, master_seed, family=mem_distribution_family)
    anchor_tau_mem_ms = geometric_mean_ms(sampled_tau_mem_ms)

    set_layer_taus_ms(
        hetero_model.network[0],
        tau_syn_ms=np.full(hidden_size, hidden_tau_syn_ms, dtype=np.float64),
        tau_mem_ms=sampled_tau_mem_ms,
        time_step=hetero_prms["time_step"],
    )
    set_layer_taus_ms(
        hetero_model.network[1],
        tau_syn_ms=np.full(int(hetero_prms["nb_outputs"]), output_tau_syn_ms, dtype=np.float64),
        tau_mem_ms=np.full(int(hetero_prms["nb_outputs"]), output_tau_mem_ms, dtype=np.float64),
        time_step=hetero_prms["time_step"],
    )
    set_layer_taus_ms(
        hom_model.network[0],
        tau_syn_ms=hidden_tau_syn_ms,
        tau_mem_ms=anchor_tau_mem_ms,
        time_step=hom_prms["time_step"],
    )
    set_layer_taus_ms(
        hom_model.network[1],
        tau_syn_ms=output_tau_syn_ms,
        tau_mem_ms=output_tau_mem_ms,
        time_step=hom_prms["time_step"],
    )

    comparison = compare_non_tau_parameters(hetero_model, hom_model)
    hetero_hidden_tau = decay_to_tau_ms(hetero_model.network[0].beta.detach().cpu().numpy(), hetero_prms["time_step"])
    hom_hidden_tau = decay_to_tau_ms(hom_model.network[0].beta.detach().cpu().numpy(), hom_prms["time_step"])

    metadata = {
        "pair_seed": int(master_seed),
        "task_key": task_key,
        "task_name": TASKS[task_key]["task_name"],
        "mem_distribution_family": mem_distribution_family,
        "hidden_tau_syn_ms": float(hidden_tau_syn_ms),
        "output_tau_syn_ms": float(output_tau_syn_ms),
        "output_tau_mem_ms": float(output_tau_mem_ms),
        "hetero_hidden_tau_mem_ms": sampled_tau_mem_ms.astype(float).tolist(),
        "hom_hidden_tau_mem_ms": [float(anchor_tau_mem_ms)] * hidden_size,
        "hidden_tau_mem_geom_mean_ms": float(anchor_tau_mem_ms),
        "linear_sync_verified": bool(comparison["matched"]),
        "linear_sync_checked": comparison["checked"],
        "linear_sync_mismatched": comparison["mismatched"],
        "hetero_hidden_tau_unique": int(np.unique(np.round(hetero_hidden_tau, 8)).size),
        "hom_hidden_tau_unique": int(np.unique(np.round(hom_hidden_tau, 8)).size),
    }
    return {
        "hetero_model": hetero_model,
        "hetero_prms": hetero_prms,
        "hom_model": hom_model,
        "hom_prms": hom_prms,
        "metadata": metadata,
    }


def build_sampling_preview_rows(master_seeds, task_key: str = "2class", mem_distribution_family: str = "lognormal"):
    rows = []
    previous_tau = None
    for master_seed in [int(seed) for seed in master_seeds]:
        pair = build_seeded_pair(
            master_seed=master_seed,
            task_key=task_key,
            mem_distribution_family=mem_distribution_family,
        )
        metadata = pair["metadata"]
        hetero_tau = np.asarray(metadata["hetero_hidden_tau_mem_ms"], dtype=np.float64)
        row = {
            "pair_seed": metadata["pair_seed"],
            "task_key": metadata["task_key"],
            "task_name": metadata["task_name"],
            "mem_distribution_family": metadata["mem_distribution_family"],
            "linear_sync_verified": metadata["linear_sync_verified"],
            "hetero_hidden_tau_unique": metadata["hetero_hidden_tau_unique"],
            "hom_hidden_tau_unique": metadata["hom_hidden_tau_unique"],
            "hidden_tau_mem_geom_mean_ms": metadata["hidden_tau_mem_geom_mean_ms"],
            "hetero_hidden_tau_min_ms": float(np.min(hetero_tau)),
            "hetero_hidden_tau_max_ms": float(np.max(hetero_tau)),
            "hetero_hidden_tau_mean_ms": float(np.mean(hetero_tau)),
            "hetero_hidden_tau_std_ms": float(np.std(hetero_tau)),
            "sample_matches_previous": False if previous_tau is None else bool(np.array_equal(previous_tau, hetero_tau)),
        }
        previous_tau = hetero_tau.copy()
        rows.append(row)
    return rows


def build_pair_summary_row(metadata: dict) -> dict:
    hetero_tau = np.asarray(metadata["hetero_hidden_tau_mem_ms"], dtype=np.float64)
    return {
        "pair_seed": metadata["pair_seed"],
        "task_key": metadata["task_key"],
        "task_name": metadata["task_name"],
        "mem_distribution_family": metadata["mem_distribution_family"],
        "hidden_tau_syn_ms": metadata["hidden_tau_syn_ms"],
        "hidden_tau_mem_geom_mean_ms": metadata["hidden_tau_mem_geom_mean_ms"],
        "hetero_hidden_tau_min_ms": float(np.min(hetero_tau)),
        "hetero_hidden_tau_max_ms": float(np.max(hetero_tau)),
        "hetero_hidden_tau_mean_ms": float(np.mean(hetero_tau)),
        "hetero_hidden_tau_std_ms": float(np.std(hetero_tau)),
        "hetero_hidden_tau_unique": metadata["hetero_hidden_tau_unique"],
        "hom_hidden_tau_unique": metadata["hom_hidden_tau_unique"],
        "linear_sync_verified": metadata["linear_sync_verified"],
        "checked_parameter_count": len(metadata["linear_sync_checked"]),
    }


def build_sampling_preview_rows(master_seeds, task_key: str = "2class", mem_distribution_family: str = "lognormal"):
    rows = []
    previous_tau = None
    for master_seed in [int(seed) for seed in master_seeds]:
        pair = build_seeded_pair(
            master_seed=master_seed,
            task_key=task_key,
            mem_distribution_family=mem_distribution_family,
        )
        metadata = pair["metadata"]
        hetero_tau = np.asarray(metadata["hetero_hidden_tau_mem_ms"], dtype=np.float64)
        row = {
            "pair_seed": metadata["pair_seed"],
            "task_key": metadata["task_key"],
            "task_name": metadata["task_name"],
            "mem_distribution_family": metadata["mem_distribution_family"],
            "linear_sync_verified": metadata["linear_sync_verified"],
            "hetero_hidden_tau_unique": metadata["hetero_hidden_tau_unique"],
            "hom_hidden_tau_unique": metadata["hom_hidden_tau_unique"],
            "hidden_tau_mem_geom_mean_ms": metadata["hidden_tau_mem_geom_mean_ms"],
            "hetero_hidden_tau_min_ms": float(np.min(hetero_tau)),
            "hetero_hidden_tau_max_ms": float(np.max(hetero_tau)),
            "hetero_hidden_tau_mean_ms": float(np.mean(hetero_tau)),
            "hetero_hidden_tau_std_ms": float(np.std(hetero_tau)),
            "sample_matches_previous": False if previous_tau is None else bool(np.array_equal(previous_tau, hetero_tau)),
        }
        previous_tau = hetero_tau.copy()
        rows.append(row)
    return rows


def build_pair_summary_row(metadata: dict) -> dict:
    hetero_tau = np.asarray(metadata["hetero_hidden_tau_mem_ms"], dtype=np.float64)
    return {
        "pair_seed": metadata["pair_seed"],
        "task_key": metadata["task_key"],
        "task_name": metadata["task_name"],
        "mem_distribution_family": metadata["mem_distribution_family"],
        "hidden_tau_syn_ms": metadata["hidden_tau_syn_ms"],
        "hidden_tau_mem_geom_mean_ms": metadata["hidden_tau_mem_geom_mean_ms"],
        "hetero_hidden_tau_min_ms": float(np.min(hetero_tau)),
        "hetero_hidden_tau_max_ms": float(np.max(hetero_tau)),
        "hetero_hidden_tau_mean_ms": float(np.mean(hetero_tau)),
        "hetero_hidden_tau_std_ms": float(np.std(hetero_tau)),
        "hetero_hidden_tau_unique": metadata["hetero_hidden_tau_unique"],
        "hom_hidden_tau_unique": metadata["hom_hidden_tau_unique"],
        "linear_sync_verified": metadata["linear_sync_verified"],
        "checked_parameter_count": len(metadata["linear_sync_checked"]),
    }


def fast_sparse_data_generator(units, times, labels, prms, shuffle=True, epoch=0, drop_last=True):
    rate = prms.get("rate", 0.0)
    p_del = prms.get("p_del", 0.0)
    if rate != 0.0 or p_del != 0.0:
        yield from sparse_data_generator(
            units,
            times,
            labels,
            prms,
            shuffle=shuffle,
            epoch=epoch,
            drop_last=drop_last,
        )
        return

    seed = prms["seed"] + epoch
    batch_size = prms["batch_size"]
    nb_steps = prms["nb_steps"]
    nb_units = prms["nb_inputs"]
    inv_dt = 1.0 / prms["time_step"]
    class_list = prms["class_list"]
    task_label_map = prms.get("task_label_map", None)

    label_arr = labels if isinstance(labels, np.ndarray) else np.array(labels[:])
    sample_index = np.where(np.isin(label_arr, class_list))[0]
    num_samples = len(sample_index)
    n_batches = (num_samples // batch_size) if drop_last else -(-num_samples // batch_size)

    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(sample_index)

    for counter in range(n_batches):
        batch_index = sample_index[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
        actual_bs = len(batch_index)
        t_arrays = [np.round(times[idx] * inv_dt).astype(np.int64) for idx in batch_index]
        u_arrays = [units[idx] for idx in batch_index]
        lengths = np.array([len(a) for a in t_arrays], dtype=np.int64)

        if lengths.sum():
            all_ts = np.concatenate(t_arrays)
            all_us = np.concatenate(u_arrays)
            all_bc = np.repeat(np.arange(actual_bs, dtype=np.int64), lengths)
            valid = all_ts < nb_steps
            all_ts, all_us, all_bc = all_ts[valid], all_us[valid], all_bc[valid]
            index_tensor = torch.from_numpy(np.stack([all_bc, all_ts, all_us]))
            values = torch.ones(all_ts.size, dtype=torch.float32)
            x_batch = torch.sparse_coo_tensor(
                index_tensor,
                values,
                torch.Size([actual_bs, nb_steps, nb_units]),
            ).to_dense()
        else:
            x_batch = torch.zeros(actual_bs, nb_steps, nb_units)

        x_batch.clamp_(max=1.0)
        if task_label_map is not None:
            y_batch = torch.tensor(
                [task_label_map[int(a)] for a in label_arr[batch_index]],
                dtype=torch.long,
            )
        else:
            y_batch = torch.tensor(
                [class_list.index(int(a)) for a in label_arr[batch_index]],
                dtype=torch.long,
            )
        yield x_batch, y_batch


def shd_generator(units, times, labels, prms, shuffle, epoch, drop_last):
    yield from fast_sparse_data_generator(
        units,
        times,
        labels,
        prms,
        shuffle=shuffle,
        epoch=epoch,
        drop_last=drop_last,
    )


def count_epoch_samples(sample_count, batch_size, drop_last):
    if drop_last:
        return (sample_count // batch_size) * batch_size
    return sample_count


def forward_logits(model, x):
    layer_recs = model(0, 0, x)
    output_layer = layer_recs[-1]
    logits, _ = torch.max(output_layer[1], dim=1)
    return logits, layer_recs


def make_optimizer(model, prms):
    weight_params, ab_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "alpha" in name or "beta" in name:
            ab_params.append(parameter)
        else:
            weight_params.append(parameter)

    param_groups = [{
        "params": weight_params,
        "lr": prms["lr"],
        "weight_decay": prms["weight_decay"],
    }]
    if ab_params:
        param_groups.append({"params": ab_params, "lr": prms["lr_ab"]})
    if prms.get("optimizer", "adam") == "adamw":
        return torch.optim.AdamW(param_groups, betas=tuple(prms["betas"]))
    return torch.optim.Adam(param_groups, betas=tuple(prms["betas"]))


@torch.no_grad()
def evaluate_batches(model, prms, units, times, labels, num_samples=None, use_amp=True):
    if num_samples is None:
        total = int(np.isin(labels[:], prms["class_list"]).sum())
        num_samples = count_epoch_samples(total, prms["batch_size"], drop_last=False)

    amp_enabled = use_amp and DEVICE.type == "cuda"
    model.eval()
    loss_acc = 0.0
    correct = 0
    for x, y in shd_generator(units, times, labels, prms, shuffle=False, epoch=0, drop_last=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
            logits, layer_recs = forward_logits(model, x)
            loss_acc += repo_loss(logits, layer_recs, y, num_samples, prms).item()
        correct += (logits.argmax(1) == y).sum().item()
    return {"loss": loss_acc, "acc": correct / max(num_samples, 1), "n": num_samples}


def train_experiment(model, prms, train_data, test_data, use_amp=True):
    amp_enabled = use_amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer = make_optimizer(model, prms)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Optional torch.compile for speed (PyTorch >= 2.0)
    if prms.get("compile_model", False) and hasattr(torch, "compile"):
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile applied (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    # LR scheduler: reduce LR by factor 0.5 when test loss plateaus for patience epochs
    scheduler_patience = prms.get("lr_patience", 0)
    scheduler = None
    if scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=prms.get("lr_factor", 0.5),
            patience=scheduler_patience,
            min_lr=prms.get("lr_min", 1e-6),
        )

    train_ctx = shd_open_cached if _is_cache(train_data) else shd_open
    test_ctx = shd_open_cached if _is_cache(test_data) else shd_open

    with train_ctx(train_data) as (u_tr, t_tr, l_tr), test_ctx(test_data) as (u_te, t_te, l_te):
        if not prms["class_list"]:
            prms["class_list"] = np.unique(l_tr[:]).tolist()

        total_tr = int(np.isin(l_tr[:], prms["class_list"]).sum())
        total_te = int(np.isin(l_te[:], prms["class_list"]).sum())
        eff_tr = count_epoch_samples(total_tr, prms["batch_size"], drop_last=bool(prms["drop_last"]))
        eff_te = count_epoch_samples(total_te, prms["batch_size"], drop_last=False)

        if prms["clip"]:
            model.apply(clipper)

        for epoch in range(1, prms["nb_epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0

            for x, y in shd_generator(u_tr, t_tr, l_tr, prms, shuffle=True, epoch=epoch, drop_last=prms["drop_last"]):
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
                    logits, layer_recs = forward_logits(model, x)
                    loss_val = repo_loss(logits, layer_recs, y, eff_tr, prms)
                scaler.scale(loss_val).backward()
                scaler.step(optimizer)
                scaler.update()
                if prms["clip"]:
                    model.apply(clipper)
                epoch_loss += loss_val.item()
                epoch_correct += (logits.argmax(1) == y).sum().item()

            test_metrics = evaluate_batches(model, prms, u_te, t_te, l_te, num_samples=eff_te, use_amp=use_amp)
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_correct / max(eff_tr, 1))
            history["test_loss"].append(test_metrics["loss"])
            history["test_acc"].append(test_metrics["acc"])

            # Step LR scheduler if configured
            if scheduler is not None:
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(test_metrics["loss"])
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr < prev_lr:
                    print(f"         LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

            # Early stopping based on test accuracy
            es_patience = prms.get("early_stop_patience", 0)
            if es_patience > 0:
                if epoch == 1:
                    _best_acc = test_metrics["acc"]
                    _best_epoch = 1
                    _no_improve = 0
                else:
                    if test_metrics["acc"] > _best_acc + 1e-4:
                        _best_acc = test_metrics["acc"]
                        _best_epoch = epoch
                        _no_improve = 0
                    else:
                        _no_improve += 1
                if _no_improve >= es_patience and epoch > 10:
                    print(f"         Early stop @ epoch {epoch}: no improvement for {es_patience} epochs "
                          f"(best {_best_acc:.3f} @ ep {_best_epoch})")
                    break

            elapsed = time.perf_counter() - epoch_start
            print(
                f"  epoch={epoch:03d}  "
                f"train_acc={epoch_correct / max(eff_tr, 1):.3f}  "
                f"test_acc={test_metrics['acc']:.3f}  "
                f"({elapsed / 60:.1f} min)"
            )

    return history


def save_model_checkpoint(path, model, history, elapsed_s, prms, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "elapsed_s": float(elapsed_s),
        "prms": {k: v for k, v in prms.items() if not isinstance(v, (torch.dtype, torch.device))},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_default_caches():
    print("Pre-loading SHD training data into RAM...")
    train_cache = SHDCache(SHD_TRAIN)
    print("Pre-loading SHD test data into RAM...")
    test_cache = SHDCache(SHD_TEST)
    return train_cache, test_cache


def get_run_directory(
    run_label: str,
    task_key: str = "2class",
    mem_distribution_family: str = "lognormal",
    checkpoint_root: Path | None = None,
) -> Path:
    checkpoint_root = Path(checkpoint_root or CHECKPOINT_ROOT)
    run_dir = checkpoint_root / run_label / f"{task_key}_{mem_distribution_family}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_pair_checkpoint_paths(
    master_seed: int,
    run_label: str,
    task_key: str = "2class",
    mem_distribution_family: str = "lognormal",
    checkpoint_root: Path | None = None,
) -> dict:
    run_dir = get_run_directory(
        run_label=run_label,
        task_key=task_key,
        mem_distribution_family=mem_distribution_family,
        checkpoint_root=checkpoint_root,
    )
    return {
        "homogeneous_anchor": run_dir / f"{task_key}_seed{master_seed:03d}_hom_geommean_{mem_distribution_family}.pt",
        "heterogeneous_sampled": run_dir / f"{task_key}_seed{master_seed:03d}_het_{mem_distribution_family}.pt",
    }


def load_result_row_from_checkpoint(path: Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu")
    history = payload.get("history", {})
    hidden_tau_mem_ms = np.asarray(payload.get("hidden_tau_mem_ms", []), dtype=np.float64)
    if hidden_tau_mem_ms.size == 0:
        hidden_tau_mem_ms = np.asarray(payload.get("paired_hetero_hidden_tau_mem_ms", []), dtype=np.float64)

    final_test_acc = float(history.get("test_acc", [float("nan")])[-1])
    final_test_loss = float(history.get("test_loss", [float("nan")])[-1])

    return {
        "run_label": payload.get("run_label"),
        "task_key": payload.get("task_key"),
        "task_name": payload.get("task"),
        "pair_seed": int(payload.get("pair_seed")),
        "pair_role": payload.get("pair_role"),
        "mem_distribution_family": payload.get("mem_distribution_family"),
        "hidden_tau_syn_ms": float(payload.get("hidden_tau_syn_ms")),
        "hidden_tau_mem_geom_mean_ms": float(payload.get("hidden_tau_mem_geom_mean_ms")),
        "hidden_tau_mem_min_ms": float(np.min(hidden_tau_mem_ms)) if hidden_tau_mem_ms.size else float("nan"),
        "hidden_tau_mem_max_ms": float(np.max(hidden_tau_mem_ms)) if hidden_tau_mem_ms.size else float("nan"),
        "hidden_tau_mem_mean_ms": float(np.mean(hidden_tau_mem_ms)) if hidden_tau_mem_ms.size else float("nan"),
        "hidden_tau_mem_std_ms": float(np.std(hidden_tau_mem_ms)) if hidden_tau_mem_ms.size else float("nan"),
        "linear_sync_verified": bool(payload.get("linear_sync_verified", False)),
        "checkpoint": str(path),
        "elapsed_s": float(payload.get("elapsed_s", float("nan"))),
        "final_test_acc": final_test_acc,
        "final_test_loss": final_test_loss,
    }


def read_manifest_rows(output_stem: Path):
    output_stem = Path(output_stem)
    json_path = output_stem.with_suffix(".json")
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def upsert_rows(existing_rows, new_rows, key_fields):
    merged = {}
    for row in existing_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = row
    for row in new_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = row
    return list(merged.values())


def run_pair_training(
    master_seed: int,
    train_cache,
    test_cache,
    task_key: str = "2class",
    mem_distribution_family: str = "lognormal",
    run_label: str = "seeded_run",
    checkpoint_root: Path | None = None,
    skip_existing: bool = True,
):
    pair = build_seeded_pair(
        master_seed=master_seed,
        task_key=task_key,
        mem_distribution_family=mem_distribution_family,
    )
    metadata = dict(pair["metadata"])
    if not metadata["linear_sync_verified"]:
        raise RuntimeError(f"Linear sync failed for seed {master_seed}: {metadata['linear_sync_mismatched']}")

    checkpoint_paths = get_pair_checkpoint_paths(
        master_seed=master_seed,
        run_label=run_label,
        task_key=task_key,
        mem_distribution_family=mem_distribution_family,
        checkpoint_root=checkpoint_root,
    )

    results = []
    train_plan = [
        ("hom", pair["hom_model"], pair["hom_prms"], "homogeneous_anchor"),
        ("het", pair["hetero_model"], pair["hetero_prms"], "heterogeneous_sampled"),
    ]
    for role_key, model, prms, role_name in train_plan:
        checkpoint_path = checkpoint_paths[role_name]
        if skip_existing and checkpoint_path.exists():
            print(f"\nSeed {master_seed} :: reusing existing {role_name} checkpoint...")
            results.append(load_result_row_from_checkpoint(checkpoint_path))
            continue

        print(f"\nSeed {master_seed} :: training {role_name}...")
        start = time.perf_counter()
        history = train_experiment(model, prms, train_cache, test_cache)
        elapsed = time.perf_counter() - start

        hidden_tau_mem_ms = metadata["hom_hidden_tau_mem_ms"] if role_key == "hom" else metadata["hetero_hidden_tau_mem_ms"]
        checkpoint = save_model_checkpoint(
            checkpoint_path,
            model,
            history,
            elapsed,
            prms,
            extra={
                "label": checkpoint_path.stem,
                "task": metadata["task_name"],
                "task_key": metadata["task_key"],
                "pair_seed": metadata["pair_seed"],
                "pair_role": role_name,
                "run_label": run_label,
                "mem_distribution_family": metadata["mem_distribution_family"],
                "hidden_tau_syn_ms": metadata["hidden_tau_syn_ms"],
                "hidden_tau_mem_ms": hidden_tau_mem_ms,
                "hidden_tau_mem_geom_mean_ms": metadata["hidden_tau_mem_geom_mean_ms"],
                "paired_hetero_hidden_tau_mem_ms": metadata["hetero_hidden_tau_mem_ms"],
                "output_tau_syn_ms": metadata["output_tau_syn_ms"],
                "output_tau_mem_ms": metadata["output_tau_mem_ms"],
                "linear_sync_verified": metadata["linear_sync_verified"],
                "linear_sync_checked": metadata["linear_sync_checked"],
            },
        )

        result_row = {
            "run_label": run_label,
            "task_key": metadata["task_key"],
            "task_name": metadata["task_name"],
            "pair_seed": metadata["pair_seed"],
            "pair_role": role_name,
            "mem_distribution_family": metadata["mem_distribution_family"],
            "hidden_tau_syn_ms": metadata["hidden_tau_syn_ms"],
            "hidden_tau_mem_geom_mean_ms": metadata["hidden_tau_mem_geom_mean_ms"],
            "hidden_tau_mem_min_ms": float(np.min(hidden_tau_mem_ms)),
            "hidden_tau_mem_max_ms": float(np.max(hidden_tau_mem_ms)),
            "hidden_tau_mem_mean_ms": float(np.mean(hidden_tau_mem_ms)),
            "hidden_tau_mem_std_ms": float(np.std(hidden_tau_mem_ms)),
            "linear_sync_verified": metadata["linear_sync_verified"],
            "checkpoint": str(checkpoint),
            "elapsed_s": float(elapsed),
            "final_test_acc": float(history["test_acc"][-1]),
            "final_test_loss": float(history["test_loss"][-1]),
        }
        results.append(result_row)

    return results, metadata


def write_manifest_rows(rows, output_stem: Path):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = output_stem.with_suffix(".csv")
    json_path = output_stem.with_suffix(".json")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    return csv_path, json_path
