"""Generate PCA variance decay figure for SNN_Analysis_Report.md."""
import sys, importlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
BASE  = PROJECT_ROOT / "Project Files"
CKPT  = BASE / "Checkpoints"
PARITY = CKPT / "Parity"
FIG_DIR = BASE / "Report Figures"

for p in [PROJECT_ROOT / "wimfo", PROJECT_ROOT / "neural_heterogeneity" / "SuGD_code"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_gen import open_file
RSNN   = importlib.import_module("model").RSNN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATALOG = [
    dict(short="ALL-LH", task="all-class",  arch="Local-Hom",            ckpt=CKPT/"local_hom_checkpoint.pt"),
    dict(short="ALL-LN", task="all-class",  arch="FittedHet-LogNorm",     ckpt=CKPT/"fittedhet_lognorm_mem_gamma_syn_checkpoint.pt"),
    dict(short="ALL-LU", task="all-class",  arch="FittedHet-LogUniform",  ckpt=CKPT/"fittedhet_loguniform_mem_gamma_syn_checkpoint.pt"),
    dict(short="2C-LH",  task="2-class",    arch="Local-Hom",             ckpt=PARITY/"2class_local_hom.pt"),
    dict(short="2C-LN",  task="2-class",    arch="FittedHet-LogNorm",     ckpt=PARITY/"2class_fittedhet_lognorm.pt"),
    dict(short="2C-LU",  task="2-class",    arch="FittedHet-LogUniform",  ckpt=PARITY/"2class_fittedhet_loguniform.pt"),
    dict(short="4C-LH",  task="4-class",    arch="Local-Hom",             ckpt=PARITY/"4class_local_hom.pt"),
    dict(short="4C-LN",  task="4-class",    arch="FittedHet-LogNorm",     ckpt=PARITY/"4class_fittedhet_lognorm.pt"),
    dict(short="4C-LU",  task="4-class",    arch="FittedHet-LogUniform",  ckpt=PARITY/"4class_fittedhet_loguniform.pt"),
]

ARCH_COLORS = {
    "Local-Hom":            "#4c78a8",
    "FittedHet-LogNorm":    "#54a24b",
    "FittedHet-LogUniform": "#e45756",
}
TASK_LS = {"all-class": "-", "2-class": "--", "4-class": ":"}


# ── SHD cache ─────────────────────────────────────────────────────────────────
class SHDCache:
    def __init__(self, path):
        units, times, labels = open_file(str(path))
        self.units  = list(units[:])
        self.times  = list(times[:])
        self.labels = np.array(labels[:])
        units._v_file.close()


def batch_gen(cache, prms, max_batches=2):
    bs        = int(prms.get("batch_size", 256))
    nb_steps  = int(prms.get("nb_steps", 1000))
    nb_units  = int(prms.get("nb_inputs", 700))
    inv_dt    = 1.0 / float(prms.get("time_step", 1e-3))
    class_list = prms.get("class_list", list(range(20)))
    label_arr  = cache.labels
    sample_idx = np.where(np.isin(label_arr, class_list))[0]
    rng = np.random.default_rng(1000)
    rng.shuffle(sample_idx)
    n_batches = min(max_batches, len(sample_idx) // bs)

    for counter in range(n_batches):
        bidx = sample_idx[bs * counter : bs * (counter + 1)]
        t_arrays = [np.round(cache.times[i] * inv_dt).astype(np.int64) for i in bidx]
        u_arrays = [cache.units[i] for i in bidx]
        lengths  = np.array([len(a) for a in t_arrays])
        if lengths.sum() > 0:
            all_ts = np.concatenate(t_arrays)
            all_us = np.concatenate(u_arrays)
            all_bc = np.repeat(np.arange(bs, dtype=np.int64), lengths)
            valid  = all_ts < nb_steps
            all_ts, all_us, all_bc = all_ts[valid], all_us[valid], all_bc[valid]
            it = torch.from_numpy(np.stack([all_bc, all_ts, all_us]))
            vt = torch.ones(all_ts.size, dtype=torch.float32)
            x_batch = torch.sparse_coo_tensor(it, vt, torch.Size([bs, nb_steps, nb_units])).to_dense()
        else:
            x_batch = torch.zeros(bs, nb_steps, nb_units)
        x_batch.clamp_(max=1.0)
        y_batch = torch.tensor(
            [class_list.index(int(a)) for a in label_arr[bidx]], dtype=torch.long
        )
        yield x_batch, y_batch


# ── Load SHD ──────────────────────────────────────────────────────────────────
print("Loading SHD test cache...")
SHD_CACHE = SHDCache(PROJECT_ROOT / "data" / "shd" / "shd_test.h5")
print("  done.")

# ── PCA loop ──────────────────────────────────────────────────────────────────
pca_profiles = {}
pca_rows     = []

for e in CATALOG:
    short = e["short"]
    print(f"  {short} ...", end=" ", flush=True)
    payload = torch.load(e["ckpt"], map_location=DEVICE, weights_only=False)
    prms    = dict(payload.get("prms", {}))
    prms.update(device=DEVICE, cuda=(DEVICE.type == "cuda"), dtype=torch.float)
    prms.setdefault("batch_size",  256)
    prms.setdefault("time_step",   1e-3)
    prms.setdefault("nb_steps",    1000)
    prms.setdefault("class_list",  list(range(20)))

    model = RSNN(prms, rec=True).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    nb_hidden = int(model.network[0].output_size)

    chunks = []
    with torch.no_grad():
        for x, _ in batch_gen(SHD_CACHE, prms, max_batches=2):
            x = x.to(DEVICE)
            layer_recs = model(0, 0, x)
            mem = layer_recs[0][1]                        # membrane potential
            arr = mem[:, ::4, :].detach().cpu().numpy()   # temporal stride=4
            arr = np.transpose(arr, (2, 0, 1)).reshape(nb_hidden, -1)
            chunks.append(arr)

    full = np.concatenate(chunks, axis=1)                 # [nb_hidden, T]

    subset_size = min(32, full.shape[0])
    sel = np.linspace(0, full.shape[0] - 1, subset_size, dtype=int)
    X   = full[sel, :].T.astype(np.float64)              # [T, subset_size]
    X  -= X.mean(axis=0, keepdims=True)

    sv  = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    ev  = (sv ** 2) / max(X.shape[0] - 1, 1)
    tot = ev.sum()
    evr = ev / tot if tot > 0 else np.zeros_like(ev)
    cum = np.cumsum(evr)

    def n_for(thresh):
        i = int(np.searchsorted(cum, thresh, side="left"))
        return min(i, cum.size - 1) + 1

    pca_profiles[short] = dict(task=e["task"], arch=e["arch"], evr=evr, cum=cum)
    row = dict(
        short=short, task=e["task"], arch=e["arch"],
        cum_pc1=float(cum[0]),
        cum_pc2=float(cum[1]) if cum.size >= 2 else float("nan"),
        cum_pc4=float(cum[3]) if cum.size >= 4 else float("nan"),
        n80=n_for(0.80), n90=n_for(0.90), n95=n_for(0.95),
    )
    pca_rows.append(row)
    print(f"n80={row['n80']}  n90={row['n90']}  cum_pc1={row['cum_pc1']:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for short, profile in pca_profiles.items():
    c  = ARCH_COLORS.get(profile["arch"], "#666")
    ls = TASK_LS.get(profile["task"], "-")
    nc = profile["cum"].size
    comps = np.arange(1, nc + 1)
    axes[0].plot(comps, profile["cum"], color=c, ls=ls, lw=1.7, label=short)
    axes[1].plot(comps, profile["evr"], color=c, ls=ls, lw=1.5, label=short)

axes[0].axhline(0.80, color="k", ls="--", lw=1.0, alpha=0.6, label="80% threshold")
axes[0].axhline(0.90, color="k", ls=":",  lw=1.0, alpha=0.6, label="90% threshold")
axes[0].set_title("PCA cumulative explained variance (all 9 networks)")
axes[0].set_xlabel("principal component index")
axes[0].set_ylabel("cumulative explained variance")
axes[0].set_ylim(0, 1.02)
axes[0].grid(alpha=0.25)

axes[1].set_yscale("log")
axes[1].set_title("PCA variance decay per component (log scale)")
axes[1].set_xlabel("principal component index")
axes[1].set_ylabel("explained variance ratio (log)")
axes[1].grid(alpha=0.25, which="both")

handles, labels = axes[0].get_legend_handles_labels()
unique_legend = dict(zip(labels, handles))
fig.legend(
    unique_legend.values(), unique_legend.keys(),
    loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.04),
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
out_path = FIG_DIR / "fig_pca_variance.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
df = pd.DataFrame(pca_rows)
print("\nPCA variance-decay summary")
print("-" * 80)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
