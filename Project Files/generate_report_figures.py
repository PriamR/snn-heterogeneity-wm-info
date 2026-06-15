"""Generate and save all figures for SNN_Analysis_Report.md."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Project Files")
CKPT        = BASE / "Checkpoints"
PARITY_CKPT = CKPT / "Parity"
FIG_DIR     = BASE / "Report Figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Catalog ───────────────────────────────────────────────────────────────────
CATALOG = [
    dict(short="ALL-LH",  task="all-class",  arch="Local-Hom",             ckpt=CKPT/"local_hom_checkpoint.pt",                                    mem_json=BASE/"sample_subset_local_hom_zero_m_zscore_sweep.json",                        spk_json=BASE/"initial_viz_all_local_hom_spk_observed_wm_sweep.json"),
    dict(short="ALL-LN",  task="all-class",  arch="FittedHet-LogNorm",      ckpt=CKPT/"fittedhet_lognorm_mem_gamma_syn_checkpoint.pt",               mem_json=BASE/"sample_subset_fittedhet_lognormmem_gammasyn_zero_m_zscore_sweep.json",    spk_json=BASE/"initial_viz_all_fittedhet_ln_spk_observed_wm_sweep.json"),
    dict(short="ALL-LU",  task="all-class",  arch="FittedHet-LogUniform",   ckpt=CKPT/"fittedhet_loguniform_mem_gamma_syn_checkpoint.pt",             mem_json=BASE/"sample_subset_fittedhet_loguniformmem_gammasyn_zero_m_zscore_sweep.json", spk_json=BASE/"initial_viz_all_fittedhet_lu_spk_observed_wm_sweep.json"),
    dict(short="2C-LH",   task="2-class",    arch="Local-Hom",             ckpt=PARITY_CKPT/"2class_local_hom.pt",                                   mem_json=BASE/"parity_2class_local_hom_mem_zero_m_zscore_sweep.json",                    spk_json=BASE/"parity_2class_local_hom_spk_zero_m_zscore_sweep.json"),
    dict(short="2C-LN",   task="2-class",    arch="FittedHet-LogNorm",      ckpt=PARITY_CKPT/"2class_fittedhet_lognorm.pt",                          mem_json=BASE/"parity_2class_fittedhet_ln_mem_zero_m_zscore_sweep.json",                  spk_json=BASE/"parity_2class_fittedhet_ln_spk_zero_m_zscore_sweep.json"),
    dict(short="2C-LU",   task="2-class",    arch="FittedHet-LogUniform",   ckpt=PARITY_CKPT/"2class_fittedhet_loguniform.pt",                       mem_json=BASE/"parity_2class_fittedhet_lu_mem_zero_m_zscore_sweep.json",                  spk_json=BASE/"parity_2class_fittedhet_lu_spk_zero_m_zscore_sweep.json"),
    dict(short="4C-LH",   task="4-class",    arch="Local-Hom",             ckpt=PARITY_CKPT/"4class_local_hom.pt",                                   mem_json=BASE/"parity_4class_local_hom_mem_zero_m_zscore_sweep.json",                    spk_json=BASE/"parity_4class_local_hom_spk_zero_m_zscore_sweep.json"),
    dict(short="4C-LN",   task="4-class",    arch="FittedHet-LogNorm",      ckpt=PARITY_CKPT/"4class_fittedhet_lognorm.pt",                          mem_json=BASE/"parity_4class_fittedhet_ln_mem_zero_m_zscore_sweep.json",                  spk_json=BASE/"parity_4class_fittedhet_ln_spk_zero_m_zscore_sweep.json"),
    dict(short="4C-LU",   task="4-class",    arch="FittedHet-LogUniform",   ckpt=PARITY_CKPT/"4class_fittedhet_loguniform.pt",                       mem_json=BASE/"parity_4class_fittedhet_lu_mem_zero_m_zscore_sweep.json",                  spk_json=BASE/"parity_4class_fittedhet_lu_spk_zero_m_zscore_sweep.json"),
]
REPO = dict(short="ALL-RepoHet", task="all-class", arch="Repo-Learned-Het",
            ckpt=CKPT/"network_A_checkpoint.pt",
            mem_json=BASE/"sample_subset_local_het_learned_zero_m_zscore_sweep.json")

ARCH_COLORS  = {"Local-Hom": "#4c78a8", "FittedHet-LogNorm": "#54a24b",
                "FittedHet-LogUniform": "#e45756", "Repo-Learned-Het": "#f58518"}
TASK_ORDER   = ["all-class", "2-class", "4-class"]
ARCH_ORDER   = ["Local-Hom", "FittedHet-LogNorm", "FittedHet-LogUniform"]
SUBSET_SIZES = [2, 4, 8, 16, 32]
TASK_MARKERS = {"all-class": "o", "2-class": "s", "4-class": "^"}
TASK_LS      = {"all-class": "-", "2-class": "--", "4-class": ":"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_sweep(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", data.get("data", data if isinstance(data, list) else []))
    return {int(r.get("subset_size", -1)): r for r in results if int(r.get("subset_size", -1)) > 0}

def rval(row, keys, default=float("nan")):
    for k in keys:
        if k in row:
            try: return float(row[k])
            except: pass
    return default

def ckpt_history(path):
    c = torch.load(path, map_location="cpu", weights_only=False)
    h = c.get("history", {})
    return np.array(h.get("train_acc", []), dtype=float), np.array(h.get("test_acc", []), dtype=float)

def final_acc(path):
    _, t = ckpt_history(path)
    return float(t[-1]) if t.size else float("nan")

def dedup(ax):
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    return list(u.values()), list(u.keys())

def savefig(name):
    p = FIG_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {p.name}")

# ── Build master dataframe ─────────────────────────────────────────────────────
rows = []
for e in CATALOG:
    acc = final_acc(e["ckpt"])
    sweep = load_sweep(e["mem_json"])
    spk_sweep = load_sweep(e.get("spk_json", ""))
    for k in SUBSET_SIZES:
        r = sweep.get(k, {})
        rs = spk_sweep.get(k, {})
        rows.append(dict(
            short=e["short"], task=e["task"], arch=e["arch"],
            subset_size=k, final_test_acc=acc,
            M=rval(r, ["observed_M_bits","M_bits"]),
            W=rval(r, ["observed_W_bits","W_bits"]),
            Mz=rval(r, ["M_zscore"]),
            Mp=rval(r, ["M_p_upper"]),
            M_spk=rval(rs, ["observed_M_bits","M_bits"]),
            W_spk=rval(rs, ["observed_W_bits","W_bits"]),
        ))

df = pd.DataFrame(rows)

# ── Figure 1: Accuracy bar chart ──────────────────────────────────────────────
print("Generating Fig 1: Accuracy bar chart...")
acc_df = df[["short","task","arch","final_test_acc"]].drop_duplicates().sort_values(["task","arch","short"])
# Add repo
acc_df = pd.concat([acc_df, pd.DataFrame([dict(short="ALL-RepoHet", task="all-class",
    arch="Repo-Learned-Het", final_test_acc=final_acc(REPO["ckpt"]))])], ignore_index=True)

SHORT_ORDER = [e["short"] for e in CATALOG]
acc_df["_ord"] = acc_df["short"].map({s: i for i, s in enumerate(SHORT_ORDER + ["ALL-RepoHet"])})
acc_df = acc_df.sort_values("_ord")

fig, ax = plt.subplots(figsize=(13, 4.5))
x = np.arange(len(acc_df))
colors = [ARCH_COLORS.get(a, "#aaaaaa") for a in acc_df["arch"]]
ax.bar(x, acc_df["final_test_acc"], color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(acc_df["short"], rotation=25)
ax.set_ylabel("Final test accuracy")
ax.set_title("Final task performance across all networks (+ repo reference)")
ax.grid(axis="y", alpha=0.25)
for i, v in enumerate(acc_df["final_test_acc"]):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for arch in list(ARCH_ORDER) + ["Repo-Learned-Het"]:
    ax.plot([], [], color=ARCH_COLORS[arch], lw=8, label=arch)
ax.legend(frameon=False, fontsize=8, loc="upper left")
# Task boundary lines
ax.axvline(2.5, color="#888", lw=1, ls="--", alpha=0.5)
ax.axvline(5.5, color="#888", lw=1, ls="--", alpha=0.5)
for tx, label in [(1, "all-class"), (4, "2-class"), (7, "4-class")]:
    ax.text(tx, ax.get_ylim()[1]*0.97, label, ha="center", fontsize=8, color="#555")
plt.tight_layout()
savefig("fig01_accuracy_bar.png")

# ── Figure 2: M and W info curves ─────────────────────────────────────────────
print("Generating Fig 2: M/W information curves...")
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
for ri, task in enumerate(TASK_ORDER):
    td = df[df["task"] == task]
    for short, grp in td.groupby("short"):
        grp = grp.sort_values("subset_size")
        c = ARCH_COLORS.get(grp["arch"].iloc[0], "#666")
        axes[ri, 0].plot(grp["subset_size"], grp["M"], marker="o", lw=1.8, color=c, label=short)
        axes[ri, 1].plot(grp["subset_size"], grp["W"], marker="o", lw=1.8, color=c, label=short)
    axes[ri, 0].set_title(f"{task}: Observed M")
    axes[ri, 0].set_ylabel("M (bits)")
    axes[ri, 0].grid(alpha=0.25)
    axes[ri, 1].set_title(f"{task}: Observed W")
    axes[ri, 1].set_ylabel("W (bits)")
    axes[ri, 1].grid(alpha=0.25)
    h, l = dedup(axes[ri, 0])
    axes[ri, 0].legend(h, l, fontsize=8, frameon=False)
for ax in axes[-1, :]:
    ax.set_xlabel("subset size (k)")
for ax in axes.flat:
    ax.set_xticks(SUBSET_SIZES)
plt.suptitle("Observed M and W Information Curves Across All Tasks", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
savefig("fig02_mw_curves.png")

# ── Figure 3: M vs accuracy scatter ───────────────────────────────────────────
print("Generating Fig 3: M vs accuracy scatter...")
scatter_df = df.groupby(["short","task","arch","final_test_acc"], as_index=False)["M"].mean().rename(columns={"M":"avg_M"})
fig, ax = plt.subplots(figsize=(10, 6))
for task in TASK_ORDER:
    for arch in ARCH_ORDER:
        sub = scatter_df[(scatter_df["task"]==task) & (scatter_df["arch"]==arch)]
        if sub.empty: continue
        ax.scatter(sub["avg_M"], sub["final_test_acc"], s=95,
                   marker=TASK_MARKERS[task], color=ARCH_COLORS[arch], alpha=0.9,
                   label=f"{task} | {arch}")
for _, row in scatter_df.iterrows():
    ax.annotate(row["short"], (row["avg_M"], row["final_test_acc"]),
                textcoords="offset points", xytext=(5,4), fontsize=8)
ax.set_xlabel("Average observed M across subsets (bits)")
ax.set_ylabel("Final test accuracy")
ax.set_title("Average M-information vs Task Performance (all 9 networks)")
ax.grid(alpha=0.25)
h, l = dedup(ax)
ax.legend(h, l, fontsize=8, frameon=False)
plt.tight_layout()
savefig("fig03_m_vs_accuracy.png")

# ── Figure 4: Z-score heatmap + -log10(p) ─────────────────────────────────────
print("Generating Fig 4: Z-score heatmaps...")
zdf = df[np.isfinite(df["Mz"])].copy()
short_order = [e["short"] for e in CATALOG]
pivot_z = zdf.pivot(index="subset_size", columns="short", values="Mz").reindex(index=SUBSET_SIZES, columns=short_order)
pivot_p = zdf.pivot(index="subset_size", columns="short", values="Mp").reindex(index=SUBSET_SIZES, columns=short_order)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
im0 = axes[0].imshow(pivot_z.to_numpy(float), aspect="auto", cmap="coolwarm")
axes[0].set_title("M z-score heatmap (membrane)")
axes[0].set_xticks(np.arange(len(short_order))); axes[0].set_xticklabels(short_order, rotation=30, ha="right")
axes[0].set_yticks(np.arange(len(SUBSET_SIZES))); axes[0].set_yticklabels(SUBSET_SIZES)
axes[0].set_xlabel("network"); axes[0].set_ylabel("subset size (k)")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

neglogp = -np.log10(np.clip(pivot_p.to_numpy(float), 1e-12, 1.0))
im1 = axes[1].imshow(neglogp, aspect="auto", cmap="magma")
axes[1].set_title("-log₁₀(M p-value) heatmap (membrane)")
axes[1].set_xticks(np.arange(len(short_order))); axes[1].set_xticklabels(short_order, rotation=30, ha="right")
axes[1].set_yticks(np.arange(len(SUBSET_SIZES))); axes[1].set_yticklabels(SUBSET_SIZES)
axes[1].set_xlabel("network"); axes[1].set_ylabel("subset size (k)")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
plt.tight_layout()
savefig("fig04_zscore_heatmaps.png")

# ── Figure 5: Z-score line plots ──────────────────────────────────────────────
print("Generating Fig 5: Z-score line plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, task in zip(axes, TASK_ORDER):
    td = zdf[zdf["task"]==task]
    for short, grp in td.groupby("short"):
        grp = grp.sort_values("subset_size")
        c = ARCH_COLORS.get(grp["arch"].iloc[0], "#666")
        ax.plot(grp["subset_size"], grp["Mz"], marker="o", lw=1.8, color=c, label=short)
    ax.set_title(f"{task}: M z-score by subset")
    ax.set_xlabel("subset size (k)")
    ax.set_xticks(SUBSET_SIZES)
    ax.grid(alpha=0.25)
    h, l = dedup(ax)
    ax.legend(h, l, fontsize=8, frameon=False)
axes[0].set_ylabel("M z-score")
plt.suptitle("M Z-Score Profiles Across Tasks (membrane representation)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
savefig("fig05_zscore_lines.png")

# ── Figure 6: Mem vs spike delta M ────────────────────────────────────────────
print("Generating Fig 6: Mem vs spike delta M...")
spk_df = df[np.isfinite(df["M_spk"])].copy()
if not spk_df.empty:
    spk_df["delta_M"] = spk_df["M_spk"] - spk_df["M"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for short, grp in spk_df.groupby("short"):
        grp = grp.sort_values("subset_size")
        c = ARCH_COLORS.get(grp["arch"].iloc[0], "#666")
        axes[0].plot(grp["subset_size"], grp["delta_M"], marker="o", lw=1.6, color=c, label=short)
    axes[0].axhline(0, color="k", ls="--", lw=1, alpha=0.7)
    axes[0].set_title("Delta M by subset (spk − mem)")
    axes[0].set_xlabel("subset size (k)"); axes[0].set_ylabel("delta M (bits)")
    axes[0].set_xticks(SUBSET_SIZES); axes[0].grid(alpha=0.25)
    h, l = dedup(axes[0]); axes[0].legend(h, l, fontsize=8, frameon=False)

    avg = spk_df.groupby(["short","arch"], as_index=False)[["M","M_spk"]].mean()
    axes[1].scatter(avg["M"], avg["M_spk"], s=80, c=[ARCH_COLORS.get(a,"#666") for a in avg["arch"]], alpha=0.9)
    for _, r in avg.iterrows():
        axes[1].annotate(r["short"], (r["M"], r["M_spk"]), textcoords="offset points", xytext=(4,4), fontsize=8)
    fm = np.isfinite(avg["M"]) & np.isfinite(avg["M_spk"])
    if fm.any():
        lo = min(avg.loc[fm,"M"].min(), avg.loc[fm,"M_spk"].min())
        hi = max(avg.loc[fm,"M"].max(), avg.loc[fm,"M_spk"].max())
        axes[1].plot([lo,hi],[lo,hi],"k--",lw=1)
    axes[1].set_title("Avg M: membrane vs spike"); axes[1].set_xlabel("avg M (mem)"); axes[1].set_ylabel("avg M (spk)")
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    savefig("fig06_mem_vs_spk.png")
else:
    print("  skipped (no spike data)")

# ── Figure 7: Percentile heatmap ──────────────────────────────────────────────
print("Generating Fig 7: Percentile heatmap...")
zdf2 = df[np.isfinite(df["Mz"])].copy()
def pct_rank(s):
    n = len(s)
    if n <= 1: return s * 0.0 + 50.0
    return (s.rank(method="average") - 1) / (n - 1) * 100.0
zdf2["pct_global"] = zdf2.groupby("subset_size")["Mz"].transform(pct_rank)
zdf2["pct_within"] = zdf2.groupby(["task","subset_size"])["Mz"].transform(pct_rank)

pivot_pct = (zdf2.pivot(index="subset_size", columns="short", values="pct_global")
             .reindex(index=SUBSET_SIZES, columns=short_order))

fig, ax = plt.subplots(figsize=(13, 4.5))
im = ax.imshow(pivot_pct.to_numpy(float), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
ax.set_title("M z-score global percentile rank (100% = highest z-score among all 9 networks)", fontsize=12)
ax.set_xticks(np.arange(len(short_order))); ax.set_xticklabels(short_order, rotation=25, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(SUBSET_SIZES))); ax.set_yticklabels(SUBSET_SIZES)
ax.set_xlabel("network"); ax.set_ylabel("subset size (k)")
for i, k in enumerate(SUBSET_SIZES):
    for j, s in enumerate(short_order):
        v = pivot_pct.loc[k, s] if (k in pivot_pct.index and s in pivot_pct.columns) else float("nan")
        if np.isfinite(v):
            tc = "black" if v < 60 else "white"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=tc, fontweight="bold")
for xb in [3, 6]:
    ax.axvline(xb - 0.5, color="white", lw=2.0, alpha=0.9)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="percentile rank (%)")
plt.tight_layout()
savefig("fig07_percentile_heatmap.png")

# ── Figure 8: Global percentile curves ────────────────────────────────────────
print("Generating Fig 8: Global percentile curves...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    td = zdf2[zdf2["task"]==task].sort_values("subset_size")
    for short, grp in td.groupby("short"):
        grp = grp.sort_values("subset_size")
        c = ARCH_COLORS.get(grp["arch"].iloc[0], "#666")
        ax.plot(grp["subset_size"], grp["pct_global"], marker="o", lw=2.0, color=c, label=short)
        last = grp.iloc[-1]
        ax.annotate(f"{last['pct_global']:.0f}%", (last["subset_size"], last["pct_global"]),
                    textcoords="offset points", xytext=(5,0), fontsize=7.5, color=c)
    ax.axhline(50, color="#888", ls="--", lw=1, alpha=0.6)
    ax.set_title(task, fontsize=11); ax.set_xlabel("subset size (k)")
    ax.set_ylim(-5, 110); ax.set_yticks([0,25,50,75,100])
    ax.set_yticklabels(["0%","25%","50%","75%","100%"])
    ax.set_xticks(SUBSET_SIZES); ax.grid(alpha=0.25)
    h, l = dedup(ax); ax.legend(h, l, fontsize=8, frameon=False)
axes[0].set_ylabel("global percentile rank")
plt.suptitle("Global Percentile Rank of M Z-Score by Task and Subset Size", fontsize=13, y=1.02)
plt.tight_layout()
savefig("fig08_percentile_curves_global.png")

# ── Figure 9: Within-task percentile ──────────────────────────────────────────
print("Generating Fig 9: Within-task percentile...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, task in zip(axes, TASK_ORDER):
    td = zdf2[zdf2["task"]==task].sort_values("subset_size")
    for short, grp in td.groupby("short"):
        grp = grp.sort_values("subset_size")
        c = ARCH_COLORS.get(grp["arch"].iloc[0], "#666")
        ax.plot(grp["subset_size"], grp["pct_within"], marker="o", lw=2.2, color=c, label=grp["arch"].iloc[0])
    ax.set_title(task, fontsize=11); ax.set_xlabel("subset size (k)")
    ax.set_ylim(-5, 110); ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%\n(lowest)", "50%", "100%\n(highest)"])
    ax.set_xticks(SUBSET_SIZES); ax.grid(alpha=0.25)
    h, l = dedup(ax)
    u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=8, frameon=False)
axes[0].set_ylabel("within-task percentile rank")
plt.suptitle("Within-Task Percentile: Which Architecture Wins at Each Subset Size?", fontsize=13, y=1.02)
plt.tight_layout()
savefig("fig09_percentile_within_task.png")

# ── Figure 10: Average global percentile bar chart ────────────────────────────
print("Generating Fig 10: Avg percentile bar chart...")
avg_pct = zdf2.groupby(["task","arch"], as_index=False)["pct_global"].mean()
x = np.arange(len(TASK_ORDER))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
for ai, arch in enumerate(ARCH_ORDER):
    vals = [avg_pct[(avg_pct["task"]==t) & (avg_pct["arch"]==arch)]["pct_global"].iloc[0]
            if not avg_pct[(avg_pct["task"]==t) & (avg_pct["arch"]==arch)].empty else float("nan")
            for t in TASK_ORDER]
    offset = (ai - 1) * width
    bars = ax.bar(x + offset, vals, width * 0.9, color=ARCH_COLORS[arch], alpha=0.9, label=arch)
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8.5)
ax.axhline(50, color="#888", ls="--", lw=1, alpha=0.6, label="median (50%)")
ax.set_xticks(x); ax.set_xticklabels(TASK_ORDER, fontsize=10)
ax.set_ylabel("average global percentile rank (%)")
ax.set_ylim(0, 115); ax.set_yticks([0,25,50,75,100])
ax.set_title("Average M Z-Score Global Percentile by Architecture and Task\n(averaged across all subset sizes)", fontsize=12)
ax.legend(fontsize=9, frameon=False); ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
savefig("fig10_avg_percentile_bar.png")

# ── Figure 11: Convergence plots ──────────────────────────────────────────────
print("Generating Fig 11: Convergence plots...")
conv_rows = []
for e in CATALOG:
    tr, te = ckpt_history(e["ckpt"])
    if te.size == 0: continue
    n = te.size
    best_idx = int(np.argmax(te))
    best_e, best_t, final_t = best_idx + 1, float(te[best_idx]), float(te[-1])
    tail = te[-5:]
    tail_d = float(tail[-1] - tail[0]) if len(tail) > 1 else 0.0
    tail_s = float(np.nanstd(tail))
    final_tr = float(tr[-1]) if tr.size else float("nan")
    conv_rows.append(dict(short=e["short"], task=e["task"], arch=e["arch"],
        n_epochs=n, best_epoch=best_e, best_test=best_t, final_test=final_t,
        peak_drop=best_t - final_t, tail_delta=tail_d, tail_std=tail_s,
        gap=final_tr - final_t if np.isfinite(final_tr) else float("nan")))

cdf = pd.DataFrame(conv_rows)
cdf["_ord"] = cdf["short"].map({s: i for i, s in enumerate(e["short"] for e in CATALOG)})
cdf = cdf.sort_values("_ord")
x = np.arange(len(cdf))
colors = [ARCH_COLORS.get(a, "#666") for a in cdf["arch"]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(x, cdf["peak_drop"], color=colors, alpha=0.9)
axes[0].axhline(0.015, color="k", ls="--", lw=1, alpha=0.7)
axes[0].set_title("Peak-to-final test accuracy drop")
axes[0].set_ylabel("best − final test acc")
axes[0].set_xticks(x); axes[0].set_xticklabels(cdf["short"], rotation=30, ha="right")
axes[0].grid(axis="y", alpha=0.25)

axes[1].bar(x, cdf["tail_delta"], color=colors, alpha=0.9)
axes[1].axhline(0, color="k", lw=1); axes[1].axhline(0.01, color="#666", ls="--", lw=0.9)
axes[1].axhline(-0.01, color="#666", ls="--", lw=0.9)
axes[1].set_title("Last-5-epoch test accuracy trend")
axes[1].set_ylabel("final − 5th-from-last test acc")
axes[1].set_xticks(x); axes[1].set_xticklabels(cdf["short"], rotation=30, ha="right")
axes[1].grid(axis="y", alpha=0.25)

for task in TASK_ORDER:
    for arch in ARCH_ORDER:
        sub = cdf[(cdf["task"]==task) & (cdf["arch"]==arch)]
        axes[2].scatter(sub["final_test"], sub["gap"], s=95, marker=TASK_MARKERS[task],
                        color=ARCH_COLORS.get(arch,"#666"), alpha=0.9, label=f"{task}|{arch}")
for _, r in cdf.iterrows():
    if np.isfinite(r["final_test"]) and np.isfinite(r["gap"]):
        axes[2].annotate(r["short"], (r["final_test"], r["gap"]),
                         textcoords="offset points", xytext=(4,4), fontsize=8)
axes[2].axhline(0, color="k", ls="--", lw=1, alpha=0.7)
axes[2].set_title("Generalization gap at final epoch")
axes[2].set_xlabel("final test accuracy"); axes[2].set_ylabel("final train − final test")
axes[2].grid(alpha=0.25)
h, l = dedup(axes[2]); axes[2].legend(h, l, fontsize=7, frameon=False)

for arch in ARCH_ORDER:
    axes[0].plot([], [], color=ARCH_COLORS[arch], lw=8, label=arch)
axes[0].legend(fontsize=7, frameon=False, loc="upper right")
plt.tight_layout()
savefig("fig11_convergence.png")

# ── Figure 12: Training curves ────────────────────────────────────────────────
print("Generating Fig 12: Training curves...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, task in zip(axes, TASK_ORDER):
    entries = [e for e in CATALOG if e["task"] == task]
    for e in entries:
        _, te = ckpt_history(e["ckpt"])
        if te.size == 0: continue
        c = ARCH_COLORS.get(e["arch"], "#666")
        ax.plot(np.arange(1, te.size+1), te, color=c, lw=1.8, label=e["short"])
    ax.set_title(f"{task}: test accuracy per epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.grid(alpha=0.25)
    h, l = dedup(ax); ax.legend(h, l, fontsize=8, frameon=False)
plt.suptitle("Training Curves — Test Accuracy per Epoch", fontsize=13)
plt.tight_layout(rect=[0,0,1,0.95])
savefig("fig12_training_curves.png")

print(f"\nAll figures saved to: {FIG_DIR}")
print("Done.")
