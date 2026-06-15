#!/usr/bin/env python
# coding: utf-8

# # Seeded Run 1 — 4-Class Task
# 
# This notebook trains five matched seeded pairs for the 4-class SHD parity×language task.
# 
# Each training cell handles one seed pair independently. Once a pair finishes, its checkpoints and summaries are already saved, so you can stop there and later continue with the next seed cell instead of rerunning a full loop.
# 
# Cell 2 defines the seeds used in this notebook. Cell 3 verifies three things before training starts:
# 1. this notebook is locked to the 4-class parity×language classification task
# 2. the homogeneous and heterogeneous model definitions are set up correctly
# 3. the log-normal hidden tau_m samples vary across the configured seeds

# In[ ]:


import sys
sys.path.insert(0, r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Project Files")

from pathlib import Path
import importlib

import pandas as pd
from IPython.display import display

import seeded_runs_common as seeded_runs_common

seeded_runs_common = importlib.reload(seeded_runs_common)

CHECKPOINT_ROOT = seeded_runs_common.CHECKPOINT_ROOT
DEVICE = seeded_runs_common.DEVICE
SHD_TEST = seeded_runs_common.SHD_TEST
SHD_TRAIN = seeded_runs_common.SHD_TRAIN
TAU_ARTIFACT_PATH = seeded_runs_common.TAU_ARTIFACT_PATH
TASKS = seeded_runs_common.TASKS
build_pair_summary_row = seeded_runs_common.build_pair_summary_row
build_sampling_preview_rows = seeded_runs_common.build_sampling_preview_rows
build_seeded_pair = seeded_runs_common.build_seeded_pair
load_default_caches = seeded_runs_common.load_default_caches
read_manifest_rows = seeded_runs_common.read_manifest_rows
run_pair_training = seeded_runs_common.run_pair_training
upsert_rows = seeded_runs_common.upsert_rows
write_manifest_rows = seeded_runs_common.write_manifest_rows

RUN_LABEL = "seeded_run_1"
TASK_KEY = "4class"
MEM_DISTRIBUTION_FAMILY = "lognormal"
MASTER_SEEDS = [101, 202, 210, 340, 440]

RUN_DIR = CHECKPOINT_ROOT / RUN_LABEL / f"{TASK_KEY}_{MEM_DISTRIBUTION_FAMILY}"
RESULT_STEM = RUN_DIR / f"{RUN_LABEL}_checkpoint_summary"
PAIR_STEM = RUN_DIR / f"{RUN_LABEL}_pair_summary"
PAIRED_ACC_CSV = RUN_DIR / f"{RUN_LABEL}_paired_accuracy.csv"

TASK_DEF = TASKS[TASK_KEY]

print(f"Device: {DEVICE}")
print(f"Train file exists: {SHD_TRAIN.exists()}")
print(f"Test file exists: {SHD_TEST.exists()}")
print(f"Tau artifact exists: {TAU_ARTIFACT_PATH.exists()}")
print(f"Run directory: {RUN_DIR}")
print(f"Master seeds: {MASTER_SEEDS}")
print(f"Task name: {TASK_DEF['task_name']}")
print(f"Task outputs: {TASK_DEF['nb_outputs']}")


# In[ ]:


assert TASK_KEY == "4class"
assert TASK_DEF["nb_outputs"] == 4
assert TASK_DEF["task_name"] == "4class_parity_language"

# Verify 4-class parity×language label map (English/German × even/odd)
expected_4class_map = {}
for i in range(20):
    is_german = int(i >= 10)
    is_odd = int(i % 2 == 1)
    expected_4class_map[i] = is_german * 2 + is_odd
assert TASK_DEF["task_label_map"] == expected_4class_map

preview = build_seeded_pair(
    master_seed=MASTER_SEEDS[0],
    task_key=TASK_KEY,
    mem_distribution_family=MEM_DISTRIBUTION_FAMILY,
)
preview_meta = preview["metadata"]

assert preview["hom_prms"]["nb_outputs"] == 4
assert preview["hetero_prms"]["nb_outputs"] == 4
assert not preview["hom_model"].network[0].alpha.requires_grad
assert not preview["hom_model"].network[0].beta.requires_grad
assert not preview["hetero_model"].network[0].alpha.requires_grad
assert not preview["hetero_model"].network[0].beta.requires_grad
assert preview_meta["linear_sync_verified"]
assert preview_meta["hom_hidden_tau_unique"] == 1
assert preview_meta["hetero_hidden_tau_unique"] > 1

sampling_rows = build_sampling_preview_rows(
    MASTER_SEEDS,
    task_key=TASK_KEY,
    mem_distribution_family=MEM_DISTRIBUTION_FAMILY,
)
sampling_df = pd.DataFrame(sampling_rows).sort_values("pair_seed").reset_index(drop=True)

assert not sampling_df["sample_matches_previous"].any()

display(sampling_df)
print("4-class parity×language task mapping verified.")
print("Homogeneous anchor and heterogeneous sampled definitions verified.")
print("Log-normal hidden tau_m sampling varies across the configured seeds.")


# In[ ]:


train_cache, test_cache = load_default_caches()


# In[ ]:


CHECKPOINT_KEY_FIELDS = ["pair_seed", "pair_role"]
PAIR_KEY_FIELDS = ["pair_seed"]

def show_run_status():
    checkpoint_rows = read_manifest_rows(RESULT_STEM)
    pair_rows = read_manifest_rows(PAIR_STEM)

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    pair_df = pd.DataFrame(pair_rows)

    if not checkpoint_df.empty:
        checkpoint_df = checkpoint_df.sort_values(["pair_seed", "pair_role"]).reset_index(drop=True)
    if not pair_df.empty:
        pair_df = pair_df.sort_values("pair_seed").reset_index(drop=True)

    paired_acc_df = pd.DataFrame()
    if not checkpoint_df.empty:
        paired_acc_df = (
            checkpoint_df.pivot(index="pair_seed", columns="pair_role", values="final_test_acc")
            .reset_index()
            .sort_values("pair_seed")
            .reset_index(drop=True)
        )
        if {"heterogeneous_sampled", "homogeneous_anchor"}.issubset(paired_acc_df.columns):
            paired_acc_df["hetero_minus_hom"] = (
                paired_acc_df["heterogeneous_sampled"] - paired_acc_df["homogeneous_anchor"]
            )
            paired_acc_df.to_csv(PAIRED_ACC_CSV, index=False)

    return pair_df, checkpoint_df, paired_acc_df

def train_one_seed(master_seed):
    run_rows, pair_meta = run_pair_training(
        master_seed=master_seed,
        train_cache=train_cache,
        test_cache=test_cache,
        task_key=TASK_KEY,
        mem_distribution_family=MEM_DISTRIBUTION_FAMILY,
        run_label=RUN_LABEL,
        skip_existing=True,
    )

    checkpoint_rows = upsert_rows(
        read_manifest_rows(RESULT_STEM),
        run_rows,
        CHECKPOINT_KEY_FIELDS,
    )
    pair_rows = upsert_rows(
        read_manifest_rows(PAIR_STEM),
        [build_pair_summary_row(pair_meta)],
        PAIR_KEY_FIELDS,
    )

    write_manifest_rows(checkpoint_rows, RESULT_STEM)
    write_manifest_rows(pair_rows, PAIR_STEM)

    pair_df, checkpoint_df, paired_acc_df = show_run_status()
    seed_df = checkpoint_df[checkpoint_df["pair_seed"] == master_seed].reset_index(drop=True)
    return seed_df, pair_df, checkpoint_df, paired_acc_df

print("Run helpers ready.")
print("Execute one seed cell at a time. Finished seeds are reused from saved checkpoints.")


# In[ ]:


# Train or reuse seed pair 101
seed_df, pair_df, checkpoint_df, paired_acc_df = train_one_seed(101)
display(seed_df[["pair_seed", "pair_role", "final_test_acc", "final_test_loss", "elapsed_s"]])
display(pair_df[pair_df["pair_seed"] == 101].reset_index(drop=True))


# In[ ]:


# Train or reuse seed pair 202
seed_df, pair_df, checkpoint_df, paired_acc_df = train_one_seed(202)
display(seed_df[["pair_seed", "pair_role", "final_test_acc", "final_test_loss", "elapsed_s"]])
display(pair_df[pair_df["pair_seed"] == 202].reset_index(drop=True))


# In[ ]:


# Train or reuse seed pair 210
seed_df, pair_df, checkpoint_df, paired_acc_df = train_one_seed(210)
display(seed_df[["pair_seed", "pair_role", "final_test_acc", "final_test_loss", "elapsed_s"]])
display(pair_df[pair_df["pair_seed"] == 210].reset_index(drop=True))


# In[ ]:


# Train or reuse seed pair 340
seed_df, pair_df, checkpoint_df, paired_acc_df = train_one_seed(340)
display(seed_df[["pair_seed", "pair_role", "final_test_acc", "final_test_loss", "elapsed_s"]])
display(pair_df[pair_df["pair_seed"] == 340].reset_index(drop=True))


# In[ ]:


# Train or reuse seed pair 440
seed_df, pair_df, checkpoint_df, paired_acc_df = train_one_seed(440)
display(seed_df[["pair_seed", "pair_role", "final_test_acc", "final_test_loss", "elapsed_s"]])
display(pair_df[pair_df["pair_seed"] == 440].reset_index(drop=True))


# In[ ]:


# Final status summary
pair_df, checkpoint_df, paired_acc_df = show_run_status()

if pair_df.empty:
    print("No saved seed summaries yet.")
else:
    display(pair_df)

if checkpoint_df.empty:
    print("No saved checkpoint summaries yet.")
else:
    display(checkpoint_df[["pair_seed", "pair_role", "final_test_acc", "final_test_loss"]])

if not paired_acc_df.empty:
    display(paired_acc_df)
    print(f"Saved paired accuracy view to: {PAIRED_ACC_CSV}")

