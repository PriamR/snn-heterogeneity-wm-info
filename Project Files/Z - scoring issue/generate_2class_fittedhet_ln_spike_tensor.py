from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


HANDOFF_DIR = Path(__file__).resolve().parent
PROJECT_FILES_DIR = HANDOFF_DIR.parent
PROJECT_ROOT = PROJECT_FILES_DIR.parent
PAPER_ROOT = PROJECT_ROOT / "neural_heterogeneity" / "SuGD_code"
SHD_TEST_PATH = PROJECT_ROOT / "data" / "shd" / "shd_test.h5"
SOURCE_CHECKPOINT = PROJECT_FILES_DIR / "Checkpoints" / "Parity" / "2class_fittedhet_lognorm.pt"
DEST_CHECKPOINT = HANDOFF_DIR / "checkpoints" / "2class_fittedhet_lognorm.pt"
DEST_SPIKE_TENSOR = HANDOFF_DIR / "spk_tensors" / "2class_fittedhet_ln_spk_tensor.npy"

if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

RSNN = importlib.import_module("model").RSNN
from data_gen import open_file  # noqa: E402


def load_parity_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    prms = dict(ckpt["prms"])
    prms["dtype"] = torch.float
    prms["device"] = device
    prms["cuda"] = device.type == "cuda"
    model = RSNN(prms, rec=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, prms


@torch.no_grad()
def collect_spike_tensor_from_file(
    model,
    prms,
    shd_test_path: Path,
    device: torch.device,
    max_batches: int = 2,
    downsample_stride: int = 4,
):
    stride = max(int(downsample_stride), 1)
    nb_hidden = int(model.network[0].output_size)

    raw_units, raw_times, raw_labels = open_file(str(shd_test_path))
    try:
        units = list(raw_units[:])
        times = list(raw_times[:])
        labels = np.array(raw_labels[:])
    finally:
        raw_units._v_file.close()

    batch_size = int(prms["batch_size"])
    nb_steps = int(prms["nb_steps"])
    nb_inputs = int(prms["nb_inputs"])
    inv_dt = 1.0 / float(prms["time_step"])
    class_list = prms["class_list"]

    sample_index = np.where(np.isin(labels, class_list))[0]
    n_batches = min(int(max_batches), -(-len(sample_index) // batch_size))

    chunks = []
    for batch_idx in range(n_batches):
        batch_index = sample_index[
            batch_size * batch_idx : min(len(sample_index), batch_size * (batch_idx + 1))
        ]
        actual_batch_size = len(batch_index)

        t_arrays = [np.round(times[idx] * inv_dt).astype(np.int64) for idx in batch_index]
        u_arrays = [units[idx] for idx in batch_index]
        lengths = np.array([len(arr) for arr in t_arrays], dtype=np.int64)

        if lengths.sum():
            all_ts = np.concatenate(t_arrays)
            all_us = np.concatenate(u_arrays)
            all_bc = np.repeat(np.arange(actual_batch_size, dtype=np.int64), lengths)
            valid = all_ts < nb_steps
            all_ts = all_ts[valid]
            all_us = all_us[valid]
            all_bc = all_bc[valid]
            indices = torch.from_numpy(np.stack([all_bc, all_ts, all_us]))
            values = torch.ones(all_ts.size, dtype=torch.float32)
            x_batch = torch.sparse_coo_tensor(
                indices,
                values,
                torch.Size([actual_batch_size, nb_steps, nb_inputs]),
            ).to_dense()
        else:
            x_batch = torch.zeros(actual_batch_size, nb_steps, nb_inputs)

        x_batch = x_batch.clamp(max=1.0).to(device)
        layer_recs = model(0, 0, x_batch)

        spk = layer_recs[0][2]
        spk = spk[:, ::stride, :].detach().cpu().numpy()
        spk = np.transpose(spk, (2, 0, 1)).reshape(nb_hidden, -1)
        chunks.append(spk)

    return np.concatenate(chunks, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Generate the 2-class log-normal spike tensor for the handoff package.")
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--downsample-stride", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {SOURCE_CHECKPOINT}")
    if not SHD_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing SHD test file: {SHD_TEST_PATH}")

    DEST_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    DEST_SPIKE_TENSOR.parent.mkdir(parents=True, exist_ok=True)

    if args.force or not DEST_CHECKPOINT.exists():
        shutil.copy2(SOURCE_CHECKPOINT, DEST_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device            : {device}")
    print(f"Source checkpoint : {SOURCE_CHECKPOINT}")
    print(f"Dest checkpoint   : {DEST_CHECKPOINT}")
    print(f"Dest spike tensor : {DEST_SPIKE_TENSOR}")

    model, prms = load_parity_model(DEST_CHECKPOINT, device)
    print(f"Loaded model      : nb_recurrent={prms['nb_recurrent']} nb_outputs={prms['nb_outputs']}")

    spk_tensor = collect_spike_tensor_from_file(
        model,
        prms,
        SHD_TEST_PATH,
        device=device,
        max_batches=args.max_batches,
        downsample_stride=args.downsample_stride,
    )

    unique_vals = np.unique(spk_tensor)
    is_binary = np.all((spk_tensor == 0.0) | (spk_tensor == 1.0))
    print(f"Tensor shape      : {spk_tensor.shape}")
    print(f"Unique values     : {unique_vals}")
    print(f"Is binary         : {is_binary}")
    print(f"Mean rate         : {float(spk_tensor.mean()):.6f}")

    np.save(DEST_SPIKE_TENSOR, spk_tensor.astype(np.float32))
    print(f"Saved spike tensor: {DEST_SPIKE_TENSOR}")


if __name__ == "__main__":
    main()
