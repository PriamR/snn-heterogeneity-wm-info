"""Extract z-score tables from sweep JSON files and print markdown tables."""
import json, pathlib

proj = pathlib.Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Project Files")

ZSCORE_MEM = {
    "ALL-LH": "sample_subset_local_hom_zero_m_zscore_sweep.json",
    "ALL-LN": "sample_subset_fittedhet_lognormmem_gammasyn_zero_m_zscore_sweep.json",
    "ALL-LU": "sample_subset_fittedhet_loguniformmem_gammasyn_zero_m_zscore_sweep.json",
    "2C-LH":  "parity_2class_local_hom_mem_zero_m_zscore_sweep.json",
    "2C-LN":  "parity_2class_fittedhet_ln_mem_zero_m_zscore_sweep.json",
    "2C-LU":  "parity_2class_fittedhet_lu_mem_zero_m_zscore_sweep.json",
    "4C-LH":  "parity_4class_local_hom_mem_zero_m_zscore_sweep.json",
    "4C-LN":  "parity_4class_fittedhet_ln_mem_zero_m_zscore_sweep.json",
    "4C-LU":  "parity_4class_fittedhet_lu_mem_zero_m_zscore_sweep.json",
}
ZSCORE_SPK = {
    "2C-LH":  "parity_2class_local_hom_spk_zero_m_zscore_sweep.json",
    "2C-LN":  "parity_2class_fittedhet_ln_spk_zero_m_zscore_sweep.json",
    "2C-LU":  "parity_2class_fittedhet_lu_spk_zero_m_zscore_sweep.json",
    "4C-LH":  "parity_4class_local_hom_spk_zero_m_zscore_sweep.json",
    "4C-LN":  "parity_4class_fittedhet_ln_spk_zero_m_zscore_sweep.json",
    "4C-LU":  "parity_4class_fittedhet_lu_spk_zero_m_zscore_sweep.json",
}

def load_results(fname):
    data = json.loads((proj / fname).read_text())
    return data["results"]

# Collect all subset sizes per tensor type
def get_sizes(fname):
    return [r["subset_size"] for r in load_results(fname)]

# Build mem table
print("=== MEMBRANE M z-score across all subset sizes ===")
for net, fname in ZSCORE_MEM.items():
    results = load_results(fname)
    for r in results:
        k = r["subset_size"]
        z = r["M_zscore"]
        m = r["observed_M_bits"]
        print(f"  {net:6s}  k={k:2d}  obs_M={m:.6f}  z={z:.1f}")

print()
print("=== SPIKE M z-score (parity only) ===")
for net, fname in ZSCORE_SPK.items():
    results = load_results(fname)
    for r in results:
        k = r["subset_size"]
        z = r["M_zscore"]
        m = r["observed_M_bits"]
        print(f"  {net:6s}  k={k:2d}  obs_M={m:.6f}  z={z:.1f}")
