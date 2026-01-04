import pickle
import re
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
ROOT = Path("/export/livia/home/vision/Aguichemerre/SAT_3")
SPLIT = "train"

METRICS = [
    "Classification_results",
    "KL_Uniform_results",
    "DBI_results",
    "ECE_results",
]

OUT_ROOT = Path("figures_by_lr_sat_3")
OUT_ROOT.mkdir(exist_ok=True)


# =========================================================
# Utils
# =========================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_curve(obj):
    """
    Robust extraction of a 1D curve from different pickle formats.
    """
    if isinstance(obj, (list, np.ndarray)):
        return np.asarray(obj)

    if isinstance(obj, dict):
        for k in ["values", "curve", "data"]:
            if k in obj:
                return np.asarray(obj[k])

    raise ValueError("Unknown pickle format")


def parse_lr(filename: str):
    """
    Parse learning rate from filename.
    Supported formats:
      - LR1em05  -> 1e-5
      - LR0p01   -> 0.01
    """
    m = re.search(r"LR([0-9]+)em([0-9]+)", filename)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * 10 ** (-exp)

    m = re.search(r"LR([0-9]+p[0-9]+)", filename)
    if m:
        return float(m.group(1).replace("p", "."))

    raise ValueError(f"LR not found in {filename}")

def parse_P(filename: str):
    """
    Parse P from filename.
    Example: _P0p1_ -> 0.1
    """
    m = re.search(r"_P([0-9]+p[0-9]+)", filename)
    if not m:
        raise ValueError(f"P not found in {filename}")
    return float(m.group(1).replace("p", "."))



def parse_K(filename: str):
    """
    Parse K from filename.
    Example: _K10p0_ -> 10.0
    """
    m = re.search(r"_K([0-9]+p[0-9]+)", filename)
    if not m:
        raise ValueError(f"K not found in {filename}")
    return float(m.group(1).replace("p", "."))


def reduce_curve(curve, mode="last"):
    """
    Reduce a curve to a scalar value.
    """
    if mode == "last":
        return curve[-1]
    if mode == "mean_last":
        return np.mean(curve[-5:])
    if mode == "min":
        return np.min(curve)
    if mode == "max":
        return np.max(curve)

    raise ValueError(f"Unknown reduction mode: {mode}")




# =========================================================
# Main loop over metrics
# =========================================================
for metric in METRICS:
    print(f"\nProcessing metric: {metric}")

    pickle_files = list(
        ROOT.rglob(f"{metric}_{SPLIT}_*.pickle")
    )

    if len(pickle_files) == 0:
        print(f"  ⚠️ No files found for {metric}")
        continue

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------
    curves_by_lr = defaultdict(list)
    curves_by_lr_k = defaultdict(lambda: defaultdict(list))

    for pkl in pickle_files:
        lr = parse_lr(pkl.name)
        K = parse_K(pkl.name)

        curve = extract_curve(load_pickle(pkl))

        curves_by_lr[lr].append(curve)
        curves_by_lr_k[lr][K].append(curve)

    metric_out = OUT_ROOT / metric
    metric_out.mkdir(exist_ok=True)

    # =====================================================
    # 1) Plot: metric vs epoch (mean ± std), one fig per LR
    # =====================================================
    for lr, curves in sorted(curves_by_lr.items()):
        curves = np.array(curves)

        min_len = min(c.shape[0] for c in curves)
        curves = curves[:, :min_len]

        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        epochs = np.arange(min_len)

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, mean, linewidth=2)
        plt.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.25
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"{metric} ({SPLIT}) — LR={lr:.1e}")
        plt.grid(True)

        out_path = metric_out / f"{SPLIT}_LR{lr:.1e}_vs_epoch.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  Saved {out_path}")

    # =====================================================
    # 2) Plot: mean curve ± std per K (one fig per LR)
    # =====================================================
    for lr, curves_by_k in sorted(curves_by_lr_k.items()):

        plt.figure(figsize=(7, 5))

        for K, curves in sorted(curves_by_k.items()):
            curves = np.array(curves)

            # aligner les longueurs (sécurité)
            min_len = min(c.shape[0] for c in curves)
            curves = curves[:, :min_len]
            epochs = np.arange(min_len)

            mean = curves.mean(axis=0)
            std = curves.std(axis=0)

            plt.plot(
                epochs,
                mean,
                linewidth=2,
                label=f"K={K}"
            )
            plt.fill_between(
                epochs,
                mean - std,
                mean + std,
                alpha=0.25
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"{metric} ({SPLIT}) — LR={lr:.1e}")
        plt.grid(True)
        plt.legend(title="K")

        out_path = metric_out / f"{SPLIT}_LR{lr:.1e}_mean_curves_by_K.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"  Saved {out_path}")

# =====================================================
# Plot: metric vs epoch (mean ± std) for each P
#        (fixed LR and fixed K)
# =====================================================
FIXED_LR = 1e-4   # 0.01
FIXED_K  = 10.0

for metric in METRICS:
    print(f"\nProcessing metric (P ablation): {metric}")

    pickle_files = list(
        ROOT.rglob(f"{metric}_{SPLIT}_*.pickle")
    )

    curves_by_P = defaultdict(list)

    for pkl in pickle_files:
        lr = parse_lr(pkl.name)
        K  = parse_K(pkl.name)
        P  = parse_P(pkl.name)

        # --- filter on fixed LR and K
        if abs(lr - FIXED_LR) > 1e-12:
            continue
        if abs(K - FIXED_K) > 1e-12:
            continue

        curve = extract_curve(load_pickle(pkl))
        curves_by_P[P].append(curve)

    if len(curves_by_P) == 0:
        print("  ⚠️ No matching files found")
        continue

    metric_out = OUT_ROOT / metric
    metric_out.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))

    for P, curves in sorted(curves_by_P.items()):
        curves = np.array(curves)

        # align curves
        min_len = min(c.shape[0] for c in curves)
        curves = curves[:, :min_len]
        epochs = np.arange(min_len)

        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)

        plt.plot(
            epochs,
            mean,
            linewidth=2,
            label=f"P={P}"
        )
        plt.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.25
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " "))
    plt.title(
        f"{metric} ({SPLIT}) — LR={FIXED_LR:.1e}, K={FIXED_K}"
    )
    plt.grid(True)
    plt.legend(title="P")

    out_path = metric_out / (
        f"{SPLIT}_LR{FIXED_LR:.1e}_K{FIXED_K}_vs_P.png"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"  Saved {out_path}")

