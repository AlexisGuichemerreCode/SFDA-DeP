import os
import json
import numpy as np
from scipy.stats import spearmanr

BASE_DIR = "/export/livia/home/vision/Aguichemerre/metrics_cvpr_da/train"
OUT_PATH = "/export/livia/home/vision/Aguichemerre/spearman_by_source_model_train.txt"

SOURCES = {"GLAS", "CAMELYON512", "CAMELYON17_512"}
WSOL_METHODS = {"PixelCAM", "LayerCAM", "GradCAMpp", "DEEPMIL", "TSCAM"}


def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def find_all_metrics_files(base_dir, debug_limit=20):
    """
    Parcourt récursivement base_dir et trouve tous les metrics.txt.
    Retourne (source, method, fold, metrics_path).
    """
    results = []
    dbg_count = 0
    for root, dirs, files in os.walk(base_dir):
        if "metrics.txt" not in files:
            continue

        rel = os.path.relpath(root, base_dir)
        parts = rel.split(os.sep)

        source, method, fold = None, None, None
        for i, p in enumerate(parts):
            if source is None and p in SOURCES:
                source = p
                # vérifier si juste après il y a un fold (nombre entier)
                if source == "CAMELYON17_512" and i + 1 < len(parts) and parts[i+1].isdigit():
                    fold = parts[i+1]
            if p in WSOL_METHODS:
                method = p

        if source is None or method is None:
            continue

        mpath = os.path.join(root, "metrics.txt")
        results.append((source, method, fold, mpath))

        if dbg_count < debug_limit:
            print(f"[DEBUG] trouvé: source={source}, fold={fold}, method={method}, path={mpath}")
            dbg_count += 1

    print(f"✅ Total metrics.txt trouvés: {len(results)}")
    return results



def collect_grouped_data(base_dir):
    grouped = {}
    items = find_all_metrics_files(base_dir)

    for source, method, fold, mpath in items:
        data = read_json(mpath)
        if not data or "shifts" not in data or "results" not in data:
            continue

        acc = data["results"].get("Default_0.5", {}).get("Accuracy", None)
        cam_perf = data.get("cam_performance", None)

        entry = {}
        entry.update(data["shifts"])
        entry["Accuracy"] = acc
        entry["cam_performance"] = cam_perf
        entry["_meta"] = {"source": source, "method": method, "fold": fold, "path": mpath}

        # inclure le fold dans la clé si présent
        if fold is not None:
            group_key = f"{source}_fold{fold}__{method}"
        else:
            group_key = f"{source}__{method}"

        grouped.setdefault(group_key, []).append(entry)

    return grouped



def compute_and_save_spearman(grouped, out_path):
    with open(out_path, "w") as f:
        for group_key, entries in sorted(grouped.items()):
            f.write(f"=== {group_key} ===\n")
            f.write(f"(N points={len(entries)})\n")

            if len(entries) < 2:
                f.write("⚠️ Pas assez de points pour Spearman\n\n")
                continue

            example = entries[0]
            shift_keys = [k for k in example.keys()
                          if k not in ("Accuracy", "cam_performance", "_meta")]

            for sk in shift_keys:
                # Corrélation vs Accuracy
                xs_acc, ys_acc = [], []
                for e in entries:
                    xv, av = e.get(sk, None), e.get("Accuracy", None)
                    if isinstance(xv, (int, float)) and isinstance(av, (int, float)):
                        if np.isfinite(xv) and np.isfinite(av):
                            xs_acc.append(xv)
                            ys_acc.append(av)
                if len(xs_acc) >= 2:
                    rho, pval = spearmanr(xs_acc, ys_acc)
                    f.write(f"{sk:>15} vs {'Accuracy':<16}: rho={rho:.3f}, p={pval:.3e}, N={len(xs_acc)}\n")
                else:
                    f.write(f"{sk:>15} vs {'Accuracy':<16}: N<2\n")

                # Corrélation vs cam_performance
                xs_cam, ys_cam = [], []
                for e in entries:
                    xv, cv = e.get(sk, None), e.get("cam_performance", None)
                    if isinstance(xv, (int, float)) and isinstance(cv, (int, float)):
                        if np.isfinite(xv) and np.isfinite(cv):
                            xs_cam.append(xv)
                            ys_cam.append(cv)
                if len(xs_cam) >= 2:
                    rho, pval = spearmanr(xs_cam, ys_cam)
                    f.write(f"{sk:>15} vs {'cam_performance':<16}: rho={rho:.3f}, p={pval:.3e}, N={len(xs_cam)}\n")
                else:
                    f.write(f"{sk:>15} vs {'cam_performance':<16}: N<2\n")

            f.write("\n")
    print(f"✅ Résultats Spearman sauvegardés dans {out_path}")


if __name__ == "__main__":
    grouped = collect_grouped_data(BASE_DIR)
    print(f"📊 Groupes trouvés: {len(grouped)}")
    for g, vals in grouped.items():
        print(f"  {g}: {len(vals)} points")
    compute_and_save_spearman(grouped, OUT_PATH)
