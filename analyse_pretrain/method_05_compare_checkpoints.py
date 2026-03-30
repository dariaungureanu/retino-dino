import argparse
import glob
import json
import os
from typing import Dict, Any, List

import pandas as pd


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_metrics(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "checkpoint": obj.get("checkpoint", ""),
        "arch": obj.get("arch", ""),
    }

    if "knn" in obj and "linear_probe" in obj:
        out.update({
            "type": "frozen_eval",
            "knn_acc": obj["knn"].get("acc"),
            "knn_bal_acc": obj["knn"].get("bal_acc"),
            "knn_macro_f1": obj["knn"].get("macro_f1"),
            "lp_acc": obj["linear_probe"].get("acc"),
            "lp_bal_acc": obj["linear_probe"].get("bal_acc"),
            "lp_macro_f1": obj["linear_probe"].get("macro_f1"),
        })
    elif "recall_at_k" in obj:
        out.update({
            "type": "retrieval",
            "split": obj.get("split"),
            "top_k": obj.get("top_k"),
            "recall_at_k": obj.get("recall_at_k"),
            "n_queries": obj.get("n_queries"),
        })
    elif "records" in obj and "num_images" in obj:
        out.update({
            "type": "visual",
            "split": obj.get("split"),
            "num_images": obj.get("num_images"),
        })
    else:
        out["type"] = "unknown"

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Example: analyse_pretrain/results/*.json")
    ap.add_argument("--out_csv", default="analyse_pretrain/checkpoint_comparison.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No json files matched: {args.glob}")

    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            obj = read_json(f)
            r = pick_metrics(obj)
            r["source_file"] = f
            rows.append(r)
        except Exception as e:
            rows.append({"type": "error", "source_file": f, "error": str(e)})

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(df.to_string(index=False))
    print(f"[INFO] Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
