#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TARGETS = {
    "ACC": 97.69,
    "AUROC": 89.93,
    "OSCR": 88.84,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate BloodMNIST K=5 open-set metrics and compare to paper targets."
    )
    parser.add_argument("--csv", required=True, help="Path to bloodmnist.csv from NirvanaOSR output")
    parser.add_argument(
        "--out-report",
        default="/Users/goodday/Downloads/pap/out/phase1_bloodmnist_report.txt",
        help="Output text report path",
    )
    parser.add_argument(
        "--out-per-split",
        default="/Users/goodday/Downloads/pap/out/phase1_bloodmnist_per_split.csv",
        help="Output per-split flattened CSV path",
    )
    parser.add_argument("--tol-acc", type=float, default=1.0, help="Tolerance for ACC")
    parser.add_argument("--tol-auroc", type=float, default=2.0, help="Tolerance for AUROC")
    parser.add_argument("--tol-oscr", type=float, default=2.0, help="Tolerance for OSCR")
    return parser.parse_args()


def load_split_table(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path, index_col=0)
    split_df = df_raw.transpose()
    split_df.index.name = "split"
    split_df = split_df.reset_index()
    split_df["split"] = split_df["split"].astype(str)
    required = ["ACC", "AUROC", "OSCR"]
    missing = [c for c in required if c not in split_df.columns]
    if missing:
        raise ValueError(f"Missing required metric columns in {csv_path}: {missing}")
    split_df = split_df[["split", "ACC", "AUROC", "OSCR"]].copy()
    for metric in ["ACC", "AUROC", "OSCR"]:
        split_df[metric] = pd.to_numeric(split_df[metric], errors="coerce")
    if split_df[["ACC", "AUROC", "OSCR"]].isna().any().any():
        raise ValueError(f"Found non-numeric values in metric columns in {csv_path}")
    return split_df


def metric_decision(name: str, mean_val: float, tol_map: dict) -> tuple[str, float, float]:
    target = DEFAULT_TARGETS[name]
    tol = tol_map[name]
    delta = mean_val - target
    status = "PASS" if abs(delta) <= tol else "FAIL"
    return status, target, delta


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_report = Path(args.out_report)
    out_per_split = Path(args.out_per_split)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_per_split.parent.mkdir(parents=True, exist_ok=True)

    split_df = load_split_table(csv_path)
    split_df.to_csv(out_per_split, index=False)

    summary = split_df[["ACC", "AUROC", "OSCR"]].agg(["mean", "std"]).transpose()

    tol_map = {
        "ACC": args.tol_acc,
        "AUROC": args.tol_auroc,
        "OSCR": args.tol_oscr,
    }

    lines = []
    lines.append("Phase 1 Reproduction Report: UCDSC on BloodMNIST (K=5)")
    lines.append(f"Source CSV: {csv_path}")
    lines.append(f"Per-split CSV: {out_per_split}")
    lines.append("")
    lines.append("Per-split metrics")
    lines.append(split_df.to_string(index=False))
    lines.append("")
    lines.append("Aggregate (mean ± std)")

    overall_pass = True
    for metric in ["ACC", "AUROC", "OSCR"]:
        m = float(summary.loc[metric, "mean"])
        s = float(summary.loc[metric, "std"])
        status, target, delta = metric_decision(metric, m, tol_map)
        overall_pass = overall_pass and (status == "PASS")
        lines.append(
            f"- {metric}: {m:.4f} ± {s:.4f} | target={target:.2f} | "
            f"delta={delta:+.4f} | tol=±{tol_map[metric]:.2f} | {status}"
        )

    lines.append("")
    lines.append(f"Overall verdict: {'PASS' if overall_pass else 'FAIL'}")

    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
