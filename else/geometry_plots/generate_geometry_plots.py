#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    import torch
except ImportError:  # pragma: no cover - runtime environment dependent
    torch = None


ROOT = Path(__file__).resolve().parent

DATASET_SPECS = [
    {
        "slug": "bloodmnist_baseline",
        "label": "BloodMNIST baseline",
        "kind": "npz",
        "path": Path(
            "/Users/goodday/Downloads/pap/out/artifacts/bloodmnist/"
            "resnet18_NirvanaOpenset_38.0_False_0.0/split_0/"
            "resume_eval_only_arrays.npz"
        ),
        "note": "Saved split-0 artifact export from the frozen baseline run.",
    },
    {
        "slug": "dermamnist_baseline",
        "label": "DermaMNIST baseline",
        "kind": "npz",
        "path": Path(
            "/Users/goodday/Downloads/pap/out/artifacts/dermamnist/"
            "resnet18_NirvanaOpenset_38.0_False_0.0/split_0/"
            "resume_eval_only_arrays.npz"
        ),
        "note": "Saved split-0 artifact export from the frozen baseline run.",
    },
    {
        "slug": "bloodmnist_fd32",
        "label": "BloodMNIST feat_dim=32",
        "kind": "npz",
        "path": Path(
            "/Users/goodday/Downloads/pap/out/artifacts/bloodmnist/"
            "resnet18_NirvanaOpenset_38.0_False_0.0_fd32/split_0/"
            "resume_eval_only_arrays.npz"
        ),
        "note": "Saved split-0 artifact export from the low-dimension run.",
    },
    {
        "slug": "pathmnist_d4",
        "label": "PathMNIST d=4",
        "kind": "pth",
        "path": Path(
            "/Users/goodday/Downloads/pap/build/mac_ucdsc/out_clean/models/pathmnist/"
            "classifier32_ed4_NirvanaOpenset_48.0_False/checkpoints/"
            "classifier32_ed4_NirvanaOpenset_0_48.0_0.0_best_criterion.pth"
        ),
        "note": (
            "Low-dimensional fallback run. The saved code skips simplex initialization when "
            "feat_dim < num_classes - 1, so non-symmetry here reflects the actual saved "
            "implementation choice."
        ),
    },
    {
        "slug": "cifar100_extension",
        "label": "CIFAR-100 extension",
        "kind": "pth",
        "path": Path(
            "/Users/goodday/Downloads/pap/tmp_analysis/windows_ucdsc_2/windows_ucdsc/"
            "windows_ucdsc/out/models/cifar100/"
            "classifier32_NirvanaOpenset_48.0_False_50/checkpoints/"
            "classifier32_NirvanaOpenset_50_0_48.0_0.0_best_criterion.pth"
        ),
        "note": "Saved split-0 criterion checkpoint from the CIFAR-100 extension run.",
    },
]


def _load_pth_centers(path: Path) -> np.ndarray:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to load .pth checkpoints. Run this script with the local venv "
            "that has torch installed."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, OrderedDict):
        if "centers" not in payload:
            raise KeyError(f"'centers' not found in OrderedDict checkpoint: {path}")
        centers = payload["centers"]
    elif isinstance(payload, dict):
        if "centers" in payload:
            centers = payload["centers"]
        elif "criterion" in payload and hasattr(payload["criterion"], "centers"):
            centers = payload["criterion"].centers
        else:
            raise KeyError(f"Could not find centers in checkpoint: {path}")
    else:
        raise TypeError(f"Unsupported checkpoint payload type for {path}: {type(payload)!r}")
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()
    return np.asarray(centers, dtype=np.float64)


def load_spec(spec: dict[str, Any]) -> dict[str, Any]:
    path = spec["path"]
    if spec["kind"] == "npz":
        data = np.load(path)
        return {
            "centers": np.asarray(data["centers"], dtype=np.float64),
            "known_features": np.asarray(data["known_features"], dtype=np.float64),
            "unknown_features": np.asarray(data["unknown_features"], dtype=np.float64),
        }
    if spec["kind"] == "pth":
        return {"centers": _load_pth_centers(path)}
    raise ValueError(f"Unsupported source kind: {spec['kind']}")


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    diffs = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def normalized_cosines(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.where(norms > 0, x / norms, x)
    return safe @ safe.T, norms.squeeze(1)


def simplex_defect(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    defects = np.zeros(n, dtype=np.float64)
    for j in range(n):
        rivals = np.delete(dist[j], j)
        mean_rival = np.mean(rivals)
        mean_sq_rival = np.mean(rivals**2)
        defects[j] = math.sqrt(mean_sq_rival) / mean_rival if mean_rival > 0 else float("nan")
    return defects


def heatmap(ax: plt.Axes, mat: np.ndarray, title: str, cmap: str, value_fmt: str) -> None:
    im = ax.imshow(mat, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xlabel("Prototype index")
    ax.set_ylabel("Prototype index")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if abs(mat[i, j]) > 0.6 * np.nanmax(np.abs(mat)) else "black"
            ax.text(j, i, value_fmt.format(mat[i, j]), ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_pairwise_heatmap(out_path: Path, mat: np.ndarray, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    heatmap(ax, mat, f"{label}: pairwise distance", cmap="viridis", value_fmt="{:.2f}")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_cosine_heatmap(out_path: Path, mat: np.ndarray, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    heatmap(ax, mat, f"{label}: normalized prototype cosine", cmap="coolwarm", value_fmt="{:.2f}")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_lambda_bars(out_path: Path, lambdas: np.ndarray, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    idx = np.arange(len(lambdas))
    ax.bar(idx, lambdas, color="#2F6B9A")
    ax.axhline(1.0, color="#B22222", linestyle="--", linewidth=1.5, label="ideal simplex")
    ax.set_title(f"{label}: simplex-defect $\\lambda_j$")
    ax.set_xlabel("Prototype index")
    ax.set_ylabel("$\\lambda_j$")
    ax.set_xticks(idx)
    ax.legend(frameon=False)
    for i, value in enumerate(lambdas):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_projection(out_path: Path, centers: np.ndarray, label: str, known: np.ndarray | None, unknown: np.ndarray | None) -> None:
    rng = np.random.default_rng(0)
    blocks = [centers]
    labels: list[tuple[str, int]] = [("center", centers.shape[0])]
    if known is not None and len(known):
        take = min(len(known), 1200)
        known = known[rng.choice(len(known), size=take, replace=False)]
        blocks.insert(0, known)
        labels.insert(0, ("known", take))
    if unknown is not None and len(unknown):
        take = min(len(unknown), 1200)
        unknown = unknown[rng.choice(len(unknown), size=take, replace=False)]
        blocks.insert(1 if known is not None and len(known) else 0, unknown)
        labels.insert(1 if known is not None and len(known) else 0, ("unknown", take))
    stacked = np.concatenate(blocks, axis=0)
    if stacked.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=0).fit_transform(stacked)
    else:
        zeros = np.zeros((stacked.shape[0], 1), dtype=np.float64)
        coords = np.hstack([stacked, zeros])[:, :2]

    slices: dict[str, np.ndarray] = {}
    start = 0
    for name, count in labels:
        slices[name] = coords[start : start + count]
        start += count

    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    if "known" in slices:
        pts = slices["known"]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.20, color="#4C78A8", label="known features")
    if "unknown" in slices:
        pts = slices["unknown"]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.18, color="#F58518", label="unknown features")
    pts = slices["center"]
    ax.scatter(pts[:, 0], pts[:, 1], s=140, marker="X", color="#111111", edgecolors="white", linewidths=0.8, label="centers")
    for i, (x, y) in enumerate(pts):
        ax.text(x, y, str(i), fontsize=9, ha="center", va="center", color="white")
    ax.set_title(f"{label}: PCA projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="best")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summary_metrics(label: str, centers: np.ndarray, dist: np.ndarray, cos: np.ndarray, lambdas: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(centers, axis=1)
    triu = np.triu_indices(dist.shape[0], 1)
    off_dist = dist[triu]
    off_cos = cos[triu]
    return {
        "label": label,
        "num_prototypes": int(centers.shape[0]),
        "feat_dim": int(centers.shape[1]),
        "center_norm_mean": float(np.mean(norms)),
        "center_norm_std": float(np.std(norms)),
        "center_norm_cv": float(np.std(norms) / np.mean(norms)) if np.mean(norms) else float("nan"),
        "pairwise_distance_mean": float(np.mean(off_dist)),
        "pairwise_distance_std": float(np.std(off_dist)),
        "pairwise_distance_cv": float(np.std(off_dist) / np.mean(off_dist)) if np.mean(off_dist) else float("nan"),
        "offdiag_cosine_mean": float(np.mean(off_cos)),
        "offdiag_cosine_std": float(np.std(off_cos)),
        "lambda_mean": float(np.mean(lambdas)),
        "lambda_max": float(np.max(lambdas)),
        "lambda_min": float(np.min(lambdas)),
    }


def save_cross_dataset_summary(out_path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.bar(x - width, [row["pairwise_distance_cv"] for row in rows], width=width, label="distance CV", color="#4C78A8")
    ax.bar(x, [row["center_norm_cv"] for row in rows], width=width, label="norm CV", color="#F58518")
    ax.bar(x + width, [row["lambda_mean"] - 1.0 for row in rows], width=width, label=r"$\lambda_{\mathrm{mean}}-1$", color="#54A24B")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Cross-dataset geometry deviation summary")
    ax.set_ylabel("Deviation from ideal symmetry")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary_files(rows: list[dict[str, Any]], notes: dict[str, str]) -> None:
    csv_path = ROOT / "geometry_summary.csv"
    json_path = ROOT / "geometry_summary.json"
    md_path = ROOT / "README.md"

    fieldnames = [
        "label",
        "num_prototypes",
        "feat_dim",
        "center_norm_mean",
        "center_norm_std",
        "center_norm_cv",
        "pairwise_distance_mean",
        "pairwise_distance_std",
        "pairwise_distance_cv",
        "offdiag_cosine_mean",
        "offdiag_cosine_std",
        "lambda_mean",
        "lambda_max",
        "lambda_min",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "notes": notes}, fh, indent=2)

    lines = [
        "# Geometry Plots",
        "",
        "This folder contains prototype-geometry evidence derived from the saved artifacts/checkpoints for each run.",
        "",
        "## Summary",
        "",
        "| Dataset | Centers | feat_dim | dist CV | norm CV | lambda_mean | lambda_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['num_prototypes']} | {row['feat_dim']} | "
            f"{row['pairwise_distance_cv']:.4f} | {row['center_norm_cv']:.4f} | "
            f"{row['lambda_mean']:.4f} | {row['lambda_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for label, note in notes.items():
        lines.append(f"- `{label}`: {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    notes: dict[str, str] = {}
    for spec in DATASET_SPECS:
        payload = load_spec(spec)
        centers = payload["centers"]
        known = payload.get("known_features")
        unknown = payload.get("unknown_features")

        dist = pairwise_distances(centers)
        cos, _ = normalized_cosines(centers)
        lambdas = simplex_defect(dist)
        rows.append(summary_metrics(spec["label"], centers, dist, cos, lambdas))
        notes[spec["label"]] = spec["note"]

        save_pairwise_heatmap(ROOT / f"{spec['slug']}_distance_heatmap.png", dist, spec["label"])
        save_cosine_heatmap(ROOT / f"{spec['slug']}_cosine_heatmap.png", cos, spec["label"])
        save_lambda_bars(ROOT / f"{spec['slug']}_lambda_bar.png", lambdas, spec["label"])
        save_projection(ROOT / f"{spec['slug']}_projection.png", centers, spec["label"], known, unknown)

    save_cross_dataset_summary(ROOT / "all_datasets_geometry_summary.png", rows)
    write_summary_files(rows, notes)


if __name__ == "__main__":
    main()
