import argparse
import json
import os
import sys
from pathlib import Path

import torch

# allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.voc import VOC2007, object_categories  # noqa: E402


def compute_cooccurrence(data_root: str, phase: str = "trainval", quantile: float = 0.5, strong_quantile: float = 0.7):
    ds = VOC2007(data_root, phase=phase, transform=None)
    num_classes = len(object_categories)

    co_mat = torch.zeros(num_classes, num_classes)
    freq = torch.zeros(num_classes)

    for _, target in ds.images:  # target: torch tensor (values in {-1,0,1})
        pos = (target > 0).nonzero(as_tuple=False).view(-1)
        if pos.numel() == 0:
            continue
        freq[pos] += 1
        idx = pos.tolist()
        for i in idx:
            for j in idx:
                co_mat[i, j] += 1

    density = torch.zeros(num_classes)
    for i in range(num_classes):
        if freq[i] > 0:
            density[i] = (co_mat[i].sum() - co_mat[i, i]) / (freq[i] * (num_classes - 1) + 1e-6)

    # choose threshold via density quantile (e.g., median when quantile=0.5)
    quantile = min(max(quantile, 0.0), 1.0)
    threshold = density.quantile(quantile).item()
    related = [i for i, d in enumerate(density.tolist()) if d >= threshold]
    discriminative = [i for i in range(num_classes) if i not in related]

    # strong co-occurrence pairs by conditional probability P(j|i)
    cond = torch.zeros_like(co_mat)
    for i in range(num_classes):
        if freq[i] > 0:
            cond[i] = co_mat[i] / (freq[i] + 1e-6)
    cond.fill_diagonal_(0)
    strong_quantile = min(max(strong_quantile, 0.0), 1.0)
    valid = cond[cond > 0]
    strong_threshold = valid.quantile(strong_quantile).item() if valid.numel() > 0 else 0.0
    strong_pairs = []
    strong_per_class = {}
    for i in range(num_classes):
        partners = []
        for j in range(num_classes):
            if i == j:
                continue
            if cond[i, j] >= strong_threshold and cond[i, j] > 0:
                strong_pairs.append([i, j])
                partners.append(j)
        if partners:
            strong_per_class[i] = partners

    meta = {
        "classes": object_categories,
        "freq": freq.tolist(),
        "cooccurrence": co_mat.tolist(),
        "density": density.tolist(),
        "threshold": threshold,
        "quantile": quantile,
        "related": related,
        "discriminative": discriminative,
        "strong_quantile": strong_quantile,
        "strong_threshold": strong_threshold,
        "strong_pairs": strong_pairs,
        "strong_per_class": strong_per_class,
        "phase": phase,
    }
    return meta


def main():
    parser = argparse.ArgumentParser(description="Compute VOC co-occurrence and split labels into two groups")
    parser.add_argument("--data-root", required=True, help="Root folder containing VOCdevkit")
    parser.add_argument("--phase", default="trainval", choices=["trainval", "train", "val", "test"], help="Split to scan")
    parser.add_argument("--output", default=str(ROOT / "cooccur_voc2007.json"), help="Path to save JSON meta")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="Quantile of density used as threshold (0.5 = median)",
    )
    parser.add_argument(
        "--strong-quantile",
        type=float,
        default=0.7,
        help="Quantile on conditional co-occurrence P(j|i) to mark strong pairs (e.g., 0.7 = top 30%)",
    )
    args = parser.parse_args()

    meta = compute_cooccurrence(args.data_root, phase=args.phase, quantile=args.quantile, strong_quantile=args.strong_quantile)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2))

    print(f"[compute_cooccurrence] saved meta to {out_path}")
    print(
        f"related: {len(meta['related'])}, discriminative: {len(meta['discriminative'])}, "
        f"quantile: {meta['quantile']:.2f}, thr: {meta['threshold']:.4f}, "
        f"strong_pairs: {len(meta['strong_pairs'])}, strong_thr: {meta['strong_threshold']:.4f}"
    )


if __name__ == "__main__":
    main()
