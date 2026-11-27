"""RCoR scatterplot on synthetic SPD matrices.

This example displays four matrix groups:
raw matrices, registered matrices, raw mean and reference matrix.
It uses classical MDS
on pairwise affine-invariant (Fisher) SPD distances. The raw mean is arithmetic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from rcor_nuclear import rr

ROOT = Path(__file__).resolve().parents[1]


def fisher_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Affine-invariant SPD geodesic distance: ||log eigenvalues(B,A)||_2."""
    values = eigh(b, a, eigvals_only=True, check_finite=True)
    if np.any(values <= 0):
        raise ValueError("The example requires strictly SPD matrices")
    return float(np.linalg.norm(np.log(values)))


def embed_fisher_mds(matrices: np.ndarray) -> np.ndarray:
    """Two-dimensional classical MDS of affine-invariant SPD distances."""
    count = len(matrices)
    d2 = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            d2[i, j] = d2[j, i] = fisher_distance(matrices[i], matrices[j]) ** 2
    center = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * center @ d2 @ center
    gram = (gram + gram.T) / 2
    values, vectors = np.linalg.eigh(gram)
    top = np.argsort(values)[::-1][:2]
    return vectors[:, top] * np.sqrt(np.maximum(values[top], 0))


def run_demo(case: str = "demo_variable_spectra_n12_d8", output: str | Path | None = None) -> Path:
    folder = ROOT / "examples" / "data" / case
    if not folder.is_dir():
        folder = ROOT / "fixtures" / case
    files = sorted(folder.glob("C_*.csv"))
    if not files:
        raise FileNotFoundError(f"No public synthetic matrices in {folder}")
    original = np.stack([np.loadtxt(path, delimiter=",") for path in files])
    reference, gamma, registered = rr(original)
    raw_mean = np.mean(original, axis=0)  # arithmetic; not Fisher/Karcher mean
    points = np.concatenate([original, registered, raw_mean[None], reference[None]])
    coords = embed_fisher_mds(points)
    n = len(original)

    fig, ax = plt.subplots(figsize=(9.1, 6.0), constrained_layout=True)
    # Pairwise movement is a visual guide in the 2D embedding, not a geodesic.
    for i in range(n):
        ax.plot([coords[i, 0], coords[n + i, 0]],
                [coords[i, 1], coords[n + i, 1]],
                color="#9cabb8", lw=0.8, alpha=0.42, zorder=1)
    ax.scatter(coords[:n, 0], coords[:n, 1], s=62, color="#3468a5",
               label="Raw SPD matrices", zorder=3, alpha=0.85)
    ax.scatter(coords[n:2*n, 0], coords[n:2*n, 1], s=67, color="#2f9a75",
               marker="s", label="Registered SPD matrices", zorder=4, alpha=0.88)
    ax.scatter(coords[-2, 0], coords[-2, 1], s=190, color="#d57c2c",
               marker="D", edgecolor="white", linewidth=1.0,
               label="Raw arithmetic mean", zorder=6)
    ax.scatter(coords[-1, 0], coords[-1, 1], s=260, color="#e6bf35",
               marker="*", edgecolor="#695526", linewidth=0.7,
               label="RCoR reference M", zorder=7)
    ax.set_title("Riemannian Co-Registration | synthetic SPD example", loc="left", pad=14)
    ax.set_xlabel("Fisher-distance MDS component 1")
    ax.set_ylabel("Fisher-distance MDS component 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    if output is None:
        output = ROOT / "examples" / "output" / "rcor_scatter_python.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output}")
    print(f"Input: {len(original)} synthetic {original.shape[-1]}x{original.shape[-1]} SPD matrices")
    print(f"Gamma orthogonality error: {np.linalg.norm(gamma.T @ gamma - np.eye(gamma.shape[0])):.3e}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="demo_variable_spectra_n12_d8", help="Name of a synthetic fixture directory")
    parser.add_argument("--output", type=Path, help="Optional path for exported PNG")
    args = parser.parse_args()
    run_demo(args.case, args.output)
