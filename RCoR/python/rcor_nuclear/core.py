"""Python implementation of Julia RR_nuclear_optimized_final.jl.

The algorithm (not merely its mathematical objective) is preserved:
- Diagonalizations.eig: eigenvalues/eigenvectors in *descending* order.
- Column-wise exhaustive sign initialization; matrix 1 is the gauge.
- Exact order of integer sign masks; first maximizing mask wins ties.
- Greedy first-improvement flips, visiting i then j, and strict f_new > f + tol.
- Nuclear objective from eigvalsh(A @ A.T), clamping round-off negatives.
- Polar factor from the SVD of the signed sum.

This is a float64, real symmetric positive-definite implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RCoRDetails:
    """Optional intermediate quantities describing the RCoR computation."""

    M: np.ndarray
    Gamma: np.ndarray
    G: np.ndarray  # shape (n, d, d)
    eigenvalues: np.ndarray  # descending within each SPD
    eigenvectors: np.ndarray  # shape (n, d, d), eigenvectors in columns
    signs_initial: np.ndarray  # shape (n, d)
    signs_final: np.ndarray  # shape (n, d)
    A_initial: np.ndarray
    A_final: np.ndarray
    nuclear_initial: float
    nuclear_final: float
    passes: int
    converged: bool


def nuclearnorm(A: np.ndarray) -> float:
    """sum(sqrt(max(eigvals(Symmetric(A*A')), 0))) as in the Julia source."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    # Julia Symmetric(X) defaults to the upper triangle.
    eigvals = np.linalg.eigvalsh(A @ A.T, UPLO="U")
    return float(np.sqrt(np.maximum(eigvals, 0.0)).sum())


def _signed_sum(U: np.ndarray, signs: np.ndarray) -> np.ndarray:
    n, d, _ = U.shape
    A = np.zeros((d, d), dtype=np.float64)
    # Preserve Julia's loop nesting: i, j, then row update.
    for i in range(n):
        for j in range(d):
            A[:, j] += signs[i, j] * U[i, :, j]
    return A


def _columnsync_init(U: np.ndarray) -> np.ndarray:
    n, d, _ = U.shape
    if n > 63:
        raise ValueError("Exhaustive column synchronization requires impractical 2^(n-1) search")
    signs = np.ones((n, d), dtype=np.float64)
    if n == 1:
        return signs

    npatterns = 1 << (n - 1)
    v = np.zeros(d, dtype=np.float64)
    for j in range(d):
        bestval = -np.inf
        bestmask = 0
        for mask in range(npatterns):
            v.fill(0.0)
            for i in range(n):
                si = 1.0 if i == 0 else (-1.0 if ((mask >> (i - 1)) & 1) == 1 else 1.0)
                v += si * U[i, :, j]
            # Julia PosDefManifold.sumOfSqr(v): sum of squared elements.
            value = float(np.dot(v, v))
            # Strict > : the first maximum encountered wins.
            if value > bestval:
                bestval = value
                bestmask = mask
        for i in range(1, n):
            signs[i, j] = -1.0 if ((bestmask >> (i - 1)) & 1) == 1 else 1.0
    return signs


def _nuclear_greedy(
    U: np.ndarray,
    signs: np.ndarray,
    A: np.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[float, int, bool]:
    n, d, _ = U.shape
    f = nuclearnorm(A)
    converged = maxiter == 0 and n == 1
    passes = 0
    for _ in range(maxiter):
        passes += 1
        improved = False
        # The order of these loops is essential; do not vectorize the flips.
        for i in range(1, n):  # Julia i = 2:n, first matrix fixed (gauge)
            for j in range(d):
                sij = signs[i, j]
                A[:, j] += (-2.0 * sij) * U[i, :, j]
                candidate = nuclearnorm(A)
                if candidate > f + tol:
                    signs[i, j] = -sij
                    f = candidate
                    improved = True
                else:
                    # Reverse the very same update as in Julia. No full recomputation.
                    A[:, j] += (2.0 * sij) * U[i, :, j]
        if not improved:
            converged = True
            break
    return f, passes, converged


def rr_from_evd(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    tol: float = 1e-12,
    maxiter: int = 100,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | RCoRDetails:
    """RCoR from prescribed eigendecompositions.

    eigenvalues[i, j] pairs with eigenvectors[i, :, j], and must be ordered
    identically to Julia Diagonalizations.eig (descending eigenvalues).
    """
    lambdas = np.asarray(eigenvalues, dtype=np.float64)
    U = np.asarray(eigenvectors, dtype=np.float64)
    if lambdas.ndim != 2:
        raise ValueError("eigenvalues must have shape (n, d)")
    n, d = lambdas.shape
    if n == 0 or d == 0 or U.shape != (n, d, d):
        raise ValueError("eigenvectors must have shape (n, d, d), n and d positive")
    if not np.isfinite(lambdas).all() or not np.isfinite(U).all():
        raise ValueError("EVD inputs must be finite")
    if np.any(lambdas <= 0):
        raise ValueError("eigenvalues must be strictly positive")
    if maxiter < 0 or tol < 0:
        raise ValueError("maxiter and tol must be nonnegative")

    signs = _columnsync_init(U)
    signs_initial = signs.copy()
    A = _signed_sum(U, signs)
    A_initial = A.copy()
    f_initial = nuclearnorm(A)
    f_final, passes, converged = _nuclear_greedy(U, signs, A, tol, maxiter)

    # Julia sv = svd(A); Gamma = sv.U * sv.Vt
    left, _, right_t = np.linalg.svd(A, full_matrices=True)
    Gamma = left @ right_t

    # Same spectral construction as Julia, with eigenvalues in descending order.
    logmean = np.log(lambdas).sum(axis=0) / n
    lambda_geo = np.exp(logmean)
    M = (Gamma * lambda_geo[np.newaxis, :]) @ Gamma.T
    G = np.empty((n, d, d), dtype=np.float64)
    for i in range(n):
        G[i] = (Gamma * lambdas[i][np.newaxis, :]) @ Gamma.T

    if return_details:
        return RCoRDetails(
            M=M, Gamma=Gamma, G=G,
            eigenvalues=lambdas.copy(), eigenvectors=U.copy(),
            signs_initial=signs_initial, signs_final=signs.copy(),
            A_initial=A_initial, A_final=A.copy(),
            nuclear_initial=f_initial, nuclear_final=f_final,
            passes=passes, converged=converged,
        )
    return M, Gamma, G


def rr(
    matrices: Sequence[np.ndarray] | np.ndarray,
    *,
    tol: float = 1e-12,
    maxiter: int = 100,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | RCoRDetails:
    """RCoR for real SPD matrices; corresponds to Julia rr(𝐂::ℍVector; tol, maxiter).

    Parameters
    ----------
    matrices : sequence or ndarray of shape (n, d, d)
        Real SPD matrices. Each row/column ordering is preserved.
    tol, maxiter : as in Julia
    return_details : bool
        Optional diagnostics without changing the default (M, Gamma, G) API.
    """
    C = np.asarray(matrices, dtype=np.float64)
    if C.ndim != 3 or C.shape[0] < 1 or C.shape[1] < 1 or C.shape[1] != C.shape[2]:
        raise ValueError("matrices must have shape (n, d, d) with n, d >= 1")
    if not np.all(np.isfinite(C)):
        raise ValueError("matrices must be finite")
    if not np.allclose(C, C.transpose(0, 2, 1), rtol=1e-12, atol=1e-12):
        raise ValueError("all matrices must be symmetric")
    n, d, _ = C.shape
    lambdas = np.empty((n, d), dtype=np.float64)
    U = np.empty((n, d, d), dtype=np.float64)
    for i in range(n):
        # eigh is ascending; Julia Diagonalizations.eig uses descending order.
        vals, vectors = np.linalg.eigh(C[i], UPLO="U")
        lambdas[i] = vals[::-1]
        U[i] = vectors[:, ::-1]

    return rr_from_evd(
        lambdas, U, tol=tol, maxiter=maxiter, return_details=return_details
    )
