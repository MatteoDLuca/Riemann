"""
Python-only implementation of the improved Riemannian Potato Field (iRPF)
translated from the uploaded Julia/RAR code:
- improvedRPF.jl
- PotatoFields.jl
- Kneedle.jl
- Pcombination.jl

Main conventions
----------------
X is always a continuous EEG matrix with shape:
    (n_samples, n_channels)

Epoch output and decisions are indexed 0-based in Python.

The implementation includes:
- FRMS/GFRMS automatic outlier rejection
- multi-potato field
- Riemannian / Euclidean / diagonal-Euclidean / phop distance features
- adaptive robust barycenter estimation with Kneedle on p-values
- geometric statistics for distances
- Fisher + Liptak combination and Tippett meta-combination
- final automatic Kneedle thresholding on sorted SQI

Dependencies:
    numpy scipy pandas
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import json

import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import eigh
from scipy.stats import norm, chi2, beta

EPS = np.finfo(float).eps


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class PotatoConfig:
    leads: List[str]
    bp: Tuple[float, float]
    kind: str = "riemann"     # riemann, euclid, diageuclid, phop
    nor: bool = False


@dataclass
class PotatoModel:
    config: PotatoConfig
    istrained: bool = False
    reg: float = 0.0
    G: Optional[np.ndarray] = None
    mu: float = 0.0
    sigma: float = 0.0
    kept: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    train_distances: Optional[np.ndarray] = None
    train_zscores: Optional[np.ndarray] = None
    train_pvalues: Optional[np.ndarray] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class PotatoFieldModel:
    potatoes: List[PotatoModel]


@dataclass
class IRPFResult:
    sqi: np.ndarray
    P: np.ndarray
    clean_mask: np.ndarray
    artifact_mask: np.ndarray
    p_threshold: float
    knee_index: Optional[int]
    sorted_sqi: np.ndarray
    sortkey: np.ndarray
    segments: List[slice]
    gfrms_clean_mask: Optional[np.ndarray] = None
    gfrms_artifact_mask: Optional[np.ndarray] = None
    gfrms_values: Optional[np.ndarray] = None
    gfrms_threshold_low: Optional[float] = None
    gfrms_threshold_high: Optional[float] = None
    gfrms_center: Optional[float] = None
    model: Optional[PotatoFieldModel] = None


# =============================================================================
# I/O helpers
# =============================================================================

def read_ascii_matrix(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def read_sensors(path: str | Path) -> List[str]:
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [t.strip() for t in txt.replace(",", " ").split() if t.strip()]


def read_potato_field(path: str | Path) -> List[PotatoConfig]:
    """Read a .pf file in the same layout used by PotatoFields.jl."""
    configs: List[PotatoConfig] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        leads = parts[:-4]
        kind = parts[-4].lower().replace("diag(euclidean)", "diageuclid")
        if kind in ("diag_euclid", "diag-euclid", "diageuclidean"):
            kind = "diageuclid"
        nor = parts[-3].lower() == "true"
        f1 = float(parts[-2].replace(",", "."))
        f2 = float(parts[-1].replace(",", "."))
        configs.append(PotatoConfig(leads=leads, bp=(f1, f2), kind=kind, nor=nor))
    return configs


# =============================================================================
# Signal processing
# =============================================================================

def bp_filter(X: np.ndarray, sr: int, band: Tuple[float, float], order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass/high-pass filter. X is samples x channels."""
    X = np.asarray(X, dtype=float)
    low, high = float(band[0]), float(band[1])
    nyq = sr / 2.0
    low = max(low, EPS)
    high = min(high, nyq - EPS)
    if high >= nyq - 1e-9:
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, X, axis=0)


def fixed_segments(n_samples: int, wl: int, slide: int = 0) -> List[slice]:
    """Fixed segmentation matching Julia segment for wl > 0."""
    if wl <= 0:
        raise NotImplementedError("Only fixed segmentation wl > 0 is implemented in this Python port.")
    if slide > 0:
        starts = range(0, (n_samples - wl) // slide + 1)
        return [slice(i * slide, i * slide + wl) for i in starts]
    return [slice(i * wl, i * wl + wl) for i in range(n_samples // wl)]


def segment_covariances(X: np.ndarray, segments: Sequence[slice]) -> List[np.ndarray]:
    """Covariances exactly like Tools.segmentCMs: X_epoch.T @ X_epoch / (len-1). No centering."""
    covs: List[np.ndarray] = []
    for s in segments:
        E = np.asarray(X[s, :], dtype=float)
        C = (E.T @ E) / max(1, E.shape[0] - 1)
        covs.append(hermitian(C))
    return covs


# =============================================================================
# Linear algebra
# =============================================================================

def hermitian(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return (A + A.T) / 2.0


def make_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = hermitian(A)
    w, V = eigh(A)
    w = np.maximum(w, eps)
    return hermitian((V * w) @ V.T)


def det1(C: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    C = make_spd(C, eps=tol)
    d = C.shape[0]
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0 or not np.isfinite(logdet):
        return C
    return hermitian(C / np.exp(logdet / d))


def affine_invariant_distance2(C: np.ndarray, G: np.ndarray) -> float:
    C = make_spd(C)
    G = make_spd(G)
    vals = eigh(C, G, eigvals_only=True)
    vals = np.maximum(vals, 1e-300)
    return float(np.sum(np.log(vals) ** 2))


def euclidean_distance2(C: np.ndarray, G: np.ndarray) -> float:
    return float(np.linalg.norm(C - G, ord="fro") ** 2)


def diagonal_euclidean_distance2(C: np.ndarray, G: np.ndarray) -> float:
    """Paper definition: ||diag(C - G)||_F^2."""
    return float(np.sum((np.diag(C - G)) ** 2))


def trace_distance2_julia(C: np.ndarray, G: np.ndarray) -> float:
    """PotatoFields.jl implementation for :diageuclid uses trace²_ = (tr(C-G))^2."""
    return float(np.trace(C - G) ** 2)


def phop_value(C: np.ndarray) -> float:
    d = C.shape[0]
    tr = float(np.trace(C))
    off = float(np.sum(C) - np.trace(C))
    return tr - off / max(1, d - 1)


def distance2(
    C: np.ndarray,
    G: np.ndarray,
    kind: str,
    nor: bool = False,
    reg: float = 0.0,
    diag_mode: str = "paper",
) -> float:
    C0 = hermitian(C)
    if reg > 0:
        C0 = hermitian(C0 + reg * np.eye(C0.shape[0]))
    if nor:
        C0 = det1(C0)
    kind = kind.lower().replace("diag_euclid", "diageuclid").replace("diag-euclid", "diageuclid")
    if kind == "riemann":
        return affine_invariant_distance2(C0, G)
    if kind == "euclid":
        return euclidean_distance2(C0, G)
    if kind == "diageuclid":
        if diag_mode == "julia_trace":
            return trace_distance2_julia(C0, G)
        return diagonal_euclidean_distance2(C0, G)
    if kind == "phop":
        return float(phop_value(C0))
    raise ValueError(f"Unknown potato kind: {kind}")


def affine_invariant_mean(covs: Sequence[np.ndarray], tol: float = 1e-8, maxiter: int = 80) -> np.ndarray:
    """Affine-invariant/Riemannian mean by fixed-point updates."""
    covs = [make_spd(C) for C in covs]
    G = make_spd(np.mean(covs, axis=0))
    for _ in range(maxiter):
        w, V = eigh(G)
        w = np.maximum(w, 1e-300)
        sqrtG = (V * np.sqrt(w)) @ V.T
        invsqrtG = (V * (1.0 / np.sqrt(w))) @ V.T
        logs = np.zeros_like(G)
        for C in covs:
            S = hermitian(invsqrtG @ C @ invsqrtG)
            ws, Vs = eigh(S)
            ws = np.maximum(ws, 1e-300)
            logs += (Vs * np.log(ws)) @ Vs.T
        logs /= len(covs)
        nrm = float(np.linalg.norm(logs, ord="fro"))
        we, Ve = eigh(hermitian(logs))
        exp_logs = (Ve * np.exp(we)) @ Ve.T
        G = make_spd(sqrtG @ exp_logs @ sqrtG)
        if nrm < tol:
            break
    return G


def geomean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-300)
    return float(np.exp(np.mean(np.log(x))))


def geosd(x: np.ndarray, gm: float) -> float:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 1e-300)
    gm = max(float(gm), 1e-300)
    return float(np.exp(np.sqrt(np.mean(np.log(x / gm) ** 2))))


def geoz(x: np.ndarray, gm: float, gsd: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    denom = np.log(float(gsd))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        denom = 1e-12
    return np.log(np.maximum(x, 1e-300) / max(float(gm), 1e-300)) / denom


# =============================================================================
# Kneedle, close to Kneedle.jl
# =============================================================================

def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    den = np.max(a) - np.min(a)
    if den == 0 or not np.isfinite(den):
        return np.zeros_like(a)
    return (a - np.min(a)) / den


def _local_maxima(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    inds = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and (y[i] > y[i - 1] or y[i] > y[i + 1]):
            inds.append(i)
    return np.asarray(inds, dtype=int)


def _local_minima(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    inds = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1] and (y[i] < y[i - 1] or y[i] < y[i + 1]):
            inds.append(i)
    return np.asarray(inds, dtype=int)


def kneedle_julia_like(
    x: Sequence[float],
    y: Sequence[float],
    curve: str = "convex",
    direction: str = "increasing",
    online: bool = True,
    S: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or np.max(y) == np.min(y):
        return np.array([]), np.array([])
    x_norm = _normalize(x)
    y_norm = _normalize(y)
    curve = curve.lower()
    direction = direction.lower()
    if direction == "decreasing":
        if curve == "concave":
            y_t = y_norm[::-1]
        else:
            y_t = np.max(y_norm) - y_norm
    elif direction == "increasing" and curve == "convex":
        y_t = (np.max(y_norm) - y_norm)[::-1]
    else:
        y_t = y_norm.copy()

    y_diff = y_t - x_norm
    maxima = _local_maxima(y_diff)
    minima = _local_minima(y_diff)
    if len(maxima) == 0:
        return np.array([]), np.array([])
    Tmx = y_diff[maxima] - (S * abs(np.mean(np.diff(x_norm))))
    all_knees: List[float] = []
    all_knees_y: List[float] = []
    threshold = 0.0
    threshold_index = 0
    maxima_threshold_index = 0
    max_set = set(maxima.tolist())
    min_set = set(minima.tolist())
    for i in range(len(y_diff)):
        if i < maxima[0]:
            continue
        if i == len(y_diff) - 1:
            break
        if i in max_set:
            threshold = Tmx[min(maxima_threshold_index, len(Tmx) - 1)]
            threshold_index = i
            maxima_threshold_index += 1
        if i in min_set:
            threshold = 0.0
        j = i + 1
        if y_diff[j] < threshold:
            if curve == "convex":
                knee_idx = threshold_index if direction == "decreasing" else len(x) - 1 - threshold_index
            else:
                knee_idx = len(x) - 1 - threshold_index if direction == "decreasing" else threshold_index
            knee = float(x[knee_idx])
            if not any(np.isclose(knee, k) for k in all_knees):
                all_knees.append(knee)
                all_knees_y.append(float(y[knee_idx]))
                if not online:
                    break
    return np.asarray(all_knees), np.asarray(all_knees_y)


# =============================================================================
# p-value combinations from Pcombination.jl
# =============================================================================

def combine_fisher(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    return float(chi2.sf(-2.0 * np.sum(np.log(p)), 2 * len(p)))


def combine_liptak(p: np.ndarray) -> float:
    """Pcombination.jl: ccdf(Normal(), sum(cquantile(Normal(), p))/sqrt(n))."""
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16)
    q = np.sum(norm.isf(p)) / math.sqrt(len(p))
    return float(norm.sf(q))


def combine_pearson(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16)
    return float(chi2.cdf(-2.0 * np.sum(np.log(1.0 - p)), 2 * len(p)))


def combine_tippett(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    return float(beta.cdf(float(np.min(p)), 1, len(p)))


def combine_min(p: np.ndarray) -> float:
    return float(np.min(np.asarray(p, dtype=float)))


def combine_max(p: np.ndarray) -> float:
    return float(np.max(np.asarray(p, dtype=float)))


def combine_row(p: np.ndarray, comb_funcs: Sequence[str] = ("liptak", "fisher"), meta: str = "tippett") -> float:
    vals = []
    for f in comb_funcs:
        fl = f.lower()
        if fl == "fisher":
            vals.append(combine_fisher(p))
        elif fl in ("liptak", "stouffer"):
            vals.append(combine_liptak(p))
        elif fl == "pearson":
            vals.append(combine_pearson(p))
        elif fl == "tippett":
            vals.append(combine_tippett(p))
        elif fl == "min":
            vals.append(combine_min(p))
        elif fl == "max":
            vals.append(combine_max(p))
        else:
            raise ValueError(f"Unknown combination function: {f}")
    vals = np.asarray(vals, dtype=float)
    ml = meta.lower()
    if ml == "tippett":
        return combine_tippett(vals)
    if ml == "min":
        return combine_min(vals)
    if ml == "max":
        return combine_max(vals)
    if ml == "fisher":
        return combine_fisher(vals)
    if ml in ("liptak", "stouffer"):
        return combine_liptak(vals)
    raise ValueError(f"Unknown meta-combination: {meta}")


# =============================================================================
# FRMS / GFRMS automatic outlier rejection
# =============================================================================

def gfrms_values(X: np.ndarray, mode: str = "paper") -> np.ndarray:
    """Return log FRMS sample-wise.

    mode='paper': sqrt(sum(x^2)/n_channels), as in the paper.
    mode='julia': sqrt(sum(x^2)/n_samples), matching uploaded Tools.jl implementation.
    """
    X = np.asarray(X, dtype=float)
    denom = X.shape[1] if mode == "paper" else X.shape[0]
    rms = np.sqrt(np.sum(X * X, axis=1) / max(1, denom))
    return np.log(np.maximum(rms, 1e-300))


def reject_outliers_frms(
    X: np.ndarray,
    sr: int,
    wl: int,
    upper_limit: float = 1.0,
    mode: str = "paper",
) -> Dict[str, object]:
    """Automatic outlier rejection equivalent to Tools.reject for fixed epochs."""
    X = np.asarray(X, dtype=float)
    ns = X.shape[0]
    segs = fixed_segments(ns, wl, slide=0)
    frms = gfrms_values(X, mode=mode)
    p = np.argsort(frms)
    lo = max(0, ns // 2 - wl)
    hi = min(ns, ns // 2 + wl + 1)
    if hi <= lo:
        lo, hi = 0, ns
    m = float(np.mean(frms[p][lo:hi]))
    low_index = min(9, len(frms) - 1)  # Julia uses frms[p][10] 1-based
    thr_down = float(frms[p][low_index])
    thr_up = float(m + (m - thr_down) * upper_limit)

    clean_mask = np.ones(len(segs), dtype=bool)
    epoch_frms_max = np.zeros(len(segs), dtype=float)
    for i, s in enumerate(segs):
        vals = frms[s]
        epoch_frms_max[i] = float(np.max(vals))
        if np.min(vals) < thr_down or np.max(vals) > thr_up:
            clean_mask[i] = False
    clean_X = np.vstack([X[s, :] for i, s in enumerate(segs) if clean_mask[i]]) if np.any(clean_mask) else np.zeros((0, X.shape[1]))
    rejected_X = np.vstack([X[s, :] for i, s in enumerate(segs) if not clean_mask[i]]) if np.any(~clean_mask) else np.zeros((0, X.shape[1]))
    return {
        "clean_X": clean_X,
        "rejected_X": rejected_X,
        "clean_mask": clean_mask,
        "artifact_mask": ~clean_mask,
        "frms": frms,
        "epoch_frms_max": epoch_frms_max,
        "center": m,
        "thr_down": thr_down,
        "thr_up": thr_up,
        "segments": segs,
    }


# =============================================================================
# Training / testing potato fields
# =============================================================================

def _indices(leads: Sequence[str], sensors: Sequence[str]) -> List[int]:
    sensors_l = [s.lower() for s in sensors]
    out = []
    for lead in leads:
        ll = lead.lower()
        if ll not in sensors_l:
            raise ValueError(f"Lead {lead!r} not found in sensors: {sensors}")
        out.append(sensors_l.index(ll))
    return out


def _unique_preserve(seq: Iterable) -> List:
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def train_potato_from_covs(
    config: PotatoConfig,
    covs_full: Sequence[np.ndarray],
    sensors_for_cov: Sequence[str],
    minsteps: int = 2,
    reg: float = 0.0,
    tol: float = 0.0,
    diag_mode: str = "paper",
    verbose: bool = False,
) -> PotatoModel:
    ind = _indices(config.leads, sensors_for_cov)
    covs = [hermitian(C[np.ix_(ind, ind)]) for C in covs_full]
    if len(covs) < 2:
        raise ValueError("At least two covariance matrices are required to train a potato.")

    if reg > 0 and config.nor:
        covs_mean = [det1(C + reg * np.eye(C.shape[0])) for C in covs]
    elif reg > 0:
        covs_mean = [hermitian(C + reg * np.eye(C.shape[0])) for C in covs]
    elif config.nor:
        covs_mean = [det1(C) for C in covs]
    else:
        covs_mean = list(covs)

    kept = np.arange(len(covs_mean), dtype=int)
    to = math.sqrt(EPS) if tol == 0 else float(tol)
    maxsteps = max(1, minsteps * 2)
    history: List[Dict[str, float]] = []
    G = make_spd(np.mean(covs_mean, axis=0))
    mu = sigma = 0.0

    for steps in range(1, maxsteps + 1):
        toli = min(1e-3, to * 10 ** (max(minsteps, 3) - steps)) if (steps < minsteps and to < 1e-6) else to
        G = affine_invariant_mean([covs_mean[i] for i in kept], tol=toli)
        d = np.asarray([distance2(covs[i], G, config.kind, nor=False, reg=0.0, diag_mode=diag_mode) for i in kept], dtype=float)
        mu = geomean(d)
        sigma = geosd(d, mu)
        z = geoz(d, mu, sigma)
        pvals = norm.sf(z)
        sortedp = np.sort(pvals)
        knees, knees_y = kneedle_julia_like(
            np.arange(1, len(sortedp) + 1, dtype=float),
            sortedp,
            curve="convex",
            direction="increasing",
            online=True,
            S=1.0,
        )
        pth = 0.0 if len(knees_y) == 0 else float(knees_y[-1])
        new_kept = kept[pvals >= pth]
        history.append({"step": steps, "pth": pth, "n_kept": int(len(new_kept)), "mu": mu, "sigma": sigma})
        if verbose:
            print(f"potato leads={config.leads} bp={config.bp} kind={config.kind} step={steps} pth={pth:.6g} kept={len(new_kept)}/{len(kept)}")
        no_change = len(new_kept) == len(kept) and np.all(new_kept == kept)
        kept = new_kept
        if (no_change and steps >= minsteps) or len(kept) < 2:
            break

    model = PotatoModel(config=config, istrained=True, reg=reg, G=G, mu=mu, sigma=sigma, kept=kept, training_history=history)
    all_d = np.asarray([distance2(C, G, config.kind, nor=config.nor, reg=reg, diag_mode=diag_mode) for C in covs], dtype=float)
    all_z = geoz(all_d, mu, sigma)
    model.train_distances = all_d
    model.train_zscores = all_z
    model.train_pvalues = norm.sf(all_z)
    return model


def train_potato_field(
    X: np.ndarray,
    sensors: Sequence[str],
    configs: Sequence[PotatoConfig],
    sr: int = 128,
    wl: int = 128,
    slide: int = 0,
    filter_order: int = 2,
    minsteps: int = 2,
    reg: float = 0.0,
    diag_mode: str = "paper",
    verbose: bool = False,
) -> PotatoFieldModel:
    X = np.asarray(X, dtype=float)
    u_leads = _unique_preserve([lead for cfg in configs for lead in cfg.leads])
    u_ind = _indices(u_leads, sensors)
    bands = _unique_preserve([cfg.bp for cfg in configs])
    models: List[Optional[PotatoModel]] = [None] * len(configs)
    for band in bands:
        Xb = bp_filter(X[:, u_ind], sr=sr, band=band, order=filter_order)
        segs = fixed_segments(Xb.shape[0], wl=wl, slide=slide)
        covs_full = segment_covariances(Xb, segs)
        for i, cfg in enumerate(configs):
            if cfg.bp == band:
                models[i] = train_potato_from_covs(
                    cfg,
                    covs_full,
                    u_leads,
                    minsteps=minsteps,
                    reg=reg,
                    diag_mode=diag_mode,
                    verbose=verbose,
                )
    return PotatoFieldModel(potatoes=[m for m in models if m is not None])


def test_potato_field(
    X: np.ndarray,
    sensors: Sequence[str],
    model: PotatoFieldModel,
    sr: int = 128,
    wl: int = 128,
    slide: int = 0,
    filter_order: int = 2,
    comb_funcs: Sequence[str] = ("liptak", "fisher"),
    meta: str = "tippett",
    diag_mode: str = "paper",
) -> Tuple[np.ndarray, np.ndarray, List[slice]]:
    X = np.asarray(X, dtype=float)
    configs = [m.config for m in model.potatoes]
    u_leads = _unique_preserve([lead for cfg in configs for lead in cfg.leads])
    u_ind = _indices(u_leads, sensors)
    bands = _unique_preserve([cfg.bp for cfg in configs])
    segs = fixed_segments(X.shape[0], wl=wl, slide=slide)
    P_cols: List[Optional[np.ndarray]] = [None] * len(model.potatoes)
    for band in bands:
        Xb = bp_filter(X[:, u_ind], sr=sr, band=band, order=filter_order)
        covs_full = segment_covariances(Xb, segs)
        for i, pm in enumerate(model.potatoes):
            if pm.config.bp != band:
                continue
            ind = _indices(pm.config.leads, u_leads)
            vals = []
            for Cfull in covs_full:
                C = hermitian(Cfull[np.ix_(ind, ind)])
                d = distance2(C, pm.G, pm.config.kind, nor=pm.config.nor, reg=pm.reg, diag_mode=diag_mode)
                z = geoz(np.asarray([d]), pm.mu, pm.sigma)[0]
                vals.append(norm.sf(z))
            P_cols[i] = np.asarray(vals, dtype=float)
    if any(c is None for c in P_cols):
        raise RuntimeError("Some potatoes did not produce p-values.")
    P = np.column_stack(P_cols)
    sqi = np.asarray([combine_row(row, comb_funcs=comb_funcs, meta=meta) for row in P], dtype=float)
    return sqi, P, segs


def final_kneedle_threshold(sqi: np.ndarray) -> Tuple[Optional[int], float, np.ndarray, np.ndarray]:
    sqi = np.asarray(sqi, dtype=float)
    sortkey = np.argsort(sqi)
    sorted_sqi = sqi[sortkey]
    knees, knees_y = kneedle_julia_like(
        np.arange(1, len(sorted_sqi) + 1, dtype=float),
        sorted_sqi,
        curve="convex",
        direction="increasing",
        online=True,
        S=1.0,
    )
    if len(knees) == 0:
        return None, 0.0, sorted_sqi, sortkey
    return int(knees[-1]), float(knees_y[-1]), sorted_sqi, sortkey


class FullIRPF:
    """Full Python-only iRPF estimator.

    Parameters
    ----------
    sensors : list[str]
        Channel names in the order of X columns.
    potato_configs : list[PotatoConfig]
        Potato field definition.
    sr : int
        Sampling rate.
    wl : int
        Epoch length in samples.
    slide : int
        Overlap/slide argument, kept for Julia compatibility.
    filter_order : int
        Butterworth order used for potato band filters.
    minsteps : int
        Minimum iterations for adaptive robust barycenter estimation.
    reg : float
        Regularization added to covariance matrices.
    diag_mode : {'paper', 'julia_trace'}
        'paper' implements true diagonal-Euclidean distance.
        'julia_trace' matches the uploaded PotatoFields.jl trace²_ behavior.
    frms_mode : {'paper', 'julia'}
        FRMS denominator. 'paper' divides by n_channels; 'julia' divides by n_samples.
    """

    def __init__(
        self,
        sensors: Sequence[str],
        potato_configs: Sequence[PotatoConfig],
        sr: int = 128,
        wl: int = 128,
        slide: int = 0,
        filter_order: int = 2,
        minsteps: int = 2,
        reg: float = 0.0,
        diag_mode: str = "paper",
        frms_mode: str = "paper",
        comb_funcs: Sequence[str] = ("liptak", "fisher"),
        meta_combination: str = "tippett",
        upper_limit: float = 1.0,
        global_prefilter: Optional[Tuple[float, float]] = None,
        global_filter_order: int = 4,
        use_gfrms_for_final_mask: bool = False,
        verbose: bool = False,
    ):
        self.sensors = list(sensors)
        self.configs = list(potato_configs)
        self.sr = int(sr)
        self.wl = int(wl)
        self.slide = int(slide)
        self.filter_order = int(filter_order)
        self.minsteps = int(minsteps)
        self.reg = float(reg)
        self.diag_mode = diag_mode
        self.frms_mode = frms_mode
        self.comb_funcs = tuple(comb_funcs)
        self.meta_combination = meta_combination
        self.upper_limit = float(upper_limit)
        self.global_prefilter = global_prefilter
        self.global_filter_order = int(global_filter_order)
        self.use_gfrms_for_final_mask = bool(use_gfrms_for_final_mask)
        self.verbose = bool(verbose)
        self.model: Optional[PotatoFieldModel] = None

    @classmethod
    def from_files(cls, sensors_file: str | Path, potato_field_file: str | Path, **kwargs) -> "FullIRPF":
        sensors = read_sensors(sensors_file)
        configs = read_potato_field(potato_field_file)
        return cls(sensors=sensors, potato_configs=configs, **kwargs)

    def fit(self, X: np.ndarray) -> "FullIRPF":
        X = np.asarray(X, dtype=float)
        if self.global_prefilter is not None:
            X = bp_filter(X, self.sr, self.global_prefilter, order=self.global_filter_order)
        # iRPF_rejectoutliers: filter in the min/max band of the potato field, then reject.
        all_freqs = [f for cfg in self.configs for f in cfg.bp]
        reject_band = (float(min(all_freqs)), float(max(all_freqs)))
        X_for_reject = bp_filter(X, self.sr, reject_band, order=self.filter_order)
        out = reject_outliers_frms(X_for_reject, self.sr, self.wl, upper_limit=self.upper_limit, mode=self.frms_mode)
        clean_X = out["clean_X"]
        if clean_X.shape[0] < self.wl:
            raise RuntimeError("GFRMS rejection left no full clean epoch for training.")
        self._last_gfrms = out
        self.model = train_potato_field(
            clean_X,
            self.sensors,
            self.configs,
            sr=self.sr,
            wl=self.wl,
            slide=self.slide,
            filter_order=self.filter_order,
            minsteps=self.minsteps,
            reg=self.reg,
            diag_mode=self.diag_mode,
            verbose=self.verbose,
        )
        return self

    def transform(self, X: np.ndarray) -> IRPFResult:
        if self.model is None:
            raise RuntimeError("Call fit first.")
        X = np.asarray(X, dtype=float)
        if self.global_prefilter is not None:
            X = bp_filter(X, self.sr, self.global_prefilter, order=self.global_filter_order)
        sqi, P, segs = test_potato_field(
            X,
            self.sensors,
            self.model,
            sr=self.sr,
            wl=self.wl,
            slide=self.slide,
            filter_order=self.filter_order,
            comb_funcs=self.comb_funcs,
            meta=self.meta_combination,
            diag_mode=self.diag_mode,
        )
        knee, pth, sorted_sqi, sortkey = final_kneedle_threshold(sqi)
        clean_mask = sqi >= pth
        gfrms_clean = None
        gfrms_art = None
        gfrms_vals = None
        thr_low = thr_high = center = None
        if hasattr(self, "_last_gfrms"):
            gfrms_clean = np.asarray(self._last_gfrms["clean_mask"], dtype=bool)
            gfrms_art = ~gfrms_clean
            gfrms_vals = np.asarray(self._last_gfrms["epoch_frms_max"], dtype=float)
            thr_low = float(self._last_gfrms["thr_down"])
            thr_high = float(self._last_gfrms["thr_up"])
            center = float(self._last_gfrms["center"])
            if self.use_gfrms_for_final_mask and len(gfrms_clean) == len(clean_mask):
                clean_mask = clean_mask & gfrms_clean
        return IRPFResult(
            sqi=sqi,
            P=P,
            clean_mask=clean_mask,
            artifact_mask=~clean_mask,
            p_threshold=pth,
            knee_index=knee,
            sorted_sqi=sorted_sqi,
            sortkey=sortkey,
            segments=segs,
            gfrms_clean_mask=gfrms_clean,
            gfrms_artifact_mask=gfrms_art,
            gfrms_values=gfrms_vals,
            gfrms_threshold_low=thr_low,
            gfrms_threshold_high=thr_high,
            gfrms_center=center,
            model=self.model,
        )

    def fit_transform(self, X: np.ndarray) -> IRPFResult:
        self.fit(X)
        return self.transform(X)


def result_to_dataframe(result: IRPFResult, prefix: str = "irpf") -> pd.DataFrame:
    df = pd.DataFrame({
        "epoch_id": np.arange(len(result.sqi)),
        f"sqi_{prefix}": result.sqi,
        f"clean_{prefix}": result.clean_mask.astype(int),
        f"artifact_{prefix}": result.artifact_mask.astype(int),
    })
    for j in range(result.P.shape[1]):
        df[f"p_potato_{j+1}_{prefix}"] = result.P[:, j]
    if result.gfrms_clean_mask is not None and len(result.gfrms_clean_mask) == len(df):
        df[f"gfrms_clean_{prefix}"] = result.gfrms_clean_mask.astype(int)
        df[f"gfrms_artifact_{prefix}"] = result.gfrms_artifact_mask.astype(int)
        df[f"gfrms_value_{prefix}"] = result.gfrms_values
    return df


def save_result_outputs(result: IRPFResult, out_dir: str | Path, prefix: str = "python_irpf") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_to_dataframe(result, prefix=prefix).to_csv(out_dir / f"{prefix}_epoch_results.csv", index=False)
    pd.DataFrame(result.P, columns=[f"p_potato_{i+1}" for i in range(result.P.shape[1])]).to_csv(out_dir / f"{prefix}_P_matrix.csv", index=False)
    pd.DataFrame({"sorted_sqi": result.sorted_sqi}).to_csv(out_dir / f"{prefix}_sorted_sqi.csv", index=False)
    with open(out_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_epochs": int(len(result.sqi)),
            "n_clean": int(np.sum(result.clean_mask)),
            "n_artifact": int(np.sum(result.artifact_mask)),
            "p_threshold": float(result.p_threshold),
            "knee_index": None if result.knee_index is None else int(result.knee_index),
            "sqi_min": float(np.min(result.sqi)),
            "sqi_mean": float(np.mean(result.sqi)),
            "sqi_max": float(np.max(result.sqi)),
            "gfrms_threshold_low": result.gfrms_threshold_low,
            "gfrms_threshold_high": result.gfrms_threshold_high,
            "gfrms_center": result.gfrms_center,
        }, f, indent=2)
