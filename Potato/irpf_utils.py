from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import combine_pvalues

from pyriemann.estimation import Covariances
from pyriemann.geometry.covariance import normalize


def bandpass_filter(
    X: np.ndarray,
    sr: float,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along the sample axis.

    Args:
        X: Array shaped ``(samples, channels)``.
        sr: Sampling rate in Hz.
        low_freq: Low cutoff in Hz.
        high_freq: High cutoff in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered array with the same shape as ``X``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (samples, channels); got shape {X.shape}")
    if sr <= 0 or order < 1:
        raise ValueError("sr must be positive and order must be at least 1")
    nyquist = sr / 2.0
    if not (0 < low_freq < high_freq < nyquist):
        raise ValueError(
            f"Cutoffs must satisfy 0 < low_freq < high_freq < Nyquist ({nyquist:g} Hz)"
        )

    sos = butter(order, [low_freq, high_freq], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, X, axis=0)


def make_epochs(X: np.ndarray, wl: int, slide: int = 0) -> np.ndarray:
    """Split continuous data into epochs.

    Args:
        X: Continuous signal shaped ``(samples, channels)``.
        wl: Epoch length in samples.
        slide: Step in samples; ``0`` means non-overlapping epochs (step ``wl``).

    Returns:
        Epoch array shaped ``(n_epochs, n_channels, n_times)``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (samples, channels); got shape {X.shape}")
    if wl <= 0:
        raise ValueError("wl must be a positive number of samples")
    if slide < 0:
        raise ValueError("slide cannot be negative")
    if X.shape[0] < wl:
        raise ValueError(f"Signal has {X.shape[0]} samples, fewer than epoch length {wl}")

    step = wl if slide == 0 else slide
    starts = range(0, X.shape[0] - wl + 1, step)
    return np.stack([X[start : start + wl, :].T for start in starts], axis=0)


def predict_clean_mask(model: Any, covariances: np.ndarray) -> np.ndarray:
    """Normalize pyRiemann predictions to a boolean clean/not-clean mask."""
    labels = np.asarray(model.predict(covariances))
    if np.issubdtype(labels.dtype, np.number) and np.any(labels < 0):
        return labels > 0
    return labels.astype(bool)


def min_fisher_stouffer(probas: np.ndarray, axis: int = 0) -> np.ndarray:
    """Combine p-values using the smaller of Fisher and Stouffer results."""
    probas = np.asarray(probas, dtype=float)
    if probas.size == 0:
        raise ValueError("probas cannot be empty")
    if np.any(~np.isfinite(probas)) or np.any((probas < 0) | (probas > 1)):
        raise ValueError("probas must contain finite p-values between 0 and 1")
    safe = np.clip(probas, 1e-300, 1.0)
    _, fisher_p = combine_pvalues(safe, method="fisher", axis=axis)
    _, stouffer_p = combine_pvalues(safe, method="stouffer", axis=axis)
    return np.minimum(fisher_p, stouffer_p)


def compute_gfrms(data_epochs: np.ndarray) -> np.ndarray:
    """Compute log-GFRMS for epochs shaped ``(epochs, channels, times)``."""
    data_epochs = np.asarray(data_epochs)
    if data_epochs.ndim != 3 or data_epochs.shape[1] == 0:
        raise ValueError("data_epochs must have shape (n_epochs, n_channels, n_times)")
    gfrms = np.sqrt(np.mean(data_epochs**2, axis=1))
    return np.log(np.maximum(gfrms, 1e-30))


def gfrms_amplitude_rejection_from_raw(
    X_raw: np.ndarray,
    config: Sequence[Mapping[str, Any]],
    sr: float,
    wl: int,
    slide: int = 0,
    upper_limit: float = 1.618,
) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    """Reject epochs outside the robust log-GFRMS amplitude limits.

    The union of configured frequency bands is filtered once before epoching.
    Returns ``(clean_mask, gfrms, lower, upper, midpoint_mean, epochs)``.
    """
    if not config:
        raise ValueError("config must contain at least one potato configuration")
    if upper_limit < 0:
        raise ValueError("upper_limit cannot be negative")
    low = min(float(cfg["low_freq"]) for cfg in config)
    high = max(float(cfg["high_freq"]) for cfg in config)
    filtered = bandpass_filter(X_raw, sr, low, high, order=4)
    epochs = make_epochs(filtered, wl=wl, slide=slide)
    gfrms = compute_gfrms(epochs)

    values = np.sort(gfrms.ravel())
    n_values = values.size
    n_times = gfrms.shape[1]
    midpoint = n_values // 2
    half_width = min(n_times, midpoint)
    middle_values = values[midpoint - half_width : midpoint + half_width]
    midpoint_mean = float(np.mean(middle_values))
    lower = float(values[min(9, n_values - 1)])
    upper = float(midpoint_mean + (midpoint_mean - lower) * upper_limit)

    epoch_min = np.min(gfrms, axis=1)
    epoch_max = np.max(gfrms, axis=1)
    clean = (epoch_min >= lower) & (epoch_max <= upper)
    return clean, gfrms, lower, upper, midpoint_mean, epochs


def compute_irpf_covariances_from_raw(
    X_raw: np.ndarray,
    sensors: Sequence[str],
    config: Sequence[Mapping[str, Any]],
    sr: float,
    wl: int,
    slide: int = 0,
) -> tuple[list[np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Filter and calculate regularized covariance matrices per potato.

    Configuration dictionaries use ``channels``, ``low_freq``, ``high_freq``,
    ``name``, and optionally ``normalization`` and ``diag_only``.
    """
    sensors = [sensor.upper() for sensor in sensors]
    cov_list: list[np.ndarray] = []
    epochs_by_potato: dict[str, np.ndarray] = {}

    for index, cfg in enumerate(config, start=1):
        name = str(cfg.get("name", f"potato_{index}"))
        channels = cfg.get("channels")
        if channels is None:
            ch_idx = list(range(len(sensors)))
            ch_used = sensors
        else:
            ch_used = [str(ch).upper() for ch in channels]
            missing = [ch for ch in ch_used if ch not in sensors]
            if missing:
                raise ValueError(f"{name}: channels not found in sensors: {missing}")
            ch_idx = [sensors.index(ch) for ch in ch_used]

        filtered = bandpass_filter(
            np.asarray(X_raw)[:, ch_idx],
            sr=sr,
            low_freq=float(cfg["low_freq"]),
            high_freq=float(cfg["high_freq"]),
            order=4,
        )
        epochs = make_epochs(filtered, wl=wl, slide=slide)
        covariances = Covariances(estimator="lwf").transform(epochs)

        normalization = cfg.get("normalization")
        if normalization is True:
            normalization = "trace"
        if normalization not in (None, False):
            covariances = normalize(covariances, normalization)

        if cfg.get("diag_only", False):
            covariances = np.eye(covariances.shape[-1])[None, :, :] * np.diagonal(
                covariances, axis1=1, axis2=2
            )[:, None, :]

        cov_list.append(covariances)
        epochs_by_potato[name] = epochs

        print(
            name,
            "| channels =", ch_used,
            "| band =", cfg["low_freq"], "-", cfg["high_freq"], "Hz",
            "| metric =", cfg.get("metric", "unknown"),
            "| normalization =", normalization,
            "| diag_only =", cfg.get("diag_only", False),
            "| covs =", covariances.shape,
        )

    epochs_for_plot = make_epochs(X_raw, wl=wl, slide=slide)
    return cov_list, epochs_by_potato, epochs_for_plot


def cfg_to_potato_config(
    cfg: Mapping[str, Any],
    sensors: Sequence[str],
    potato_config_class: type,
) -> Any:
    """Convert a notebook-style potato dict to ``full_irpf_python_only`` config.

    ``potato_config_class`` is passed in to avoid coupling this utility module to
    the pipeline implementation; typically pass its ``PotatoConfig`` class.
    """
    kind = "diageuclid" if cfg.get("diag_only", False) else cfg["metric"]
    channels = cfg.get("channels")
    leads = [sensor.upper() for sensor in sensors] if channels is None else list(channels)
    normalization = cfg.get("normalization")
    nor = normalization not in (None, False)
    return potato_config_class(
        leads=leads,
        bp=(float(cfg["low_freq"]), float(cfg["high_freq"])),
        kind=kind,
        nor=bool(nor),
    )
