from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def _segment_start_stop(segment: Any) -> tuple[int, int]:
    """Return start/stop sample indices for common segment representations."""
    if isinstance(segment, slice):
        if segment.stop is None:
            raise ValueError("Segment slice must have a stop value")
        start = 0 if segment.start is None else int(segment.start)
        return start, int(segment.stop)

    if isinstance(segment, (tuple, list, np.ndarray)) and len(segment) == 2:
        return int(segment[0]), int(segment[1])

    indices = np.asarray(list(segment), dtype=int)
    if indices.size == 0:
        raise ValueError("Segment cannot be empty")
    return int(indices[0]), int(indices[-1]) + 1


def plot_diagnostics(
    result: Any,
    sorted_epochs: np.ndarray,
    sensors: Sequence[str],
    sr: int,
    threshold_epoch: int,
    potato_config: Sequence[Mapping[str, Any]] | None = None,
    *,
    window_sec: float = 10,
    step_sec: float = 1,
    show: bool = True,
) -> dict[str, Any]:
    """Display an interactive signal view and per-potato activation heatmap.

    Args:
        result: ``FullIRPF`` result with ``sqi``, ``P``, masks, and segments.
        sorted_epochs: Epochs arranged from lower to higher SQI, shaped
            ``(n_epochs, n_channels, n_times)``.
        sensors: Channel labels in the same order as the epoch channel axis.
        sr: Sampling rate in Hz.
        threshold_epoch: 1-based ordered epoch at the iRPF/Kneedle threshold.
        potato_config: Optional configuration dictionaries used to name heatmap
            rows. If omitted or mismatched, generic potato names are shown.
        window_sec: Signal window width in seconds.
        step_sec: Slider and navigation step in seconds.
        show: Call ``plt.show()`` and display the widget controls.

    Returns:
        A dictionary containing the figure, axes, widgets and update callback.
    """
    sorted_epochs = np.asarray(sorted_epochs)
    sensors = [str(sensor).upper() for sensor in sensors]
    if sorted_epochs.ndim != 3:
        raise ValueError("sorted_epochs must have shape (epochs, channels, samples)")
    n_epochs, n_channels, n_times = sorted_epochs.shape
    if n_epochs == 0 or n_channels == 0 or n_times == 0:
        raise ValueError("sorted_epochs cannot have an empty dimension")
    if len(sensors) != n_channels:
        raise ValueError(f"Received {len(sensors)} sensor names for {n_channels} channels")
    if sr <= 0 or window_sec <= 0 or step_sec <= 0:
        raise ValueError("sr, window_sec and step_sec must be positive")

    sqi = np.asarray(result.sqi)
    p_original = np.asarray(result.P)
    clean_original = np.asarray(result.clean_mask, dtype=bool)
    if sqi.shape[0] != n_epochs or p_original.shape[0] != n_epochs:
        raise ValueError("Result epoch count does not match sorted_epochs")
    if clean_original.shape[0] != n_epochs:
        raise ValueError("clean_mask length does not match sorted_epochs")

    # The sorted epoch array is ordered using the SQI values from the original
    # recording, so this index maps each displayed row back to its source epoch.
    sorted_idx = np.argsort(sqi)
    p_ord = p_original[sorted_idx, :]
    clean_ord = clean_original[sorted_idx]
    n_potatoes = p_ord.shape[1]
    potato_names = [
        str(cfg.get("name", f"potato_{i + 1}"))
        for i, cfg in enumerate(potato_config or [])
    ]
    if len(potato_names) != n_potatoes:
        potato_names = [f"potato_{i + 1}" for i in range(n_potatoes)]

    segments = list(result.segments)
    if len(segments) != n_epochs:
        raise ValueError("Result segment count does not match the number of epochs")
    starts_stops = [_segment_start_stop(segment) for segment in segments]
    orig_start_sec = np.asarray([item[0] for item in starts_stops], dtype=float) / sr
    orig_stop_sec = np.asarray([item[1] for item in starts_stops], dtype=float) / sr

    p_safe = np.clip(p_ord, 1e-15, 1.0)
    activation_ord = -np.log10(p_safe)
    best_potato_idx = np.argmin(p_ord, axis=1)
    best_potato_p = np.min(p_ord, axis=1)
    best_potato_activation = -np.log10(np.clip(best_potato_p, 1e-15, 1.0))

    X_concat = np.transpose(sorted_epochs, (0, 2, 1)).reshape(-1, n_channels)
    window_samples = int(round(window_sec * sr))
    step_samples = max(1, int(round(step_sec * sr)))
    if X_concat.shape[0] < window_samples:
        raise ValueError(
            f"Signal is {X_concat.shape[0] / sr:.2f}s; shorter than the "
            f"{window_sec:g}s display window"
        )
    max_start = X_concat.shape[0] - window_samples

    max_abs = np.nanmax(np.abs(X_concat))
    if max_abs < 1e-3:
        X_plot = X_concat * 1e6
        unit = "uV"
    else:
        X_plot = X_concat.copy()
        unit = "uV"
    X_plot -= np.nanmean(X_plot, axis=0, keepdims=True)
    amplitude = np.nanpercentile(np.abs(X_plot), 95)
    if not np.isfinite(amplitude) or amplitude == 0:
        amplitude = 1.0
    spacing_base = amplitude * 4

    epoch_sec = n_times / sr
    threshold_sec = int(threshold_epoch) * epoch_sec
    times = np.arange(window_samples) / sr
    fig, (ax_sig, ax_heat) = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        gridspec_kw={"height_ratios": [3.5, 1.8]},
        sharex=False,
    )
    offsets = np.arange(n_channels)[::-1] * spacing_base
    initial_segment = X_plot[:window_samples]
    lines = [
        ax_sig.plot(times, initial_segment[:, ch] + offsets[ch], linewidth=0.8)[0]
        for ch in range(n_channels)
    ]
    threshold_line = ax_sig.axvline(0, linestyle="--", linewidth=2, color="black")
    threshold_text = ax_sig.text(
        0, offsets[0], "soglia iRPF", rotation=90, va="bottom", ha="right", color="black"
    )
    ax_sig.set_yticks(offsets)
    ax_sig.set_yticklabels(sensors)
    ax_sig.set_ylabel(f"Ampiezza + offset [{unit}]")
    ax_sig.set_xlim(0, window_sec)
    ax_sig.grid(True, axis="x", alpha=0.3)
    title = ax_sig.set_title("")

    heat_img = ax_heat.imshow(
        np.zeros((n_potatoes, 1)),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    ax_heat.set_yticks(np.arange(n_potatoes))
    ax_heat.set_yticklabels(potato_names, fontsize=8)
    ax_heat.set_ylabel("-log10(p)\nper potato")
    heat_cbar = fig.colorbar(heat_img, ax=ax_heat, pad=0.01)
    heat_cbar.set_label("attivazione")
    plt.tight_layout()

    prev_button = widgets.Button(
        description="Precedente", layout=widgets.Layout(width="140px")
    )
    next_button = widgets.Button(
        description="Successiva", layout=widgets.Layout(width="140px")
    )
    start_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max_start,
        step=step_samples,
        description="Start",
        continuous_update=False,
        layout=widgets.Layout(width="750px"),
    )
    scale_slider = widgets.FloatSlider(
        value=1.0,
        min=0.3,
        max=5.0,
        step=0.1,
        description="Scala",
        continuous_update=False,
        layout=widgets.Layout(width="750px"),
    )
    diagnostic_label = widgets.HTML()

    def update_plot() -> None:
        start = int(start_slider.value)
        end = start + window_samples
        t0, t1 = start / sr, end / sr
        spacing = spacing_base / float(scale_slider.value)
        offsets_local = np.arange(n_channels)[::-1] * spacing
        segment = X_plot[start:end]
        for ch, line in enumerate(lines):
            line.set_ydata(segment[:, ch] + offsets_local[ch])
        ax_sig.set_yticks(offsets_local)
        ax_sig.set_yticklabels(sensors)
        ymin = np.nanmin(segment.T + offsets_local[:, None])
        ymax = np.nanmax(segment.T + offsets_local[:, None])
        ax_sig.set_ylim(ymin - spacing, ymax + spacing)

        if t0 <= threshold_sec <= t1:
            x_threshold = threshold_sec - t0
            threshold_line.set_xdata([x_threshold, x_threshold])
            threshold_line.set_visible(True)
            threshold_text.set_position((x_threshold, ymax))
            threshold_text.set_visible(True)
        else:
            threshold_line.set_visible(False)
            threshold_text.set_visible(False)

        epoch_start = max(0, int(np.floor(t0 / epoch_sec)))
        epoch_end = min(n_epochs, int(np.ceil(t1 / epoch_sec)))
        epochs_window = np.arange(epoch_start, epoch_end)
        start_1based = epoch_start + 1
        end_1based = epoch_end
        center = int(np.floor(((t0 + t1) / 2) / epoch_sec))
        center = max(0, min(center, n_epochs - 1))
        center_1based = center + 1
        epoch_times = epochs_window * epoch_sec - t0

        heat_window = activation_ord[epochs_window, :].T
        if heat_window.size == 0:
            heat_window = np.zeros((n_potatoes, 1))
        heat_img.set_data(heat_window)
        if len(epoch_times):
            heat_img.set_extent(
                [epoch_times[0], epoch_times[-1] + epoch_sec, -0.5, n_potatoes - 0.5]
            )
        else:
            heat_img.set_extent([0, window_sec, -0.5, n_potatoes - 0.5])
        vmax = np.nanpercentile(heat_window, 98)
        heat_img.set_clim(0, vmax if np.isfinite(vmax) and vmax > 0 else 1.0)
        ax_heat.set_xlim(0, window_sec)
        title.set_text(
            f"iRPF diagnostic plot | finestra ordinata {t0:.1f}-{t1:.1f}s\n"
            f"Epoche ordinate: {start_1based}-{end_1based} | epoca centrale: {center_1based} "
            f"| soglia iRPF/Kneedle: epoca {threshold_epoch}"
        )

        rows = []
        for ordered_epoch in epochs_window:
            original_epoch = int(sorted_idx[ordered_epoch])
            potato_idx = int(best_potato_idx[ordered_epoch])
            rows.append(
                "<tr>"
                f"<td>{ordered_epoch + 1}</td>"
                f"<td>{original_epoch + 1}</td>"
                f"<td>{orig_start_sec[original_epoch]:.2f}-{orig_stop_sec[original_epoch]:.2f}</td>"
                f"<td>{sqi[original_epoch]:.3e}</td>"
                f"<td>{potato_names[potato_idx]}</td>"
                f"<td>{best_potato_p[ordered_epoch]:.3e}</td>"
                f"<td>{best_potato_activation[ordered_epoch]:.2f}</td>"
                f"<td>{'clean' if clean_ord[ordered_epoch] else 'artifact'}</td>"
                "</tr>"
            )
        diagnostic_label.value = (
            f"<b>Finestra ordinata:</b> {t0:.1f}-{t1:.1f} s"
            f"&nbsp;&nbsp; | &nbsp;&nbsp;<b>Epoche ordinate:</b> {start_1based}-{end_1based}"
            f"&nbsp;&nbsp; | &nbsp;&nbsp;<b>Epoca centrale:</b> {center_1based}"
            f"&nbsp;&nbsp; | &nbsp;&nbsp;<b>Soglia iRPF:</b> epoca {threshold_epoch}, "
            f"tempo ordinato {threshold_sec:.1f} s<br><br>"
            "<table border='1' style='border-collapse:collapse; font-size:12px;'>"
            "<tr><th>Epoca ordinata</th><th>Epoca originale</th>"
            "<th>Secondi originali</th><th>SQI</th><th>Potato più attiva</th>"
            "<th>p min</th><th>-log10(p)</th><th>Stato</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        fig.canvas.draw_idle()

    def go_previous(_: Any) -> None:
        start_slider.value = max(0, int(start_slider.value) - step_samples)

    def go_next(_: Any) -> None:
        start_slider.value = min(max_start, int(start_slider.value) + step_samples)

    prev_button.on_click(go_previous)
    next_button.on_click(go_next)
    start_slider.observe(lambda _: update_plot(), names="value")
    scale_slider.observe(lambda _: update_plot(), names="value")
    update_plot()

    if show:
        plt.show()
        display(
            widgets.HBox([prev_button, next_button]),
            start_slider,
            scale_slider,
            diagnostic_label,
        )

    return {
        "figure": fig,
        "axes": (ax_sig, ax_heat),
        "previous_button": prev_button,
        "next_button": next_button,
        "start_slider": start_slider,
        "scale_slider": scale_slider,
        "diagnostic_label": diagnostic_label,
        "update_plot": update_plot,
    }