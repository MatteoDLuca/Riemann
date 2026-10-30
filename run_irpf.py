from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from full_irpf_python_only import FullIRPF, PotatoConfig, save_result_outputs
from irpf_utils import cfg_to_potato_config, make_epochs
from potato_config import IRPF_CONFIG


# Main settings. Update these paths and parameters for your recording.
SR = 512
WL = SR                 # 1-second epochs
SLIDE = 0                # 0 means non-overlapping epochs
GFRMS_UPPER_LIMIT = 1.618
DB_DIR = Path("/content")
SENSOR_FILE = DB_DIR / "8 canali.txt"
EEG_FILE = DB_DIR / "PD_P__AlAn75__rest_pre_OC.txt"
OUTPUT_DIR = Path("potato1")

# Manual epoch-selection settings.
# Choices: "after_threshold", "from_ordered_epoch", "best_n", "all".
EPOCH_SAVE_MODE = "from_ordered_epoch"
N_BEST_EPOCHS = 136
SAVE_FROM_ORDERED_EPOCH = 5


def _read_sensor_names(path: Path) -> list[str]:
    """Read sensor names, ignoring blank lines and a count-only first line."""
    if not path.is_file():
        raise FileNotFoundError(f"Sensor file not found: {path}")
    sensors: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.isdigit():
                continue
            sensors.extend(line.replace(",", " ").replace("\t", " ").split())
    sensors = [sensor.upper() for sensor in sensors]
    if not sensors:
        raise ValueError(f"No sensor names found in {path}")
    return sensors


def _read_eeg_text(path: Path, n_sensors: int) -> np.ndarray:
    """Read numeric samples from a whitespace-, comma-, or tab-delimited file."""
    if not path.is_file():
        raise FileNotFoundError(f"EEG file not found: {path}")
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            try:
                row = [float(value) for value in parts]
            except ValueError:
                # Ignore header lines.
                continue
            if row and np.all(np.isfinite(row)):
                rows.append(row)
    if not rows:
        raise ValueError(f"No numeric EEG samples found in {path}")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("EEG file has rows with different numbers of columns")
    data = np.asarray(rows, dtype=float)
    if data.shape[1] > n_sensors:
        data = data[:, :n_sensors]
    elif data.shape[1] < n_sensors:
        raise ValueError(
            f"EEG has {data.shape[1]} channels but sensor file lists {n_sensors}; "
            "check the files or their column order"
        )
    return data


def _select_epochs(
    sorted_epochs: np.ndarray,
    threshold_epoch: int,
    mode: str,
    n_best: int,
    save_from_ordered_epoch: int,
) -> tuple[np.ndarray, str]:
    """Select epochs from the SQI-sorted array (1-based manual index)."""
    n_epochs = sorted_epochs.shape[0]
    if mode == "after_threshold":
        # threshold_epoch is 1-based; preserve the original notebook behavior.
        selected = sorted_epochs[threshold_epoch:]
        suffix = f"clean_after_threshold_{len(selected)}"
    elif mode == "from_ordered_epoch":
        if not 1 <= save_from_ordered_epoch <= n_epochs:
            raise ValueError(f"SAVE_FROM_ORDERED_EPOCH must be between 1 and {n_epochs}")
        selected = sorted_epochs[save_from_ordered_epoch - 1 :]
        suffix = f"clean_from_ordered_epoch_{save_from_ordered_epoch}_n_{len(selected)}"
    elif mode == "best_n":
        if n_best <= 0:
            raise ValueError("N_BEST_EPOCHS must be positive")
        count = min(n_best, n_epochs)
        selected = sorted_epochs[-count:]
        suffix = f"best_{count}"
    elif mode == "all":
        selected = sorted_epochs
        suffix = f"all_{n_epochs}"
    else:
        raise ValueError(
            "EPOCH_SAVE_MODE must be 'after_threshold', 'from_ordered_epoch', "
            "'best_n' or 'all'"
        )
    return selected, suffix


def run_pipeline(
    sensor_file: str | Path = SENSOR_FILE,
    eeg_file: str | Path = EEG_FILE,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    sr: int = SR,
    wl: int = WL,
    slide: int = SLIDE,
    gfrms_upper_limit: float = GFRMS_UPPER_LIMIT,
    epoch_save_mode: str = EPOCH_SAVE_MODE,
    n_best_epochs: int = N_BEST_EPOCHS,
    save_from_ordered_epoch: int = SAVE_FROM_ORDERED_EPOCH,
    show_plot: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load input files, run iRPF, save selected epochs and numeric outputs."""
    sensor_file = Path(sensor_file)
    eeg_file = Path(eeg_file)
    output_dir = Path(output_dir)
    if sr <= 0 or wl <= 0:
        raise ValueError("sr and wl must be positive")

    sensors = _read_sensor_names(sensor_file)
    X_raw = _read_eeg_text(eeg_file, len(sensors))
    if verbose:
        print("Sensor file:", sensor_file.resolve())
        print("EEG file:", eeg_file.resolve())
        print("Sensors:", sensors)
        print("EEG shape:", X_raw.shape)
        print("Duration [s]:", X_raw.shape[0] / sr)

    # Skip potatoes that require sensors absent from this recording.
    config = [
        cfg for cfg in IRPF_CONFIG
        if cfg["channels"] is None
        or all(channel.upper() in sensors for channel in cfg["channels"])
    ]
    if not config:
        raise ValueError("No potato configurations match the sensors in this recording")
    if verbose:
        print("Potatoes used:", [cfg["name"] for cfg in config])

    potato_configs = [cfg_to_potato_config(cfg, sensors, PotatoConfig) for cfg in config]

    irpf = FullIRPF(
        sensors=sensors,
        potato_configs=potato_configs,
        sr=sr,
        wl=wl,
        slide=slide,
        filter_order=4,
        minsteps=2,
        reg=0.0,
        diag_mode="paper",
        frms_mode="paper",
        comb_funcs=("liptak", "fisher"),
        meta_combination="tippett",
        upper_limit=gfrms_upper_limit,
        global_prefilter=None,
        use_gfrms_for_final_mask=False,
        verbose=verbose,
    )
    result = irpf.fit_transform(X_raw)

    epochs = make_epochs(X_raw, wl=wl, slide=slide)
    order = np.asarray(result.sortkey, dtype=int)
    if len(order) != len(epochs):
        raise ValueError("FullIRPF sortkey length does not match the epoch count")
    sorted_epochs = epochs[order]
    sorted_sqi = np.asarray(result.sorted_sqi)
    if result.knee_index is None:
        threshold_epoch = int(np.sum(~np.asarray(result.clean_mask)))
    else:
        threshold_epoch = int(result.knee_index)
    threshold_epoch = max(1, min(threshold_epoch, len(sorted_epochs)))

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_sqi, ".-", markersize=3, label="iRPF SQI / p-value")
        plt.axhline(
            result.p_threshold,
            linestyle="--",
            color="black",
            label=f"p-th = {result.p_threshold:.3g}",
        )
        plt.axvline(
            threshold_epoch,
            linestyle="--",
            color="red",
            label="soglia iRPF / Kneedle",
        )
        plt.yscale("log")
        plt.xlabel("Epoche ordinate")
        plt.ylabel("iRPF combined p-value / SQI")
        plt.title("iRPF: epoche ordinate da più sporche a più pulite")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    selected_epochs, suffix = _select_epochs(
        sorted_epochs,
        threshold_epoch,
        epoch_save_mode,
        n_best_epochs,
        save_from_ordered_epoch,
    )
    if selected_epochs.size == 0:
        raise ValueError("Epoch selection is empty; nothing to save")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_continuous = np.transpose(selected_epochs, (0, 2, 1)).reshape(
        -1, len(sensors)
    )
    output_file = output_dir / f"{eeg_file.stem}_{suffix}.txt"
    with output_file.open("w", encoding="utf-8") as stream:
        stream.write("# " + "\t".join(sensors) + "\n")
        np.savetxt(stream, selected_continuous, delimiter="\t", fmt="%.10f")

    results_dir = output_dir / f"{eeg_file.stem}_full_irpf_results"
    save_result_outputs(result, results_dir, prefix="full_irpf_python_only")

    if verbose:
        print("Epoche totali:", len(sorted_epochs))
        print("Epoche selezionate:", len(selected_epochs))
        print("Modalità salvataggio:", epoch_save_mode)
        print("Soglia epoca:", threshold_epoch)
        print("File EEG pulito:", output_file)
        print("Risultati numerici:", results_dir)

    return {
        "result": result,
        "sensors": sensors,
        "epochs_sorted": sorted_epochs,
        "sorted_sqi": sorted_sqi,
        "threshold_epoch": threshold_epoch,
        "selected_epochs": selected_epochs,
        "output_file": output_file,
        "results_dir": results_dir,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    run_pipeline()


if __name__ == "__main__":
    main()