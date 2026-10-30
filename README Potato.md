# iRPF EEG Artifact Detection Pipeline

This project organizes the Python-only iRPF pipeline into reusable modules, a separate potato configuration, and an example notebook. It is intended to be run from Jupyter or Google Colab.

## Project files

- `run_irpf.py`: loads the input files, runs `FullIRPF`, and saves the selected epochs and numeric results.
- `irpf_utils.py`: reusable helpers for filtering, epoching, GFRMS, covariance calculation, and configuration conversion.
- `potato_config.py`: the `IRPF_CONFIG` list defining potato channels, frequency bands, metrics, and targets.
- `diagnostic_plot.py`: the interactive `plot_diagnostics` function for inspecting SQI-sorted EEG and potato activations.
- `requirements.txt`: Python dependencies used by the included modules.
- `example_irpf.ipynb`: a guided example notebook for trying the pipeline with your own data.
- `full_irpf_python_only.py`: **required, but not included here**. Add the existing module that defines `FullIRPF`, `PotatoConfig`, and `save_result_outputs`.

Keep all Python modules in the same folder, or make that folder importable in Python.

## Installation

In Google Colab, upload or clone the project folder, then install the dependencies from a notebook cell:

```python
%pip install -r requirements.txt
```

After installation, a runtime restart may be needed for interactive plots. To enable the interactive backend, run this in a notebook cell before plotting:

```python
%matplotlib widget
```

Use `%matplotlib inline` if static plots are sufficient.

## Input files

Edit the settings near the top of `run_irpf.py`, or pass paths directly to `run_pipeline`:

```python
DB_DIR = Path("/content")
SENSOR_FILE = DB_DIR / "8 channels.txt"
EEG_FILE = DB_DIR / "recording.txt"
OUTPUT_DIR = Path("potato1")
```

The sensor file should contain channel names separated by spaces, commas, or tabs. A line containing only the channel count is ignored. The EEG file should have one sample per row and one channel per column; spaces, commas, and tabs are supported. Non-numeric header lines are skipped.

Check `SR`, `WL`, and `SLIDE`. `WL` and `SLIDE` are measured in samples. `SLIDE = 0` means non-overlapping epochs.

## Run from a notebook

With the project modules and the input files available in the working directory:

```python
from run_irpf import run_pipeline

run = run_pipeline(show_plot=True)
```

Paths and selection options can be passed explicitly:

```python
run = run_pipeline(
    sensor_file="/content/8 channels.txt",
    eeg_file="/content/recording.txt",
    output_dir="/content/cleaned_output",
    sr=512,
    wl=512,
    slide=0,
    epoch_save_mode="from_ordered_epoch",
    save_from_ordered_epoch=5,
)
```

Supported `epoch_save_mode` values:

- `"after_threshold"`: select epochs after the iRPF/Kneedle threshold.
- `"from_ordered_epoch"`: select from the 1-based ordered epoch given by `save_from_ordered_epoch`.
- `"best_n"`: select the `n_best_epochs` epochs with the highest SQI.
- `"all"`: select all epochs in SQI order.

`run_pipeline` returns a dictionary with the iRPF result, sorted epochs, threshold, and output paths.

## Interactive diagnostic plot

After running the pipeline:

```python
from diagnostic_plot import plot_diagnostics
from potato_config import IRPF_CONFIG
from run_irpf import SR

plot_diagnostics(
    result=run["result"],
    sorted_epochs=run["epochs_sorted"],
    sensors=run["sensors"],
    sr=SR,
    threshold_epoch=run["threshold_epoch"],
    potato_config=IRPF_CONFIG,
)
```

The plot shows the concatenated EEG in SQI order, a `-log10(p)` heatmap for each potato, and a table mapping ordered epochs to their original epoch and time.

## Customize the potatoes

Edit `IRPF_CONFIG` in `potato_config.py` to adjust channel groups, frequency bands, metrics, normalization, and artifact targets. Channel names must match the sensor file. `run_pipeline` skips configurations that require channels absent from the recording.

## Output files

The selected output directory contains:

- a text file with the selected epochs concatenated into a continuous signal, with channel names in a header;
- a `<recording>_full_irpf_results` folder containing numeric outputs written by `save_result_outputs`.

The cleaned signal filename includes the selection mode and selection range.
