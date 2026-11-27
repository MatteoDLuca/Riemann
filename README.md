<p align="center">
  <img src="RCoR/assets/intense-logo.png" alt="INTENSE Project Logo" width="220"/>
</p>

# INTENSE Physiological Signal Processing Toolkit

A collection of physiological signal processing algorithms, artifact-removal pipelines, and research utilities, with a primary focus on electroencephalography (EEG) and related biosignals.

This repository brings together implementations, ports, adaptations, and reproducible workflows developed or maintained within the **INTENSE Project**. It spans complementary methodological families—including **Riemannian geometry on symmetric positive-definite (SPD) manifolds** and **multivariate signal decomposition / subspace reconstruction**—to support standardized preprocessing, quality control, and downstream machine learning applications.

Each module is self-contained and documented with its theoretical background, software dependencies, parameter settings, usage examples, and known limitations.

> **Project status:** Modules are under active research development and may vary in maturity or supported languages (e.g., Python, Julia). Please consult the `README.md` inside each method's folder before applying a pipeline to new datasets.

---

## Table of Contents

- [Available Methods](#-available-methods)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Data Validation and Usage Guidelines](#-data-validation-and-usage-guidelines)
- [About the Authors and Acknowledgments](#-about-the-authors-and-acknowledgments)
- [Contributing](#-contributing)
- [References](#-references)
- [License](#-license)

---

# Available Methods

The repository currently organizes core signal processing contributions into three main modules, alongside space for future extensions:

### Potato-Based Artifact Detection (`Potato/`)
Methods based on *Potato Field* concepts and Riemannian distance metrics on covariance matrices for automated EEG quality assessment and outlier rejection.
- **Main use cases:** automatic detection of artifact-contaminated epochs, continuous signal quality monitoring, and baseline-referenced outlier filtering.
- **Contents:** modular Python pipeline (`potato_config.py`, `irpf_utils.py`, `run_irpf.py`, `full_irpf_python_only.py`), diagnostic plotting utilities (`diagnostic_plot.py`), and an interactive walkthrough (`example_irpf.ipynb`).

### Riemannian Co-Registration (`RCoR/`)
Tools for co-registering collections of real symmetric positive-definite (SPD) matrices into a common orthogonal eigenvector basis while resolving eigenvector sign ambiguities via nuclear-norm maximization.
- **Main use cases:** cross-subject, cross-session, or cross-condition alignment of covariance representations and computation of Riemannian-aligned reference matrices.
- **Contents:** independent **Python** (`python/rcor_nuclear/`) and **Julia** (`julia/RR_nuclear_optimized_final.jl`) implementations, synthetic benchmark datasets, and comparative visualization scripts (`examples/`).

### MEMD-ASR (`MEMD-ASR/`)
Preprocessing and artifact-attenuation pipelines combining **Multivariate Empirical Mode Decomposition (MEMD)** with **Artifact Subspace Reconstruction (ASR)**.
- **Main use cases:** data-driven multi-scale decomposition and non-stationary artifact suppression in low- and high-density EEG recordings without relying on fixed basis functions.
- **Contents:** implementation variants, parameter guidelines, and usage examples detailed in the module's dedicated documentation.

### Planned & Extensible Modules
The repository is structured to accommodate additional physiological signal processing workflows, including:
- spectral parameterization, filtering, and narrowband denoising utilities;
- multimodal biosignal synchronization and quality control tools;
- feature extraction and tangent-space / manifold representations for machine learning.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)

---

## Repository Structure

```text
Riemann/  # (or INTENSE-Physiological-Signal-Processing)
├── Potato/
│   ├── README.md
│   ├── requirements.txt
│   ├── potato_config.py
│   ├── irpf_utils.py
│   ├── diagnostic_plot.py
│   ├── full_irpf_python_only.py
│   ├── run_irpf.py
│   └── example_irpf.ipynb
│
├── RCoR/
│   ├── README.md
│   ├── pyproject.toml
│   ├── assets/
│   ├── examples/
│   ├── julia/
│   └── python/
│       └── rcor_nuclear/
│
├── MEMD-ASR/
│   ├── README.md
│   └── ...
│
├── LICENSE
└── README.md
```

## Installation

Each method may have specific dependencies. Install the requirements provided within the corresponding folder. For example:

```bash
cd Potato
pip install -r requirements.txt
```

Or, for the Python package in RCoR/:

```bash
cd RCoR
python -m pip install -e ".[plot]"
```

Method-specific instructions, Julia setup details, notebooks, and examples are provided alongside each implementation.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)

## Data Validation and Usage Guidelines
These tools are intended for research and development purposes. Before applying a method to a new dataset, ensure that:
- data format and channel ordering are correct;
- signal units and sampling rates are consistent with the method requirements;
- assumptions behind the processing procedure are appropriate for the target data;
- outputs are validated against references, publications, or established implementations whenever possible.

For reproducibility, always document:
- preprocessing parameters;
- software versions;
- channel selections;
- filtering settings;
- artifact rejection criteria.

Do not upload identifiable recordings, credentials, or other sensitive data to the repository. Whenever possible, provide instructions for obtaining or preparing example datasets instead of distributing raw data.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)

## About the Authors and Acknowledgments
This repository was developed within the INTENSE Project, whose objective is the development and validation of advanced methods for physiological signal processing and analysis.

The work has been financially supported by the Ministry of Enterprises and Made in Italy (MIMIT).

The repository collects original implementations, adaptations, and ports of methods proposed in the scientific literature. Individual modules may contain additional acknowledgments and references specific to the original authors and software projects from which they are derived. For authorship information, please refer to the repository contributors and commit history.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)

## Contributing
Contributions, suggestions, bug reports, and feature requests are welcome. If you encounter a problem, please open a GitHub Issue and provide:
- the affected method or processing pipeline;
- software version and dependency information;
- operating system details;
- a minimal reproducible example;
- expected and observed behavior.

To protect privacy and ensure compliance with data-protection regulations, please do not upload identifiable recordings, private datasets, credentials, or proprietary validation artifacts.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)

## References
Each method folder contains references to the original scientific publications and software projects on which the implementation is based. Repository-wide references may be listed here as the project evolves.

General References

Schiavon, J. (2022). Geometria differenziale delle matrici simmetriche e definite positive per applicazioni statistiche. PhD Thesis, University of Padua, Italy. https://hdl.handle.net/11577/3449438

Additional references are provided within the documentation of the corresponding methods.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)

## License
Please verify both the repository license and the licenses of any original implementations, ports, or source projects before redistribution. Appropriate attribution should always be maintained for third-party methods and software.

The license adopted by this repository is specified in the LICENSE file.

[▲ Back to top](#intense-physiological-signal-processing-toolkit)
