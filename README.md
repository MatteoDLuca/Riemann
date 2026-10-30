# Python Porting of Riemannian Methods for Biological Signals

This repository brings together Python ports and tools based on Riemannian geometry for processing biological signals, with a particular focus on EEG.

The goal is to make these methods more accessible and reusable in Python workflows, while documenting each implementation's reference algorithm, requirements, parameters, and limitations.

> **Project status:** implementations and documentation may be at different stages of maturity. Check the README in each method's folder before using it with new data.

## Methods

The repository currently covers four areas:

### Potatoes

Ports and tools for methods based on *Potato Field* and Riemannian geometry, applied to EEG artifact detection. The `Potato/` folder contains the Python pipeline, potato configurations, and an example notebook.

### Peak Alignment

An implementation for aligning peaks in biological signals. The method-specific documentation will describe the reference algorithm, expected inputs, and output format.

### Riemannian Registration

Tools for registering signals or representations using Riemannian distances and geometry. The relevant folder will document the available modes and parameters.

### MEMD-ASR

A port or integration of a pipeline combining **MEMD** (*Multivariate Empirical Mode Decomposition*) and **ASR** (*Artifact Subspace Reconstruction*) for processing biological signals, particularly EEG. The dedicated folder will describe the implemented variants and required dependencies.

## Repository structure

```text
Riemann/
├── Potato/
│   ├── README.md
│   ├── requirements.txt
│   ├── example_irpf.ipynb
│   └── ...
├── PeakAlignment/          # folder name to confirm
├── RiemannianRegistration/ # folder name to confirm
└── MEMD-ASR/               # folder name to confirm
```

The folder names shown for methods that are not yet documented are placeholders. Update them to match the actual repository structure.

## Requirements and installation

Each method may have its own dependencies. Install the packages listed in that method's folder. For example, for the Potato project:

```bash
cd Potato
pip install -r requirements.txt
```

Instructions for using notebooks, scripts, or Python APIs are kept alongside each implementation.

## Data and validation

These tools are intended for research and development. Before applying them to a new dataset:

- verify the data format, units, sampling rate, and channel order;
- check that the method's assumptions suit the data;
- compare results with appropriate implementations or references;
- document the preprocessing choices and parameters used.

Do not add identifiable EEG recordings, credentials, or other sensitive data to the repository. Where possible, provide instructions for obtaining or preparing example data instead.

## References

Each method folder should cite the scientific papers and software projects on which the method or port is based. Add repository-wide references here when available.

## Contributing

Contributions, bug reports, and suggestions are welcome. To make reports easier to review, include:

- the affected method and version;
- operating system and Python/dependency versions;
- a minimal reproducible example that contains no sensitive data;
- the expected and observed results.

## License

The repository license and the licenses of the original methods or source code must be checked before distribution. Add the selected license here and provide proper attribution for each port.
