
INTENSE Signal Preprocessing Toolkit
A collection of preprocessing methods, pipelines, and utilities for physiological signals, with a particular focus on EEG.
The repository brings together implementations, ports, and adaptations of state-of-the-art signal processing approaches developed or maintained within the INTENSE project. The aim is to provide reproducible, documented, and reusable preprocessing workflows that can be integrated into research and applied machine learning pipelines.
Each method is documented with its scientific background, implementation details, requirements, parameters, and known limitations.
Project status: methods are under active development and documentation may be incomplete in some modules. Please consult the README available in each method's folder before applying a pipeline to new datasets.

Available Methods
The repository currently includes several families of preprocessing approaches.
Potato-Based Artifact Detection
Tools based on Potato Field and Riemannian geometry for EEG artifact detection and quality assessment.
Typical applications include:

- automatic artifact identification;
- signal quality monitoring;
- rejection of contaminated EEG epochs.
The dedicated folder contains configurations, examples, and utilities for integrating Potato-based processing into existing pipelines.

Riemannian Co-Registration (RCoR)
Tools for aligning and registering EEG representations across subjects, sessions, or recording conditions using Riemannian geometry.
Available implementations may include different registration strategies and optimization approaches, together with practical examples and benchmarking scripts.


MEMD-ASR
Implementation and integration of preprocessing pipelines combining:

- MEMD (Multivariate Empirical Mode Decomposition)
- ASR (Artifact Subspace Reconstruction)
These methods are primarily intended for artifact removal and signal cleaning in EEG recordings.
Documentation specifies dependencies, supported variants, and recommended parameter settings.

Additional Methods
The repository is designed to host additional preprocessing techniques, including:

- filtering and denoising methods;
- artifact-removal pipelines;
- signal quality assessment tools;
- feature-oriented preprocessing procedures;
- multimodal physiological signal processing workflows.

Repository Structure
INTENSE_Preprocessing/

├── Potato/
│   ├── README.md
│   ├── requirements.txt
│   ├── examples/
│   └── ...
│
├── RCoR/
│   ├── README.md
│   ├── julia/
│   ├── python/
│   ├── examples/
│   └── ...
│
├── MEMD-ASR/
│   ├── README.md
│   └── ...
│
└── ...
The structure will evolve as new methods and preprocessing pipelines are added.

Installation
Each preprocessing method may have specific dependencies.
Install the requirements provided within the corresponding folder. For example:
cd Potato
pip install -r requirements.txt
Method-specific instructions, notebooks, and examples are provided alongside each implementation.

Data Validation and Usage Guidelines
These tools are intended for research and development purposes.
Before applying a method to a new dataset, ensure that:

- data format and channel ordering are correct;
- signal units and sampling rates are consistent with the method requirements;
- assumptions behind the preprocessing procedure are appropriate for the target data;
- outputs are validated against references, publications, or established implementations when possible.
For reproducibility, always document:

- preprocessing parameters;
- software versions;
- channel selections;
- filtering settings;
- artifact rejection criteria.
Do not upload identifiable recordings, credentials, or other sensitive data to the repository. Whenever possible, provide instructions for obtaining or preparing example datasets instead of distributing raw data.

License
Please verify both the repository license and the licenses of any original implementations, ports, or source projects before redistribution.
Appropriate attribution should always be maintained for third-party methods and software

About the Authors and Acknowledgments
This repository was developed within the INTENSE Project, whose objective is the development and validation of advanced methods for physiological signal processing and analysis.
The work has been financially supported by the Ministry of Enterprises and Made in Italy (MIMIT).
The repository collects original implementations, adaptations, and ports of methods proposed in the scientific literature. Individual modules may contain additional acknowledgments and references specific to the original authors and software projects from which they are derived.
For authorship information, please refer to the repository contributors and commit history.

Contributing
Contributions, suggestions, bug reports, and feature requests are welcome.
If you encounter a problem, please open a GitHub Issue and provide:

- the affected method or preprocessing pipeline;
- software version and dependency information;
- operating system details;
- a minimal reproducible example;
- expected and observed behavior.
To protect privacy and ensure compliance with data-protection regulations, please do not upload identifiable recordings, private datasets, credentials, or proprietary validation artifacts.

References
Each method folder contains references to the original scientific publications and software projects on which the implementation is based.
Repository-wide references may be listed here as the project evolves.
Example:

- Schiavon, J. (2022). Geometria differenziale delle matrici simmetriche e definite positive per applicazioni statistiche. PhD Thesis, University of Padua, Italy. https://hdl.handle.net/11577/3449438
Additional references are provided within the documentation of the corresponding methods.
