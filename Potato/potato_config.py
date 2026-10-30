"""Default iRPF potato definitions.

Edit ``IRPF_CONFIG`` to match the available sensors and the artifact families
of interest. The pipeline filters out configurations whose channels are absent.
"""

IRPF_CONFIG = [
    # OCULAR / BLINK
    {
        "name": "p1_ocular_cov_riemann",
        "family": "ocular",
        "channels": ["FP1", "FP2", "FZ"],
        "low_freq": 0.5,
        "high_freq": 10.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "frontal ocular covariance",
        "diag_only": False,
    },
    {
        "name": "p2_ocular_amp_euclid",
        "family": "ocular",
        "channels": ["FP1", "FP2", "FZ"],
        "low_freq": 0.5,
        "high_freq": 10.0,
        "metric": "euclid",
        "normalization": None,
        "target": "frontal ocular amplitude",
        "diag_only": False,
    },

    # LOCAL COVARIANCE / CONTACT / POP
    {
        "name": "p3_frontal_local_cov",
        "family": "local_cov",
        "channels": ["FP1", "FP2", "FZ"],
        "low_freq": 1.0,
        "high_freq": 20.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "frontal local covariance/contact",
        "diag_only": False,
    },
    {
        "name": "p4_central_local_cov",
        "family": "local_cov",
        "channels": ["C5", "CZ", "C6"],
        "low_freq": 1.0,
        "high_freq": 20.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "central local covariance/contact",
        "diag_only": False,
    },
    {
        "name": "p5_occipital_local_cov",
        "family": "local_cov",
        "channels": ["O1", "O2"],
        "low_freq": 1.0,
        "high_freq": 20.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "occipital local covariance/contact",
        "diag_only": False,
    },

    # SINGLE-CHANNEL LOW/MID TRANSIENTS
    {
        "name": "p6_frontal_single_lowmid",
        "family": "single_channel",
        "channels": ["FP1", "FP2", "FZ"],
        "low_freq": 0.5,
        "high_freq": 20.0,
        "metric": "euclid",
        "normalization": None,
        "target": "frontal single-channel low/mid transient",
        "diag_only": True,
    },
    {
        "name": "p7_central_single_lowmid",
        "family": "single_channel",
        "channels": ["C5", "CZ", "C6"],
        "low_freq": 0.5,
        "high_freq": 20.0,
        "metric": "euclid",
        "normalization": None,
        "target": "central single-channel low/mid transient",
        "diag_only": True,
    },
    {
        "name": "p8_occipital_single_lowmid",
        "family": "single_channel",
        "channels": ["O1", "O2"],
        "low_freq": 0.5,
        "high_freq": 20.0,
        "metric": "euclid",
        "normalization": None,
        "target": "occipital single-channel low/mid transient",
        "diag_only": True,
    },

    # HIGH FREQUENCY MUSCLE
    {
        "name": "p9_frontal_muscle_highfreq",
        "family": "muscle",
        "channels": ["FP1", "FP2", "FZ"],
        "low_freq": 20.0,
        "high_freq": 44.0,
        "metric": "euclid",
        "normalization": None,
        "target": "frontal high-frequency muscle",
        "diag_only": True,
    },
    {
        "name": "p10_posterior_muscle_highfreq",
        "family": "muscle",
        "channels": ["C5", "CZ", "C6", "O1", "O2"],
        "low_freq": 20.0,
        "high_freq": 44.0,
        "metric": "euclid",
        "normalization": None,
        "target": "central/posterior high-frequency muscle",
        "diag_only": True,
    },

    # GLOBAL ARTIFACTS
    {
        "name": "p11_global_cov_riemann",
        "family": "global",
        "channels": ["FP1", "FZ", "FP2", "C5", "CZ", "C6", "O1", "O2"],
        "low_freq": 0.5,
        "high_freq": 20.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "global covariance structure",
        "diag_only": False,
    },
    {
        "name": "p12_global_amp_euclid",
        "family": "global",
        "channels": ["FP1", "FZ", "FP2", "C5", "CZ", "C6", "O1", "O2"],
        "low_freq": 0.5,
        "high_freq": 20.0,
        "metric": "euclid",
        "normalization": None,
        "target": "global amplitude/covariance change",
        "diag_only": False,
    },
]
