# src/evaluation/metrics.py

from __future__ import annotations

import numpy as np

from src.physics.spectral import (
    divergence,
    energy_spectrum,
)


EPS = 1e-12


# ============================================================
# Basic Error Metrics
# ============================================================

def l2_error(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Absolute L2 error.
    """

    prediction = np.asarray(prediction)
    reference = np.asarray(reference)

    return np.linalg.norm(
        prediction - reference
    )


def relative_l2_error(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Relative L2 error.
    """

    prediction = np.asarray(prediction)
    reference = np.asarray(reference)

    numerator = np.linalg.norm(
        prediction - reference
    )

    denominator = (
        np.linalg.norm(reference)
        + EPS
    )

    return numerator / denominator


def linf_error(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Maximum absolute error.
    """

    prediction = np.asarray(prediction)
    reference = np.asarray(reference)

    return np.max(
        np.abs(prediction - reference)
    )


def mse(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Mean squared error.
    """

    prediction = np.asarray(prediction)
    reference = np.asarray(reference)

    return np.mean(
        (prediction - reference) ** 2
    )


def rmse(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Root mean squared error.
    """

    return np.sqrt(
        mse(
            prediction,
            reference,
        )
    )


def mae(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Mean absolute error.
    """

    prediction = np.asarray(prediction)
    reference = np.asarray(reference)

    return np.mean(
        np.abs(
            prediction - reference
        )
    )


# ============================================================
# Statistical Similarity
# ============================================================

def correlation(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Pearson correlation coefficient.
    """

    pred = np.asarray(
        prediction
    ).ravel()

    ref = np.asarray(
        reference
    ).ravel()

    pred_std = np.std(pred)
    ref_std = np.std(ref)

    if pred_std < EPS:
        return 0.0

    if ref_std < EPS:
        return 0.0

    return np.corrcoef(
        pred,
        ref,
    )[0, 1]


def cosine_similarity(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Cosine similarity.
    """

    pred = np.asarray(
        prediction
    ).ravel()

    ref = np.asarray(
        reference
    ).ravel()

    numerator = np.dot(
        pred,
        ref,
    )

    denominator = (
        np.linalg.norm(pred)
        * np.linalg.norm(ref)
        + EPS
    )

    return numerator / denominator


# ============================================================
# Fluid Metrics
# ============================================================

def kinetic_energy(
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """
    Domain-averaged kinetic energy.

    E = 1/2 <u² + v²>
    """

    return (
        0.5
        * np.mean(
            u**2 + v**2
        )
    )


def enstrophy(
    omega: np.ndarray,
) -> float:
    """
    Domain-averaged enstrophy.

    Ω = 1/2 <ω²>
    """

    return (
        0.5
        * np.mean(
            omega**2
        )
    )


def energy_error(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
) -> float:
    """
    Relative kinetic energy error.
    """

    e_pred = kinetic_energy(
        u_pred,
        v_pred,
    )

    e_ref = kinetic_energy(
        u_ref,
        v_ref,
    )

    return abs(
        e_pred - e_ref
    ) / (
        abs(e_ref)
        + EPS
    )


def enstrophy_error(
    omega_pred: np.ndarray,
    omega_ref: np.ndarray,
) -> float:
    """
    Relative enstrophy error.
    """

    ens_pred = enstrophy(
        omega_pred
    )

    ens_ref = enstrophy(
        omega_ref
    )

    return abs(
        ens_pred - ens_ref
    ) / (
        abs(ens_ref)
        + EPS
    )


# ============================================================
# Divergence Diagnostics
# ============================================================

def divergence_l2(
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """
    L2 divergence norm.
    """

    div = divergence(
        u,
        v,
    )

    return (
        np.linalg.norm(div)
        / np.sqrt(div.size)
    )


def divergence_linf(
    u: np.ndarray,
    v: np.ndarray,
) -> float:
    """
    Linf divergence norm.
    """

    div = divergence(
        u,
        v,
    )

    return np.max(
        np.abs(div)
    )


# ============================================================
# Spectrum Metrics
# ============================================================

def spectrum_l2_error(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
) -> float:
    """
    Compare isotropic energy spectra.
    """

    k1, E_pred = energy_spectrum(
        u_pred,
        v_pred,
    )

    k2, E_ref = energy_spectrum(
        u_ref,
        v_ref,
    )

    n = min(
        len(E_pred),
        len(E_ref),
    )

    return relative_l2_error(
        E_pred[:n],
        E_ref[:n],
    )


def spectrum_correlation(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
) -> float:
    """
    Correlation of isotropic spectra.
    """

    _, E_pred = energy_spectrum(
        u_pred,
        v_pred,
    )

    _, E_ref = energy_spectrum(
        u_ref,
        v_ref,
    )

    n = min(
        len(E_pred),
        len(E_ref),
    )

    return correlation(
        E_pred[:n],
        E_ref[:n],
    )


# ============================================================
# Field Comparison Helpers
# ============================================================

def field_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
):
    """
    Standard comparison bundle.
    """

    return {
        "l2": l2_error(
            prediction,
            reference,
        ),
        "relative_l2": relative_l2_error(
            prediction,
            reference,
        ),
        "linf": linf_error(
            prediction,
            reference,
        ),
        "rmse": rmse(
            prediction,
            reference,
        ),
        "mae": mae(
            prediction,
            reference,
        ),
        "correlation": correlation(
            prediction,
            reference,
        ),
        "cosine_similarity": cosine_similarity(
            prediction,
            reference,
        ),
    }


# ============================================================
# Velocity Benchmark
# ============================================================

def velocity_metrics(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
):
    """
    Combined velocity-field metrics.
    """

    u_stats = field_metrics(
        u_pred,
        u_ref,
    )

    v_stats = field_metrics(
        v_pred,
        v_ref,
    )

    return {
        "u": u_stats,
        "v": v_stats,
        "energy_error": energy_error(
            u_pred,
            v_pred,
            u_ref,
            v_ref,
        ),
        "divergence_l2": divergence_l2(
            u_pred,
            v_pred,
        ),
        "divergence_linf": divergence_linf(
            u_pred,
            v_pred,
        ),
        "spectrum_l2": spectrum_l2_error(
            u_pred,
            v_pred,
            u_ref,
            v_ref,
        ),
        "spectrum_corr": spectrum_correlation(
            u_pred,
            v_pred,
            u_ref,
            v_ref,
        ),
    }


# ============================================================
# Vorticity Benchmark
# ============================================================

def vorticity_metrics(
    omega_pred: np.ndarray,
    omega_ref: np.ndarray,
):
    """
    Vorticity benchmark bundle.
    """

    stats = field_metrics(
        omega_pred,
        omega_ref,
    )

    stats["enstrophy_error"] = (
        enstrophy_error(
            omega_pred,
            omega_ref,
        )
    )

    return stats


# ============================================================
# Summary Printer
# ============================================================

def print_metrics(
    metrics: dict,
):
    """
    Pretty print benchmark results.
    """

    for key, value in metrics.items():

        if isinstance(
            value,
            dict,
        ):

            print(f"\n{key}")

            for k, v in value.items():

                if isinstance(
                    v,
                    float,
                ):
                    print(
                        f"  {k:<20}: {v:.6e}"
                    )
                else:
                    print(
                        f"  {k:<20}: {v}"
                    )

        else:

            if isinstance(
                value,
                float,
            ):
                print(
                    f"{key:<20}: {value:.6e}"
                )
            else:
                print(
                    f"{key:<20}: {value}"
                )