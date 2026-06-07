# src/evaluation/benchmark.py

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from scipy.interpolate import (
    RegularGridInterpolator,
)

from src.evaluation.metrics import (
    l2_error,
    linf_error,
    correlation_coefficient,
)

from src.physics.spectral import (
    compute_vorticity_spectral,
    energy_spectrum,
)

from src.physics.taylor_green import (
    tg_velocity_np,
)


# ============================================================
# Reference Interpolation
# ============================================================

def interpolate_reference_to_grid(
    u_ref,
    v_ref,
    x_target,
    y_target,
    domain_length=2.0 * np.pi,
):
    """
    Interpolate reference velocity field
    onto target grid.
    """

    nx_ref, ny_ref = u_ref.shape

    xr = np.linspace(
        0,
        domain_length,
        nx_ref,
        endpoint=False,
    )

    yr = np.linspace(
        0,
        domain_length,
        ny_ref,
        endpoint=False,
    )

    interp_u = RegularGridInterpolator(
        (xr, yr),
        u_ref,
    )

    interp_v = RegularGridInterpolator(
        (xr, yr),
        v_ref,
    )

    pts = np.stack(
        [x_target.ravel(), y_target.ravel()],
        axis=-1,
    )

    u = interp_u(pts).reshape(
        x_target.shape
    )

    v = interp_v(pts).reshape(
        x_target.shape
    )

    return u, v


# ============================================================
# Reference Loader
# ============================================================

def load_reference_data(
    filepath,
):
    """
    Load reference_data_safe.npz
    """

    data = np.load(
        filepath,
        allow_pickle=True,
    )

    return {
        key: data[key]
        for key in data.files
    }


# ============================================================
# Velocity Metrics
# ============================================================

def velocity_metrics(
    u_pred,
    v_pred,
    u_ref,
    v_ref,
):
    """
    Standard benchmark metrics.
    """

    return {
        "u_l2":
            l2_error(
                u_pred,
                u_ref,
            ),

        "u_linf":
            linf_error(
                u_pred,
                u_ref,
            ),

        "u_corr":
            correlation_coefficient(
                u_pred,
                u_ref,
            ),

        "v_l2":
            l2_error(
                v_pred,
                v_ref,
            ),

        "v_linf":
            linf_error(
                v_pred,
                v_ref,
            ),

        "v_corr":
            correlation_coefficient(
                v_pred,
                v_ref,
            ),
    }


# ============================================================
# Vorticity Metrics
# ============================================================

def vorticity_metrics(
    omega_pred,
    omega_ref,
):
    """
    Vorticity benchmark metrics.
    """

    return {
        "omega_l2":
            l2_error(
                omega_pred,
                omega_ref,
            ),

        "omega_linf":
            linf_error(
                omega_pred,
                omega_ref,
            ),

        "omega_corr":
            correlation_coefficient(
                omega_pred,
                omega_ref,
            ),
    }


# ============================================================
# Spectrum Metrics
# ============================================================

def spectrum_metrics(
    u_pred,
    v_pred,
    u_ref,
    v_ref,
):
    """
    Compare energy spectra.
    """

    k_pred, E_pred = (
        energy_spectrum(
            u_pred,
            v_pred,
        )
    )

    k_ref, E_ref = (
        energy_spectrum(
            u_ref,
            v_ref,
        )
    )

    n = min(
        len(E_pred),
        len(E_ref),
    )

    spec_l2 = np.linalg.norm(
        E_pred[:n]
        - E_ref[:n]
    ) / (
        np.linalg.norm(
            E_ref[:n]
        )
        + 1e-12
    )

    return {
        "spectrum_l2":
            float(spec_l2),
        "k_pred":
            k_pred,
        "E_pred":
            E_pred,
        "k_ref":
            k_ref,
        "E_ref":
            E_ref,
    }


# ============================================================
# Runtime Benchmark
# ============================================================

def benchmark_runtime(
    func,
    *args,
    **kwargs,
):
    """
    Measure wall-clock runtime.
    """

    t0 = time.time()

    result = func(
        *args,
        **kwargs,
    )

    runtime = (
        time.time() - t0
    )

    return result, runtime


# ============================================================
# Taylor-Green Benchmark
# ============================================================

def benchmark_taylor_green(
    u_pred,
    v_pred,
    x,
    y,
    t,
    nu,
):
    """
    Compare prediction against
    exact Taylor-Green vortex.
    """

    u_ref, v_ref = (
        tg_velocity_np(
            x,
            y,
            t,
            nu=nu,
        )
    )

    return velocity_metrics(
        u_pred,
        v_pred,
        u_ref,
        v_ref,
    )


# ============================================================
# Reference Dataset Benchmark
# ============================================================

def benchmark_against_reference(
    u_pred,
    v_pred,
    x,
    y,
    reference_path,
):
    """
    Compare against
    reference_data_safe.npz.
    """

    ref = load_reference_data(
        reference_path,
    )

    u_ref = ref["u"]
    v_ref = ref["v"]

    u_ref_grid, v_ref_grid = (
        interpolate_reference_to_grid(
            u_ref,
            v_ref,
            x,
            y,
        )
    )

    omega_pred = (
        compute_vorticity_spectral(
            u_pred,
            v_pred,
        )
    )

    omega_ref = (
        compute_vorticity_spectral(
            u_ref_grid,
            v_ref_grid,
        )
    )

    results = {}

    results.update(
        velocity_metrics(
            u_pred,
            v_pred,
            u_ref_grid,
            v_ref_grid,
        )
    )

    results.update(
        vorticity_metrics(
            omega_pred,
            omega_ref,
        )
    )

    results.update(
        spectrum_metrics(
            u_pred,
            v_pred,
            u_ref_grid,
            v_ref_grid,
        )
    )

    return results


# ============================================================
# Model Comparison Table
# ============================================================

def compare_models(
    model_results: dict,
):
    """
    Convert benchmark results
    into dataframe.

    Example
    -------
    {
      "PINN": {...},
      "VQPINN": {...},
      "VQLS": {...}
    }
    """

    rows = []

    for name, metrics in (
        model_results.items()
    ):

        row = {
            "model": name,
        }

        row.update(metrics)

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Summary Printer
# ============================================================

def print_benchmark_report(
    metrics,
):
    """
    Console report.
    """

    print("=" * 60)
    print("Benchmark Report")
    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            print(
                f"{key:20s}: "
                f"{value:.6e}"
            )

    print("=" * 60)