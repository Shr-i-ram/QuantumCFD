# src/evaluation/interpolation.py

from __future__ import annotations

import numpy as np

from scipy.interpolate import (
    RegularGridInterpolator,
    griddata,
)


# ============================================================
# Structured Grid Interpolation
# ============================================================

def interpolate_field(
    field: np.ndarray,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    method: str = "linear",
):
    """
    Interpolate a field from one structured grid
    to another structured grid.

    Parameters
    ----------
    field : ndarray
        Source field (nx, ny)

    x_source : ndarray
        Source X meshgrid

    y_source : ndarray
        Source Y meshgrid

    x_target : ndarray
        Target X meshgrid

    y_target : ndarray
        Target Y meshgrid

    method : str
        linear | nearest

    Returns
    -------
    ndarray
        Interpolated field
    """

    x = x_source[:, 0]
    y = y_source[0, :]

    interpolator = RegularGridInterpolator(
        (x, y),
        field,
        method=method,
        bounds_error=False,
        fill_value=None,
    )

    points = np.column_stack(
        [
            x_target.ravel(),
            y_target.ravel(),
        ]
    )

    values = interpolator(points)

    return values.reshape(
        x_target.shape
    )


# ============================================================
# Multiple Field Interpolation
# ============================================================

def interpolate_solution(
    solution: dict,
    x_target: np.ndarray,
    y_target: np.ndarray,
    method: str = "linear",
):
    """
    Interpolate all fields in a solution dict.

    Expected keys:
        x
        y
        u
        v
        omega
        psi

    Returns
    -------
    dict
    """

    x_src = solution["x"]
    y_src = solution["y"]

    out = {
        "x": x_target,
        "y": y_target,
    }

    for key in solution:

        if key in ["x", "y", "t"]:
            continue

        out[key] = interpolate_field(
            solution[key],
            x_src,
            y_src,
            x_target,
            y_target,
            method=method,
        )

    if "t" in solution:
        out["t"] = solution["t"]

    return out


# ============================================================
# Unstructured -> Structured
# ============================================================

def scattered_to_grid(
    x_points: np.ndarray,
    y_points: np.ndarray,
    values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    method: str = "linear",
):
    """
    Convert scattered data to a structured grid.

    Useful for:
        PINN predictions
        VQPINN predictions

    Parameters
    ----------
    x_points : (N,)
    y_points : (N,)
    values   : (N,)

    Returns
    -------
    field : (nx, ny)
    """

    pts = np.column_stack(
        [
            x_points,
            y_points,
        ]
    )

    field = griddata(
        pts,
        values,
        (x_grid, y_grid),
        method=method,
    )

    return field


# ============================================================
# PINN Prediction Reshaping
# ============================================================

def pinn_prediction_to_grid(
    x_points: np.ndarray,
    y_points: np.ndarray,
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
):
    """
    Convert PINN output vectors into
    structured CFD grids.

    Returns
    -------
    u_grid
    v_grid
    """

    u_grid = scattered_to_grid(
        x_points,
        y_points,
        u_pred,
        x_grid,
        y_grid,
    )

    v_grid = scattered_to_grid(
        x_points,
        y_points,
        v_pred,
        x_grid,
        y_grid,
    )

    return u_grid, v_grid


# ============================================================
# Reference Alignment
# ============================================================

def align_to_reference_grid(
    field: np.ndarray,
    x_field: np.ndarray,
    y_field: np.ndarray,
    x_reference: np.ndarray,
    y_reference: np.ndarray,
):
    """
    Interpolate arbitrary field
    onto reference CFD grid.

    Used by:
        PINN evaluation
        VQPINN evaluation
        VQLS evaluation
    """

    return interpolate_field(
        field,
        x_field,
        y_field,
        x_reference,
        y_reference,
    )


# ============================================================
# Downsampling
# ============================================================

def downsample_field(
    field: np.ndarray,
    factor: int,
):
    """
    Uniform downsampling.

    Example
    -------
    128x128 -> factor=4 -> 32x32
    """

    return field[::factor, ::factor]


def downsample_solution(
    solution: dict,
    factor: int,
):
    """
    Downsample every field
    in a solution dictionary.
    """

    out = {}

    for key, value in solution.items():

        if isinstance(
            value,
            np.ndarray,
        ) and value.ndim == 2:

            out[key] = downsample_field(
                value,
                factor,
            )

        else:

            out[key] = value

    return out


# ============================================================
# Common Benchmark Helper
# ============================================================

def match_grids(
    pred_field: np.ndarray,
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
):
    """
    Convenience wrapper.

    Returns prediction interpolated
    onto the reference grid.
    """

    return interpolate_field(
        pred_field,
        pred_x,
        pred_y,
        ref_x,
        ref_y,
    )


# ============================================================
# Resolution Utilities
# ============================================================

def same_resolution(
    a: np.ndarray,
    b: np.ndarray,
):
    """
    Check if two fields already
    share the same grid shape.
    """

    return a.shape == b.shape


def resolution_ratio(
    coarse: np.ndarray,
    fine: np.ndarray,
):
    """
    Compute refinement ratio.

    Example
    -------
    32x32 vs 128x128 -> 4
    """

    return (
        fine.shape[0]
        // coarse.shape[0]
    )