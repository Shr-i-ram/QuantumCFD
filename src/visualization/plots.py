# src/visualization/plots.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def _prepare_save_path(
    save_path,
):
    if save_path is None:
        return None

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return save_path


def save_or_show(
    fig,
    save_path=None,
    show=True,
    dpi=300,
):
    """
    Common figure handler.
    """

    save_path = _prepare_save_path(
        save_path
    )

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    else:
        plt.close(fig)


# ============================================================
# Scalar Field Plot
# ============================================================

def plot_scalar_field(
    field,
    title="Field",
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Generic scalar field visualization.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        field,
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )

    ax.set_title(title)

    fig.colorbar(
        im,
        ax=ax,
    )

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Velocity Magnitude
# ============================================================

def plot_velocity_magnitude(
    u,
    v,
    title="Velocity Magnitude",
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    |u|
    """

    vel_mag = np.sqrt(
        u**2 + v**2
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        vel_mag,
        origin="lower",
        cmap=cmap,
        aspect="auto",
    )

    ax.set_title(title)

    fig.colorbar(
        im,
        ax=ax,
        label="|u|",
    )

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Vorticity
# ============================================================

def plot_vorticity(
    omega,
    title="Vorticity",
    cmap="RdBu_r",
    save_path=None,
    show=True,
):
    """
    Vorticity field.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    vmax = np.max(
        np.abs(omega)
    )

    im = ax.imshow(
        omega,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_title(title)

    fig.colorbar(
        im,
        ax=ax,
        label="ω",
    )

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Velocity Quiver
# ============================================================

def plot_velocity_vectors(
    u,
    v,
    stride=2,
    title="Velocity Field",
    save_path=None,
    show=True,
):
    """
    Quiver plot.
    """

    ny, nx = u.shape

    x = np.arange(nx)
    y = np.arange(ny)

    X, Y = np.meshgrid(
        x,
        y,
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
    )

    ax.set_title(title)

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Prediction vs Reference
# ============================================================

def plot_comparison(
    prediction,
    reference,
    pred_title="Prediction",
    ref_title="Reference",
    cmap="viridis",
    save_path=None,
    show=True,
):
    """
    Side-by-side comparison.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    vmin = min(
        prediction.min(),
        reference.min(),
    )

    vmax = max(
        prediction.max(),
        reference.max(),
    )

    im0 = axes[0].imshow(
        prediction,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(
        pred_title
    )

    axes[1].imshow(
        reference,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title(
        ref_title
    )

    fig.colorbar(
        im0,
        ax=axes.ravel().tolist(),
    )

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Error Map
# ============================================================

def plot_error_map(
    prediction,
    reference,
    title="Absolute Error",
    save_path=None,
    show=True,
):
    """
    |prediction-reference|
    """

    error = np.abs(
        prediction
        - reference
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(
        error,
        origin="lower",
        cmap="inferno",
    )

    ax.set_title(title)

    fig.colorbar(
        im,
        ax=ax,
    )

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Training Curves
# ============================================================

def plot_loss_history(
    losses,
    title="Training Loss",
    log_scale=True,
    save_path=None,
    show=True,
):
    """
    Single loss curve.
    """

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(losses)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(title)

    ax.grid(True)

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Multi-Loss Curves
# ============================================================

def plot_training_history(
    history,
    title="Training History",
    save_path=None,
    show=True,
):
    """
    PINN/VQPINN histories.

    Expects:
    {
      total,
      ic,
      pde,
      continuity
    }
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for key, values in (
        history.items()
    ):

        ax.plot(
            values,
            label=key,
        )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(title)

    ax.legend()

    ax.grid(True)

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Energy Spectrum
# ============================================================

def plot_energy_spectrum(
    k,
    E,
    label="Spectrum",
    save_path=None,
    show=True,
):
    """
    Single energy spectrum.
    """

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.loglog(
        k,
        E,
        marker="o",
        label=label,
    )

    ax.set_xlabel(
        "k"
    )

    ax.set_ylabel(
        "E(k)"
    )

    ax.set_title(
        "Energy Spectrum"
    )

    ax.grid(True)

    ax.legend()

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Spectrum Comparison
# ============================================================

def plot_spectrum_comparison(
    k_ref,
    E_ref,
    k_pred,
    E_pred,
    ref_label="Reference",
    pred_label="Prediction",
    save_path=None,
    show=True,
):
    """
    Compare energy spectra.
    """

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.loglog(
        k_ref,
        E_ref,
        marker="o",
        label=ref_label,
    )

    ax.loglog(
        k_pred,
        E_pred,
        marker="s",
        label=pred_label,
    )

    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")

    ax.set_title(
        "Energy Spectrum Comparison"
    )

    ax.legend()
    ax.grid(True)

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Model Benchmark Bar Plot
# ============================================================

def plot_model_comparison(
    dataframe,
    metric,
    title=None,
    save_path=None,
    show=True,
):
    """
    Compare models from benchmark dataframe.
    """

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        dataframe["model"],
        dataframe[metric],
    )

    ax.set_ylabel(metric)

    if title is None:
        title = (
            f"Model Comparison "
            f"({metric})"
        )

    ax.set_title(title)

    ax.grid(
        axis="y"
    )

    save_or_show(
        fig,
        save_path,
        show,
    )


# ============================================================
# Taylor-Green Dashboard
# ============================================================

def plot_taylor_green_summary(
    u,
    v,
    omega,
    save_path=None,
    show=True,
):
    """
    Compact dashboard used
    across notebooks.
    """

    vel_mag = np.sqrt(
        u**2 + v**2
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    im0 = axes[0].imshow(
        vel_mag,
        origin="lower",
        cmap="viridis",
    )

    axes[0].set_title(
        "Velocity Magnitude"
    )

    fig.colorbar(
        im0,
        ax=axes[0],
    )

    vmax = np.max(
        np.abs(omega)
    )

    im1 = axes[1].imshow(
        omega,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    axes[1].set_title(
        "Vorticity"
    )

    fig.colorbar(
        im1,
        ax=axes[1],
    )

    save_or_show(
        fig,
        save_path,
        show,
    )