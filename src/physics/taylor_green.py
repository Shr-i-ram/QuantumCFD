# src/physics/taylor_green.py

from __future__ import annotations

import numpy as np
import torch

from src.config import (
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    KINEMATIC_VISCOSITY,
)


# ============================================================
# NumPy Implementations
# ============================================================

def tg_velocity_np(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Analytical Taylor-Green velocity field.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    t : float
    nu : float

    Returns
    -------
    u : ndarray
    v : ndarray
    """

    decay = np.exp(-2.0 * nu * t)

    u = np.sin(x) * np.cos(y) * decay
    v = -np.cos(x) * np.sin(y) * decay

    return u, v


def tg_pressure_np(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Analytical pressure field.
    """

    decay = np.exp(-4.0 * nu * t)

    p = -0.25 * (
        np.cos(2.0 * x) +
        np.cos(2.0 * y)
    ) * decay

    return p


def tg_vorticity_np(
    x: np.ndarray,
    y: np.ndarray,
    t: float = 0.0,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Analytical vorticity field.

    ω = dv/dx - du/dy
    """

    decay = np.exp(-2.0 * nu * t)

    omega = (
        2.0
        * np.sin(x)
        * np.sin(y)
        * decay
    )

    return omega


# ============================================================
# Torch Implementations
# ============================================================

def tg_velocity_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Differentiable Taylor-Green velocity.
    """

    decay = torch.exp(-2.0 * nu * t)

    u = torch.sin(x) * torch.cos(y) * decay
    v = -torch.cos(x) * torch.sin(y) * decay

    return u, v


def tg_pressure_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Differentiable Taylor-Green pressure.
    """

    decay = torch.exp(-4.0 * nu * t)

    p = -0.25 * (
        torch.cos(2.0 * x)
        + torch.cos(2.0 * y)
    ) * decay

    return p


def tg_vorticity_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Differentiable Taylor-Green vorticity.
    """

    decay = torch.exp(-2.0 * nu * t)

    omega = (
        2.0
        * torch.sin(x)
        * torch.sin(y)
        * decay
    )

    return omega


# ============================================================
# Initial Condition Helpers
# ============================================================

def initial_velocity_field(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Taylor-Green initial condition.
    """

    return tg_velocity_np(x, y, t=0.0)


def initial_vorticity_field(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Taylor-Green initial vorticity.
    """

    return tg_vorticity_np(x, y, t=0.0)


def initial_pressure_field(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Taylor-Green initial pressure.
    """

    return tg_pressure_np(x, y, t=0.0)


# ============================================================
# Energy / Enstrophy
# ============================================================

def kinetic_energy_density(
    u,
    v,
):
    """
    Pointwise kinetic energy density.
    """

    return 0.5 * (u**2 + v**2)


def total_kinetic_energy(
    u,
    v,
):
    """
    Domain averaged kinetic energy.
    """

    return np.mean(
        kinetic_energy_density(u, v)
    )


def enstrophy(
    omega,
):
    """
    Domain averaged enstrophy.
    """

    return 0.5 * np.mean(omega**2)


# ============================================================
# Decay Factors
# ============================================================

def velocity_decay_factor(
    t: float,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Velocity amplitude decay.
    """

    return np.exp(-2.0 * nu * t)


def pressure_decay_factor(
    t: float,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Pressure amplitude decay.
    """

    return np.exp(-4.0 * nu * t)


# ============================================================
# Grid Generation
# ============================================================

def generate_uniform_grid(
    nx: int,
    ny: int,
):
    """
    Periodic Taylor-Green grid.
    """

    x = np.linspace(
        0.0,
        DOMAIN_LENGTH_X,
        nx,
        endpoint=False,
    )

    y = np.linspace(
        0.0,
        DOMAIN_LENGTH_Y,
        ny,
        endpoint=False,
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    return X, Y


# ============================================================
# Exact Solution Dictionary
# ============================================================

def exact_solution(
    x,
    y,
    t,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Convenience wrapper used by PINN
    evaluation and benchmarking.

    Returns
    -------
    dict:
        {
            "u": ...,
            "v": ...,
            "p": ...,
            "omega": ...
        }
    """

    u, v = tg_velocity_np(x, y, t, nu)
    p = tg_pressure_np(x, y, t, nu)
    omega = tg_vorticity_np(x, y, t, nu)

    return {
        "u": u,
        "v": v,
        "p": p,
        "omega": omega,
    }