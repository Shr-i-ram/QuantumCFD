# src/datasets/training_data.py

from __future__ import annotations

import numpy as np
import torch

from src.config import (
    DEVICE,
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
)

from src.physics.taylor_green import (
    tg_velocity_np,
    tg_pressure_np,
)


# ============================================================
# Random Sampling Utilities
# ============================================================

def sample_domain(
    n_points: int,
    t_final: float,
):
    """
    Uniformly sample interior collocation points.

    Returns
    -------
    x, y, t
        shape = (N,1)
    """

    x = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_X,
        (n_points, 1),
    )

    y = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_Y,
        (n_points, 1),
    )

    t = np.random.uniform(
        0.0,
        t_final,
        (n_points, 1),
    )

    return x, y, t


def sample_initial_condition(
    n_points: int,
):
    """
    Sample points at t=0.
    """

    x = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_X,
        (n_points, 1),
    )

    y = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_Y,
        (n_points, 1),
    )

    t = np.zeros(
        (n_points, 1),
        dtype=np.float32,
    )

    return x, y, t


def sample_boundary_condition(
    n_points: int,
    t_final: float,
):
    """
    Sample periodic boundary points.

    Returns
    -------
    left
    right
    bottom
    top
    """

    n_side = n_points // 4

    t = np.random.uniform(
        0.0,
        t_final,
        (n_side, 1),
    )

    # Left / Right

    y_lr = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_Y,
        (n_side, 1),
    )

    x_left = np.zeros_like(y_lr)

    x_right = (
        DOMAIN_LENGTH_X
        * np.ones_like(y_lr)
    )

    # Bottom / Top

    x_bt = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_X,
        (n_side, 1),
    )

    y_bottom = np.zeros_like(x_bt)

    y_top = (
        DOMAIN_LENGTH_Y
        * np.ones_like(x_bt)
    )

    return {
        "left": (
            x_left,
            y_lr,
            t,
        ),
        "right": (
            x_right,
            y_lr,
            t,
        ),
        "bottom": (
            x_bt,
            y_bottom,
            t,
        ),
        "top": (
            x_bt,
            y_top,
            t,
        ),
    }


# ============================================================
# Taylor-Green Initial Conditions
# ============================================================

def generate_initial_condition_targets(
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Exact Taylor-Green values.
    """

    u, v = tg_velocity_np(
        x,
        y,
        t=0.0,
    )

    p = tg_pressure_np(
        x,
        y,
        t=0.0,
    )

    return {
        "u": u,
        "v": v,
        "p": p,
    }


# ============================================================
# Tensor Conversion
# ============================================================

def to_tensor(
    array,
    requires_grad=False,
):
    """
    NumPy -> Torch tensor
    """

    return torch.tensor(
        array,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=requires_grad,
    )


def collocation_tensors(
    x,
    y,
    t,
):
    """
    Convert collocation points
    to differentiable tensors.
    """

    return {
        "x": to_tensor(
            x,
            requires_grad=True,
        ),
        "y": to_tensor(
            y,
            requires_grad=True,
        ),
        "t": to_tensor(
            t,
            requires_grad=True,
        ),
    }


def supervised_tensors(
    x,
    y,
    t,
    u,
    v,
    p,
):
    """
    Convert IC data to tensors.
    """

    return {
        "x": to_tensor(
            x,
            requires_grad=True,
        ),
        "y": to_tensor(
            y,
            requires_grad=True,
        ),
        "t": to_tensor(
            t,
            requires_grad=True,
        ),
        "u": to_tensor(u),
        "v": to_tensor(v),
        "p": to_tensor(p),
    }


# ============================================================
# Collocation Dataset
# ============================================================

def generate_collocation_dataset(
    n_collocation: int,
    t_final: float,
):
    """
    PDE training points.
    """

    x, y, t = sample_domain(
        n_collocation,
        t_final,
    )

    return collocation_tensors(
        x,
        y,
        t,
    )


# ============================================================
# Initial Condition Dataset
# ============================================================

def generate_ic_dataset(
    n_initial: int,
):
    """
    Taylor-Green IC training set.
    """

    x, y, t = sample_initial_condition(
        n_initial,
    )

    targets = (
        generate_initial_condition_targets(
            x,
            y,
        )
    )

    return supervised_tensors(
        x,
        y,
        t,
        targets["u"],
        targets["v"],
        targets["p"],
    )


# ============================================================
# Boundary Dataset
# ============================================================

def generate_bc_dataset(
    n_boundary: int,
    t_final: float,
):
    """
    Periodic BC samples.
    """

    boundaries = sample_boundary_condition(
        n_boundary,
        t_final,
    )

    bc = {}

    for key, value in boundaries.items():

        x, y, t = value

        bc[key] = {
            "x": to_tensor(
                x,
                requires_grad=True,
            ),
            "y": to_tensor(
                y,
                requires_grad=True,
            ),
            "t": to_tensor(
                t,
                requires_grad=True,
            ),
        }

    return bc


# ============================================================
# Full PINN Dataset
# ============================================================

def generate_training_data(
    n_collocation: int,
    n_initial: int,
    n_boundary: int,
    t_final: float,
):
    """
    Complete PINN/VQPINN dataset.
    """

    return {
        "collocation":
            generate_collocation_dataset(
                n_collocation,
                t_final,
            ),

        "initial":
            generate_ic_dataset(
                n_initial,
            ),

        "boundary":
            generate_bc_dataset(
                n_boundary,
                t_final,
            ),
    }


# ============================================================
# Evaluation Grid
# ============================================================

def generate_evaluation_grid(
    nx: int = 128,
    ny: int = 128,
    t: float = 0.1,
):
    """
    Structured evaluation grid.
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

    T = (
        np.ones_like(X)
        * t
    )

    return {
        "x": X,
        "y": Y,
        "t": T,
    }


# ============================================================
# Curriculum Sampling
# ============================================================

def curriculum_collocation_points(
    n_points: int,
    current_epoch: int,
    max_epochs: int,
    t_final: float,
):
    """
    Time curriculum used in the PINN notebook.

    Early epochs:
        small time horizon

    Later epochs:
        full horizon
    """

    frac = max(
        0.05,
        current_epoch / max_epochs,
    )

    current_tmax = (
        frac * t_final
    )

    x = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_X,
        (n_points, 1),
    )

    y = np.random.uniform(
        0.0,
        DOMAIN_LENGTH_Y,
        (n_points, 1),
    )

    t = np.random.uniform(
        0.0,
        current_tmax,
        (n_points, 1),
    )

    return collocation_tensors(
        x,
        y,
        t,
    )