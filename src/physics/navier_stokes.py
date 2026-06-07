# src/physics/navier_stokes.py

from __future__ import annotations

import torch

from src.config import KINEMATIC_VISCOSITY


# ============================================================
# Autograd Utilities
# ============================================================

def grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
):
    """
    First derivative helper.

    Returns
    -------
    d(outputs)/d(inputs)
    """

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def second_grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
):
    """
    Second derivative helper.
    """

    first = grad(
        outputs,
        inputs,
    )

    second = torch.autograd.grad(
        first,
        inputs,
        grad_outputs=torch.ones_like(first),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return second


# ============================================================
# Navier-Stokes Derivatives
# ============================================================

def velocity_derivatives(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
):
    """
    Compute all first derivatives
    required by Navier-Stokes.
    """

    u_x = grad(u, x)
    u_y = grad(u, y)
    u_t = grad(u, t)

    v_x = grad(v, x)
    v_y = grad(v, y)
    v_t = grad(v, t)

    return {
        "u_x": u_x,
        "u_y": u_y,
        "u_t": u_t,
        "v_x": v_x,
        "v_y": v_y,
        "v_t": v_t,
    }


def velocity_second_derivatives(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """
    Compute second-order derivatives.
    """

    u_xx = second_grad(u, x)
    u_yy = second_grad(u, y)

    v_xx = second_grad(v, x)
    v_yy = second_grad(v, y)

    return {
        "u_xx": u_xx,
        "u_yy": u_yy,
        "v_xx": v_xx,
        "v_yy": v_yy,
    }


def pressure_derivatives(
    p: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """
    Pressure gradients.
    """

    p_x = grad(p, x)
    p_y = grad(p, y)

    return {
        "p_x": p_x,
        "p_y": p_y,
    }


# ============================================================
# Continuity Equation
# ============================================================

def continuity_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """
    Incompressibility constraint.

        u_x + v_y = 0
    """

    u_x = grad(u, x)
    v_y = grad(v, y)

    return (
        u_x +
        v_y
    )


# ============================================================
# Momentum Residuals
# ============================================================

def momentum_residuals(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Compute Navier-Stokes PDE residuals.

    Equations:

        u_t + u*u_x + v*u_y
        + p_x
        - nu*(u_xx + u_yy)

        v_t + u*v_x + v*v_y
        + p_y
        - nu*(v_xx + v_yy)
    """

    d1 = velocity_derivatives(
        u,
        v,
        x,
        y,
        t,
    )

    d2 = velocity_second_derivatives(
        u,
        v,
        x,
        y,
    )

    dp = pressure_derivatives(
        p,
        x,
        y,
    )

    f_u = (
        d1["u_t"]
        + u * d1["u_x"]
        + v * d1["u_y"]
        + dp["p_x"]
        - nu * (
            d2["u_xx"]
            + d2["u_yy"]
        )
    )

    f_v = (
        d1["v_t"]
        + u * d1["v_x"]
        + v * d1["v_y"]
        + dp["p_y"]
        - nu * (
            d2["v_xx"]
            + d2["v_yy"]
        )
    )

    return f_u, f_v


# ============================================================
# Full Residual Evaluation
# ============================================================

def compute_residuals(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    nu: float = KINEMATIC_VISCOSITY,
):
    """
    Shared PINN/VQPINN residual computation.

    Expected model output:

        u, v, p
    """

    outputs = model(
        torch.cat(
            [x, y, t],
            dim=1,
        )
    )

    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]

    f_u, f_v = momentum_residuals(
        u,
        v,
        p,
        x,
        y,
        t,
        nu,
    )

    continuity = continuity_residual(
        u,
        v,
        x,
        y,
    )

    return {
        "u": u,
        "v": v,
        "p": p,
        "f_u": f_u,
        "f_v": f_v,
        "continuity": continuity,
    }


# ============================================================
# Physics Loss
# ============================================================

def physics_loss(
    residuals: dict,
):
    """
    Mean-squared PDE loss.
    """

    f_u = residuals["f_u"]
    f_v = residuals["f_v"]
    cont = residuals["continuity"]

    loss_u = torch.mean(
        f_u**2
    )

    loss_v = torch.mean(
        f_v**2
    )

    loss_cont = torch.mean(
        cont**2
    )

    total = (
        loss_u
        + loss_v
        + loss_cont
    )

    return {
        "total": total,
        "momentum_u": loss_u,
        "momentum_v": loss_v,
        "continuity": loss_cont,
    }


# ============================================================
# Vorticity Utilities
# ============================================================

def vorticity_from_velocity(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
):
    """
    ω = dv/dx - du/dy
    """

    v_x = grad(v, x)
    u_y = grad(u, y)

    return (
        v_x -
        u_y
    )


# ============================================================
# Evaluation Helper
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
):
    """
    Convenience inference wrapper.
    """

    outputs = model(
        torch.cat(
            [x, y, t],
            dim=1,
        )
    )

    return {
        "u": outputs[:, 0:1],
        "v": outputs[:, 1:2],
        "p": outputs[:, 2:3],
    }