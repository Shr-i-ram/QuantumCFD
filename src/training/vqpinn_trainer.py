# src/training/vqpinn_trainer.py

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

from src.config import DEVICE

from src.models.vqpinn import (
    VQ_PINN_NS,
)

from src.physics.taylor_green import (
    tg_velocity_np,
)


# ============================================================
# Autograd Utilities
# ============================================================

def grad(
    outputs,
    inputs,
):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def second_grad(
    outputs,
    inputs,
):
    g = grad(
        outputs,
        inputs,
    )

    return torch.autograd.grad(
        g,
        inputs,
        grad_outputs=torch.ones_like(g),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


# ============================================================
# VQPINN Residual
# ============================================================

def navier_stokes_residual(
    model,
    x,
    y,
    t,
    nu=0.01,
):
    """
    Pressure-free residual used
    in notebook.

    u_t + u*u_x + v*u_y
      - nu*(u_xx+u_yy)

    v_t + u*v_x + v*v_y
      - nu*(v_xx+v_yy)

    continuity:
        u_x + v_y
    """

    xyt = torch.cat(
        [x, y, t],
        dim=1,
    )

    uv = model(xyt)

    u = uv[:, 0:1]
    v = uv[:, 1:2]

    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)

    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)

    u_xx = second_grad(
        u,
        x,
    )

    u_yy = second_grad(
        u,
        y,
    )

    v_xx = second_grad(
        v,
        x,
    )

    v_yy = second_grad(
        v,
        y,
    )

    f_u = (
        u_t
        + u * u_x
        + v * u_y
        - nu * (
            u_xx + u_yy
        )
    )

    f_v = (
        v_t
        + u * v_x
        + v * v_y
        - nu * (
            v_xx + v_yy
        )
    )

    continuity = (
        u_x + v_y
    )

    return {
        "u": u,
        "v": v,
        "f_u": f_u,
        "f_v": f_v,
        "continuity": continuity,
    }


# ============================================================
# VQPINN Trainer
# ============================================================

class VQPINNTrainer:

    def __init__(
        self,
        model: VQ_PINN_NS,
        learning_rate: float = 1e-3,
        reynolds_number: float = 100.0,
        n_collocation: int = 1024,
        n_initial: int = 512,
        device=DEVICE,
    ):

        self.model = model.to(
            device
        )

        self.device = device

        self.re = reynolds_number
        self.nu = (
            1.0 / reynolds_number
        )

        self.n_collocation = (
            n_collocation
        )

        self.n_initial = (
            n_initial
        )

        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )
        )

        self.scheduler = (
            StepLR(
                self.optimizer,
                step_size=500,
                gamma=0.7,
            )
        )

        # notebook weights

        self.lambda_ic = 1.0
        self.lambda_pde = 0.5
        self.lambda_cont = 0.5

        self.loss_history = []
        self.ic_history = []
        self.pde_history = []
        self.cont_history = []

    # =====================================================
    # Data Sampling
    # =====================================================

    def sample_collocation_points(
        self,
    ):

        x = np.random.uniform(
            0.0,
            2.0 * np.pi,
            (
                self.n_collocation,
                1,
            ),
        )

        y = np.random.uniform(
            0.0,
            2.0 * np.pi,
            (
                self.n_collocation,
                1,
            ),
        )

        t = np.random.uniform(
            0.0,
            0.10,
            (
                self.n_collocation,
                1,
            ),
        )

        return (
            torch.tensor(
                x,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            ),
            torch.tensor(
                t,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            ),
        )

    def sample_initial_points(
        self,
    ):

        x = np.random.uniform(
            0.0,
            2.0 * np.pi,
            (
                self.n_initial,
                1,
            ),
        )

        y = np.random.uniform(
            0.0,
            2.0 * np.pi,
            (
                self.n_initial,
                1,
            ),
        )

        t = np.zeros_like(x)

        u0, v0 = tg_velocity_np(
            x,
            y,
            t=0.0,
            nu=self.nu,
        )

        return (
            torch.tensor(
                x,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            ),
            torch.tensor(
                t,
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            ),
            torch.tensor(
                u0,
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                v0,
                dtype=torch.float32,
                device=self.device,
            ),
        )

    # =====================================================
    # Loss Functions
    # =====================================================

    def initial_condition_loss(
        self,
        x,
        y,
        t,
        u0,
        v0,
    ):

        pred = self.model(
            torch.cat(
                [x, y, t],
                dim=1,
            )
        )

        u = pred[:, 0:1]
        v = pred[:, 1:2]

        return (
            torch.mean(
                (u - u0) ** 2
            )
            +
            torch.mean(
                (v - v0) ** 2
            )
        )

    def pde_loss(
        self,
        x,
        y,
        t,
    ):

        residuals = (
            navier_stokes_residual(
                self.model,
                x,
                y,
                t,
                self.nu,
            )
        )

        loss_pde = (
            torch.mean(
                residuals["f_u"] ** 2
            )
            +
            torch.mean(
                residuals["f_v"] ** 2
            )
        )

        loss_cont = (
            torch.mean(
                residuals["continuity"] ** 2
            )
        )

        return (
            loss_pde,
            loss_cont,
        )

    # =====================================================
    # Training Step
    # =====================================================

    def train_step(
        self,
    ):

        self.optimizer.zero_grad()

        x_f, y_f, t_f = (
            self.sample_collocation_points()
        )

        x_i, y_i, t_i, u0, v0 = (
            self.sample_initial_points()
        )

        loss_ic = (
            self.initial_condition_loss(
                x_i,
                y_i,
                t_i,
                u0,
                v0,
            )
        )

        (
            loss_pde,
            loss_cont,
        ) = self.pde_loss(
            x_f,
            y_f,
            t_f,
        )

        total_loss = (
            self.lambda_ic
            * loss_ic
            +
            self.lambda_pde
            * loss_pde
            +
            self.lambda_cont
            * loss_cont
        )

        total_loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return {
            "total":
                total_loss.item(),
            "ic":
                loss_ic.item(),
            "pde":
                loss_pde.item(),
            "cont":
                loss_cont.item(),
        }

    # =====================================================
    # Training Loop
    # =====================================================

    def train(
        self,
        epochs: int = 2000,
        verbose=True,
    ):

        start = time.time()

        for epoch in range(
            epochs
        ):

            losses = (
                self.train_step()
            )

            self.loss_history.append(
                losses["total"]
            )

            self.ic_history.append(
                losses["ic"]
            )

            self.pde_history.append(
                losses["pde"]
            )

            self.cont_history.append(
                losses["cont"]
            )

            if (
                verbose
                and (
                    epoch % 25 == 0
                    or epoch == epochs - 1
                )
            ):

                print(
                    f"Epoch "
                    f"{epoch:5d}/{epochs} | "
                    f"Loss={losses['total']:.4e} | "
                    f"IC={losses['ic']:.4e} | "
                    f"PDE={losses['pde']:.4e} | "
                    f"CONT={losses['cont']:.4e}"
                )

        elapsed = (
            time.time() - start
        )

        print(
            f"\nTraining completed "
            f"in {elapsed:.2f}s"
        )

    # =====================================================
    # Inference
    # =====================================================

    @torch.no_grad()
    def predict(
        self,
        x,
        y,
        t,
    ):

        x = torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        )

        y = torch.tensor(
            y,
            dtype=torch.float32,
            device=self.device,
        )

        t = torch.tensor(
            t,
            dtype=torch.float32,
            device=self.device,
        )

        uv = self.model(
            torch.cat(
                [x, y, t],
                dim=1,
            )
        )

        return {
            "u":
                uv[:, 0:1]
                .cpu()
                .numpy(),

            "v":
                uv[:, 1:2]
                .cpu()
                .numpy(),
        }

    # =====================================================
    # Checkpointing
    # =====================================================

    def save(
        self,
        filepath,
    ):

        filepath = Path(
            filepath
        )

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "scheduler_state_dict":
                    self.scheduler.state_dict(),

                "loss_history":
                    self.loss_history,
            },
            filepath,
        )

    def load(
        self,
        filepath,
    ):

        checkpoint = (
            torch.load(
                filepath,
                map_location=self.device,
            )
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        self.scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        self.loss_history = (
            checkpoint.get(
                "loss_history",
                [],
            )
        )

    # =====================================================
    # Diagnostics
    # =====================================================

    def get_history(
        self,
    ):

        return {
            "total":
                self.loss_history,

            "ic":
                self.ic_history,

            "pde":
                self.pde_history,

            "continuity":
                self.cont_history,
        }