# src/training/pinn_trainer.py

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    DEVICE,
)

from src.physics.navier_stokes import (
    compute_residuals,
)

from src.datasets.training_data import (
    curriculum_collocation_points,
)

from src.physics.taylor_green import (
    tg_velocity_np,
)

from src.models.pinn import (
    PINN_NavierStokes,
)


class PINNTrainer:
    """
    Notebook-faithful trainer.

    Curriculum:

        Stage 1:
            t ∈ [0, 0.03]
            4000 epochs

        Stage 2:
            t ∈ [0, 0.06]
            6000 epochs

        Stage 3:
            t ∈ [0, 0.10]
            8000 epochs

    Total:
        18000 epochs
    """

    def __init__(
        self,
        model: PINN_NavierStokes,
        learning_rate: float = 1e-3,
        reynolds_number: float = 100.0,
        lambda_ic: float = 10.0,
        lambda_pde: float = 1.0,
        lambda_cont: float = 1.0,
        n_collocation: int = 5000,
        n_initial: int = 1000,
        device=DEVICE,
    ):

        self.model = model.to(device)

        self.device = device

        self.re = reynolds_number
        self.nu = 1.0 / reynolds_number

        self.lambda_ic = lambda_ic
        self.lambda_pde = lambda_pde
        self.lambda_cont = lambda_cont

        self.n_collocation = n_collocation
        self.n_initial = n_initial

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.loss_history = []
        self.ic_history = []
        self.pde_history = []
        self.cont_history = []

    # =========================================================
    # Initial Condition Dataset
    # =========================================================

    def generate_ic_batch(self):

        x = np.random.uniform(
            0.0,
            2.0 * np.pi,
            (self.n_initial, 1),
        )

        y = np.random.uniform(
            0.0,
            2.0 * np.pi,
            (self.n_initial, 1),
        )

        t = np.zeros_like(x)

        u0, v0 = tg_velocity_np(
            x,
            y,
            t=0.0,
            nu=self.nu,
        )

        x = torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        y = torch.tensor(
            y,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        t = torch.tensor(
            t,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        u0 = torch.tensor(
            u0,
            dtype=torch.float32,
            device=self.device,
        )

        v0 = torch.tensor(
            v0,
            dtype=torch.float32,
            device=self.device,
        )

        return (
            x,
            y,
            t,
            u0,
            v0,
        )

    # =========================================================
    # Physics Loss
    # =========================================================

    def pde_loss(
        self,
        x,
        y,
        t,
    ):

        residuals = compute_residuals(
            self.model,
            x,
            y,
            t,
            nu=self.nu,
        )

        loss_fu = torch.mean(
            residuals["f_u"] ** 2
        )

        loss_fv = torch.mean(
            residuals["f_v"] ** 2
        )

        loss_cont = torch.mean(
            residuals["continuity"] ** 2
        )

        loss_pde = (
            loss_fu +
            loss_fv
        )

        return (
            loss_pde,
            loss_cont,
        )

    # =========================================================
    # Initial Condition Loss
    # =========================================================

    def ic_loss(
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

        loss_u = torch.mean(
            (u - u0) ** 2
        )

        loss_v = torch.mean(
            (v - v0) ** 2
        )

        return (
            loss_u +
            loss_v
        )

    # =========================================================
    # Single Training Step
    # =========================================================

    def train_step(
        self,
        current_tmax,
    ):

        self.optimizer.zero_grad()

        collocation = curriculum_collocation_points(
            self.n_collocation,
            current_epoch=1,
            max_epochs=1,
            t_final=current_tmax,
        )

        x_f = collocation["x"]
        y_f = collocation["y"]
        t_f = collocation["t"]

        (
            x_i,
            y_i,
            t_i,
            u0,
            v0,
        ) = self.generate_ic_batch()

        loss_ic = self.ic_loss(
            x_i,
            y_i,
            t_i,
            u0,
            v0,
        )

        loss_pde, loss_cont = self.pde_loss(
            x_f,
            y_f,
            t_f,
        )

        total_loss = (
            self.lambda_ic * loss_ic
            +
            self.lambda_pde * loss_pde
            +
            self.lambda_cont * loss_cont
        )

        total_loss.backward()

        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "ic": loss_ic.item(),
            "pde": loss_pde.item(),
            "cont": loss_cont.item(),
        }

    # =========================================================
    # Stage Training
    # =========================================================

    def train_stage(
        self,
        epochs,
        t_max,
        verbose=True,
    ):

        start = time.time()

        for epoch in range(epochs):

            losses = self.train_step(
                t_max,
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
                    epoch % 100 == 0
                    or epoch == epochs - 1
                )
            ):

                print(
                    f"[t≤{t_max:.3f}] "
                    f"Epoch {epoch:5d}/{epochs} | "
                    f"Loss={losses['total']:.4e} | "
                    f"IC={losses['ic']:.4e} | "
                    f"PDE={losses['pde']:.4e} | "
                    f"CONT={losses['cont']:.4e}"
                )

        elapsed = time.time() - start

        print(
            f"Stage completed in "
            f"{elapsed:.2f} seconds."
        )

    # =========================================================
    # Full Curriculum
    # =========================================================

    def train(
        self,
        verbose=True,
    ):

        curriculum = [
            (4000, 0.03),
            (6000, 0.06),
            (8000, 0.10),
        ]

        for epochs, tmax in curriculum:

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Curriculum Stage "
                f"(t ≤ {tmax})"
            )

            print(
                "=" * 60
            )

            self.train_stage(
                epochs=epochs,
                t_max=tmax,
                verbose=verbose,
            )

    # =========================================================
    # Inference
    # =========================================================

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

        pred = self.model(
            torch.cat(
                [x, y, t],
                dim=1,
            )
        )

        return {
            "u": pred[:, 0:1].cpu().numpy(),
            "v": pred[:, 1:2].cpu().numpy(),
            "p": pred[:, 2:3].cpu().numpy(),
        }

    # =========================================================
    # Save / Load
    # =========================================================

    def save(
        self,
        filepath,
    ):

        filepath = Path(filepath)

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
                "loss_history":
                    self.loss_history,
            },
            filepath,
        )

        print(
            f"Model saved to "
            f"{filepath}"
        )

    def load(
        self,
        filepath,
    ):

        checkpoint = torch.load(
            filepath,
            map_location=self.device,
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

        self.loss_history = checkpoint.get(
            "loss_history",
            [],
        )

        print(
            f"Loaded checkpoint "
            f"from {filepath}"
        )

    # =========================================================
    # Diagnostics
    # =========================================================

    def get_history(self):

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