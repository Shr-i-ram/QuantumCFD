# src/models/pinn.py

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ============================================================
# Fourier Feature Encoding
# ============================================================

class FourierFeatures(nn.Module):
    """
    Notebook-faithful Fourier encoder.

    Input:
        xyt -> (N,3)

    Features:

        x:
            sin(2πkx), cos(2πkx)
            k = 1..6

        y:
            sin(2πky), cos(2πky)
            k = 1..6

        t:
            sin(2πkt), cos(2πkt)
            k = 1..2

    Output dimension:
        28
    """

    def __init__(
        self,
        n_fxy: int = 6,
        n_ft: int = 2,
    ):
        super().__init__()

        self.n_fxy = n_fxy
        self.n_ft = n_ft

    @property
    def output_dim(self):
        return (
            4 * self.n_fxy
            + 2 * 2 * self.n_ft
        )

    def _encode_coord(
        self,
        coord: torch.Tensor,
        n_freq: int,
    ):
        features = []

        for k in range(1, n_freq + 1):

            features.append(
                torch.sin(
                    2.0 * math.pi * k * coord
                )
            )

            features.append(
                torch.cos(
                    2.0 * math.pi * k * coord
                )
            )

        return torch.cat(
            features,
            dim=1,
        )

    def forward(
        self,
        xyt: torch.Tensor,
    ):

        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]

        # ---------------------------------
        # Notebook normalization
        # ---------------------------------

        x = x / (2.0 * math.pi)
        y = y / (2.0 * math.pi)

        # t already normalized in notebook
        # sampling pipeline

        fx = self._encode_coord(
            x,
            self.n_fxy,
        )

        fy = self._encode_coord(
            y,
            self.n_fxy,
        )

        ft = self._encode_coord(
            t,
            self.n_ft,
        )

        return torch.cat(
            [
                fx,
                fy,
                ft,
            ],
            dim=1,
        )


# ============================================================
# Weight Initialization
# ============================================================

def initialize_weights(
    module,
):
    """
    Xavier initialization used
    throughout repository.
    """

    if isinstance(
        module,
        nn.Linear,
    ):

        nn.init.xavier_normal_(
            module.weight,
        )

        nn.init.zeros_(
            module.bias,
        )


# ============================================================
# PINN Network
# ============================================================

class PINN_NavierStokes(nn.Module):
    """
    Physics-Informed Neural Network
    for 2D incompressible Navier-Stokes.

    Architecture extracted directly
    from notebook:

        FourierFeatures (28)

            ↓

        Linear(28,128)
        Tanh

        Linear(128,128)
        Tanh

        Linear(128,128)
        Tanh

        Linear(128,128)
        Tanh

        Linear(128,128)
        Tanh

        Linear(128,128)
        Tanh

        Linear(128,128)
        Tanh

        Linear(128,3)

    Outputs:
        u,v,p
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 8,
        n_fxy: int = 6,
        n_ft: int = 2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.depth = depth

        self.encoder = FourierFeatures(
            n_fxy=n_fxy,
            n_ft=n_ft,
        )

        in_dim = self.encoder.output_dim

        layers = []

        layers.append(
            nn.Linear(
                in_dim,
                hidden_dim,
            )
        )

        layers.append(
            nn.Tanh()
        )

        for _ in range(depth - 2):

            layers.append(
                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                )
            )

            layers.append(
                nn.Tanh()
            )

        layers.append(
            nn.Linear(
                hidden_dim,
                3,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

        self.apply(
            initialize_weights
        )

    def forward(
        self,
        xyt: torch.Tensor,
    ):
        """
        Parameters
        ----------
        xyt : (N,3)

        Returns
        -------
        (N,3)

        columns:
            u
            v
            p
        """

        features = self.encoder(
            xyt
        )

        return self.network(
            features
        )

    def predict_uvp(
        self,
        xyt: torch.Tensor,
    ):
        """
        Convenience wrapper.
        """

        outputs = self.forward(
            xyt
        )

        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]

        return u, v, p

    def predict_u(
        self,
        xyt: torch.Tensor,
    ):
        return self.forward(
            xyt
        )[:, 0:1]

    def predict_v(
        self,
        xyt: torch.Tensor,
    ):
        return self.forward(
            xyt
        )[:, 1:2]

    def predict_p(
        self,
        xyt: torch.Tensor,
    ):
        return self.forward(
            xyt
        )[:, 2:3]


# ============================================================
# Utilities
# ============================================================

def count_parameters(
    model: nn.Module,
):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def model_summary(
    model: nn.Module,
):
    """
    Lightweight model summary.
    """

    n_params = count_parameters(
        model
    )

    print(
        "=" * 60
    )
    print(
        model.__class__.__name__
    )
    print(
        "=" * 60
    )
    print(
        f"Trainable Parameters: {n_params:,}"
    )
    print(
        "=" * 60
    )


# ============================================================
# Factory
# ============================================================

def create_pinn():
    """
    Notebook default model.
    """

    return PINN_NavierStokes(
        hidden_dim=128,
        depth=8,
        n_fxy=6,
        n_ft=2,
    )