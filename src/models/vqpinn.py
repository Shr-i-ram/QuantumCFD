# src/models/vqpinn.py

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.quantum_layers import (
    N_QUBITS,
    QuantumLayer,
    QuantumRegressionHead,
    count_quantum_parameters,
)


# ============================================================
# Weight Initialization
# ============================================================

def initialize_weights(
    module,
):
    """
    Xavier initialization
    for classical layers.
    """

    if isinstance(
        module,
        nn.Linear,
    ):

        nn.init.xavier_normal_(
            module.weight
        )

        nn.init.zeros_(
            module.bias
        )


# ============================================================
# VQPINN Model
# ============================================================

class VQ_PINN_NS(nn.Module):
    """
    Variational Quantum Physics-Informed
    Neural Network.

    Architecture extracted directly
    from notebook:

        (x,y,t)
            ↓
        Fourier Features
            ↓
        Keep 6 Features
            ↓
        AngleEmbedding(Y)
            ↓
        StronglyEntanglingLayers
            ↓
        <Z0>...<Z5>

            ↓

        Linear(6,32)
        Tanh
        Linear(32,2)

    Outputs:
        u,v
    """

    def __init__(
        self,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.quantum = QuantumLayer()

        self.head = QuantumRegressionHead(
            in_dim=N_QUBITS,
            hidden_dim=hidden_dim,
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

        columns:
            x,y,t

        Returns
        -------
        (N,2)

        columns:
            u,v
        """

        quantum_features = (
            self.quantum(
                xyt
            )
        )

        return self.head(
            quantum_features
        )

    # =====================================================
    # Convenience Methods
    # =====================================================

    def predict_uv(
        self,
        xyt: torch.Tensor,
    ):

        outputs = self.forward(
            xyt
        )

        u = outputs[:, 0:1]
        v = outputs[:, 1:2]

        return u, v

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

    # =====================================================
    # Diagnostics
    # =====================================================

    def quantum_features(
        self,
        xyt: torch.Tensor,
    ):
        """
        Return raw expectation values.

        Shape:
            (N,6)
        """

        return self.quantum(
            xyt
        )

    def latent_representation(
        self,
        xyt: torch.Tensor,
    ):
        """
        Alias for quantum feature vector.
        """

        return self.quantum_features(
            xyt
        )


# ============================================================
# Utilities
# ============================================================

def count_parameters(
    model: nn.Module,
):
    """
    Total trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def model_summary(
    model: VQ_PINN_NS,
):
    """
    Repository summary helper.
    """

    total_params = (
        count_parameters(model)
    )

    quantum_params = (
        count_quantum_parameters(
            model
        )
    )

    print(
        "=" * 60
    )
    print(
        "VQ_PINN_NS"
    )
    print(
        "=" * 60
    )

    print(
        f"Total Parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable Parameters: "
        f"{quantum_params:,}"
    )

    print(
        f"Quantum Qubits: "
        f"{N_QUBITS}"
    )

    print(
        f"Outputs: u,v"
    )

    print(
        "=" * 60
    )


# ============================================================
# Factory
# ============================================================

def create_vqpinn():
    """
    Notebook default model.
    """

    return VQ_PINN_NS(
        hidden_dim=32
    )