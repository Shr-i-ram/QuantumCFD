# src/models/vqls.py

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# Notebook Constants
# ============================================================

VQLS_LAYERS = 8
DEFAULT_LR = 1e-2
PRETRAIN_EPOCHS = 100
TRAIN_EPOCHS = 600


# ============================================================
# Utilities
# ============================================================

def compute_n_qubits(
    vector_size: int,
):
    """
    n_qubits = ceil(log2(vector_size))
    """

    return int(
        np.ceil(
            np.log2(vector_size)
        )
    )


def amp_embed(
    vec_flat,
    n_qubits: int,
):
    """
    Notebook amplitude embedding.

    Pads vector to 2^n_qubits
    and normalizes.
    """

    vec_flat = np.asarray(
        vec_flat,
        dtype=np.float32,
    )

    pad_len = (
        2**n_qubits
        - vec_flat.size
    )

    padded = np.concatenate(
        [
            vec_flat,
            np.zeros(
                pad_len,
                dtype=vec_flat.dtype,
            ),
        ]
    )

    norm = (
        np.linalg.norm(
            padded
        )
        + 1e-12
    )

    return padded / norm


# ============================================================
# Hardware Efficient Ansatz
# ============================================================

def he_ansatz(
    params,
    n_qubits,
):
    """
    Notebook ansatz:

        RY
        RZ

    then ring CNOTs
    """

    n_layers = (
        params.shape[0]
    )

    wires = list(
        range(
            n_qubits
        )
    )

    for layer in range(
        n_layers
    ):

        for w in wires:

            qml.RY(
                params[
                    layer,
                    w,
                    0,
                ],
                wires=w,
            )

            qml.RZ(
                params[
                    layer,
                    w,
                    1,
                ],
                wires=w,
            )

        for w in wires:

            qml.CNOT(
                wires=[
                    w,
                    (w + 1)
                    % n_qubits,
                ]
            )


# ============================================================
# VQLS Model
# ============================================================

class VQLS_Solver(
    nn.Module
):
    """
    Notebook-faithful VQLS implementation.

    IMPORTANT:

    This model does NOT return
    a linear-system solution directly.

    It returns:

        loss,
        x_amp

    exactly as the notebook does.
    """

    def __init__(
        self,
        vector_size: int,
        n_layers: int = VQLS_LAYERS,
        seed: int = 42,
    ):
        super().__init__()

        self.vector_size = (
            vector_size
        )

        self.n_qubits = (
            compute_n_qubits(
                vector_size
            )
        )

        self.n_layers = (
            n_layers
        )

        self.seed = seed

        self.dev = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            seed=seed,
        )

        self.params = nn.Parameter(
            0.01
            * torch.randn(
                n_layers,
                self.n_qubits,
                2,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "calls",
            torch.zeros(
                1,
                dtype=torch.long,
            ),
        )

        @qml.qnode(
            self.dev,
            interface="torch",
            diff_method="parameter-shift",
            shots=None,
        )
        def state_prep(
            params,
            b_amp,
        ):

            qml.AmplitudeEmbedding(
                b_amp,
                wires=range(
                    self.n_qubits
                ),
                normalize=True,
            )

            he_ansatz(
                params,
                self.n_qubits,
            )

            return [
                qml.expval(
                    qml.PauliZ(i)
                )
                for i in range(
                    self.n_qubits
                )
            ]

        self.state_prep = (
            state_prep
        )

    # =====================================================
    # Expansion Trick
    # =====================================================

    def expand_amplitudes(
        self,
        x_amp,
    ):
        """
        Notebook repeats the
        expectation vector until
        N² entries are obtained.
        """

        if (
            x_amp.numel()
            < self.vector_size
        ):

            reps = int(
                np.ceil(
                    self.vector_size
                    / x_amp.numel()
                )
            )

            x_amp = (
                x_amp.repeat(
                    reps
                )[
                    : self.vector_size
                ]
            )

        return x_amp

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        A_apply,
        b_vec,
    ):
        """
        Parameters
        ----------
        A_apply : callable

            Function implementing:

                y -> A y

        b_vec : ndarray

            RHS vector

        Returns
        -------
        loss
        x_amp

        exactly matching notebook
        behavior.
        """

        b_amp = torch.tensor(
            amp_embed(
                b_vec.astype(
                    np.float32
                ),
                self.n_qubits,
            ),
            dtype=torch.float32,
        )

        psi = torch.stack(
            self.state_prep(
                self.params,
                b_amp,
            )
        )

        self.calls += 1

        x_amp = psi.float()

        x_amp = (
            self.expand_amplitudes(
                x_amp
            )
        )

        # --------------------------------
        # Preconditioned RHS
        # --------------------------------

        with torch.no_grad():

            Mb = torch.tensor(
                b_vec,
                dtype=torch.float32,
            )

        # --------------------------------
        # Matrix-vector product
        # --------------------------------

        AMy_np = A_apply(
            x_amp.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        ).astype(
            np.float32
        )

        AMy = torch.tensor(
            AMy_np,
            dtype=torch.float32,
        )

        # --------------------------------
        # Residual
        # --------------------------------

        residual = (
            AMy - Mb
        )

        # --------------------------------
        # Smoothness Penalty
        # --------------------------------

        side = int(
            np.sqrt(
                self.vector_size
            )
        )

        x_grid = x_amp.view(
            side,
            side,
        )

        smooth_penalty = (
            torch.mean(
                (
                    x_grid[1:]
                    - x_grid[:-1]
                )
                ** 2
            )
            +
            torch.mean(
                (
                    x_grid[:, 1:]
                    - x_grid[:, :-1]
                )
                ** 2
            )
        )

        loss = (
            torch.mean(
                residual**2
            )
            +
            1e-2
            * smooth_penalty
        )

        return (
            loss,
            x_amp,
        )

    # =====================================================
    # Pretraining Loss
    # =====================================================

    def pretraining_loss(
        self,
        target_field,
        A_apply,
        rhs,
    ):
        """
        Notebook warm-start:

            mse(x_amp, omega)
        """

        _, x_amp = self.forward(
            A_apply,
            rhs,
        )

        return torch.mean(
            (
                x_amp
                - target_field
            )
            ** 2
        )

    # =====================================================
    # Diagnostics
    # =====================================================

    def call_count(
        self,
    ):
        return int(
            self.calls.item()
        )


# ============================================================
# Utilities
# ============================================================

def count_parameters(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def model_summary(
    model,
):
    print("=" * 60)
    print("VQLS Solver")
    print("=" * 60)

    print(
        f"Vector Size : "
        f"{model.vector_size}"
    )

    print(
        f"Qubits      : "
        f"{model.n_qubits}"
    )

    print(
        f"Layers      : "
        f"{model.n_layers}"
    )

    print(
        f"Parameters  : "
        f"{count_parameters(model):,}"
    )

    print("=" * 60)