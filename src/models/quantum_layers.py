# src/models/quantum_layers.py

from __future__ import annotations

import math
import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# Constants extracted from notebook
# ============================================================

N_QUBITS = 6
N_LAYERS = 4
T_FINAL = 0.10


# ============================================================
# Fourier Feature Encoding
# ============================================================

def fourier_features(
    xyt: torch.Tensor,
):
    """
    Notebook feature map.

    Input
    -----
    xyt : (N,3)

    columns:
        x,y,t

    Features:

        x
        y
        tt

        sin(2x)
        cos(2x)
        sin(2y)
        cos(2y)
        sin(2tt)
        cos(2tt)

        sin(3x)
        cos(3x)
        sin(3y)
        cos(3y)
        sin(3tt)
        cos(3tt)

    Total:
        15 features
    """

    x = xyt[:, 0:1]
    y = xyt[:, 1:2]
    t = xyt[:, 2:3]

    tt = (
        2.0
        * math.pi
        * (t / T_FINAL)
    )

    feats = [
        x,
        y,
        tt,
    ]

    for k in (2, 3):

        feats.extend(
            [
                torch.sin(k * x),
                torch.cos(k * x),

                torch.sin(k * y),
                torch.cos(k * y),

                torch.sin(k * tt),
                torch.cos(k * tt),
            ]
        )

    return torch.cat(
        feats,
        dim=1,
    )


# ============================================================
# Qubit Projection
# ============================================================

def pad_to_qubits(
    features: torch.Tensor,
    n_qubits: int = N_QUBITS,
):
    """
    Notebook keeps first six
    encoded features.

    Output:
        (N,6)
    """

    if features.shape[1] >= n_qubits:

        return features[:, :n_qubits]

    pad = torch.zeros(
        features.shape[0],
        n_qubits - features.shape[1],
        device=features.device,
        dtype=features.dtype,
    )

    return torch.cat(
        [
            features,
            pad,
        ],
        dim=1,
    )


# ============================================================
# PennyLane Device
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS,
)


# ============================================================
# Quantum Circuit
# ============================================================

@qml.qnode(
    dev,
    interface="torch",
    diff_method="parameter-shift",
)
def quantum_circuit(
    inputs,
    weights,
):
    """
    AngleEmbedding
        ->
    StronglyEntanglingLayers
        ->
    <Z_i>
    """

    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y",
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS),
    )

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(N_QUBITS)
    ]


# ============================================================
# TorchLayer Wrapper
# ============================================================

def create_quantum_torch_layer():
    """
    PennyLane TorchLayer.

    Weight shape extracted
    from notebook.

    (4,6,3)
    """

    weight_shapes = {
        "weights": (
            N_LAYERS,
            N_QUBITS,
            3,
        )
    }

    return qml.qnn.TorchLayer(
        quantum_circuit,
        weight_shapes,
    )


# ============================================================
# Standalone Quantum Layer
# ============================================================

class QuantumLayer(nn.Module):
    """
    Quantum feature extractor.

    Input:
        x,y,t

    Output:
        six expectation values
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
    ):
        super().__init__()

        self.n_qubits = n_qubits

        self.qlayer = (
            create_quantum_torch_layer()
        )

    def encode(
        self,
        xyt: torch.Tensor,
    ):
        feats = fourier_features(
            xyt
        )

        feats = pad_to_qubits(
            feats,
            self.n_qubits,
        )

        return feats

    def forward(
        self,
        xyt: torch.Tensor,
    ):

        feats = self.encode(
            xyt
        )

        return self.qlayer(
            feats
        )


# ============================================================
# Hybrid Head
# ============================================================

class QuantumRegressionHead(
    nn.Module
):
    """
    Notebook classical head.

    6
      ↓
    32
      ↓
    2

    Outputs:
        u,v
    """

    def __init__(
        self,
        in_dim: int = N_QUBITS,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


# ============================================================
# Combined Quantum Block
# ============================================================

class QuantumFeatureBlock(
    nn.Module
):
    """
    QuantumLayer
        ->
    RegressionHead

    Returns:
        u,v
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.quantum = (
            QuantumLayer()
        )

        self.head = (
            QuantumRegressionHead()
        )

    def forward(
        self,
        xyt,
    ):

        q = self.quantum(
            xyt
        )

        return self.head(
            q
        )


# ============================================================
# Utilities
# ============================================================

def count_quantum_parameters(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def quantum_summary():
    """
    Notebook config helper.
    """

    print(
        "=" * 60
    )
    print(
        "Quantum Configuration"
    )
    print(
        "=" * 60
    )

    print(
        f"Qubits : {N_QUBITS}"
    )

    print(
        f"Layers : {N_LAYERS}"
    )

    print(
        "Embedding : AngleEmbedding(Y)"
    )

    print(
        "Ansatz : StronglyEntanglingLayers"
    )

    print(
        "Measurements : <Z_i>"
    )

    print(
        "=" * 60
    )