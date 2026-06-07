# src/config.py

from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np


# ============================================================
# Project Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = DATA_DIR / "outputs"

MODEL_DIR = ROOT_DIR / "models"

REFERENCE_FILE = REFERENCE_DIR / "reference_data_safe.npz"


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


# ============================================================
# Device Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Physical Domain
# ============================================================

DOMAIN_LENGTH_X = 2.0 * np.pi
DOMAIN_LENGTH_Y = 2.0 * np.pi

X_MIN = 0.0
X_MAX = DOMAIN_LENGTH_X

Y_MIN = 0.0
Y_MAX = DOMAIN_LENGTH_Y


# ============================================================
# Fluid Parameters
# ============================================================

REYNOLDS_NUMBER = 100.0

KINEMATIC_VISCOSITY = 1.0 / REYNOLDS_NUMBER


# ============================================================
# Taylor-Green Parameters
# ============================================================

TG_U0 = 1.0
TG_V0 = 1.0


# ============================================================
# Spectral Solver Defaults
# ============================================================

DEFAULT_GRID_SIZE = 32

DEFAULT_NX = DEFAULT_GRID_SIZE
DEFAULT_NY = DEFAULT_GRID_SIZE

REFERENCE_GRID_SIZE = 128

DEFAULT_DT = 1e-3

DEFAULT_T_FINAL = 0.1

DEALIAS_FRACTION = 2.0 / 3.0


# ============================================================
# PINN Training Defaults
# ============================================================

PINN_HIDDEN_DIM = 128
PINN_NUM_HIDDEN_LAYERS = 6

PINN_OUTPUT_DIM = 3  # u,v,p

PINN_INPUT_DIM = 3   # x,y,t

FOURIER_FEATURES_DIM = 64

PINN_LEARNING_RATE = 1e-3

PINN_EPOCHS = 5000

PINN_BATCH_SIZE = 8192

PINN_WEIGHT_DECAY = 0.0


# ============================================================
# Training Sample Counts
# ============================================================

N_INITIAL_CONDITION = 2000

N_BOUNDARY = 2000

N_COLLOCATION = 20000


# ============================================================
# Loss Weights
# ============================================================

LAMBDA_PDE = 1.0

LAMBDA_IC = 10.0

LAMBDA_BC = 1.0

LAMBDA_CONTINUITY = 1.0


# ============================================================
# VQPINN Defaults
# ============================================================

N_QUBITS = 4

N_Q_LAYERS = 4

QML_INTERFACE = "torch"

QML_DIFF_METHOD = "backprop"

QML_DEVICE_NAME = "default.qubit"

QUANTUM_OUTPUT_DIM = 8


# ============================================================
# VQLS Defaults
# ============================================================

VQLS_MAX_ITER = 100

VQLS_TOLERANCE = 1e-8

PICARD_MAX_ITER = 25

PICARD_TOLERANCE = 1e-8


# ============================================================
# Evaluation
# ============================================================

EPS = 1e-12


# ============================================================
# Visualization
# ============================================================

DEFAULT_FIGSIZE = (8, 6)

DEFAULT_CMAP = "viridis"

DPI = 150


# ============================================================
# Structured Config Objects
# ============================================================

@dataclass
class FluidConfig:
    reynolds_number: float = REYNOLDS_NUMBER
    viscosity: float = KINEMATIC_VISCOSITY


@dataclass
class SpectralConfig:
    nx: int = DEFAULT_NX
    ny: int = DEFAULT_NY
    dt: float = DEFAULT_DT
    t_final: float = DEFAULT_T_FINAL


@dataclass
class PINNConfig:
    hidden_dim: int = PINN_HIDDEN_DIM
    num_hidden_layers: int = PINN_NUM_HIDDEN_LAYERS
    learning_rate: float = PINN_LEARNING_RATE
    epochs: int = PINN_EPOCHS
    fourier_features_dim: int = FOURIER_FEATURES_DIM


@dataclass
class QuantumConfig:
    n_qubits: int = N_QUBITS
    n_layers: int = N_Q_LAYERS
    device_name: str = QML_DEVICE_NAME


@dataclass
class DatasetConfig:
    n_collocation: int = N_COLLOCATION
    n_initial: int = N_INITIAL_CONDITION
    n_boundary: int = N_BOUNDARY