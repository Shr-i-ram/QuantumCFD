# FLUID_SOLVERS

A research-oriented framework for solving the incompressible Navier–Stokes equations using a combination of classical numerical methods, Physics-Informed Neural Networks (PINNs), Variational Quantum Physics-Informed Neural Networks (VQPINNs), and Variational Quantum Linear Solvers (VQLS).

The repository provides a unified environment for comparing traditional computational fluid dynamics (CFD) approaches with modern machine learning and quantum-inspired techniques on benchmark fluid flow problems. The primary test case implemented in this project is the two-dimensional Taylor–Green vortex, a well-known analytical solution of the incompressible Navier–Stokes equations frequently used for validating numerical solvers.

---

# Table of Contents

* [Overview](#overview)
* [Motivation](#motivation)
* [Key Features](#key-features)
* [Repository Structure](#repository-structure)
* [Scientific Background](#scientific-background)
* [Taylor–Green Vortex Problem](#taylor-green-vortex-problem)
* [Governing Equations](#governing-equations)
* [Numerical Approaches Included in this Repository](#numerical-approaches-included-in-this-repository)

---

# Overview

Computational Fluid Dynamics (CFD) has traditionally relied on numerical discretization techniques such as finite difference methods, finite volume methods, finite element methods, and spectral methods. These approaches are highly accurate but often become computationally expensive when solving large-scale nonlinear partial differential equations.

Recent advances in machine learning and quantum computing have introduced alternative paradigms for approximating PDE solutions:

1. Physics-Informed Neural Networks (PINNs)
2. Quantum Machine Learning based PINNs (VQPINNs)
3. Variational Quantum Linear Solvers (VQLS)
4. Hybrid Classical–Quantum PDE Solvers

This repository explores and compares all of these approaches within a common framework.

The objective is not merely to solve a fluid mechanics problem but to study how different computational paradigms behave when applied to the same governing equations.

The framework allows direct comparison between:

| Method                 | Type                          |
| ---------------------- | ----------------------------- |
| Pseudo-Spectral Solver | Classical Numerical           |
| PINN                   | Deep Learning                 |
| VQPINN                 | Quantum Machine Learning      |
| VQLS Solver            | Variational Quantum Algorithm |

All implementations are organized into reusable Python modules rather than standalone notebooks, making the repository suitable for experimentation, benchmarking, and future research extensions.

---

# Motivation

The Navier–Stokes equations govern a vast range of physical systems including:

* Atmospheric dynamics
* Aerodynamics
* Ocean circulation
* Combustion systems
* Industrial fluid processes
* Turbulence modeling
* Weather prediction
* Plasma dynamics

While numerical solvers have achieved remarkable success, several challenges remain:

* High computational cost
* Large memory requirements
* Difficulties in uncertainty quantification
* Scaling limitations for large systems
* Expensive parameter sweeps

Physics-Informed Neural Networks attempt to address some of these challenges by replacing explicit numerical discretization with neural network function approximators constrained by physical laws.

Similarly, variational quantum algorithms have emerged as promising candidates for solving large-scale linear systems and optimization problems that arise throughout scientific computing.

This repository was created to investigate the strengths and limitations of these approaches when applied to a common fluid dynamics benchmark.

---

# Key Features

## Classical CFD

* Pseudo-spectral Navier–Stokes solver
* FFT-based derivatives
* Periodic boundary conditions
* High-order spatial accuracy
* Taylor–Green vortex validation

## Physics-Informed Neural Networks

* Fully connected neural networks
* Fourier feature encoding
* Automatic differentiation
* PDE residual minimization
* Curriculum learning strategy

## Variational Quantum PINNs

* PennyLane implementation
* Angle embedding
* Strongly entangling layers
* Hybrid classical–quantum architecture
* Physics-informed training

## Variational Quantum Linear Solver

* Amplitude embedding
* Hardware-efficient ansatz
* Quantum residual minimization
* Spectral preconditioning
* Picard linearization framework

## Evaluation Framework

* L2 error
* L∞ error
* Correlation coefficient
* Energy spectrum analysis
* Vorticity comparison
* Runtime benchmarking

## Visualization Tools

* Velocity magnitude plots
* Vorticity plots
* Energy spectrum plots
* Training curves
* Prediction vs reference comparisons

---

# Repository Structure

```text
FLUID_SOLVERS/
│
├── data/
│   ├── outputs/
│   └── reference/
│
├── models/
│   └── pinn_ns.pt
│
├── notebooks/
│   ├── PINN_NavierStokes_TaylorGreen.ipynb
│   ├── PseudoSpectral_TaylorGreen_Solver.ipynb
│   ├── VQPINN_TaylorGreen_PennyLane.ipynb
│   └── VQLS_NavierStokes_Solver.ipynb
│
├── src/
│   ├── config.py
│   │
│   ├── datasets/
│   │   └── training_data.py
│   │
│   ├── evaluation/
│   │   ├── benchmark.py
│   │   ├── interpolation.py
│   │   └── metrics.py
│   │
│   ├── models/
│   │   ├── pinn.py
│   │   ├── quantum_layers.py
│   │   ├── vqpinn.py
│   │   └── vqls.py
│   │
│   ├── physics/
│   │   ├── navier_stokes.py
│   │   ├── spectral.py
│   │   └── taylor_green.py
│   │
│   ├── solvers/
│   │   ├── pseudo_spectral_solver.py
│   │   └── picard_solver.py
│   │
│   ├── training/
│   │   ├── pinn_trainer.py
│   │   └── vqpinn_trainer.py
│   │
│   └── visualization/
│       └── plots.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Scientific Background

Fluid motion is governed by conservation laws.

The three primary conservation principles are:

### Conservation of Mass

Mass cannot be created or destroyed.

### Conservation of Momentum

Newton's Second Law applied to fluid motion.

### Conservation of Energy

Energy is conserved throughout fluid interactions.

For incompressible fluids, these conservation principles lead to the Navier–Stokes equations.

The incompressibility assumption implies:

[
\nabla \cdot \mathbf{u}=0
]

where

[
\mathbf{u}=(u,v)
]

represents the velocity field.

---

# Taylor–Green Vortex Problem

The Taylor–Green vortex is one of the most widely used benchmark problems in computational fluid dynamics.

It possesses a known analytical solution and therefore provides an ideal environment for evaluating numerical and machine learning solvers.

The velocity field is given by:

[
u(x,y,t)=
\sin(x)
\cos(y)
e^{-2\nu t}
]

[
v(x,y,t)=
-\cos(x)
\sin(y)
e^{-2\nu t}
]

where:

* (x) and (y) denote spatial coordinates
* (t) denotes time
* (\nu) is the kinematic viscosity

The corresponding pressure field is:

[
p(x,y,t)=
-\frac14
\left(
\cos(2x)+\cos(2y)
\right)
e^{-4\nu t}
]

This exact solution enables direct quantitative validation of:

* PINN predictions
* VQPINN predictions
* Spectral solver solutions
* VQLS approximations

---

# Governing Equations

The two-dimensional incompressible Navier–Stokes equations are

### Momentum Equation

[
\frac{\partial \mathbf{u}}{\partial t}
+
(\mathbf{u}\cdot\nabla)\mathbf{u}
=================================

-\nabla p
+
\nu\nabla^2\mathbf{u}
]

### Continuity Equation

[
\nabla\cdot\mathbf{u}=0
]

Expanding the momentum equations into component form:

[
u_t
+
u u_x
+
v u_y
=====

-p_x
+
\nu
(u_{xx}+u_{yy})
]

[
v_t
+
u v_x
+
v v_y
=====

-p_y
+
\nu
(v_{xx}+v_{yy})
]

subject to the incompressibility constraint

[
u_x+v_y=0
]

These equations constitute the physical constraints enforced throughout the machine learning models implemented in this repository.

---

# Numerical Approaches Included in this Repository

This repository contains four fundamentally different approaches for solving fluid flow problems.

### 1. Pseudo-Spectral Solver

A traditional CFD solver based on Fourier transforms.

Spatial derivatives are computed in spectral space using FFTs, resulting in very high accuracy for periodic domains.

### 2. Physics-Informed Neural Network (PINN)

A deep neural network approximates the solution field directly while minimizing both data loss and PDE residuals.

### 3. Variational Quantum PINN (VQPINN)

A hybrid classical–quantum architecture where quantum circuits act as trainable feature extractors within a physics-informed framework.

### 4. Variational Quantum Linear Solver (VQLS)

A variational quantum algorithm designed to approximately solve linear systems arising from Picard-linearized Navier–Stokes equations.

Together these methods provide a comprehensive platform for studying classical numerical computing, scientific machine learning, and quantum computing techniques for fluid dynamics.

# Pseudo-Spectral Navier–Stokes Solver

## Overview

The pseudo-spectral solver serves as the classical baseline against which all machine learning and quantum approaches are compared.

Spectral methods are among the most accurate numerical techniques available for solving PDEs on periodic domains because derivatives are computed in Fourier space rather than through finite difference approximations.

The solver implemented in this repository operates on a two-dimensional periodic domain:

[
(x,y)\in [0,2\pi]^2
]

and evolves the incompressible Navier–Stokes equations through time.

---

## Why Spectral Methods?

Traditional finite difference methods approximate derivatives locally.

For example,

[
\frac{\partial u}{\partial x}
\approx
\frac{u_{i+1}-u_{i-1}}
{2\Delta x}
]

Such approximations introduce truncation errors.

Spectral methods instead represent the solution as a Fourier series:

[
u(x,y)
======

\sum_k
\hat{u}_k
e^{ikx}
]

Differentiation becomes:

[
\frac{\partial u}{\partial x}
=============================

\sum_k
ik\hat{u}_k
e^{ikx}
]

which is exact in spectral space.

---

## Solver Workflow

The pseudo-spectral solver performs the following operations:

1. Transform velocity field to Fourier space.
2. Compute spatial derivatives spectrally.
3. Evaluate nonlinear convective terms.
4. Apply incompressibility projection.
5. Advance solution using time integration.
6. Transform back to physical space.

The workflow can be summarized as:

```text
Velocity Field
      ↓
FFT
      ↓
Spectral Derivatives
      ↓
Nonlinear Terms
      ↓
Pressure Projection
      ↓
Time Integration
      ↓
IFFT
      ↓
Updated Velocity Field
```

---

## Advantages

* Spectral accuracy
* Fast FFT-based operations
* Ideal for periodic domains
* Well-established CFD benchmark
* High-quality reference solutions

---

# Physics-Informed Neural Network (PINN)

## Concept

A Physics-Informed Neural Network approximates the solution of a PDE using a neural network while enforcing physical laws through the loss function.

Rather than discretizing the governing equations on a mesh, the network learns a continuous mapping:

[
(x,y,t)
\rightarrow
(u,v,p)
]

The PDE itself becomes part of the optimization objective.

---

## PINN Architecture

The PINN implemented in this repository follows:

```text
Input
(x,y,t)

      ↓

Fourier Feature Encoding

      ↓

Fully Connected Network

      ↓

(u,v,p)
```

---

## Fourier Feature Encoding

High-frequency PDE solutions are often difficult for neural networks to learn.

To address this issue, Fourier features are introduced.

For spatial coordinates:

[
\sin(2\pi kx)
]

[
\cos(2\pi kx)
]

[
\sin(2\pi ky)
]

[
\cos(2\pi ky)
]

for

[
k=1,\ldots,6
]

Time features are similarly encoded.

This transforms the original coordinates into a richer representation that improves convergence and accuracy.

---

## Network Architecture

The PINN architecture is:

```text
Input Features
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
```

Outputs:

```text
u
v
p
```

The final network contains approximately one hundred thousand trainable parameters.

---

## Automatic Differentiation

The PDE residuals are computed through automatic differentiation.

For example:

[
u_x
===

\frac{\partial u}
{\partial x}
]

[
u_t
===

\frac{\partial u}
{\partial t}
]

[
u_{xx}
======

\frac{\partial^2 u}
{\partial x^2}
]

These derivatives are computed exactly through backpropagation rather than finite differences.

---

## PDE Residuals

The PINN minimizes:

### x-Momentum Residual

[
f_u
===

u_t
+
uu_x
+
vu_y
+
p_x
---

\nu
(u_{xx}+u_{yy})
]

### y-Momentum Residual

[
f_v
===

v_t
+
uv_x
+
vv_y
+
p_y
---

\nu
(v_{xx}+v_{yy})
]

### Continuity Residual

[
f_c
===

u_x
+
v_y
]

---

## Initial Condition Loss

At

[
t=0
]

the network is constrained using the analytical Taylor–Green vortex solution.

The loss is:

[
L_{IC}
======

MSE(u,u_0)
+
MSE(v,v_0)
]

---

## Total PINN Loss

The total objective is:

[
L
=

10L_{IC}
+
L_{PDE}
+
L_{CONT}
]

where

[
L_{PDE}
=======

MSE(f_u)
+
MSE(f_v)
]

and

[
L_{CONT}
========

MSE(f_c)
]

---

## Curriculum Learning Strategy

Training is performed progressively.

Stage 1:

```text
t ≤ 0.03
```

Stage 2:

```text
t ≤ 0.06
```

Stage 3:

```text
t ≤ 0.10
```

This curriculum stabilizes optimization and improves convergence.

---

# Variational Quantum Physics-Informed Neural Network (VQPINN)

## Motivation

While classical neural networks can approximate PDE solutions effectively, they may require large parameter counts and extensive training.

Quantum machine learning introduces parameterized quantum circuits as trainable nonlinear feature extractors.

The VQPINN explores whether quantum representations can encode fluid dynamics information efficiently.

---

## Architecture Overview

```text
(x,y,t)
      ↓
Fourier Features
      ↓
Quantum Circuit
      ↓
Expectation Values
      ↓
Classical Head
      ↓
(u,v)
```

---

## Feature Encoding

The VQPINN uses Fourier feature encoding similar to the PINN.

Features include:

```text
x
y
t

sin(2x)
cos(2x)

sin(2y)
cos(2y)

sin(2t)
cos(2t)

sin(3x)
cos(3x)

sin(3y)
cos(3y)

sin(3t)
cos(3t)
```

These are projected to the quantum feature space.

---

## Quantum Device

The implementation uses PennyLane.

```python
qml.device(
    "default.qubit",
    wires=6
)
```

Number of qubits:

[
N_q = 6
]

---

## Angle Embedding

Classical features are encoded into quantum states using:

[
R_Y(\theta)
]

rotations.

This is implemented through:

```python
qml.AngleEmbedding(...)
```

---

## Variational Ansatz

The quantum circuit employs:

```python
qml.StronglyEntanglingLayers(...)
```

with

[
4
]

variational layers.

Each layer contains trainable parameters and entangling operations across all qubits.

---

## Quantum Measurements

The circuit outputs expectation values:

[
\langle Z_0\rangle
]

[
\langle Z_1\rangle
]

[
\langle Z_2\rangle
]

[
\langle Z_3\rangle
]

[
\langle Z_4\rangle
]

[
\langle Z_5\rangle
]

creating a six-dimensional quantum feature vector.

---

## Classical Readout Head

The measured quantum features are passed into:

```text
Linear(6,32)
Tanh
Linear(32,2)
```

producing:

```text
u
v
```

Unlike the classical PINN, pressure is not explicitly modeled.

---

## VQPINN Residuals

The pressure-free formulation minimizes:

[
u_t
+
uu_x
+
vu_y
----

\nu
(u_{xx}+u_{yy})
]

[
v_t
+
uv_x
+
vv_y
----

\nu
(v_{xx}+v_{yy})
]

along with the incompressibility constraint:

[
u_x+v_y=0
]

---

## VQPINN Loss

The objective function is:

[
L
=

L_{IC}
+
0.5L_{PDE}
+
0.5L_{CONT}
]

This weighting differs from the classical PINN and was chosen to improve training stability for the hybrid quantum architecture.

---

# Variational Quantum Linear Solver (VQLS)

## Motivation

Linear systems appear throughout scientific computing.

After linearization, many PDEs can be written as:

[
Ax=b
]

Quantum linear system algorithms seek efficient ways to approximate solutions to these systems.

The VQLS implementation in this repository investigates variational approaches to this problem.

---

## Picard Linearization

The nonlinear Navier–Stokes equations are transformed into a sequence of linear systems.

At iteration (n):

[
A^{(n)}
=======

## I

\Delta t
\left(
\nu\nabla^2
-----------

## u^{(n)}\partial_x

v^{(n)}\partial_y
\right)
]

This creates a sparse linear system that must be solved at every iteration.

---

## Spectral Preconditioning

To improve conditioning:

[
M
\approx
(I-\Delta t\nu\nabla^2)^{-1}
]

is applied.

The effective system becomes:

[
(MAM)y
======

Mb
]

followed by:

[
x=My
]

This significantly improves optimization behavior.

---

## Amplitude Embedding

The right-hand-side vector is embedded into a quantum state using:

```python
qml.AmplitudeEmbedding(...)
```

The vector is padded and normalized automatically.

---

## Hardware Efficient Ansatz

The VQLS circuit uses:

* RY rotations
* RZ rotations
* Ring CNOT entanglement

The parameter tensor has shape:

[
(n_{layers},
n_{qubits},
2)
]

with:

[
n_{layers}=8
]

---

## Quantum Measurements

Expectation values of all qubits are measured:

[
\langle Z_i\rangle
]

for

[
i=0,\ldots,n_q-1
]

These measurements form a compact latent representation of the solution.

---

## Expansion Strategy

The PDE system contains:

[
1024
]

degrees of freedom.

The quantum circuit produces only:

[
10
]

measurements.

The notebook therefore expands the quantum output by repetition until the full state dimension is reached.

This allows experimentation with quantum-inspired representations while maintaining computational tractability.

---

## VQLS Objective Function

The residual is:

[
r
=

## A(Mx)

Mb
]

The optimization objective is:

[
L
=

|r|^2
+
10^{-2}
L_{smooth}
]

where

[
L_{smooth}
]

penalizes sharp spatial oscillations in the reconstructed field.

---

## Optimization Procedure

### Pretraining

100 epochs:

[
MSE(x,\omega)
]

### Main Training

Adam optimizer:

```text
Learning Rate = 1e-2
Epochs = 600
```

The resulting solution is then projected back through the spectral preconditioner to recover the final vorticity field.

# Installation

## Clone the Repository

```bash
git clone https://github.com/Shr-i-ram/QuantumCFD.git

cd FLUID_SOLVERS
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python -c "import torch, pennylane, scipy, numpy; print('Installation Successful')"
```

---

# Quick Start

This repository provides three independent workflows:

1. Classical Pseudo-Spectral CFD Solver
2. Physics-Informed Neural Network Solver
3. Variational Quantum PINN Solver
4. Variational Quantum Linear Solver

Each workflow can be executed independently.

---

# Running the Pseudo-Spectral Solver

The pseudo-spectral solver provides high-accuracy reference solutions.

```python
from src.solvers.pseudo_spectral_solver import (
    PseudoSpectralSolver
)

solver = PseudoSpectralSolver(
    N=64,
    reynolds_number=100
)

results = solver.solve(
    t_final=0.1
)
```

The returned dictionary contains:

```python
{
    "u": u,
    "v": v,
    "omega": omega,
    "time": t
}
```

---

# Training a PINN

## Create Model

```python
from src.models.pinn import (
    create_pinn
)

model = create_pinn()
```

---

## Create Trainer

```python
from src.training.pinn_trainer import (
    PINNTrainer
)

trainer = PINNTrainer(
    model=model,
    reynolds_number=100
)
```

---

## Train

```python
trainer.train(
    epochs=5000
)
```

---

## Save Model

```python
trainer.save(
    "models/pinn_ns.pt"
)
```

---

## Load Model

```python
trainer.load(
    "models/pinn_ns.pt"
)
```

---

## Predict

```python
predictions = trainer.predict(
    x,
    y,
    t
)

u = predictions["u"]
v = predictions["v"]
p = predictions["p"]
```

---

# Training a VQPINN

## Create Model

```python
from src.models.vqpinn import (
    create_vqpinn
)

model = create_vqpinn()
```

---

## Create Trainer

```python
from src.training.vqpinn_trainer import (
    VQPINNTrainer
)

trainer = VQPINNTrainer(
    model=model,
    reynolds_number=100
)
```

---

## Train

```python
trainer.train(
    epochs=2000
)
```

---

## Save Checkpoint

```python
trainer.save(
    "models/vqpinn.pt"
)
```

---

## Predict

```python
predictions = trainer.predict(
    x,
    y,
    t
)

u = predictions["u"]
v = predictions["v"]
```

---

# Running the VQLS Solver

The VQLS implementation solves Picard-linearized Navier–Stokes systems using a variational quantum algorithm.

---

## Create Solver

```python
from src.solvers.picard_solver import (
    PicardLinearizedSolver
)

solver = PicardLinearizedSolver(
    N=32,
    reynolds_number=100,
    dt=0.05
)
```

---

## Classical Solve

```python
omega_next, runtime, kappa = (
    solver.classical_step(
        omega,
        u,
        v
    )
)
```

---

## Quantum Solve

```python
omega_next, runtime, calls, kappa = (
    solver.quantum_step(
        omega,
        u,
        v
    )
)
```

Returned values:

| Variable   | Description                 |
| ---------- | --------------------------- |
| omega_next | Updated vorticity           |
| runtime    | Execution time              |
| calls      | Quantum circuit evaluations |
| kappa      | Estimated condition number  |

---

# Benchmarking

The repository includes a comprehensive benchmarking framework.

---

## Evaluate Against Taylor–Green Vortex

```python
from src.evaluation.benchmark import (
    benchmark_taylor_green
)

results = benchmark_taylor_green(
    u_pred,
    v_pred,
    x,
    y,
    t,
    nu
)
```

---

## Evaluate Against Reference Dataset

```python
from src.evaluation.benchmark import (
    benchmark_against_reference
)

results = benchmark_against_reference(
    u_pred,
    v_pred,
    x_grid,
    y_grid,
    "data/reference/reference_data_safe.npz"
)
```

---

## Available Metrics

### Velocity Metrics

* Relative L2 Error
* Relative L∞ Error
* Correlation Coefficient

### Vorticity Metrics

* Relative L2 Error
* Relative L∞ Error
* Correlation Coefficient

### Spectral Metrics

* Energy Spectrum Error
* Spectral Correlation

### Runtime Metrics

* Execution Time
* Training Time
* Circuit Evaluation Count

---

# Visualization

The visualization package contains utilities for examining flow fields and training performance.

---

## Velocity Magnitude

```python
from src.visualization.plots import (
    plot_velocity_magnitude
)

plot_velocity_magnitude(
    u,
    v
)
```

---

## Vorticity

```python
plot_vorticity(
    omega
)
```

---

## Training Curves

```python
plot_training_history(
    trainer.get_history()
)
```

---

## Energy Spectrum

```python
plot_energy_spectrum(
    k,
    E
)
```

---

## Prediction vs Reference

```python
plot_comparison(
    prediction,
    reference
)
```

---

# Expected Outputs

During execution the repository can generate:

```text
data/
└── outputs/
    ├── velocity.png
    ├── vorticity.png
    ├── spectrum.png
    ├── benchmark.csv
    └── loss_history.png
```

---

# Example Research Questions

This repository may be used to investigate:

* Can PINNs accurately reproduce analytical fluid solutions?
* How do quantum-enhanced PINNs compare against classical PINNs?
* How does a VQLS compare against classical Krylov solvers?
* What are the effects of spectral preconditioning on quantum optimization?
* Can hybrid quantum-classical methods improve scientific computing workflows?
* How do training dynamics differ between classical and quantum architectures?

---

# Extending the Repository

The framework is intentionally modular.

New physics problems can be added by extending:

```text
src/physics/
```

New machine learning architectures can be added through:

```text
src/models/
```

New numerical solvers can be added through:

```text
src/solvers/
```

New benchmarks and metrics can be added through:

```text
src/evaluation/
```

---

# Current Limitations

This repository is intended primarily as a research and educational framework.

Several limitations should be noted:

* Simulated quantum devices are used
* Noisy quantum hardware is not currently supported
* Three-dimensional flows are not implemented
* Turbulent flow regimes are not the primary focus
* Large-scale distributed training is not included
* Adaptive mesh refinement is not implemented

Future work will address many of these limitations.

---

# Future Work

Planned extensions include:

### Physics

* Three-dimensional Navier–Stokes equations
* Burgers equation
* Shallow water equations
* Magnetohydrodynamics
* Compressible flows

### Machine Learning

* Fourier Neural Operators
* DeepONets
* Transformer-based PDE solvers
* Physics-Informed Graph Neural Networks

### Quantum Computing

* Quantum Convolutional Neural Networks
* Quantum Fourier Neural Operators
* QSVT-based linear solvers
* HHL-inspired architectures
* Error-mitigated quantum workflows

### High Performance Computing

* GPU acceleration
* Distributed training
* Multi-node simulations
* Hybrid HPC–Quantum pipelines

---

# Citation

If you use this repository in academic work, please cite:

```bibtex
@software{fluid_solvers,
  title={FLUID_SOLVERS: Classical, PINN, VQPINN and VQLS Framework for Fluid Dynamics},
  author={Shriram},
  year={2026},
  url={https://github.com/yourusername/FLUID_SOLVERS}
}
```

---

# Acknowledgements

This project builds upon ideas from several fields:

* Computational Fluid Dynamics
* Scientific Machine Learning
* Physics-Informed Neural Networks
* Quantum Machine Learning
* Variational Quantum Algorithms
* Numerical Linear Algebra

The implementation draws inspiration from research in:

* Spectral Methods
* PINNs
* Variational Quantum Circuits
* Quantum Linear Solvers
* Hybrid Quantum-Classical Computing

---

# License

This project is released under the MIT License.

See the LICENSE file for details.

---

# Contact

For questions, suggestions, collaborations, or research discussions:

GitHub Issues and Pull Requests are welcome.

Researchers interested in quantum computing, scientific machine learning, computational fluid dynamics, or hybrid quantum-classical methods are encouraged to contribute.

---

**FLUID_SOLVERS** aims to provide a unified platform for exploring the intersection of numerical simulation, machine learning, and quantum computing in fluid dynamics.
