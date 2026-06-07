# src/solvers/pseudo_spectral_solver.py

from __future__ import annotations

import numpy as np

from src.physics.spectral import (
    fft,
    ifft,
    dealias,
    create_wavenumbers,
    nonlinear_term,
    poisson_solve,
    velocity_from_streamfunction,
    kinetic_energy,
    enstrophy,
)

from src.physics.taylor_green import (
    generate_uniform_grid,
    initial_vorticity_field,
)


class PseudoSpectralSolver:
    """
    2D incompressible Navier-Stokes solver
    in vorticity-streamfunction form.

    Governing equation

        dω/dt + u·∇ω = ν∇²ω
    """

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        viscosity: float = 0.01,
        dt: float = 1e-3,
        t_final: float = 0.1,
        lx: float = 2.0 * np.pi,
        ly: float = 2.0 * np.pi,
    ):

        self.nx = nx
        self.ny = ny

        self.lx = lx
        self.ly = ly

        self.nu = viscosity

        self.dt = dt
        self.t_final = t_final

        self.time = 0.0

        (
            self.KX,
            self.KY,
            self.K2,
        ) = create_wavenumbers(
            nx,
            ny,
            lx,
            ly,
        )

        self.X, self.Y = generate_uniform_grid(
            nx,
            ny,
        )

        self.omega = initial_vorticity_field(
            self.X,
            self.Y,
        )

        self.omega_hat = fft(self.omega)

        self.energy_history = []
        self.enstrophy_history = []
        self.time_history = []

    # =====================================================
    # Diagnostics
    # =====================================================

    def velocity_field(self):
        """
        Recover velocity field from vorticity.
        """

        psi_hat = poisson_solve(
            self.omega_hat,
            self.KX,
            self.KY,
        )

        u, v = velocity_from_streamfunction(
            psi_hat,
            self.KX,
            self.KY,
        )

        return u, v

    def streamfunction(self):
        """
        Return physical-space streamfunction.
        """

        psi_hat = poisson_solve(
            self.omega_hat,
            self.KX,
            self.KY,
        )

        return ifft(psi_hat)

    def diagnostics(self):

        u, v = self.velocity_field()

        omega = ifft(self.omega_hat)

        self.energy_history.append(
            kinetic_energy(u, v)
        )

        self.enstrophy_history.append(
            enstrophy(omega)
        )

        self.time_history.append(
            self.time
        )

    # =====================================================
    # RHS
    # =====================================================

    def rhs(
        self,
        omega_hat,
    ):
        """
        Compute

            dω/dt

        in Fourier space.
        """

        adv_hat = nonlinear_term(
            omega_hat,
            self.KX,
            self.KY,
        )

        diffusion_hat = (
            -self.nu
            * self.K2
            * omega_hat
        )

        return (
            -adv_hat
            + diffusion_hat
        )

    # =====================================================
    # RK4
    # =====================================================

    def rk4_step(self):

        dt = self.dt

        k1 = self.rhs(
            self.omega_hat
        )

        k2 = self.rhs(
            self.omega_hat
            + 0.5 * dt * k1
        )

        k3 = self.rhs(
            self.omega_hat
            + 0.5 * dt * k2
        )

        k4 = self.rhs(
            self.omega_hat
            + dt * k3
        )

        self.omega_hat = (
            self.omega_hat
            + dt
            * (
                k1
                + 2 * k2
                + 2 * k3
                + k4
            )
            / 6.0
        )

        self.omega_hat = dealias(
            self.omega_hat
        )

        self.time += dt

    # =====================================================
    # Time Integration
    # =====================================================

    def step(self):

        self.rk4_step()

    def run(self):

        self.diagnostics()

        nsteps = int(
            np.round(
                self.t_final
                / self.dt
            )
        )

        for _ in range(nsteps):

            self.step()

            self.diagnostics()

        return self.solution()

    # =====================================================
    # Outputs
    # =====================================================

    def solution(self):

        omega = ifft(
            self.omega_hat
        )

        psi_hat = poisson_solve(
            self.omega_hat,
            self.KX,
            self.KY,
        )

        u, v = velocity_from_streamfunction(
            psi_hat,
            self.KX,
            self.KY,
        )

        psi = ifft(psi_hat)

        return {
            "x": self.X,
            "y": self.Y,
            "u": u,
            "v": v,
            "omega": omega,
            "psi": psi,
            "t": self.time,
        }

    # =====================================================
    # Save Reference Data
    # =====================================================

    def save_reference(
        self,
        filename,
    ):
        """
        Save benchmark solution.

        Compatible with
        reference_data_safe.npz
        workflow.
        """

        sol = self.solution()

        np.savez_compressed(
            filename,
            x=sol["x"],
            y=sol["y"],
            u=sol["u"],
            v=sol["v"],
            omega=sol["omega"],
            psi=sol["psi"],
            t=sol["t"],
            energy=np.array(
                self.energy_history
            ),
            enstrophy=np.array(
                self.enstrophy_history
            ),
            time_history=np.array(
                self.time_history
            ),
        )

    # =====================================================
    # Convenience
    # =====================================================

    def get_velocity(self):

        return self.velocity_field()

    def get_vorticity(self):

        return ifft(
            self.omega_hat
        )

    def get_streamfunction(self):

        psi_hat = poisson_solve(
            self.omega_hat,
            self.KX,
            self.KY,
        )

        return ifft(
            psi_hat
        )