# src/solvers/picard_solver.py

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from src.models.vqls import (
    VQLS_Solver,
    VQLS_LAYERS,
)


# ============================================================
# Finite Difference Operators
# ============================================================

def circ_diff_1d(
    n: int,
    dx: float,
):
    """
    Periodic centered difference.

        [-1/2, 0, +1/2] / dx
    """

    e = np.ones(n)

    D = sp.diags(
        [
            -0.5 * e,
             0.5 * e,
        ],
        offsets=[-1, 1],
        shape=(n, n),
        format="lil",
    ) / dx

    D[0, -1] = -0.5 / dx
    D[-1, 0] = 0.5 / dx

    return D.tocsc()


def periodic_laplacian_1d(
    n: int,
    dx: float,
):
    """
    Periodic second derivative.
    """

    e = np.ones(n)

    L = sp.diags(
        [
            -2.0 * e,
             1.0 * e,
             1.0 * e,
        ],
        offsets=[0, 1, -1],
        shape=(n, n),
        format="lil",
    ) / (dx**2)

    L[0, -1] = 1.0 / dx**2
    L[-1, 0] = 1.0 / dx**2

    return L.tocsc()


# ============================================================
# Matrix Assembly
# ============================================================

def build_operators(
    N: int,
    dx: float,
):
    """
    Construct periodic FD operators.
    """

    Dx1 = circ_diff_1d(
        N,
        dx,
    )

    Dy1 = circ_diff_1d(
        N,
        dx,
    )

    Ix = sp.identity(
        N,
        format="csc",
    )

    Iy = sp.identity(
        N,
        format="csc",
    )

    Dx = sp.kron(
        Iy,
        Dx1,
        format="csc",
    )

    Dy = sp.kron(
        Dy1,
        Ix,
        format="csc",
    )

    L1 = periodic_laplacian_1d(
        N,
        dx,
    )

    L2D = (
        sp.kron(
            Iy,
            L1,
            format="csc",
        )
        +
        sp.kron(
            L1,
            Ix,
            format="csc",
        )
    )

    I_N2 = sp.identity(
        N * N,
        format="csc",
    )

    return {
        "Dx": Dx,
        "Dy": Dy,
        "L2D": L2D,
        "I": I_N2,
    }


# ============================================================
# Picard Solver
# ============================================================

class PicardLinearizedSolver:
    """
    Notebook-faithful Picard
    linearization framework.
    """

    def __init__(
        self,
        N: int = 32,
        L: float = 2.0 * np.pi,
        reynolds_number: float = 100.0,
        dt: float = 0.05,
    ):

        self.N = N
        self.L = L

        self.dx = L / N

        self.re = reynolds_number
        self.nu = (
            1.0 / reynolds_number
        )

        self.dt = dt

        ops = build_operators(
            N,
            self.dx,
        )

        self.Dx = ops["Dx"]
        self.Dy = ops["Dy"]
        self.L2D = ops["L2D"]
        self.I = ops["I"]

    # =====================================================
    # Matrix Assembly
    # =====================================================

    def assemble_An(
        self,
        u_freeze,
        v_freeze,
    ):
        """
        A^(n)

        I - dt(
              ν∇²
            - u∂x
            - v∂y
        )
        """

        U = sp.diags(
            u_freeze.ravel(),
            0,
            shape=(
                self.N * self.N,
                self.N * self.N,
            ),
            format="csc",
        )

        V = sp.diags(
            v_freeze.ravel(),
            0,
            shape=(
                self.N * self.N,
                self.N * self.N,
            ),
            format="csc",
        )

        A = (
            self.I
            -
            self.dt
            * (
                self.nu * self.L2D
                - U @ self.Dx
                - V @ self.Dy
            )
        )

        return A

    # =====================================================
    # Condition Number
    # =====================================================

    def cond_number_estimate(
        self,
        A,
    ):

        try:

            smax = spla.svds(
                A,
                k=1,
                which="LM",
                return_singular_vectors=False,
            )[0]

            smin = spla.svds(
                A,
                k=1,
                which="SM",
                return_singular_vectors=False,
            )[0]

            return float(
                abs(
                    smax / smin
                )
            )

        except Exception:

            return np.nan

    # =====================================================
    # Spectral Preconditioner
    # =====================================================

    def _wave_numbers(
        self,
    ):

        kx = (
            np.fft.fftfreq(
                self.N,
                d=self.dx,
            )
            * 2.0
            * np.pi
        )

        ky = (
            np.fft.fftfreq(
                self.N,
                d=self.dx,
            )
            * 2.0
            * np.pi
        )

        return np.meshgrid(
            kx,
            ky,
            indexing="ij",
        )

    def apply_M(
        self,
        vec_flat,
    ):
        """
        M ≈
        (I - dt ν ∇²)^(-1)
        """

        w = vec_flat.reshape(
            self.N,
            self.N,
        )

        KX, KY = (
            self._wave_numbers()
        )

        denom = (
            1.0
            +
            self.dt
            * self.nu
            * (
                KX**2
                + KY**2
            )
        )

        w_hat = np.fft.fft2(
            w
        )

        z_hat = (
            w_hat / denom
        )

        z = np.real(
            np.fft.ifft2(
                z_hat
            )
        )

        return z.ravel()

    def apply_M_inv(
        self,
        vec_flat,
    ):

        w = vec_flat.reshape(
            self.N,
            self.N,
        )

        KX, KY = (
            self._wave_numbers()
        )

        num = (
            1.0
            +
            self.dt
            * self.nu
            * (
                KX**2
                + KY**2
            )
        )

        w_hat = np.fft.fft2(
            w
        )

        z_hat = (
            w_hat * num
        )

        z = np.real(
            np.fft.ifft2(
                z_hat
            )
        )

        return z.ravel()

    # =====================================================
    # M A M
    # =====================================================

    def A_precond_matvec(
        self,
        vec,
        u_freeze,
        v_freeze,
    ):
        """
        M A M y
        """

        z = self.apply_M(
            vec
        )

        A = self.assemble_An(
            u_freeze,
            v_freeze,
        )

        w = A @ z

        return self.apply_M(
            w
        )

    # =====================================================
    # Classical CG
    # =====================================================

    def classical_step(
        self,
        omega_n,
        u_freeze,
        v_freeze,
        tol=1e-10,
        maxit=1000,
    ):
        """
        Solve:

            (MAM)y = Mb

        then:

            x = My
        """

        A = self.assemble_An(
            u_freeze,
            v_freeze,
        )

        b = (
            omega_n.ravel()
            .astype(np.float64)
        )

        kappa_raw = (
            self.cond_number_estimate(
                A
            )
        )

        def mv(y):

            return (
                self.A_precond_matvec(
                    y,
                    u_freeze,
                    v_freeze,
                )
            )

        Aeff = spla.LinearOperator(
            (
                self.N * self.N,
                self.N * self.N,
            ),
            matvec=mv,
            dtype=np.float64,
        )

        Mb = self.apply_M(
            b
        )

        t0 = time.time()

        y, info = spla.cg(
            Aeff,
            Mb,
            rtol=tol,
            atol=0.0,
            maxiter=maxit,
        )

        t1 = time.time()

        if info != 0:

            print(
                f"[CG] Warning "
                f"(info={info})"
            )

        x = self.apply_M(
            y
        )

        omega_np1 = x.reshape(
            self.N,
            self.N,
        )

        return (
            omega_np1,
            t1 - t0,
            kappa_raw,
        )

    # =====================================================
    # Quantum Step
    # =====================================================

    def quantum_step(
        self,
        omega_n,
        u_freeze,
        v_freeze,
        epochs=600,
        lr=1e-2,
    ):
        """
        Notebook VQLS solve.
        """

        b = (
            omega_n.ravel()
            .astype(np.float32)
        )

        def mv_numpy(
            y_np,
        ):
            return (
                self.A_precond_matvec(
                    y_np,
                    u_freeze,
                    v_freeze,
                )
                .astype(np.float32)
            )

        model = VQLS_Solver(
            vector_size=self.N
            * self.N,
            n_layers=VQLS_LAYERS,
        )

        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        omega_flat = (
            torch.tensor(
                omega_n.ravel(),
                dtype=torch.float32,
            )
        )

        # -------------------------------
        # Pretraining
        # -------------------------------

        for _ in range(100):

            opt.zero_grad()

            _, x_amp = model(
                mv_numpy,
                b,
            )

            pre_loss = torch.mean(
                (
                    x_amp
                    - omega_flat
                )
                ** 2
            )

            pre_loss.backward()

            opt.step()

        print(
            "[Pretrain] done"
        )

        # -------------------------------
        # Main solve
        # -------------------------------

        t0 = time.time()

        for ep in range(
            epochs
        ):

            opt.zero_grad()

            loss, y = model(
                mv_numpy,
                b,
            )

            loss.backward()

            opt.step()

            if ep % 50 == 0:

                print(
                    f"[VQLS] "
                    f"ep {ep:4d} | "
                    f"loss={loss.item():.3e} | "
                    f"calls={model.call_count()}"
                )

        t1 = time.time()

        y_np = (
            y.detach()
            .cpu()
            .numpy()
        )

        x = self.apply_M(
            y_np
        ).reshape(
            self.N,
            self.N,
        )

        kappa_raw = (
            self.cond_number_estimate(
                self.assemble_An(
                    u_freeze,
                    v_freeze,
                )
            )
        )

        return (
            x,
            t1 - t0,
            model.call_count(),
            kappa_raw,
        )