# src/physics/spectral.py

from __future__ import annotations

import numpy as np

from numpy.fft import (
    fft2,
    ifft2,
    fftfreq,
)


# ============================================================
# Wavenumber Utilities
# ============================================================

def create_wavenumbers(
    nx: int,
    ny: int,
    lx: float = 2.0 * np.pi,
    ly: float = 2.0 * np.pi,
):
    """
    Construct Fourier wavenumber grids.

    Returns
    -------
    KX, KY, K2
    """

    kx = fftfreq(nx, d=lx / nx) * 2.0 * np.pi
    ky = fftfreq(ny, d=ly / ny) * 2.0 * np.pi

    KX, KY = np.meshgrid(
        kx,
        ky,
        indexing="ij",
    )

    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-20

    return KX, KY, K2


# ============================================================
# FFT Wrappers
# ============================================================

def fft(field):
    return fft2(field)


def ifft(field_hat):
    return np.real(ifft2(field_hat))


# ============================================================
# Spectral Derivatives
# ============================================================

def spectral_derivative_x(
    field: np.ndarray,
    KX: np.ndarray,
):
    """
    ∂f/∂x
    """

    field_hat = fft2(field)

    return np.real(
        ifft2(
            1j * KX * field_hat
        )
    )


def spectral_derivative_y(
    field: np.ndarray,
    KY: np.ndarray,
):
    """
    ∂f/∂y
    """

    field_hat = fft2(field)

    return np.real(
        ifft2(
            1j * KY * field_hat
        )
    )


# ============================================================
# 2/3 Dealiasing Rule
# ============================================================

def dealias(
    arr_hat: np.ndarray,
):
    """
    Same implementation used in notebook.

    Applies the 2/3-rule filter.
    """

    arr_hat = arr_hat.copy()

    n = arr_hat.shape[0]

    cutoff = n // 3

    arr_hat[cutoff:-cutoff, :] = 0
    arr_hat[:, cutoff:-cutoff] = 0

    return arr_hat


# ============================================================
# Poisson Solver
# ============================================================

def poisson_solve(
    omega_hat: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
):
    """
    Solve

        ∇²ψ = -ω

    in Fourier space.
    """

    k2 = KX**2 + KY**2

    psi_hat = np.zeros_like(
        omega_hat,
        dtype=complex,
    )

    mask = k2 != 0

    psi_hat[mask] = (
        -omega_hat[mask]
        / k2[mask]
    )

    return psi_hat


# ============================================================
# Streamfunction -> Velocity
# ============================================================

def velocity_from_streamfunction(
    psi_hat: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
):
    """
    u =  ∂ψ/∂y
    v = -∂ψ/∂x
    """

    u_hat = 1j * KY * psi_hat
    v_hat = -1j * KX * psi_hat

    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))

    return u, v


# ============================================================
# Vorticity -> Velocity
# ============================================================

def velocity_from_vorticity(
    omega: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
):
    """
    Reconstruct velocity field from vorticity.
    """

    omega_hat = fft2(omega)

    psi_hat = poisson_solve(
        omega_hat,
        KX,
        KY,
    )

    return velocity_from_streamfunction(
        psi_hat,
        KX,
        KY,
    )


# ============================================================
# Velocity -> Vorticity
# ============================================================

def vorticity_from_uv(
    u: np.ndarray,
    v: np.ndarray,
):
    """
    ω = dv/dx - du/dy

    Uses same convention as notebook.
    """

    nx, ny = u.shape

    KX, KY, _ = create_wavenumbers(
        nx,
        ny,
    )

    u_hat = fft2(u)
    v_hat = fft2(v)

    omega_hat = (
        1j * KX * v_hat
        - 1j * KY * u_hat
    )

    omega = np.real(
        ifft2(omega_hat)
    )

    return omega


# ============================================================
# Divergence
# ============================================================

def divergence(
    u: np.ndarray,
    v: np.ndarray,
):
    """
    Compute ∇·u.
    """

    nx, ny = u.shape

    KX, KY, _ = create_wavenumbers(
        nx,
        ny,
    )

    u_hat = fft2(u)
    v_hat = fft2(v)

    div_hat = (
        1j * KX * u_hat
        + 1j * KY * v_hat
    )

    return np.real(
        ifft2(div_hat)
    )


def divergence_l2_norm(
    u: np.ndarray,
    v: np.ndarray,
):
    """
    Same diagnostic used in notebook.
    """

    div = divergence(u, v)

    nx, ny = div.shape

    return (
        np.linalg.norm(div)
        / np.sqrt(nx * ny)
    )


# ============================================================
# Nonlinear Advection Term
# ============================================================

def nonlinear_term(
    omega_hat: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
):
    """
    Exact notebook implementation.
    """

    psi_hat = poisson_solve(
        omega_hat,
        KX,
        KY,
    )

    u, v = velocity_from_streamfunction(
        psi_hat,
        KX,
        KY,
    )

    omega = -np.real(
        ifft2(omega_hat)
    )

    domega_dx = -np.real(
        ifft2(
            1j * KX * omega_hat
        )
    )

    domega_dy = -np.real(
        ifft2(
            1j * KY * omega_hat
        )
    )

    adv = (
        u * domega_dx
        + v * domega_dy
    )

    adv_hat = fft2(adv)

    return dealias(adv_hat)


# ============================================================
# Energy
# ============================================================

def kinetic_energy(
    u: np.ndarray,
    v: np.ndarray,
):
    """
    Domain-averaged kinetic energy.
    """

    return 0.5 * np.mean(
        u**2 + v**2
    )


def enstrophy(
    omega: np.ndarray,
):
    """
    Domain-averaged enstrophy.
    """

    return 0.5 * np.mean(
        omega**2
    )


# ============================================================
# Isotropic Energy Spectrum
# ============================================================

def energy_spectrum(
    u: np.ndarray,
    v: np.ndarray,
):
    """
    Same implementation used in notebook.
    """

    nx, ny = u.shape

    u_hat = fft2(u)
    v_hat = fft2(v)

    E2D = (
        0.5
        * (
            np.abs(u_hat) ** 2
            + np.abs(v_hat) ** 2
        )
        / (nx * ny) ** 2
    )

    kx = (
        fftfreq(nx, d=(2 * np.pi) / nx)
        * nx
        / (2 * np.pi)
    )

    ky = (
        fftfreq(ny, d=(2 * np.pi) / ny)
        * ny
        / (2 * np.pi)
    )

    KX, KY = np.meshgrid(
        kx,
        ky,
        indexing="ij",
    )

    K = np.sqrt(
        KX**2 + KY**2
    )

    kmax = int(
        np.floor(K.max())
    )

    E1D = np.zeros(kmax + 1)
    counts = np.zeros(kmax + 1)

    for i in range(nx):
        for j in range(ny):

            kbin = int(
                np.round(K[i, j])
            )

            if kbin <= kmax:
                E1D[kbin] += E2D[i, j]
                counts[kbin] += 1

    counts[counts == 0] = 1.0

    return (
        np.arange(kmax + 1),
        E1D / counts,
    )