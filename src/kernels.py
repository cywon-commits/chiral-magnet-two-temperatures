"""Monte Carlo kernels: 2D classical Heisenberg + Bloch-type DMI on the square lattice.

H = -J sum_<ij> S_i.S_j - D sum_<ij> r_ij.(S_i x S_j),  |S|=1, PBC.
Kernels follow prelim/tstar.py (same sign conventions, checked against it).
One "sweep" = 1 Metropolis sweep (sequential) + 2 over-relaxation sweeps (checkerboard).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def local_field(S, i, j, L, J, D):
    ip = (i + 1) % L; im = (i - 1) % L; jp = (j + 1) % L; jm = (j - 1) % L
    bx = J * (S[ip, j, 0] + S[im, j, 0] + S[i, jp, 0] + S[i, jm, 0])
    by = J * (S[ip, j, 1] + S[im, j, 1] + S[i, jp, 1] + S[i, jm, 1])
    bz = J * (S[ip, j, 2] + S[im, j, 2] + S[i, jp, 2] + S[i, jm, 2])
    # + D * sum_j S_j x r_ij ;  S x (+x) = (0, Sz, -Sy) ; S x (+y) = (-Sz, 0, Sx)
    by += D * (S[ip, j, 2] - S[im, j, 2]); bz += -D * (S[ip, j, 1] - S[im, j, 1])
    bx += -D * (S[i, jp, 2] - S[i, jm, 2]); bz += D * (S[i, jp, 0] - S[i, jm, 0])
    return bx, by, bz


@njit(cache=True)
def metropolis(S, T, J, D, sig):
    L = S.shape[0]; acc = 0
    for i in range(L):
        for j in range(L):
            bx, by, bz = local_field(S, i, j, L, J, D)
            nx = S[i, j, 0] + sig * np.random.randn(); ny = S[i, j, 1] + sig * np.random.randn(); nz = S[i, j, 2] + sig * np.random.randn()
            r = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz); nx *= r; ny *= r; nz *= r
            dE = -((nx - S[i, j, 0]) * bx + (ny - S[i, j, 1]) * by + (nz - S[i, j, 2]) * bz)
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                S[i, j, 0] = nx; S[i, j, 1] = ny; S[i, j, 2] = nz; acc += 1
    return acc / (L * L)


@njit(cache=True)
def overrelax(S, J, D):
    L = S.shape[0]
    for par in range(2):
        for i in range(L):
            for j in range(L):
                if (i + j) % 2 != par: continue
                bx, by, bz = local_field(S, i, j, L, J, D)
                b2 = bx * bx + by * by + bz * bz
                if b2 < 1e-12: continue
                f = 2.0 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz) / b2
                S[i, j, 0] = f * bx - S[i, j, 0]; S[i, j, 1] = f * by - S[i, j, 1]; S[i, j, 2] = f * bz - S[i, j, 2]


@njit(cache=True)
def sweep(S, T, J, D, sig):
    a = metropolis(S, T, J, D, sig); overrelax(S, J, D); overrelax(S, J, D)
    return a


@njit(cache=True)
def energy(S, J, D):
    """Energy per spin."""
    L = S.shape[0]; e = 0.0
    for i in range(L):
        for j in range(L):
            bx, by, bz = local_field(S, i, j, L, J, D)
            e -= 0.5 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz)
    return e / (L * L)


@njit(cache=True)
def seed(n):
    np.random.seed(n)


def random_state(L, rng):
    S = rng.standard_normal((L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
    return S
