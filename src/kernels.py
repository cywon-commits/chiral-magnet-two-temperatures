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


@njit(cache=True)
def bond_sums(S):
    """Per spin: x = sum over the 2 bonds (+x,+y) of S_i.S_j ; c = sum over the 2 bonds of r_ij.(S_i x S_j).
    Energy per spin e = -J x - D c ; bond chiral correlation <C> = c/2 ; exchange per bond <S.S> = x/2."""
    L = S.shape[0]; x = 0.0; c = 0.0
    for i in range(L):
        for j in range(L):
            ip = (i + 1) % L; jp = (j + 1) % L
            ax, ay, az = S[i, j, 0], S[i, j, 1], S[i, j, 2]
            bx, by, bz = S[ip, j, 0], S[ip, j, 1], S[ip, j, 2]          # r_ij = +x
            x += ax * bx + ay * by + az * bz; c += ay * bz - az * by      # x-component of S_i x S_j
            bx, by, bz = S[i, jp, 0], S[i, jp, 1], S[i, jp, 2]          # r_ij = +y
            x += ax * bx + ay * by + az * bz; c += az * bx - ax * bz      # y-component of S_i x S_j
    return x / (L * L), c / (L * L)


@njit(cache=True)
def bond_sums_dir(S):
    """Per spin, by bond direction: (xx, xy, cx, cy) with xx = sum_i S_i.S_{i+x}, cx = sum_i x.(S_i x S_{i+x}), etc.
    e = -J (xx + xy) - D (cx + cy)."""
    L = S.shape[0]; xx = 0.0; xy = 0.0; cx = 0.0; cy = 0.0
    for i in range(L):
        for j in range(L):
            ip = (i + 1) % L; jp = (j + 1) % L
            ax, ay, az = S[i, j, 0], S[i, j, 1], S[i, j, 2]
            bx, by, bz = S[ip, j, 0], S[ip, j, 1], S[ip, j, 2]
            xx += ax * bx + ay * by + az * bz; cx += ay * bz - az * by
            bx, by, bz = S[i, jp, 0], S[i, jp, 1], S[i, jp, 2]
            xy += ax * bx + ay * by + az * bz; cy += az * bx - ax * bz
    n = L * L
    return xx / n, xy / n, cx / n, cy / n


@njit(cache=True)
def accum_moments(S, acc):
    """acc[..., 0:3] += S ; acc[..., 3:9] += (SxSx, SySy, SzSz, SxSy, SxSz, SySz)."""
    L = S.shape[0]
    for i in range(L):
        for j in range(L):
            x, y, z = S[i, j, 0], S[i, j, 1], S[i, j, 2]
            acc[i, j, 0] += x; acc[i, j, 1] += y; acc[i, j, 2] += z
            acc[i, j, 3] += x * x; acc[i, j, 4] += y * y; acc[i, j, 5] += z * z
            acc[i, j, 6] += x * y; acc[i, j, 7] += x * z; acc[i, j, 8] += y * z
