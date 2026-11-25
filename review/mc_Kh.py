"""Heisenberg + Bloch DMI + uniaxial anisotropy -K (S^z)^2 + field -h.S on the square lattice (cooling run).
Per temperature: block energies, magnetisation, and the spin-component structure-factor matrix
P_ab(k) = <F_a(k) conj(F_b(k))>, F = FFT(S)/L, on the window |n_x|, |n_y| <= W.
usage (repository root): python3 review/mc_Kh.py D K hx hz L seed tag"""
import sys, numpy as np
from numba import njit
sys.path.insert(0, 'prelim')
from tstar import local_field, seed

D, K, hx, hz = (float(x) for x in sys.argv[1:5]); L = int(sys.argv[5]); sd = int(sys.argv[6]); tag = sys.argv[7]
J = 1.0; NEQ, NMEAS, EVERY, NB, W = 700, 3200, 10, 8, 6
Ts = np.round(np.arange(1.10, 0.549, -0.025), 4)


@njit(cache=True)
def sweep(S, T, J, D, K, hx, hz, sig):
    L = S.shape[0]
    for i in range(L):                       # Metropolis
        for j in range(L):
            bx, by, bz = local_field(S, i, j, L, J, D); bx += hx; bz += hz
            nx = S[i, j, 0] + sig * np.random.randn(); ny = S[i, j, 1] + sig * np.random.randn(); nz = S[i, j, 2] + sig * np.random.randn()
            r = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz); nx *= r; ny *= r; nz *= r
            dE = -((nx - S[i, j, 0]) * bx + (ny - S[i, j, 1]) * by + (nz - S[i, j, 2]) * bz) - K * (nz * nz - S[i, j, 2] ** 2)
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                S[i, j, 0] = nx; S[i, j, 1] = ny; S[i, j, 2] = nz
    for rep in range(2):                     # over-relaxation about (exchange + DMI + field); anisotropy by Metropolis test
        for par in range(2):
            for i in range(L):
                for j in range(L):
                    if (i + j) % 2 != par: continue
                    bx, by, bz = local_field(S, i, j, L, J, D); bx += hx; bz += hz
                    b2 = bx * bx + by * by + bz * bz
                    if b2 < 1e-12: continue
                    f = 2.0 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz) / b2
                    nz = f * bz - S[i, j, 2]
                    if K != 0.0:
                        dE = -K * (nz * nz - S[i, j, 2] ** 2)
                        if dE > 0.0 and np.random.random() >= np.exp(-dE / T): continue
                    S[i, j, 0] = f * bx - S[i, j, 0]; S[i, j, 1] = f * by - S[i, j, 1]; S[i, j, 2] = nz


@njit(cache=True)
def energy(S, J, D, K, hx, hz):
    L = S.shape[0]; e = 0.0
    for i in range(L):
        for j in range(L):
            bx, by, bz = local_field(S, i, j, L, J, D)
            e += -0.5 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz) - K * S[i, j, 2] ** 2 - hx * S[i, j, 0] - hz * S[i, j, 2]
    return e / (L * L)


idx = np.r_[0:W + 1, L - W:L]                 # FFT indices of n = 0..W, -W..-1
seed(sd); rng = np.random.default_rng(sd)
S = rng.standard_normal((L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
E = np.zeros((len(Ts), NB)); M = np.zeros((len(Ts), 3)); P = np.zeros((len(Ts), 2 * W + 1, 2 * W + 1, 3, 3), dtype=np.complex64)
for it, T in enumerate(Ts):
    sig = min(1.5, 1.2 * np.sqrt(T))
    for _ in range(NEQ): sweep(S, T, J, D, K, hx, hz, sig)
    acc = np.zeros((2 * W + 1, 2 * W + 1, 3, 3), dtype=complex); n = 0; per = NMEAS // NB
    for s in range(NMEAS):
        sweep(S, T, J, D, K, hx, hz, sig); E[it, s // per] += energy(S, J, D, K, hx, hz) / per
        if s % EVERY == 0:
            F = (np.fft.fft2(S, axes=(0, 1)) / L)[np.ix_(idx, idx)]
            acc += F[..., :, None] * np.conj(F[..., None, :]); M[it] += S.mean((0, 1)); n += 1
    P[it] = acc / n; M[it] /= n
    print('%s T=%.3f U=%.4f m=(%.3f, %.3f, %.3f)' % (tag, T, E[it].mean(), M[it, 0], M[it, 1], M[it, 2]), flush=True)
np.savez_compressed('review/data/Kh_%s_s%d.npz' % (tag, sd), T=Ts, E=E, M=M, P=P, par=np.array([D, K, hx, hz, L]))
