"""Heisenberg + Bloch DMI on the TRIANGULAR lattice (cooling run), review-session check that the two temperatures are not square-lattice artefacts.
H = -J sum_<ij> S_i.S_j - D sum_<ij> r_ij.(S_i x S_j), |r_ij| = 1, six neighbours.  Axial coordinates r = i a1 + j a2, a1 = (1,0), a2 = (1/2, sqrt3/2).
Per temperature: block means of (S.S per site, DMI per site), their per-sweep covariance, and the total structure factor
S(k) = sum_a <|FFT(S_a)|^2>/N on the window |n1|, |n2| <= W  (k.a1 = 2 pi n1/L, k.a2 = 2 pi n2/L).
usage (repository root): python3 review/mc_tri.py D L seed"""
import sys, numpy as np
from numba import njit
D = float(sys.argv[1]); L = int(sys.argv[2]); sd = int(sys.argv[3])
J = 1.0; NEQ, NMEAS, EVERY, NB, W = 400, 2400, 8, 12, 20
Ts = np.round(np.arange(2.40, 0.799, -0.04), 4)
R3 = np.sqrt(3.0) / 2
NBR = np.array([[1, 0, 1.0, 0.0], [-1, 0, -1.0, 0.0], [0, 1, 0.5, R3], [0, -1, -0.5, -R3], [-1, 1, -0.5, R3], [1, -1, 0.5, -R3]])

@njit(cache=True)
def field(S, i, j, L, J, D, NBR):
    bx = 0.0; by = 0.0; bz = 0.0
    for n in range(6):
        ii = (i + int(NBR[n, 0])) % L; jj = (j + int(NBR[n, 1])) % L; dx = NBR[n, 2]; dy = NBR[n, 3]
        sx = S[ii, jj, 0]; sy = S[ii, jj, 1]; sz = S[ii, jj, 2]
        bx += J * sx + D * (-sz * dy); by += J * sy + D * (sz * dx); bz += J * sz + D * (sx * dy - sy * dx)
    return bx, by, bz

@njit(cache=True)
def sweep(S, T, J, D, sig, NBR):
    L = S.shape[0]
    for i in range(L):
        for j in range(L):
            bx, by, bz = field(S, i, j, L, J, D, NBR)
            nx = S[i, j, 0] + sig * np.random.randn(); ny = S[i, j, 1] + sig * np.random.randn(); nz = S[i, j, 2] + sig * np.random.randn()
            r = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz); nx *= r; ny *= r; nz *= r
            dE = -((nx - S[i, j, 0]) * bx + (ny - S[i, j, 1]) * by + (nz - S[i, j, 2]) * bz)
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                S[i, j, 0] = nx; S[i, j, 1] = ny; S[i, j, 2] = nz
    for rep in range(2):                      # sequential over-relaxation (energy conserving reflection about the local field)
        for i in range(L):
            for j in range(L):
                bx, by, bz = field(S, i, j, L, J, D, NBR); b2 = bx * bx + by * by + bz * bz
                if b2 < 1e-12: continue
                f = 2.0 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz) / b2
                S[i, j, 0] = f * bx - S[i, j, 0]; S[i, j, 1] = f * by - S[i, j, 1]; S[i, j, 2] = f * bz - S[i, j, 2]

@njit(cache=True)
def e2(S, NBR):                               # (sum_bonds S.S, sum_bonds r.(S x S')) per site
    L = S.shape[0]; ej = 0.0; ed = 0.0
    for i in range(L):
        for j in range(L):
            bx, by, bz = field(S, i, j, L, 1.0, 0.0, NBR); ej += 0.5 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz)
            bx, by, bz = field(S, i, j, L, 0.0, 1.0, NBR); ed += 0.5 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz)
    return ej / (L * L), ed / (L * L)

@njit(cache=True)
def seed(s): np.random.seed(s)

idx = np.r_[0:W + 1, L - W:L]
seed(sd); rng = np.random.default_rng(sd)
S = rng.standard_normal((L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
nT = len(Ts); EB = np.zeros((nT, NB, 2)); EC = np.zeros((nT, 2, 2)); SK = np.zeros((nT, 2 * W + 1, 2 * W + 1), dtype=np.float32)
for it, T in enumerate(Ts):
    sig = min(1.5, 1.0 * np.sqrt(T / 1.7))
    for _ in range(NEQ): sweep(S, T, J, D, sig, NBR)
    es = np.zeros((NMEAS, 2)); acc = np.zeros((2 * W + 1, 2 * W + 1)); n = 0
    for s in range(NMEAS):
        sweep(S, T, J, D, sig, NBR); es[s] = e2(S, NBR)
        if s % EVERY == 0:
            F = np.fft.fft2(S, axes=(0, 1))[np.ix_(idx, idx)]; acc += (np.abs(F) ** 2).sum(-1) / (L * L); n += 1
    EB[it] = es.reshape(NB, -1, 2).mean(1); EC[it] = np.cov(es.T); SK[it] = acc / n
    print('D=%.2f s%d T=%.2f U=%.4f' % (D, sd, T, -J * es[:, 0].mean() - D * es[:, 1].mean()), flush=True)
np.savez_compressed('review/data/tri_D%03d_L%d_s%d.npz' % (round(D * 100), L, sd), T=Ts, EB=EB, EC=EC, SK=SK, par=np.array([D, L, sd, W]))
