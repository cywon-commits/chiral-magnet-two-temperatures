"""2D Heisenberg + DMI (Bloch type, d_ij = r_ij) on the square lattice.
Cooling run; per temperature: energy blocks, S(k) radial profile, second-moment xi."""
import numpy as np, sys, json, time
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
    L = S.shape[0]
    for i in range(L):
        for j in range(L):
            bx, by, bz = local_field(S, i, j, L, J, D)
            nx = S[i, j, 0] + sig * np.random.randn(); ny = S[i, j, 1] + sig * np.random.randn(); nz = S[i, j, 2] + sig * np.random.randn()
            r = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz); nx *= r; ny *= r; nz *= r
            dE = -((nx - S[i, j, 0]) * bx + (ny - S[i, j, 1]) * by + (nz - S[i, j, 2]) * bz)
            if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
                S[i, j, 0] = nx; S[i, j, 1] = ny; S[i, j, 2] = nz


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
def energy(S, J, D):
    L = S.shape[0]; e = 0.0
    for i in range(L):
        for j in range(L):
            bx, by, bz = local_field(S, i, j, L, J, D)
            e -= 0.5 * (S[i, j, 0] * bx + S[i, j, 1] * by + S[i, j, 2] * bz)
    return e / (L * L)


@njit(cache=True)
def seed(n):
    np.random.seed(n)


def main(D, L, Ts, neq, nmeas, nblk, tag, sd):
    J = 1.0; seed(sd); rng = np.random.default_rng(sd)
    S = rng.standard_normal((L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
    kx = np.fft.fftfreq(L) * L; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KR = np.rint(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    out = []; t0 = time.time()
    for T in Ts:
        sig = min(1.5, 1.2 * np.sqrt(T))
        for _ in range(neq):
            metropolis(S, T, J, D, sig); overrelax(S, J, D); overrelax(S, J, D)
        eb = np.zeros(nblk); e2b = np.zeros(nblk); per = nmeas // nblk
        Sk = np.zeros((L, L)); nsk = 0
        for b in range(nblk):
            for s in range(per):
                metropolis(S, T, J, D, sig); overrelax(S, J, D); overrelax(S, J, D)
                e = energy(S, J, D); eb[b] += e; e2b[b] += e * e
                if s % 10 == 0:
                    F = np.fft.fft2(S, axes=(0, 1)); Sk += (np.abs(F) ** 2).sum(-1) / (L * L); nsk += 1
            eb[b] /= per; e2b[b] /= per
        Sk /= nsk
        rad = np.bincount(KR.ravel(), Sk.ravel()) / np.bincount(KR.ravel())
        S0 = Sk[0, 0]; S1 = 0.25 * (Sk[1, 0] + Sk[-1, 0] + Sk[0, 1] + Sk[0, -1])
        xi = float(np.sqrt(max(S0 / S1 - 1, 0)) / (2 * np.sin(np.pi / L)))
        out.append(dict(T=float(T), eb=eb.tolist(), e2b=e2b.tolist(), rad=rad[:L // 2].tolist(), xi=xi))
        print('D=%.2f L=%d T=%.3f U=%.4f C=%.3f xi=%.2f  t=%.0fs' % (D, L, T, eb.mean(), L * L * (e2b.mean() - eb.mean() ** 2) / T ** 2, xi, time.time() - t0), flush=True)
        json.dump(dict(D=D, L=L, data=out), open('data/ts_%s.json' % tag, 'w'))


if __name__ == '__main__':
    D = float(sys.argv[1]); L = int(sys.argv[2]); neq = int(sys.argv[3]); nmeas = int(sys.argv[4])
    Thi = float(sys.argv[5]); Tlo = float(sys.argv[6]); tag = sys.argv[7]; sd = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    Ts = np.round(np.arange(Thi, Tlo - 1e-9, -0.025), 4)
    main(D, L, Ts, neq, nmeas, 10, tag, sd)
