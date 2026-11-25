"""Chirality-resolved structure factors on the CPC temperature grid, for the linear-VAE (probabilistic PCA) estimate.
Cooling run; per temperature saves p_a(k) = <|e_a(k)^* . F(k)|^2>, a in (+, -, 0), F = FFT(S)/sqrt(N).
usage: python3 review/linvae_spectra.py D L seed   (run from the repository root)"""
import sys, numpy as np
sys.path.insert(0, 'prelim')
from tstar import metropolis, overrelax, seed

D = float(sys.argv[1]); L = int(sys.argv[2]); sd = int(sys.argv[3]); J = 1.0
NEQ, NMEAS, EVERY = 600, 1500, 10
Ts = np.round(np.concatenate([np.arange(1.5, 0.95, -0.1), np.arange(0.9, 0.49, -0.025), [0.4, 0.3, 0.2, 0.1]]), 4)

k = 2 * np.pi * np.fft.fftfreq(L); KX, KY = np.meshgrid(k, k, indexing='ij')
sx, sy = np.sin(KX), np.sin(KY); sn = np.hypot(sx, sy); ok = sn > 1e-12
hx = np.where(ok, sx / np.where(ok, sn, 1), 1.0); hy = np.where(ok, sy / np.where(ok, sn, 1), 0.0)   # s-hat (arbitrary where s = 0)
# e1 = z, e2 = s-hat x z = (hy, -hx, 0);  e_pm = (e1 +- i e2)/sqrt2 ; e_0 = s-hat


def channels(S):
    F = np.fft.fft2(S, axes=(0, 1)) / L
    f0 = hx * F[..., 0] + hy * F[..., 1]
    f1 = F[..., 2]; f2 = hy * F[..., 0] - hx * F[..., 1]
    fp = (f1 - 1j * f2) / np.sqrt(2); fm = (f1 + 1j * f2) / np.sqrt(2)      # e_+^* . F, e_-^* . F
    return np.stack([np.abs(fp) ** 2, np.abs(fm) ** 2, np.abs(f0) ** 2])


seed(sd); rng = np.random.default_rng(sd)
S = rng.standard_normal((L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
out = []
for T in Ts:
    sig = min(1.5, 1.2 * np.sqrt(T))
    for _ in range(NEQ):
        metropolis(S, T, J, D, sig); overrelax(S, J, D); overrelax(S, J, D)
    P = np.zeros((3, L, L)); n = 0
    for s in range(NMEAS):
        metropolis(S, T, J, D, sig); overrelax(S, J, D); overrelax(S, J, D)
        if s % EVERY == 0:
            P += channels(S); n += 1
    out.append((P / n).astype(np.float32))
    print('D=%.2f seed=%d T=%.3f sum=%.4f' % (D, sd, T, P.sum() / n / (L * L)), flush=True)
np.savez_compressed('review/data/linvae_D%03d_L%d_s%d.npz' % (round(D * 100), L, sd), T=Ts, P=np.array(out), D=D, L=L)
