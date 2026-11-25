"""Linear-VAE (probabilistic PCA) prediction of the VAE order parameter eta(T) from measured structure factors.
A linear Gaussian VAE trained on the whole temperature grid keeps the n_z dataset-covariance modes with the largest
variance sbar, provided sbar > theta (theta = beta * decoder variance; sum-reduced MSE with beta = 1 gives theta = 1/2),
and reconstructs each kept mode with gain g = 1 - theta/sbar.  Then
    eta_rms(T)^2 = (1/N) sum_kept g^2 s_T(mode),     (1/N) sum_all s_T = 1.
Modes are (k, chirality channel); run from the repository root after review/linvae_spectra.py."""
import numpy as np, glob, json, sys


def sym(A):                      # average over the square point group acting on k (FFT index order)
    f0 = lambda X: np.roll(X[::-1], 1, axis=0); f1 = lambda X: np.roll(X[:, ::-1], 1, axis=1)
    out = 0
    for B in (A, A.T):
        out = out + B + f0(B) + f1(B) + f0(f1(B))
    return out / 8


def load(D, L=128):
    fs = sorted(glob.glob('review/data/linvae_D%03d_L%d_s*.npz' % (round(D * 100), L)))
    P = np.mean([np.load(f)['P'] for f in fs], axis=0).astype(float); T = np.load(fs[0])['T']
    P = np.array([[sym(P[i, a]) for a in range(3)] for i in range(len(T))])
    return T, P, len(fs)


def eta(T, P, nz, theta):
    N = P.shape[-1] ** 2; sbar = P.mean(0).ravel(); order = np.argsort(sbar)[::-1][:nz]
    g = np.clip(1 - theta / sbar[order], 0, None)
    return np.sqrt((P.reshape(len(T), -1)[:, order] * g ** 2).sum(1) / N), int((g > 0).sum())


def t_infl(T, e):
    m = (T >= 0.5) & (T <= 0.9); t = T[m][::-1]; y = e[m][::-1]; d = -np.gradient(y, t); i = int(np.argmax(d))
    if i < 2 or i > len(t) - 3: return float('nan')
    p = np.polyfit(t[i - 2:i + 3], d[i - 2:i + 3], 2)
    return float(-p[1] / (2 * p[0])) if p[0] < 0 else float('nan')


def t_half(T, e):
    h = e.max() / 2; return float(np.interp(-h, -e[::-1], T[::-1])) if e.min() < h else float('nan')


res = {}
show = [1.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1]
for D in (0.0, 0.3, 0.5):
    T, P, ns = load(D); idx = [int(np.argmin(np.abs(T - t))) for t in show]
    print('\nD = %.1f (%d chains, L = 128); columns T =' % (D, ns), show)
    for nz, th in [(100, 0.5), (25, 0.5), (400, 0.5), (100, 5.0), (100, 20.0), (100, 60.0), (100, 150.0)]:
        e, nk = eta(T, P, nz, th)
        res['D%.1f_nz%d_th%g' % (D, nz, th)] = dict(T=T.tolist(), eta=e.tolist(), kept=nk, T_infl=t_infl(T, e), T_half=t_half(T, e))
        print('n_z=%3d theta=%5.1f kept=%3d  eta:' % (nz, th, nk), ' '.join('%.2f' % e[i] for i in idx), ' T_infl=%.3f T_half=%.3f' % (t_infl(T, e), t_half(T, e)))
    # largest single-mode power vs T (sets where a given water level theta cuts in)
    pk = P.reshape(len(T), -1).max(1); print('max single-mode power s_T:', ' '.join('%.0f' % pk[i] for i in idx))
json.dump(res, open('review/data/linvae_eta.json', 'w'))
