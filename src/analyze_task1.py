"""Task 1 analysis: specific-heat peak position T* per (D, L, seed, direction).

Estimators (each applied to every run separately; seeds give the statistical error):
  fl  : fluctuation C = N var(e)/T^2 on the 0.01 grid, peak by quadratic fit in a window around the max
  du  : C = dU/dT by central finite difference on the 0.01 grid, same peak finder
  mh  : multiple-histogram (MBAR / Ferrenberg-Swendsen) reweighting of all grid temperatures of the run,
        continuous C(T) on a 0.001 grid, argmax restricted to T where every reweighted point has Kish ESS>=50
Peak-finder window half-widths 0.03, 0.04, 0.05 -> spread used as a systematic error.
Output: data/task1_summary.json
"""
import numpy as np, glob, json, os, sys
from scipy.special import logsumexp

NEQ_DROP = None  # use the run's own neq
WINDOWS = (0.03, 0.04, 0.05)
SUB = 5          # subsample energy series every SUB sweeps for MBAR (tau_int ~ 7)

def tau_int(x, c=6.0):
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]
    t = 0.5
    for w in range(1, n):
        t += acf[w]
        if w >= c * t: break
    return t

def qpeak(T, y, w):
    i = int(np.argmax(y)); sel = np.abs(T - T[i]) <= w + 1e-9
    if sel.sum() < 4: return np.nan
    c = np.polyfit(T[sel] - T[i], y[sel], 2)
    if c[0] >= 0: return np.nan
    t = T[i] - c[1] / (2 * c[0])
    return t if T[sel].min() <= t <= T[sel].max() else np.nan

def mbar(Ts, Es, N, Tgrid):
    """Ts: (K,), Es: list of arrays of energy per spin. Returns C(T) and min Kish ESS on Tgrid."""
    b = 1.0 / np.asarray(Ts); X = np.concatenate(Es) * N; Nk = np.array([len(e) for e in Es], float)
    X0 = X.mean(); X = X - X0
    u = np.outer(b, X)                     # (K, Ntot) reduced energies
    from scipy.optimize import minimize
    lN = np.log(Nk)
    def obj(g):  # convex MBAR objective, f_0 fixed to 0
        f = np.r_[0.0, g]; a = lN[:, None] + f[:, None] - u; den = logsumexp(a, axis=0)
        w = np.exp(a - den[None, :]).sum(1)
        return den.sum() - (Nk * f).sum(), (w - Nk)[1:]
    # initial guess: f_k by trapezoidal integration of <u> (thermodynamic integration)
    Um = np.array([e.mean() * N for e in Es]) - X0; f0 = np.r_[0, np.cumsum(0.5 * (Um[1:] + Um[:-1]) * np.diff(b))]
    sol = minimize(obj, f0[1:], jac=True, method='L-BFGS-B', options=dict(maxiter=5000, gtol=1e-6, ftol=1e-15))
    f = np.r_[0.0, sol.x]
    den = logsumexp(np.log(Nk)[:, None] + f[:, None] - u, axis=0)
    C = np.empty(len(Tgrid)); ess = np.empty(len(Tgrid)); U = np.empty(len(Tgrid))
    for j, T in enumerate(Tgrid):
        lw = -X / T - den; lw -= lw.max(); w = np.exp(lw); w /= w.sum()
        m1 = (w * X).sum(); m2 = (w * (X - m1) ** 2).sum()
        U[j] = (m1 + X0) / N; C[j] = m2 / N / T ** 2; ess[j] = 1.0 / (w ** 2).sum()
    return U, C, ess

def xi_oz(Sk_low, L, kc):
    """Ornstein-Zernike fit 1/S(k) = A (1 + xi^2 khat^2) on the unaveraged low-k crop, k != 0,
    self-consistent window khat*xi < 2 (at least the 4 smallest |k| shells)."""
    n = np.arange(-kc, kc + 1); NX, NY = np.meshgrid(n, n, indexing='ij')
    k2 = (4 * np.sin(np.pi * NX / L) ** 2 + 4 * np.sin(np.pi * NY / L) ** 2).ravel(); y = 1.0 / Sk_low.ravel()
    nz = k2 > 0; k2 = k2[nz]; y = y[nz]; shells = np.unique(np.round(k2, 12))
    xg = 5.0
    for it in range(10):
        sel = np.sqrt(k2) * xg < 2.0
        if len(np.unique(np.round(k2[sel], 12))) < 4: sel = k2 <= shells[3] + 1e-12
        c = np.polyfit(k2[sel], y[sel], 1); xg = np.sqrt(max(c[0] / c[1], 1e-6))
    return float(xg)

def analyze_run(fn):
    d = np.load(fn); N = int(d['L']) ** 2; neq = int(d['neq'])
    out = []
    for dirn in np.unique(d['dirn']):
        m = d['dirn'] == dirn; T = d['T'][m]; E = d['E'][m][:, neq:]; Eeq = d['E'][m][:, :neq]
        o = np.argsort(T); T = T[o]; E = E[o]; Eeq = Eeq[o]
        U = E.mean(1); Cfl = N * E.var(1) / T ** 2
        # statistical error of U and C_fl per T from 20 blocks
        nb = 20; Eb = E[:, :E.shape[1] // nb * nb].reshape(len(T), nb, -1)
        sU = Eb.mean(2).std(1, ddof=1) / np.sqrt(nb)
        Cb = N * Eb.var(2) / T[:, None] ** 2; sC = Cb.std(1, ddof=1) / np.sqrt(nb)
        drift = (E[:, E.shape[1] // 2:].mean(1) - E[:, :E.shape[1] // 2].mean(1)) / (2 * sU)  # in units of sigma of the half-difference
        eqdrift = (Eeq[:, neq // 2:].mean(1) - U) / sU
        Tc = T[1:-1]; Cdu = (U[2:] - U[:-2]) / (T[2:] - T[:-2]); sCdu = np.sqrt(sU[2:] ** 2 + sU[:-2] ** 2) / (T[2:] - T[:-2])
        Tg = np.round(np.arange(T.min(), T.max() + 1e-9, 0.001), 4)
        Umh, Cmh, ess = mbar(T, [e[::SUB] for e in E], N, Tg)
        ok = ess >= 50
        Tmh = Tg[ok][np.argmax(Cmh[ok])] if ok.any() else np.nan
        r = dict(file=os.path.basename(fn), proto=str(d['proto']), D=float(d['D']), L=int(d['L']), seed=int(d['seed']), dirn=str(dirn),
                 T=T.tolist(), U=U.tolist(), sU=sU.tolist(), Cfl=Cfl.tolist(), sCfl=sC.tolist(), Tdu=Tc.tolist(), Cdu=Cdu.tolist(), sCdu=sCdu.tolist(),
                 drift_sigma=drift.tolist(), eqdrift_sigma=eqdrift.tolist(),
                 tau=[float(tau_int(e)) for e in E[::5]],
                 Tmh_grid=Tg[::5].tolist(), Cmh=Cmh[::5].tolist(), ess_min=float(ess.min()), Tstar_mh=float(Tmh),
                 Tstar_fl={str(w): float(qpeak(T, Cfl, w)) for w in WINDOWS},
                 Tstar_du={str(w): float(qpeak(Tc, Cdu, w)) for w in WINDOWS},
                 Cmax_fl=float(Cfl.max()),
                 S0=d['S0'][m][o].tolist(), Sk_rad=d['Sk_rad'][m][o].tolist())
        if 'cos4' in d:
            r['cos4'] = d['cos4'][m][o].tolist(); r['e2'] = d['e2'][m][o].tolist()
            r['xi_oz'] = [xi_oz(sk, int(d['L']), int(d['kc'])) for sk in d['Sk_low'][m][o]]
        out.append(r)
    return out

if __name__ == '__main__':
    files = sorted(glob.glob('data/raw/t1_*.npz'))
    res = []
    for fn in files:
        d = np.load(fn)
        if len(d['T']) < 42 and str(d['proto']) == 'anneal': print('incomplete', fn, len(d['T'])); continue
        if len(d['T']) < 21: print('incomplete', fn, len(d['T'])); continue
        for r in analyze_run(fn):
            res.append(r)
            print('%-28s %-6s T*: mh=%.3f(ess %.0f)  fl=%s  du=%s  Cmax=%.3f  max|drift|=%.1f' % (
                r['file'], r['dirn'], r['Tstar_mh'], r['ess_min'],
                ' '.join('%.3f' % v for v in r['Tstar_fl'].values()), ' '.join('%.3f' % v for v in r['Tstar_du'].values()),
                r['Cmax_fl'], np.max(np.abs(r['drift_sigma']))), flush=True)
    json.dump(res, open('data/raw/task1_runs.json', 'w'))
