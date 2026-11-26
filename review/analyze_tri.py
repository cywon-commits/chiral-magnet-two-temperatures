"""Triangular-lattice check: T*, T_ring and kappa = xi_H(T_ring)/lambda with the same estimators as for the square lattice.
xi_H: D = 0, fit 1/S = r + c1 k^2 + c2 k^4 on k <= 2/xi (iterated), xi = sqrt(c1/r).   T_ring: c1/r = 0 (smooth quadratic crossing), windows w q.
k^2 is the lattice form (4/z) sum_delta (1 - cos k.delta) on both lattices.  Square inputs: data/task2_extract (L = 512, D = 0) and review/data/t6_kappa_window.json."""
import numpy as np, glob, json
def khat2_tri(L, W):
    n = np.arange(-W, W + 1); a, b = np.meshgrid(2 * np.pi * n / L, 2 * np.pi * n / L, indexing='ij')
    return (4 / 3) * (3 - np.cos(a) - np.cos(b) - np.cos(b - a))
def khat2_sq(L, W):
    n = np.arange(-W, W + 1); a, b = np.meshgrid(2 * np.pi * n / L, 2 * np.pi * n / L, indexing='ij')
    return 4 * np.sin(a / 2) ** 2 + 4 * np.sin(b / 2) ** 2
def fit(S, k2, kmax):
    m = (k2 <= kmax ** 2) & (k2 > 0)
    A = np.vstack([np.ones(m.sum()), k2[m], k2[m] ** 2]).T; return np.linalg.lstsq(A, 1 / S[m], rcond=None)[0], int(m.sum())
def xi_of(S, k2):
    kmax = np.sqrt(k2.max()); xi = np.nan
    for _ in range(6):
        c, n = fit(S, k2, kmax)
        if c[0] <= 0 or c[1] <= 0: return np.nan
        xi = np.sqrt(c[1] / c[0]); ks = np.sort(np.sqrt(k2[k2 > 0]).ravel()); kmax = max(2 / xi, ks[min(24, len(ks) - 1)])
    return xi
def cross(T, y, t0, h=0.14):
    s = (T >= t0 - h) & (T <= t0 + h); c = np.polyfit(T[s], y[s], 2); r = np.roots(c); r = r[np.isreal(r)].real; r = r[(r > t0 - h - 0.05) & (r < t0 + h + 0.05)]
    return float(r[np.argmin(abs(r - t0))]) if len(r) else np.nan
def shift(S): return np.fft.fftshift(S, axes=(-2, -1)) if False else S
def load_tri(D):
    fs = sorted(glob.glob('review/data/tri_D%03d_L96_s*.npz' % round(D * 100))); ds = [np.load(f) for f in fs]
    W = int(ds[0]['par'][3]); idx = np.r_[W + 1:2 * W + 1, 0:W + 1]             # reorder FFT window (0..W, -W..-1) -> (-W..W)
    return ds[0]['T'], [d['SK'][:, idx][:, :, idx] for d in ds], ds
def tstar(T, U, lo=0.85, hi=1.6):                       # peak of dU/dT from a 7th-order polynomial fit of U on [lo, hi + margin]
    s = (T >= lo - 0.05) & (T <= hi + 0.3); c = np.polyfit(T[s] - 1.2, U[s], 7); tt = np.linspace(lo, hi, 3001); return float(tt[np.argmax(np.polyval(np.polyder(c), tt - 1.2))])
out = {}
# ---------- D = 0: xi_H(T) and T*(0) on the triangular lattice
T, SKs, ds = load_tri(0.0); L = 96; k2 = khat2_tri(L, 20)
xi_seeds = np.array([[xi_of(S[t], k2) for t in range(len(T))] for S in SKs]); xi_tri = np.nanmean(xi_seeds, 0)
def U_of(d, D): return -d['EB'][..., 0].mean(1) - D * d['EB'][..., 1].mean(1)
def jk(vals_fn, ds):
    n = len(ds); full = vals_fn(ds); js = [vals_fn(ds[:i] + ds[i + 1:]) for i in range(n)]; return full, float(np.sqrt((n - 1) / n * np.sum((np.array(js) - np.mean(js)) ** 2)))
ts0, e0 = jk(lambda dd: tstar(T, np.mean([U_of(d, 0.0) for d in dd], 0)), ds)
xiT = lambda t: float(np.exp(np.interp(t, T[::-1], np.log(xi_tri[::-1]))))
out['tri_D0'] = dict(T=T.tolist(), xi=xi_tri.tolist(), xi_err=(np.nanstd(xi_seeds, 0, ddof=1) / np.sqrt(len(SKs))).tolist(), Tstar=ts0, Tstar_err=e0)
print('TRI D=0: T* = %.4f +- %.4f   xi_H(T*) = %.2f' % (ts0, e0, xiT(ts0)))
print('  xi_H(T):', ' '.join('%.2f:%.2f' % (t, x) for t, x in zip(T, xi_tri) if 1.0 <= t <= 2.2 and abs((t * 100) % 20) < 1e-6))
# ---------- square: xi_H with the same estimator (L = 512, D = 0)
dq = [np.load(f) for f in sorted(glob.glob('data/task2_extract/t2_main_D000_L512_s*.npz'))]; Tq = dq[0]['T']; k2q = khat2_sq(512, 24)
xi_sq = np.nanmean([[xi_of(d['low_tr'][t], k2q) for t in range(len(Tq))] for d in dq], 0); o = np.argsort(Tq)
xiS = lambda t: float(np.exp(np.interp(t, Tq[o], np.log(xi_sq[o]))))
jx = json.load(open('data/task6_judge.json'))['xiH']; print('SQ  D=0: xi_H same estimator vs judge file at T=0.70, 0.85, 1.06:', [(round(xiS(t), 2), round(float(np.exp(np.interp(t, jx['T'][::-1], np.log(jx['xi'][::-1])))), 2)) for t in (0.70, 0.85, 1.06)])
sqk = json.load(open('review/data/t6_kappa_window.json')); sq_T = {0.3: sqk['030_176']['TL'], 0.5: sqk['050_112']['TL']}; sq_Ts = {0.0: 0.680, 0.3: 0.708, 0.5: 0.729}
# ---------- D = 0.3, 0.5 on the triangular lattice
def pitch_tri(D):
    k = np.linspace(1e-4, 1.5, 200001); return k[np.argmax(np.cos(k) + 2 * np.cos(k / 2) + D * (np.sin(k) + np.sin(k / 2)))]
for D, t0 in ((0.3, 1.40), (0.5, 1.75)):
    T, SKs, ds = load_tri(D); q = pitch_tri(D); lam = 2 * np.pi / q; r = {}
    for w in (1.0, 1.2, 1.5):
        def tr(idx_list, w=w):
            y = np.mean([[(lambda c: c[1] / c[0])(fit(SKs[i][t], k2, w * q)[0]) for t in range(len(T))] for i in idx_list], 0); return cross(T, y, t0)
        n = len(SKs); full = tr(list(range(n))); js = [tr([j for j in range(n) if j != i]) for i in range(n)]; err = float(np.sqrt((n - 1) / n * np.sum((np.array(js) - np.mean(js)) ** 2)))
        r['w%.1f' % w] = dict(T_ring=full, err=err, kappa=xiT(full) / lam, kappa_lo=xiT(full + err) / lam, kappa_hi=xiT(full - err) / lam)
    ts, e = jk(lambda dd: tstar(T, np.mean([U_of(d, D) for d in dd], 0)), ds)
    lq = 2 * np.pi / np.arctan(D); iw = {1.0: 1, 1.2: 2, 1.5: 3}
    r.update(Tstar=ts, Tstar_err=e, shift=ts / ts0 - 1, lam=lam, sq={('w%.1f' % w): dict(T_ring=sq_T[D][iw[w]], kappa=xiS(sq_T[D][iw[w]]) / lq) for w in (1.0, 1.2, 1.5)}, sq_shift=sq_Ts[D] / sq_Ts[0.0] - 1)
    out['tri_D%.1f' % D] = r
    print('TRI D=%.1f lam=%.2f  T* = %.4f +- %.4f (shift %+.1f%%; square %+.1f%%)' % (D, lam, ts, e, 100 * r['shift'], 100 * r['sq_shift']))
    for w in ('w1.0', 'w1.2', 'w1.5'):
        print('   %s  T_ring = %.4f +- %.4f  kappa_tri = %.3f [%.3f, %.3f]   | square: T_ring = %.3f kappa = %.3f   | T_ring/T*: tri %.3f sq %.3f' % (w, r[w]['T_ring'], r[w]['err'], r[w]['kappa'], r[w]['kappa_lo'], r[w]['kappa_hi'], r['sq'][w]['T_ring'], r['sq'][w]['kappa'], r[w]['T_ring'] / ts, r['sq'][w]['T_ring'] / sq_Ts[D]))
json.dump(out, open('review/data/tri_analysis.json', 'w'), indent=1)
