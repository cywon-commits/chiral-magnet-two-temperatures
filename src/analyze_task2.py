"""Task 2 analysis of t2_<tag>_*.npz runs.

Per temperature, on the unaveraged low-k S(k) (crop |kx|,|ky|<=kc, k != 0 excluded only for D=0):
  Landau-Brazovskii fit  1/S(k) = r + c1 khat^2 + c2 khat^4 ,  khat^2 = 4 sin^2(kx/2) + 4 sin^2(ky/2)
  (weighted linear least squares on 1/S with weights S^2, window |k| <= w q, w = 1.2, 1.5, 2.0; q = arctan(D)).
  * disorder line T_L : c1(T) = 0  (S''(0) changes sign; max of S leaves k=0)
  * for c1 < 0: k0^2 = -c1/(2 c2), helix correlation length from the expansion near the ring,
    1/S ~ (r - c1^2/(4 c2)) + 4 c2 k0^2 (k-k0)^2  ->  xi_q = sqrt(4 c2 k0^2 / (r - c1^2/(4 c2)))
  * T_xi : xi_q(T) = lambda = 2 pi / q   (also reported: xi_q = lambda/2pi, i.e. q xi_q = 1)
D=0: Ornstein-Zernike xi_H from the same fit restricted to khat*xi < 2 (as task 1).
Bond chiral correlation <C> = <c>/2 and exchange <S.S> = <x>/2 with block errors.
Output data/raw/task2_<tag>_runs.json
"""
import numpy as np, glob, json, os, sys
WIN = (1.2, 1.5, 2.0)

def khat2(L, kc):
    n = np.arange(-kc, kc + 1); NX, NY = np.meshgrid(n, n, indexing='ij')
    kx, ky = 2 * np.pi * NX / L, 2 * np.pi * NY / L
    return (4 * np.sin(kx / 2) ** 2 + 4 * np.sin(ky / 2) ** 2).ravel(), np.sqrt(kx ** 2 + ky ** 2).ravel()

def lb_fit(Sk, k2h, kabs, kw):
    sel = kabs <= kw + 1e-12; s = Sk.ravel()[sel]; x = k2h[sel]
    A = np.vstack([np.ones_like(x), x, x * x]).T * s[:, None]; b = np.ones_like(s)   # weight S: (1/S_fit)*S ~ 1
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c, int(sel.sum())

def xi_q_from(c):
    r, c1, c2 = c
    if c1 >= 0 or c2 <= 0: return np.nan, np.nan
    k02 = -c1 / (2 * c2); gap = r - c1 * c1 / (4 * c2)
    if gap <= 0: return np.sqrt(k02), np.inf
    return np.sqrt(k02), np.sqrt(4 * c2 * k02 / gap)

def oz_xi(Sk, k2h):
    y = 1.0 / Sk.ravel(); nz = k2h > 0; x = k2h[nz]; y = y[nz]; sh = np.unique(np.round(x, 12)); xg = 5.0
    for _ in range(10):
        sel = np.sqrt(x) * xg < 2.0
        if len(np.unique(np.round(x[sel], 12))) < 4: sel = x <= sh[3] + 1e-12
        cc = np.polyfit(x[sel], y[sel], 1); xg = np.sqrt(max(cc[0] / cc[1], 1e-6))
    return float(xg)

def crossing(T, y, level=0.0):
    """Temperatures where y(T)-level changes sign (linear interpolation), T sorted descending as in the run."""
    o = np.argsort(T); T = np.asarray(T)[o]; y = np.asarray(y, float)[o] - level; out = []
    for i in range(len(T) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] * y[i + 1] < 0:
            out.append(float(T[i] - y[i] * (T[i + 1] - T[i]) / (y[i + 1] - y[i])))
    return out

def analyze(fn):
    d = np.load(fn); D = float(d['D']); L = int(d['L']); kc = int(d['kc']); neq = int(d['neq'])
    k2h, kabs = khat2(L, kc); q = np.arctan(D) if D > 0 else 0.0; lam = 2 * np.pi / q if D > 0 else np.nan
    T = d['T']; r = dict(file=os.path.basename(fn), D=D, L=L, seed=int(d['seed']), T=T.tolist(), dirn=d['dirn'].tolist(), lam=lam)
    X = d['X'][:, neq:]; Cb = d['Cb'][:, neq:]; nb = 20; m = X.shape[1] // nb * nb
    r['SS'] = (X.mean(1) / 2).tolist(); r['C'] = (Cb.mean(1) / 2).tolist()
    r['C_err'] = (Cb[:, :m].reshape(len(T), nb, -1).mean(2).std(1, ddof=1) / np.sqrt(nb) / 2).tolist()
    e = -X - D * Cb; r['U'] = e.mean(1).tolist(); r['Cv'] = (L * L * e.var(1) / T ** 2).tolist()
    r['S0'] = [float(s[kc, kc]) for s in d['Sk_low']]
    rad = d['Sk_rad']; kb = (np.arange(rad.shape[1]) + 0.5) * 0.25 * 2 * np.pi / L
    r['kmax_rad'] = [float(kb[np.argmax(p)]) if np.argmax(p) > 0 else 0.0 for p in rad]
    r['cos4'] = d['cos4'].tolist(); r['e2'] = d['e2'].tolist()
    if D > 0:
        for w in WIN:
            cs = [lb_fit(s, k2h, kabs, w * q)[0] for s in d['Sk_low']]
            r['c1_w%.1f' % w] = [float(c[1] / c[0]) for c in cs]                    # c1/r: sign gives curvature
            xq = [xi_q_from(c) for c in cs]; r['k0_w%.1f' % w] = [float(a) for a, b in xq]; r['xiq_w%.1f' % w] = [float(b) for a, b in xq]
            r['TL_w%.1f' % w] = crossing(T, r['c1_w%.1f' % w])
            r['Txi_w%.1f' % w] = crossing(T, np.log(np.nan_to_num(np.array(r['xiq_w%.1f' % w]), nan=1e-9, posinf=1e9)), np.log(lam))
            r['Txi1_w%.1f' % w] = crossing(T, np.log(np.nan_to_num(np.array(r['xiq_w%.1f' % w]), nan=1e-9, posinf=1e9)), np.log(lam / (2 * np.pi)))
    r['xi_oz'] = [oz_xi(s, k2h) for s in d['Sk_low']]
    return r

if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
    res = []
    for fn in sorted(f for f in glob.glob('data/raw/t2_%s_*.npz' % tag) if not f.endswith('.ckpt.npz')):
        r = analyze(fn); res.append(r)
        if r['D'] > 0:
            print('%-26s lam=%5.1f  T_L(w=1.2,1.5,2.0)=%s  T_xi(xi_q=lam)=%s  T(q xi_q=1)=%s' % (r['file'], r['lam'],
                  [['%.3f' % t for t in r['TL_w%.1f' % w]] for w in WIN], [['%.3f' % t for t in r['Txi_w%.1f' % w]] for w in WIN],
                  [['%.3f' % t for t in r['Txi1_w%.1f' % w]] for w in WIN]))
        else:
            print('%-26s D=0 xi_OZ:' % r['file'], ' '.join('%.2f:%.1f' % (t, x) for t, x in zip(r['T'][::3], r['xi_oz'][::3])))
    json.dump(res, open('data/raw/task2_%s_runs.json' % tag, 'w'))
