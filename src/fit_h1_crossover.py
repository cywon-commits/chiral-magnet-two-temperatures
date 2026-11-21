"""[V-3] Crossover form  y = T*(D)/T*(0) - 1 = b ( sqrt(D^2 + D0^2) - D0 ),  D0 = 1/xi_H(T*(0)).
xi_H(T): Ornstein-Zernike fit of the unaveraged low-k S(k), D=0, L=256 (task-1 runs, 4 seeds x cool/heat),
interpolated in log xi at T*(0) = D=0 L=256 dU/dT peak. Uncertainty of D0 from (a) seed scatter of xi,
(b) T*(0) total error, (c) L=128 vs 256 difference.
Fits: D0 fixed (0.1, 1/xi_H, and its range), D0 free; errors 'du' (dU/dT only) and 'tot'; T*(0) error ignored
(variant i) and propagated independently (variant ii). 6 points: L/lambda~8 (D=0.2..0.5) + D=0.1, 0.15 (L/lambda~4).
Output: data/task1_crossover.json"""
import json, numpy as np
from scipy.optimize import minimize_scalar
H = json.load(open('data/task1_h1.json'))['rows']; S = json.load(open('data/task1_summary.json'))
r0 = H['0_256']; T0 = r0['T']
def xi_at(L, T):
    vals = []
    for d in ('cool', 'heat'):
        e = [x for x in S if x['proto'] == 'anneal' and x['D'] == 0 and x['L'] == L and x['dirn'] == d][0]
        c = e['curve_xi_oz']; vals.append(np.exp(np.interp(T, c['T'][::-1] if c['T'][0] > c['T'][-1] else c['T'],
                                                       np.log(c['mean'][::-1] if c['T'][0] > c['T'][-1] else c['mean']))))
    return float(np.mean(vals))
xi0 = xi_at(256, T0)
e = [x for x in S if x['proto'] == 'anneal' and x['D'] == 0 and x['L'] == 256 and x['dirn'] == 'cool'][0]['curve_xi_oz']
i = int(np.argmin(np.abs(np.array(e['T']) - T0))); xi_seed = e['err'][i]
xi_lo, xi_hi = xi_at(256, T0 + r0['tot']), xi_at(256, T0 - r0['tot']); xi128 = xi_at(128, T0)
print('T*(0) = %.4f +- %.4f ; xi_H(T*(0)) L=256 = %.2f (seed err %.2f; T*(0)+-sigma -> %.2f..%.2f; L=128: %.2f)' % (T0, r0['tot'], xi0, xi_seed, xi_lo, xi_hi, xi128))
D0c = 1 / xi0; D0lo, D0hi = 1 / xi_hi, 1 / xi_lo
print('D0 = 1/xi_H = %.4f  (range from T*(0) error: %.4f .. %.4f)' % (D0c, D0lo, D0hi))
pts = [(0.1, 256), (0.15, 168), (0.2, 256), (0.3, 176), (0.4, 128), (0.5, 112)]
D = np.array([p[0] for p in pts]); T = np.array([H['%g_%d' % p]['T'] for p in pts])
def sig(r, kind): return r['tot'] if kind == 'tot' else float(np.sqrt(r['stat'] ** 2 + r['s_win'] ** 2 + r['s_hys'] ** 2))
y = T / T0 - 1
out = dict(T0=T0, xi0=xi0, xi_seed_err=xi_seed, xi_range=[xi_lo, xi_hi], xi128=xi128, D0=D0c, D0_range=[D0lo, D0hi], fits={})
for kind in ('du', 'tot'):
    for var in ('i', 'ii'):
        s = np.array([sig(H['%g_%d' % p], kind) for p in pts]) / T0
        if var == 'ii': s = np.hypot(s, T / T0 * sig(r0, kind) / T0)
        w = 1 / s ** 2
        def fit(D0):
            g = np.sqrt(D * D + D0 * D0) - D0; b = (w * g * y).sum() / (w * g * g).sum()
            return b, float(np.sqrt(1 / (w * g * g).sum())), float((w * (y - b * g) ** 2).sum())
        res = {}
        for nm, D0 in (('D0=0.1', 0.1), ('D0=1/xi_H', D0c), ('D0=1/xi_H lo', D0lo), ('D0=1/xi_H hi', D0hi), ('D0=0 (|D|)', 0.0)):
            b, eb, c2 = fit(D0); res[nm] = dict(D0=D0, b=b, err=eb, chi2=c2, dof=len(D) - 1)
        m = minimize_scalar(lambda d: fit(abs(d))[2], bounds=(0, 1.0), method='bounded'); D0f = abs(m.x); bf, ebf, c2f = fit(D0f)
        grid = np.linspace(0, 0.6, 601); c2g = np.array([fit(d)[2] for d in grid]); ok = grid[c2g <= c2f + 1]
        res['D0 free'] = dict(D0=D0f, D0_1sigma=[float(ok.min()), float(ok.max())], b=bf, err=ebf, chi2=c2f, dof=len(D) - 2)
        # pure D^2 for reference
        b2 = (w * D ** 2 * y).sum() / (w * D ** 4).sum(); res['a D^2'] = dict(a=float(b2), chi2=float((w * (y - b2 * D ** 2) ** 2).sum()), dof=len(D) - 1)
        out['fits']['%s|%s' % (kind, var)] = res
        print('\n== sigma=%s, T*(0) error %s' % (kind, 'ignored' if var == 'i' else 'propagated'))
        for nm, r in res.items():
            if nm == 'a D^2': print('  %-14s a=%.3f  chi2=%.1f/%d' % (nm, r['a'], r['chi2'], r['dof']))
            elif nm == 'D0 free': print('  %-14s D0=%.3f (chi2+1: %.3f..%.3f)  b=%.3f+-%.3f  chi2=%.1f/%d' % (nm, r['D0'], *r['D0_1sigma'], r['b'], r['err'], r['chi2'], r['dof']))
            else: print('  %-14s D0=%.4f  b=%.3f+-%.3f  chi2=%.1f/%d' % (nm, r['D0'], r['b'], r['err'], r['chi2'], r['dof']))
json.dump(out, open('data/task1_crossover.json', 'w'), indent=1)
