"""[V-2] Quantitative check of revised H1: T*(D)/T*(0) - 1 = a D^2.
Input data/task1_h1.json (rows). Errors: 'tot' (pre-registered sigma) or 'du' (stat (+) window (+) cool/heat).
Variants: (i) T*(0) error ignored, (ii) T*(0) error propagated into every point independently,
(iii) full generalized least squares with the common-T*(0) correlation.
Also z of each point vs three reference forms a = 1/3, 1/2, 1/4. Output data/task1_d2fit.json"""
import json, numpy as np, sys
H = json.load(open('data/task1_h1.json'))['rows']
sets = {'L/lambda~8': [(0.2, 256), (0.3, 176), (0.4, 128), (0.5, 112)], 'L/lambda~4': [(0.2, 128), (0.3, 88), (0.4, 64), (0.5, 56)]}
extra = [(0.1, 256), (0.15, 168)]
REFS = {'sqrt(1+2D^2/3)': lambda D: np.sqrt(1 + 2 * D * D / 3), 'sqrt(1+D^2)': lambda D: np.sqrt(1 + D * D), '(1+sqrt(1+D^2))/2': lambda D: (1 + np.sqrt(1 + D * D)) / 2}
def sig(r, kind): return r['tot'] if kind == 'tot' else float(np.sqrt(r['stat'] ** 2 + r['s_win'] ** 2 + r['s_hys'] ** 2))
r0 = H['0_256']; T0 = r0['T']
out = {}
for sname, pts0 in sets.items():
    for withextra in (False, True):
        pts = pts0 + ([p for p in extra if '%g_%d' % p in H] if withextra else [])
        if withextra and len(pts) == len(pts0): continue
        key = sname + (' + D=0.1,0.15 (L/lambda~4)' if withextra else '')
        for kind in ('tot', 'du'):
            D = np.array([p[0] for p in pts]); T = np.array([H['%g_%d' % p]['T'] for p in pts]); s = np.array([sig(H['%g_%d' % p], kind) for p in pts]); s0 = sig(r0, kind)
            y = T / T0 - 1; x = D ** 2
            res = {}
            for var in ('i', 'ii', 'iii'):
                if var == 'i': C = np.diag((s / T0) ** 2)
                elif var == 'ii': C = np.diag((s / T0) ** 2 + (T * s0 / T0 ** 2) ** 2)
                else: C = np.diag((s / T0) ** 2) + np.outer(T, T) * s0 ** 2 / T0 ** 4
                Ci = np.linalg.inv(C); va = 1.0 / (x @ Ci @ x); a = va * (x @ Ci @ y); rr = y - a * x; chi2 = float(rr @ Ci @ rr)
                res[var] = dict(a=float(a), err=float(np.sqrt(va)), chi2=chi2, dof=len(x) - 1)
                # also free linear + quadratic: y = b D + a D^2
                A = np.vstack([D, x]).T; cov = np.linalg.inv(A.T @ Ci @ A); p = cov @ A.T @ Ci @ y
                vb = 1.0 / (D @ Ci @ D); b1 = vb * (D @ Ci @ y); rb = y - b1 * D
                res[var]['lin_only'] = dict(b=float(b1), err=float(np.sqrt(vb)), chi2=float(rb @ Ci @ rb))
                res[var]['lin+quad'] = dict(b=float(p[0]), b_err=float(np.sqrt(cov[0, 0])), a=float(p[1]), a_err=float(np.sqrt(cov[1, 1])))
            zref = {}
            for nm, f in REFS.items():
                Tp = T0 * f(D); zref[nm] = ((T - Tp) / np.sqrt(s ** 2 + (f(D) * s0) ** 2)).tolist()
            out['%s | %s' % (key, kind)] = dict(D=D.tolist(), y=y.tolist(), fit=res, z_vs_ref=zref, sig=s.tolist())
            print('\n== %s | sigma=%s' % (key, kind))
            for var, r in res.items():
                print('  (%s) a D^2: a = %.3f +- %.3f  chi2=%.1f/%d | b D: b = %.3f +- %.3f chi2=%.1f/%d | b D + a D^2: b=%.3f+-%.3f a=%.3f+-%.3f' % (var, r['a'], r['err'], r['chi2'], r['dof'], r['lin_only']['b'], r['lin_only']['err'], r['lin_only']['chi2'], r['dof'], r['lin+quad']['b'], r['lin+quad']['b_err'], r['lin+quad']['a'], r['lin+quad']['a_err']))
            for nm, z in zref.items(): print('  z vs %-18s' % nm, ' '.join('%+.1f' % v for v in z), '  (D=%s)' % ','.join('%g' % d for d in D))
json.dump(out, open('data/task1_d2fit.json', 'w'), indent=1)
