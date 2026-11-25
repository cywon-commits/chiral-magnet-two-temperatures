"""Task 6b judgement, fixed (and committed) before any task-6 result was looked at.
Pre-registered criteria: REQUEST.md task 6, (6-1)-(6-4).  Definitions of T_L and sigma as in [R-6]/[R-8] (src/judge_h2.py).

Inputs
  data/raw/task2_main_runs.json   (D = 0, 0.1-0.5; src/analyze_task2.py main)
  data/raw/task2_small_runs.json  (D = 0.05: L=512 s1,s2 and L=1024 s1; D = 0.075: L=384 s1,s2; src/analyze_task2.py small)

T_L (per D and L): seed-averaged c1/r curve -> lowest-T sign change; sigma_stat = std over seeds / sqrt(n) (1.5q window);
  sigma_syst = half range over the windows 1.2, 1.5, 2.0 q.  For the single-seed L=1024 run sigma_stat is not available
  and is taken from the L=512 value at the same D (stated in the output).
T* for the task-6 runs (rule fixed here because task 6 has cooling runs only, no heating branch):
  C = dU/dT by central difference on the 0.01 grid, quadratic peak finder with half-widths 0.03, 0.04, 0.05 (as task 1, 'du').
  T* = peak of the seed-averaged U(T) with the 0.04 window; sigma_win = half range over the three windows;
  sigma_stat = max(std over seeds / sqrt(n), parametric bootstrap from the block errors of U (500 resamples));
  no hysteresis term (no heating run) -> stated as a limitation.
  The same rule applied to the task-2 main runs (cooling only, 0.03 grid -> windows scaled to 3, 4, 5 grid spacings) at
  D = 0.1-0.5 is printed as a calibration against task 1 (not used in any verdict).
(6-1) D = 0.05, L = 512: Delta = T_L - T*, sigma = sqrt(sigma_TL^2 + sigma_T*^2);
  Delta < 0 and |Delta| > 2 sigma -> 'inversion supported'; |Delta| <= 2 sigma -> 'undecided'; Delta > 2 sigma -> 'inversion rejected'.
  L = 1024 and D = 0.075 reported alongside (not part of the verdict).
(6-2) 8 points D = 0.05, 0.075 (task 6) + 0.1-0.5 (task 2), weights 1/sigma_TL^2:
  (i) log 1/T = s ln(1/D) + c ; (ii) linear T = T0 + b D ; (iii) T = T0 + a D^2 ; (iv) xi_H(T) = kappa lambda (one parameter).
  A form is 'supported' if its chi^2 is lower than every other form's by more than 9, else 'not distinguishable'.
(6-3) kappa_i = xi_H(T_L,i)/lambda_i with xi_H from D = 0, L = 512 (log interpolation); sigma from sigma_TL * |d xi_H/dT|;
  weighted constant, chi^2 p > 0.05 -> 'constant supported', p < 0.01 -> 'rejected', else 'undecided'.
  Points with T_L < 0.60 (L/xi_H < 15 at L = 512, [R-10]) are flagged.
(6-4) Large-N (task 6a, data/task6a_largeN_TL.json): kappa stays ~constant (0.136 -> 0.137) as D -> 0 and kappa increases as D
  decreases. 'Same direction' if the MC kappa(D) slope over D <= 0.1 is <= 0 within 2 sigma (kappa does not fall as D decreases)
  and the MC kappa at D = 0.05, 0.075 is finite (T_L exists).  Reported qualitatively, as REQUEST.md asks.
Output: data/task6_judge.json
"""
import json, numpy as np
from scipy.stats import chi2 as chi2d

WIN = ('1.2', '1.5', '2.0'); TWIN = (0.03, 0.04, 0.05)
R = json.load(open('data/raw/task2_main_runs.json')) + json.load(open('data/raw/task2_small_runs.json'))
rng = np.random.default_rng(6)


def crossing_min(T, y, level=0.0):
    o = np.argsort(T); T = np.asarray(T)[o]; y = np.asarray(y, float)[o] - level; c = []
    for i in range(len(T) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] * y[i + 1] < 0: c.append(T[i] - y[i] * (T[i + 1] - T[i]) / (y[i + 1] - y[i]))
    return min(c) if c else np.nan


def qpeak(T, y, w):
    i = int(np.argmax(y)); sel = np.abs(T - T[i]) <= w + 1e-9
    if sel.sum() < 4: return np.nan
    c = np.polyfit(T[sel] - T[i], y[sel], 2)
    if c[0] >= 0: return np.nan
    t = T[i] - c[1] / (2 * c[0])
    return t if T[sel].min() <= t <= T[sel].max() else np.nan


def tl(rs):
    T = np.array(rs[0]['T'])
    byw = {w: crossing_min(T, np.nanmean([np.array(r['c1_w' + w], float) for r in rs], 0)) for w in WIN}
    seeds = [crossing_min(T, np.array(r['c1_w1.5'], float)) for r in rs]; sv = np.array([s for s in seeds if np.isfinite(s)])
    stat = sv.std(ddof=1) / np.sqrt(len(sv)) if len(sv) > 1 else np.nan
    wv = [v for v in byw.values() if np.isfinite(v)]; syst = (max(wv) - min(wv)) / 2 if len(wv) > 1 else np.nan
    return dict(T=byw['1.5'], stat=stat, syst=syst, by_window=byw, seeds=seeds)


def tstar(rs):
    T = np.array(rs[0]['T']); o = np.argsort(T); T = T[o]
    U = np.array([np.array(r['U'])[o] for r in rs]); sU = np.array([np.array(r['sU'])[o] for r in rs])
    Tc = T[1:-1]; h = float(np.median(np.diff(T))); sc = h / 0.01   # windows scale with the grid spacing (0.03-0.05 on the 0.01 grid)
    def peak(u, w): return qpeak(Tc, (u[2:] - u[:-2]) / (T[2:] - T[:-2]), w * sc)
    um = U.mean(0); sm = np.sqrt((sU ** 2).sum(0)) / len(rs)
    byw = {w: peak(um, w) for w in TWIN}
    seeds = [peak(u, 0.04) for u in U]; sv = np.array([s for s in seeds if np.isfinite(s)])
    s_seed = sv.std(ddof=1) / np.sqrt(len(sv)) if len(sv) > 1 else np.nan
    bs = np.array([peak(um + sm * rng.standard_normal(len(um)), 0.04) for _ in range(500)]); bs = bs[np.isfinite(bs)]
    s_boot = bs.std(ddof=1); stat = np.nanmax([s_seed, s_boot])
    wv = [v for v in byw.values() if np.isfinite(v)]; s_win = (max(wv) - min(wv)) / 2
    return dict(T=byw[0.04], stat=float(stat), s_seed=float(s_seed), s_boot=float(s_boot), s_win=float(s_win), tot=float(np.hypot(stat, s_win)),
                by_window={str(k): v for k, v in byw.items()}, seeds=seeds, n_boot_ok=int(len(bs)))


# xi_H(T) at D = 0, L = 512
r0 = [r for r in R if r['D'] == 0]; T0 = np.array(r0[0]['T']); xiH = np.exp(np.mean([np.log(r['xi_oz']) for r in r0], 0))
o0 = np.argsort(T0); To, lxo = T0[o0], np.log(xiH[o0])
def xi_at(T): return float(np.exp(np.interp(T, To, lxo)))
def dlnxi_dT(T, h=0.015): return (np.log(xi_at(T + h)) - np.log(xi_at(T - h))) / (2 * h)

groups = {}
for r in R:
    if r['D'] > 0: groups.setdefault((r['D'], r['L']), []).append(r)
main_L = {0.05: 512, 0.075: 384, 0.1: 512, 0.15: 336, 0.2: 256, 0.3: 176, 0.4: 128, 0.5: 112}
res = {}
for (D, L), rs in sorted(groups.items()):
    e = tl(rs); res[(D, L)] = dict(D=D, L=L, lam=rs[0]['lam'], n_seeds=len(rs), TL=e, Tstar=tstar(rs))
for (D, L), v in res.items():   # single-seed runs: borrow sigma_stat from the main-L run at the same D
    if not np.isfinite(v['TL']['stat']) and (D, main_L[D]) in res and main_L[D] != L:
        v['TL']['stat'] = res[(D, main_L[D])]['TL']['stat']; v['TL']['stat_borrowed_from_L'] = main_L[D]
    v['TL']['tot'] = float(np.hypot(v['TL']['stat'], v['TL']['syst']))

print('%-6s %-5s %-6s | %-44s | %s' % ('D', 'L', 'lam', 'T_L  (stat, syst) [w1.2, w1.5, w2.0]', 'T* dU/dT cooling (stat=max(seed,boot), win) [0.03,0.04,0.05]'))
for k, v in sorted(res.items()):
    a, b = v['TL'], v['Tstar']
    print('%-6.3f %-5d %-6.1f | %.4f (%.4f, %.4f) [%s] | %.4f (%.4f [%.4f,%.4f], %.4f) [%s]' % (v['D'], v['L'], v['lam'], a['T'], a['stat'], a['syst'],
          ', '.join('%.3f' % a['by_window'][w] for w in WIN), b['T'], b['stat'], b['s_seed'], b['s_boot'], b['s_win'],
          ', '.join('%.4f' % x for x in b['by_window'].values())))
print('  task-1 T* (dU/dT, cool+heat): 0.1:0.6843 0.15:0.6925 0.2:0.6961 0.3:0.7083 0.4:0.7086 0.5:0.7292 (calibration of the cooling-only rule)')

J = {}
# (6-1)
for key in ((0.05, 512), (0.05, 1024), (0.075, 384)):
    if key not in res: continue
    v = res[key]; d = v['TL']['T'] - v['Tstar']['T']; s = float(np.hypot(v['TL']['tot'], v['Tstar']['tot']))
    verdict = 'inversion supported' if (d < 0 and abs(d) > 2 * s) else ('inversion rejected' if d > 2 * s else 'undecided')
    J['6-1_D%g_L%d' % key] = dict(Delta=float(d), sigma=s, z=float(d / s), verdict=verdict, primary=(key == (0.05, 512)))
    print('(6-1) D=%g L=%d: T_L - T* = %+.4f +- %.4f (z=%+.1f) -> %s%s' % (key[0], key[1], d, s, d / s, verdict, '  [primary]' if key == (0.05, 512) else ''))

# (6-2), (6-3)
pts = [res[(D, L)] for D, L in sorted(main_L.items()) if (D, L) in res and np.isfinite(res[(D, L)]['TL']['T'])]
D_ = np.array([p['D'] for p in pts]); T_ = np.array([p['TL']['T'] for p in pts]); S_ = np.array([p['TL']['tot'] for p in pts]); lam_ = 2 * np.pi / np.arctan(D_)
print('\n8-point set: D = %s' % ', '.join('%g' % d for d in D_))
def wlin(x, y, s):
    w = 1 / s ** 2; A = np.vstack([x, np.ones_like(x)]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); p = cov @ A.T @ (w * y)
    return p, cov, float((w * (y - A @ p) ** 2).sum())
forms = {}
p, cov, c2 = wlin(np.log(1 / D_), 1 / T_, S_ / T_ ** 2); forms['(i) log'] = dict(chi2=c2, dof=len(D_) - 2, s=float(p[0]), c=float(p[1]))
p, cov, c2 = wlin(D_, T_, S_); forms['(ii) linear'] = dict(chi2=c2, dof=len(D_) - 2, b=float(p[0]), T0=float(p[1]))
p, cov, c2 = wlin(D_ ** 2, T_, S_); forms['(iii) D^2'] = dict(chi2=c2, dof=len(D_) - 2, a=float(p[0]), T0=float(p[1]))
def Tpred(kap): return np.interp(np.log(kap * lam_), lxo[::-1], To[::-1], left=np.nan, right=np.nan)
kg = np.exp(np.linspace(np.log(0.02), np.log(2), 4000)); c2g = np.array([np.sum(((T_ - Tpred(k)) / S_) ** 2) if np.all(np.isfinite(Tpred(k))) else np.inf for k in kg])
kb = kg[np.argmin(c2g)]; forms['(iv) kappa*lambda'] = dict(chi2=float(c2g.min()), dof=len(D_) - 1, kappa=float(kb), pred=Tpred(kb).tolist())
for nm, f in forms.items():
    f['p'] = float(1 - chi2d.cdf(f['chi2'], f['dof'])); print('  (6-2) %-18s chi2=%7.2f/%d p=%.3g  %s' % (nm, f['chi2'], f['dof'], f['p'], {k: round(x, 4) for k, x in f.items() if k not in ('chi2', 'dof', 'p', 'pred')}))
cs = sorted((f['chi2'], nm) for nm, f in forms.items()); best = cs[0][1] if cs[1][0] - cs[0][0] > 9 else None
J['6-2'] = dict(forms=forms, best=best, verdict=('%s supported' % best) if best else 'not distinguishable', gap=float(cs[1][0] - cs[0][0]))
print('  (6-2) -> %s (best-second chi2 gap %.2f)' % (J['6-2']['verdict'], J['6-2']['gap']))
xs = np.array([xi_at(t) for t in T_]); k = xs / lam_; sk = xs * np.abs([dlnxi_dT(t) for t in T_]) * S_ / lam_
wk = 1 / sk ** 2; km = float((wk * k).sum() / wk.sum()); c2k = float((wk * (k - km) ** 2).sum()); pk = float(1 - chi2d.cdf(c2k, len(k) - 1))
v3 = 'constant supported' if pk > 0.05 else ('constant rejected' if pk < 0.01 else 'undecided'); flag = [bool(t < 0.60) for t in T_]
J['6-3'] = dict(D=D_.tolist(), kappa=k.tolist(), err=sk.tolist(), mean=km, chi2=c2k, dof=len(k) - 1, p=pk, flag_T_lt_060=flag, verdict=v3)
print('  (6-3) kappa = ' + ' '.join('%g:%.3f+-%.3f%s' % (d, a, b, '*' if f else '') for d, a, b, f in zip(D_, k, sk, flag)) + ' | mean %.3f chi2=%.1f/%d p=%.3g -> %s' % (km, c2k, len(k) - 1, pk, v3))
sel = D_ <= 0.1 + 1e-9
if sel.sum() >= 2:
    p, cov, _ = wlin(D_[sel], k[sel], sk[sel]); sl, se = float(p[0]), float(np.sqrt(cov[0, 0]))
    same = bool(sl - 2 * se <= 0)
    J['6-4'] = dict(kappa_slope_D_le_01=sl, err=se, same_direction_as_largeN=same, largeN_kappa={'0.05': 0.1364, '0.075': 0.1357, '0.1': 0.1349})
    print('  (6-4) MC d kappa/dD (D<=0.1) = %.3f +- %.3f ; large-N: 0.1364, 0.1357, 0.1349 (D=0.05,0.075,0.1) -> same direction: %s' % (sl, se, same))
json.dump(dict(per_run={'%g_%d' % k: v for k, v in res.items()}, xiH=dict(T=T0.tolist(), xi=xiH.tolist()), judge=J), open('data/task6_judge.json', 'w'), indent=1, default=float)
