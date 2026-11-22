"""H2 judgement exactly as pre-registered in REPORT.md [R-6] (main definition T_L per review V-4).
Inputs: data/raw/task2_main_runs.json (src/analyze_task2.py main), data/task1_h1.json (T*).
Definitions (per D):
  seed-averaged curve (mean over seeds of c1/r, or of log xi_q with NaN ignored) -> lowest-T sign change (linear interp.)
  sigma_stat = std over seeds of the per-seed value / sqrt(n);  sigma_syst = half range over windows 1.2, 1.5, 2.0 q
  T_L: c1/r = 0 ;  T_xi1: q xi_q = 1 ;  T_xi: xi_q = lambda (only if reached for the seed-averaged curve with the 1.5q window)
xi_H(T): D=0 L=512, mean over 2 seeds of the OZ xi, log-interpolated; flagged if T < 0.55 (L/xi < 5).
Output data/task2_h2.json and printed table."""
import json, numpy as np
from scipy.stats import chi2 as chi2d
R = json.load(open('data/raw/task2_main_runs.json')); H1 = json.load(open('data/task1_h1.json'))['rows']
WIN = ('1.2', '1.5', '2.0')
def crossing_min(T, y, level=0.0):
    o = np.argsort(T); T = np.asarray(T)[o]; y = np.asarray(y, float)[o] - level; c = []
    for i in range(len(T) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] * y[i + 1] < 0: c.append(T[i] - y[i] * (T[i + 1] - T[i]) / (y[i + 1] - y[i]))
    return min(c) if c else np.nan
# xi_H(T)
r0 = [r for r in R if r['D'] == 0]; T0 = np.array(r0[0]['T']); xiH = np.exp(np.mean([np.log(r['xi_oz']) for r in r0], 0))
xiH_seed = np.array([r['xi_oz'] for r in r0])
def xi_at(T):
    o = np.argsort(T0); return float(np.exp(np.interp(T, T0[o], np.log(xiH[o]))))
def dlnxi_dT(T, h=0.015): return (np.log(xi_at(T + h)) - np.log(xi_at(T - h))) / (2 * h)
Ds = sorted({r['D'] for r in R if r['D'] > 0}); res = {}
for D in Ds:
    rs = [r for r in R if r['D'] == D]; T = np.array(rs[0]['T']); lam = rs[0]['lam']; q = 2 * np.pi / lam; e = {}
    for name, key, lev, uselog in (('T_L', 'c1_w', 0.0, False), ('T_xi1', 'xiq_w', np.log(1 / q), True), ('T_xi', 'xiq_w', np.log(lam), True)):
        def curve(r, w):
            y = np.array(r[key + w], float)
            if uselog: y = np.log(np.where(np.isfinite(y) & (y > 0), y, np.nan))
            return y
        byw = {w: crossing_min(T, np.nanmean([curve(r, w) for r in rs], 0), lev) for w in WIN}
        seeds = [crossing_min(T, curve(r, '1.5'), lev) for r in rs]; sv = np.array([s for s in seeds if np.isfinite(s)])
        stat = sv.std(ddof=1) / np.sqrt(len(sv)) if len(sv) > 1 else np.nan
        wv = [v for v in byw.values() if np.isfinite(v)]; syst = (max(wv) - min(wv)) / 2 if len(wv) > 1 else np.nan
        e[name] = dict(T=byw['1.5'], stat=stat, syst=syst, tot=float(np.hypot(stat, syst)) if np.isfinite(syst) else stat, by_window=byw, seeds=seeds, n_windows=len(wv))
    res[D] = dict(L=rs[0]['L'], lam=lam, n_seeds=len(rs), **e)
# T* (task 1): L/lambda~8 where available, D=0.1/0.15 L/lambda~4; dU/dT-only sigma
tkey = {0.1: '0.1_256', 0.15: '0.15_168', 0.2: '0.2_256', 0.3: '0.3_176', 0.4: '0.4_128', 0.5: '0.5_112'}
def ts(D): r = H1[tkey[D]]; return r['T'], float(np.sqrt(r['stat'] ** 2 + r['s_win'] ** 2 + r['s_hys'] ** 2))
out = dict(xiH=dict(T=T0.tolist(), xi=xiH.tolist(), xi_seeds=xiH_seed.tolist()), per_D={str(D): v for D, v in res.items()}, judge={})
print('%-5s %-4s %-6s | %-28s | %-28s | %-28s | T* (dU/dT)' % ('D', 'L', 'lam', 'T_L  (stat, syst) [w1.2,2.0]', 'T_xi1 (q xi_q=1)', 'T_xi (xi_q=lam)'))
for D in Ds:
    r = res[D]; t, s = ts(D)
    f = lambda e: '%.4f (%.4f, %.4f) [%.3f,%.3f]' % (e['T'], e['stat'], e['syst'], e['by_window']['1.2'], e['by_window']['2.0'])
    print('%-5.2f %-4d %-6.1f | %s | %s | %s | %.4f +- %.4f' % (D, r['L'], r['lam'], f(r['T_L']), f(r['T_xi1']), f(r['T_xi']), t, s))
for name in ('T_L', 'T_xi1', 'T_xi'):
    J = {}; pts = [(D, res[D][name]) for D in Ds if np.isfinite(res[D][name]['T']) and np.isfinite(res[D][name]['tot'])]
    if name == 'T_xi': pts = [(D, e) for D, e in pts if res[D][name]['by_window']['1.5'] == res[D][name]['by_window']['1.5']]
    D_ = np.array([p[0] for p in pts]); T_ = np.array([p[1]['T'] for p in pts]); S_ = np.array([p[1]['tot'] for p in pts])
    print('\n==== %s  (%d D values: %s)' % (name, len(D_), ', '.join('%g' % d for d in D_)))
    w = 1 / S_ ** 2
    def wlin(x, y, w):
        A = np.vstack([x, np.ones_like(x)]).T * np.sqrt(w)[:, None]; cov = np.linalg.inv(A.T @ A); p = cov @ A.T @ (y * np.sqrt(w)); c2 = float((w * (y - A[:, 0] / np.sqrt(w) * p[0] - p[1]) ** 2).sum())
        return p, cov, c2
    # H2a
    p, cov, c2 = wlin(D_, T_, w); z = p[0] / np.sqrt(cov[0, 0])
    v = 'support' if z > 3 else ('reject' if abs(z) < 2 else 'undecided'); J['H2a'] = dict(slope=float(p[0]), err=float(np.sqrt(cov[0, 0])), z=float(z), verdict=v)
    print('  H2a: dT_q/dD = %.3f +- %.3f (z=%.1f, chi2=%.1f/%d) -> %s' % (p[0], np.sqrt(cov[0, 0]), z, c2, len(D_) - 2, v))
    # H2b
    if 0.1 in D_ and 0.5 in D_:
        Dl = {d: (res[d][name]['T'] - ts(d)[0], float(np.hypot(res[d][name]['tot'], ts(d)[1]))) for d in D_}
        dd = Dl[0.1][0] - Dl[0.5][0]; sd = np.hypot(Dl[0.1][1], Dl[0.5][1]); z = abs(dd) / sd
        absd = np.array([abs(Dl[d][0]) for d in D_]); mono = np.polyfit(D_, absd, 1)[0] < 0
        v = 'support' if (z > 3 and mono) else ('reject' if z < 2 else 'undecided')
        J['H2b'] = dict(Delta={str(d): Dl[d] for d in D_}, diff_01_05=float(dd), err=float(sd), z=float(z), absDelta_grows_as_D_decreases=bool(mono), verdict=v)
        print('  H2b: Delta(D)=T_q-T*: ' + ' '.join('%g:%+.3f' % (d, Dl[d][0]) for d in D_) + '  | Delta(0.1)-Delta(0.5)=%+.4f+-%.4f z=%.1f, |Delta| grows as D->small: %s -> %s' % (dd, sd, z, mono, v))
    # H2c
    xs = np.array([xi_at(t) for t in T_]); sx = xs * np.abs([dlnxi_dT(t) for t in T_]) * S_; lam_ = 2 * np.pi / np.arctan(D_)
    k = xs / lam_; sk = sx / lam_; wk = 1 / sk ** 2; km = (wk * k).sum() / wk.sum(); c2k = float((wk * (k - km) ** 2).sum()); pk = 1 - chi2d.cdf(c2k, len(k) - 1)
    flag = [bool(t < 0.55) for t in T_]
    v = 'support' if pk > 0.05 else ('reject' if pk < 0.01 else 'undecided')
    J['H2c'] = dict(kappa=k.tolist(), kappa_err=sk.tolist(), kappa_mean=float(km), chi2=c2k, dof=len(k) - 1, p=float(pk), xi_flag_T_lt_055=flag, verdict=v)
    print('  H2c: xi_H(T_q)/lam = ' + ' '.join('%.3f+-%.3f%s' % (a, b, '*' if f else '') for a, b, f in zip(k, sk, flag)) + '  mean %.3f chi2=%.1f/%d p=%.3g -> %s  (* T<0.55, xi_H unreliable)' % (km, c2k, len(k) - 1, pk, v))
    # functional forms
    forms = {}
    x = np.log(1 / D_); y = 1 / T_; sy = S_ / T_ ** 2; p, cov, c2 = wlin(x, y, 1 / sy ** 2)
    forms['(i) 1/T=s ln(1/D)+c'] = dict(chi2=c2, dof=len(D_) - 2, s=float(p[0]), s_err=float(np.sqrt(cov[0, 0])), z_vs_1_2pi=float((p[0] - 1 / (2 * np.pi)) / np.sqrt(cov[0, 0])))
    p, cov, c2 = wlin(D_, T_, w); forms['(ii) T=T0+b|D|'] = dict(chi2=c2, dof=len(D_) - 2, T0=float(p[1]), b=float(p[0]))
    p, cov, c2 = wlin(D_ ** 2, T_, w); forms['(iii) T=T0+aD^2'] = dict(chi2=c2, dof=len(D_) - 2, T0=float(p[1]), a=float(p[0]))
    To = np.sort(T0); lxo = np.log(xiH[np.argsort(T0)])
    def Tpred(kap):  # invert xi_H(T) = kap*lam (xi_H decreasing in T); restrict to T>=0.55
        sel = To >= 0.55; return np.interp(np.log(kap * lam_), lxo[sel][::-1], To[sel][::-1], left=np.nan, right=np.nan)
    kg = np.exp(np.linspace(np.log(0.01), np.log(3), 3000)); c2g = np.array([np.nansum(((T_ - Tpred(kk)) / S_) ** 2) + 1e6 * np.isnan(Tpred(kk)).sum() for kk in kg])
    kb = kg[np.argmin(c2g)]; forms['(iv) xi_H(T)=kappa lam'] = dict(chi2=float(c2g.min()), dof=len(D_) - 1, kappa=float(kb), n_outside_xi_range=int(np.isnan(Tpred(kb)).sum()))
    for nm, f in forms.items():
        f['p'] = float(1 - chi2d.cdf(f['chi2'], f['dof'])) if f['chi2'] < 1e5 else 0.0
        print('  %-24s chi2=%7.1f/%d p=%.3g  %s' % (nm, f['chi2'], f['dof'], f['p'], {k: round(v, 4) for k, v in f.items() if k not in ('chi2', 'dof', 'p')}))
    best_nonan = min(forms['(i) 1/T=s ln(1/D)+c']['chi2'], forms['(iv) xi_H(T)=kappa lam']['chi2'])
    ok = (forms['(i) 1/T=s ln(1/D)+c']['p'] > 0.05 or forms['(iv) xi_H(T)=kappa lam']['p'] > 0.05) and \
         forms['(ii) T=T0+b|D|']['chi2'] - best_nonan > 9 and forms['(iii) T=T0+aD^2']['chi2'] - best_nonan > 9
    J['forms'] = forms; J['forms_distinguished'] = bool(ok)
    print('  functional form distinguished from analytic (ii),(iii): %s' % ok)
    out['judge'][name] = J
json.dump(out, open('data/task2_h2.json', 'w'), indent=1, default=float)
