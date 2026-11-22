"""Task 2 main runs (L/lambda~8, 4 seeds; D=0.1 and D=0: L=512, 2 seeds).
(a) characteristic temperatures vs D: T_L (disorder line, main), T_xi1 (q xi_q = 1), T_xi (xi_q = lambda), T* (task 1 C peak);
    inner bar = stat (seed scatter), outer = total (stat (+) fit-window syst).
(b) 1/T vs ln(1/D): log-form fits (i) and one-parameter xi_H(T_q) = kappa*lambda curves (iv) from the measured D=0 xi_H(T).
(c) xi_H(T_q)/lambda for T_L and T_xi1 (H2c), RPA prediction 0.098 for T_L.
(d) bond chiral correlation <C>/D vs T. Run from repo root."""
import sys, json; sys.path.insert(0, 'figures')
from _style import *
H = json.load(open('data/task2_h2.json')); H1 = json.load(open('data/task1_h1.json'))['rows']; S = json.load(open('data/task2_main_summary.json'))
Ds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]; P = H['per_D']
DC = dict(DCOL); DC[0.1] = '#7a5af5'; DC[0.15] = '#008300'
defs = (('T_L', 'T_L (disorder line)', 'o', DCOL[0.0]), ('T_xi1', 'T(q ξ_q = 1)', 's', DCOL[0.2]), ('T_xi', 'T(ξ_q = λ)', '^', DCOL[0.3]))
tkey = {0.1: '0.1_256', 0.15: '0.15_168', 0.2: '0.2_256', 0.3: '0.3_176', 0.4: '0.4_128', 0.5: '0.5_112'}
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
a, b, c, d = ax
for k, lab, mk, col in defs:
    T = np.array([P[str(D)][k]['T'] for D in Ds]); st = np.array([P[str(D)][k]['stat'] for D in Ds]); tot = np.array([P[str(D)][k]['tot'] for D in Ds])
    a.errorbar(Ds, T, yerr=tot, color=col, lw=1, capsize=0, ls='none'); a.errorbar(Ds, T, yerr=st, color=col, marker=mk, ms=6, lw=1.6, capsize=3, label=lab)
ts = np.array([H1[tkey[D]]['T'] for D in Ds]); tse = np.array([np.sqrt(H1[tkey[D]]['stat'] ** 2 + H1[tkey[D]]['s_win'] ** 2 + H1[tkey[D]]['s_hys'] ** 2) for D in Ds])
a.errorbar(Ds, ts, yerr=tse, color=INK, marker='D', mfc='white', ms=5, lw=1.2, capsize=3, label='T* (C peak, task 1)')
a.axhspan(0.0, 0.55, color='#f1efe8', zorder=0); a.text(0.3, 0.45, 'T < 0.55: ξ_H(D=0) not reliable at L=512', fontsize=7, color=MUTED, ha='center')
a.set_xlabel('D  [J]'); a.set_ylabel('T  [J/k_B]'); a.set_ylim(0.25, 1.12); a.legend(fontsize=7.5, loc='upper left'); a.set_title('(a) characteristic temperatures', fontsize=9.5)
xT = np.array(H['xiH']['T']); xx = np.array(H['xiH']['xi']); o = np.argsort(xT); xT, xx = xT[o], xx[o]; sel = xT >= 0.55
Dg = np.exp(np.linspace(np.log(0.04), np.log(0.6), 100)); lamg = 2 * np.pi / np.arctan(Dg)
for k, lab, mk, col in defs[:2]:
    T = np.array([P[str(D)][k]['T'] for D in Ds]); tot = np.array([P[str(D)][k]['tot'] for D in Ds])
    b.errorbar(np.log(1 / np.array(Ds)), 1 / T, yerr=tot / T ** 2, color=col, marker=mk, ms=6, ls='none', capsize=3, label=lab)
    f = H['judge'][k]['forms']['(i) 1/T=s ln(1/D)+c']; x = np.log(1 / np.array(Ds)); y = 1 / T; w = (T ** 2 / tot) ** 2
    p = np.polyfit(x, y, 1, w=np.sqrt(w)); xg = np.log(1 / Dg); b.plot(xg, np.polyval(p, xg), color=col, lw=1, ls='--', label='(i) s = %.3f' % f['s'])
    kap = H['judge'][k]['forms']['(iv) xi_H(T)=kappa lam']['kappa']; Tp = np.interp(np.log(kap * lamg), np.log(xx[sel])[::-1], xT[sel][::-1], left=np.nan, right=np.nan)
    b.plot(xg, 1 / Tp, color=col, lw=1.4, alpha=0.6, label='(iv) ξ_H = %.3f λ' % kap)
b.plot(xg, xg / (2 * np.pi) + 1.0, color=MUTED, lw=0.8, ls=':', label='slope 1/2π (offset arbitrary)')
b.set_xlabel('ln(J/D)'); b.set_ylabel('1/T  [k_B/J]'); b.legend(fontsize=6.5); b.set_title('(b) log form vs measured ξ_H(T)', fontsize=9.5)
for k, lab, mk, col in defs[:2]:
    j = H['judge'][k]['H2c']; c.errorbar(Ds, j['kappa'], yerr=j['kappa_err'], color=col, marker=mk, ms=6, capsize=3, ls='none', label='%s: mean %.3f, p=%.2f' % (lab, j['kappa_mean'], j['p']))
c.axhline(0.098, color=DCOL[0.0], lw=0.8, ls=':', label='RPA for T_L: √(3/8)/2π = 0.098')
c.set_xlabel('D  [J]'); c.set_ylabel('ξ_H(T_q) / λ'); c.legend(fontsize=7); c.set_title('(c) H2c: ξ_H(T_q) = κλ ?', fontsize=9.5)
for D in Ds:
    rs = [r for r in S if r['D'] == D]; T = np.array(rs[0]['T']); C = np.array([r['C'] for r in rs]).mean(0) / D
    d.plot(T, C, color=DC[D], marker=DMRK.get(D, 'o') if D in DMRK else ('P' if D == 0.1 else 'X'), ms=3.5, lw=1.2, label='D=%g' % D)
d.set_xlabel('T  [J/k_B]'); d.set_ylabel('⟨C⟩/D  (bond chiral correlation)'); d.legend(fontsize=7, ncol=2); d.set_title('(d) ⟨C⟩/D, seed mean (err < 1.3e-3)', fontsize=9.5)
fig.savefig('figures/fig_t2_main.png')
