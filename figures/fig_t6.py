"""Task 6b figure: (a) T_L(D) and T*(D) with the four fitted forms and the pre-registered predictions at D=0.05, 0.075;
(b) kappa = xi_H(T_L)/lambda vs D with the large-N kappa; (c) c1/r(T) at D = 0.05 (L=512 seeds, L=1024) and 0.075.
Inputs data/task6_judge.json, data/raw/task2_small_runs.json, data/task6a_largeN_TL.json -> figures/t6_inversion.png"""
import sys; sys.path.insert(0, 'figures')
from _style import *
J = json.load(open('data/task6_judge.json')); P = J['per_run']; G = J['judge']
main = ['0.05_512', '0.075_384', '0.1_512', '0.15_336', '0.2_256', '0.3_176', '0.4_128', '0.5_112']
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
a = ax[0]; D = np.array([P[k]['D'] for k in main]); TL = np.array([P[k]['TL']['T'] for k in main]); sTL = np.array([P[k]['TL']['tot'] for k in main])
a.errorbar(D, TL, sTL, fmt='o', color=C1, capsize=2, label='$T_L$ (MC, $L/\\lambda\\approx$4–8)', zorder=5)
Ts = np.array([P[k]['Tstar']['T'] for k in main[:2]]); sTs = np.array([P[k]['Tstar']['tot'] for k in main[:2]])
a.errorbar(D[:2], Ts, sTs, fmt='s', color=C2, capsize=2, label='$T^*$ (task 6, cooling dU/dT)', zorder=5)
t1 = {0.1: (0.6843, 0.0046), 0.15: (0.6925, 0.0034), 0.2: (0.6961, 0.0018), 0.3: (0.7083, 0.0053), 0.4: (0.7086, 0.0097), 0.5: (0.7292, 0.0104)}
a.errorbar(list(t1), [v[0] for v in t1.values()], [v[1] for v in t1.values()], fmt='s', mfc='white', color=C2, capsize=2, label='$T^*$ (task 1)')
if '0.05_1024' in P and np.isfinite(P['0.05_1024']['TL']['T']):
    a.errorbar([0.05], [P['0.05_1024']['TL']['T']], [P['0.05_1024']['TL']['tot']], fmt='o', mfc='white', color=C1, capsize=2, label='$T_L$, $L$=1024')
Dg = np.linspace(0.03, 0.52, 200); F = G['6-2']['forms']
a.plot(Dg, 1 / (F['(i) log']['s'] * np.log(1 / Dg) + F['(i) log']['c']), color=MUTED, ls='--', lw=1, label='log fit $\\chi^2$=%.1f' % F['(i) log']['chi2'])
a.plot(Dg, F['(ii) linear']['T0'] + F['(ii) linear']['b'] * Dg, color=C3, lw=1, label='linear fit $\\chi^2$=%.1f' % F['(ii) linear']['chi2'])
a.plot(Dg, F['(iii) D^2']['T0'] + F['(iii) D^2']['a'] * Dg ** 2, color=C4, lw=1, ls=':', label='$D^2$ fit $\\chi^2$=%.1f' % F['(iii) D^2']['chi2'])
a.plot(D, F['(iv) kappa*lambda']['pred'], 'x', color=INK, label='$\\kappa\\lambda$ fit $\\chi^2$=%.1f' % F['(iv) kappa*lambda']['chi2'])
a.plot([0.05, 0.075], [0.633, 0.664], '_', ms=14, mew=2, color='#8a5cd0', label='pre-registered $\\kappa\\lambda$ prediction')
a.set_xlabel('D / J'); a.set_ylabel('T / J'); a.set_xlim(0, 0.53); a.legend(fontsize=7, loc='upper left'); a.set_title('(a) order inversion at small D', fontsize=10)
b = ax[1]; k = G['6-3']; b.errorbar(k['D'], k['kappa'], k['err'], fmt='o', color=C1, capsize=2, label='MC $\\xi_H(T_L)/\\lambda$')
LN = json.load(open('data/task6a_largeN_TL.json'))['1024']; b.plot([r['D'] for r in LN], [r['kappa'] for r in LN], '-', color=C2, label='large-N')
b.axhline(k['mean'], color=MUTED, lw=0.8, ls='--', label='weighted mean %.3f ($p$=%.2g)' % (k['mean'], k['p']))
b.set_xlabel('D / J'); b.set_ylabel('$\\kappa$'); b.set_xlim(0, 0.53); b.legend(fontsize=7); b.set_title('(b) $\\kappa$ grows as D decreases', fontsize=10)
c = ax[2]; R = json.load(open('data/raw/task2_small_runs.json'))
for r in R:
    if not len(r['T']) or 'c1_w1.5' not in r: continue
    col = C1 if r['D'] == 0.05 else C3; ls = '-' if r['L'] != 1024 else '--'
    c.plot(r['T'], r['c1_w1.5'], ls, color=col, lw=1, marker='o' if r['seed'] == 1 else 's', ms=2.5, label='D=%g L=%d s%d' % (r['D'], r['L'], r['seed']))
for kk, col in (('0.05_512', C1), ('0.075_384', C3)): c.axvline(P[kk]['TL']['T'], color=col, lw=0.8, ls=':'); c.axvline(P[kk]['Tstar']['T'], color=C2, lw=0.8)
c.axhline(0, color=INK, lw=0.6); c.set_ylim(-400, 120); c.set_xlim(0.5, 0.8); c.set_xlabel('T / J'); c.set_ylabel('$c_1/r$ (window 1.5q)')
c.legend(fontsize=7, loc='lower right'); c.set_title('(c) curvature at k=0; dotted $T_L$, solid $T^*$', fontsize=10)
fig.tight_layout(); fig.savefig('figures/t6_inversion.png')
