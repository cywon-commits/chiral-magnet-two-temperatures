"""Task 1: (a) U_heat - U_cool vs T for every (D, L) (seed mean, error = seed scatter (+) per-direction).
(b) D=0.3: C = dU/dT for the CPC-style quench protocol (L=128) vs annealed cooling/heating (L=88, 176). Run from repo root."""
import sys; sys.path.insert(0, 'figures')
from _style import *
S = summary(); sizes = {0.0: (64, 128, 256), 0.2: (128, 256), 0.3: (88, 176), 0.4: (64, 128), 0.5: (56, 112)}
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8))
for D, Ls in sizes.items():
    for k, L in enumerate(Ls):
        c, h = get(S, 'anneal', D, L, 'cool'), get(S, 'anneal', D, L, 'heat')
        if c is None or h is None: continue
        T = np.array(c['curve_U']['T']); d = np.array(h['curve_U']['mean']) - np.array(c['curve_U']['mean'])
        e = np.hypot(c['curve_U']['err'], h['curve_U']['err'])
        a.errorbar(T + 0.0015 * (list(sizes).index(D) - 2), d * 1e3, yerr=e * 1e3, color=DCOL[D], marker=DMRK[D], mfc=DCOL[D] if k == len(Ls) - 1 else 'white',
                   lw=0.8, label='D=%.1f L=%d' % (D, L))
a.axhline(0, color=MUTED, lw=0.8); a.set_xlabel('T  [J/k_B]'); a.set_ylabel('U_heat − U_cool  [10⁻³ J per spin]'); a.legend(fontsize=6.5, ncol=2)
a.set_title('(a) hysteresis of the energy', fontsize=9.5)
for proto, L, dirn, ls, lab in (('quench', 128, 'quench', '-', 'quench→heat (CPC), L=128'), ('anneal', 88, 'cool', ':', 'anneal cool, L=88'),
                                 ('anneal', 176, 'cool', '--', 'anneal cool, L=176'), ('anneal', 176, 'heat', '-.', 'anneal heat, L=176')):
    e = get(S, proto, 0.3, L, dirn)
    if e is None: continue
    c = e['curve_Cdu']; b.errorbar(c['T'], c['mean'], yerr=c['err'], ls=ls, color=DCOL[0.3] if proto == 'anneal' else INK, label=lab, lw=1.3)
b.set_xlabel('T  [J/k_B]'); b.set_ylabel('C = dU/dT  [k_B per spin]'); b.legend(fontsize=7.5); b.set_title('(b) D=0.3: quench protocol vs annealing', fontsize=9.5)
fig.savefig('figures/fig_t1_hysteresis.png')
