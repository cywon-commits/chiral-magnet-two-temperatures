"""Task 1: specific heat C(T) per spin. One panel per D, both sizes; dU/dT (lines) and fluctuation C (open markers).
Seed-averaged (4 seeds), error bars = seed scatter / sqrt(4). Run from repo root: python3 figures/fig_t1_C.py"""
import sys; sys.path.insert(0, 'figures')
from _style import *
S = summary(); sizes = {0.0: (64, 128, 256), 0.2: (128, 256), 0.3: (88, 176), 0.4: (64, 128), 0.5: (56, 112)}
fig, axs = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
for ax, (D, Ls) in zip(axs, sizes.items()):
    for k, L in enumerate(Ls):
        for dirn, ls in (('cool', '-'), ('heat', '--')):
            e = get(S, 'anneal', D, L, dirn)
            if e is None: continue
            alpha = 0.45 + 0.55 * (k + 1) / len(Ls)
            c = e['curve_Cdu']; ax.errorbar(c['T'], c['mean'], yerr=c['err'], ls=ls, color=DCOL[D], alpha=alpha, lw=1.2 + 0.4 * k,
                                            label='L=%d %s' % (L, dirn))
            if k == len(Ls) - 1 and dirn == 'cool':
                c = e['curve_Cfl']; ax.errorbar(c['T'], c['mean'], yerr=c['err'], ls='none', marker=DMRK[D], mfc='none', color=DCOL[D], alpha=0.6, label='L=%d C_fl' % L)
    ax.set_title('D = %.1f' % D); ax.set_xlabel('T  [J/k_B]'); ax.legend(fontsize=7)
axs[0].set_ylabel('C per spin  [k_B]')
fig.suptitle('Specific heat: lines = dU/dT (central difference, solid cool / dashed heat), open markers = N var(e)/T^2', fontsize=10)
fig.savefig('figures/fig_t1_C.png')
