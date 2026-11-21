"""Task 1: specific-heat peak T*(D). Filled = L/lambda~8 (D=0: L=256), open = L/lambda~4 (D=0: L=128, 64 grey).
Inner error bar = statistical (seed scatter), outer = total (stat (+) syst: estimators, window, cool/heat).
Crosses: preliminary dU/dT values (REQUEST.md sec. 4). Run from repo root."""
import sys, json; sys.path.insert(0, 'figures')
from _style import *
H = json.load(open('data/task1_h1.json'))['rows']
prelim = {0.0: 0.678, 0.2: 0.695, 0.3: 0.704, 0.4: 0.699, 0.5: 0.725}
big = {0.0: 256, 0.2: 256, 0.3: 176, 0.4: 128, 0.5: 112}; small = {0.0: 128, 0.2: 128, 0.3: 88, 0.4: 64, 0.5: 56}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for sz, off, filled, lab in ((big, 0.006, True, 'L/λ≈8 (D=0: L=256)'), (small, -0.006, False, 'L/λ≈4 (D=0: L=128)')):
    for i, (D, L) in enumerate(sz.items()):
        r = H.get('%g_%d' % (D, L))
        if r is None: continue
        ax.errorbar(D + off, r['T'], yerr=r['tot'], color=DCOL[D], capsize=0, lw=1)
        ax.errorbar(D + off, r['T'], yerr=r['stat'], color=DCOL[D], capsize=3, lw=2, marker=DMRK[D], mfc=DCOL[D] if filled else 'white', ms=6,
                    label=lab if i == 0 else None)
r = H.get('0_64')
if r: ax.errorbar(-0.012, r['T'], yerr=r['tot'], color=MUTED, marker='o', mfc='white', ms=5, lw=1, label='D=0, L=64')
ax.plot(list(prelim), list(prelim.values()), 'x', color=INK, ms=6, label='prelim (dU/dT, 1 seed)')
ax.set_xlabel('D  [J]'); ax.set_ylabel('T*  [J/k_B]  (peak of dU/dT)'); ax.legend(fontsize=7.5, loc='upper left')
ax.set_title('Specific-heat peak vs D (inner bar: stat, outer: total)', fontsize=9.5)
fig.savefig('figures/fig_t1_Tstar.png')
