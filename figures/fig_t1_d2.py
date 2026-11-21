"""[V-2] Revised H1 check: Delta T*/T*(0) = T*(D)/T*(0) - 1 vs D^2, reference T*(0) = D=0, L=256.
Filled: L/lambda~8, open: L/lambda~4 (D=0.1: L=256, D=0.15: L=168 are L/lambda~4).
Inner error bar: dU/dT-only sigma (stat (+) window (+) cool/heat), outer: pre-registered total sigma; T*(0) error propagated.
Lines: three reference energy scales (a = 1/3, 1/2, 1/4), weighted fits a D^2 and b|D| (dU/dT-only sigma, L/lambda~8 + D=0.1,0.15).
Run from repo root."""
import sys, json; sys.path.insert(0, 'figures')
from _style import *
H = json.load(open('data/task1_h1.json'))['rows']; r0 = H['0_256']; T0 = r0['T']
big = [(0.2, 256), (0.3, 176), (0.4, 128), (0.5, 112)]; small = [(0.1, 256), (0.15, 168), (0.2, 128), (0.3, 88), (0.4, 64), (0.5, 56)]
DC = dict(DCOL); DC[0.1] = '#7a5af5'; DC[0.15] = '#008300'; DM = dict(DMRK); DM[0.1] = 'P'; DM[0.15] = 'X'
sdu = lambda r: np.sqrt(r['stat'] ** 2 + r['s_win'] ** 2 + r['s_hys'] ** 2)
fig, ax = plt.subplots(figsize=(6, 4.2))
x = np.linspace(0, 0.27, 200); Dx = np.sqrt(x)
for nm, f, ls in (('√(1+2D²/3)  (a=1/3)', lambda D: np.sqrt(1 + 2 * D * D / 3), (0, (6, 2))), ('√(1+D²)  (a=1/2)', lambda D: np.sqrt(1 + D * D), '--'),
                  ('(1+√(1+D²))/2  (a=1/4)', lambda D: (1 + np.sqrt(1 + D * D)) / 2, ':')):
    ax.plot(x, f(Dx) - 1, ls=ls, color='#a8a7a0', lw=1.0, label=nm)
# fits on (L/lambda~8) + D=0.1, 0.15 with dU/dT-only errors, T*(0) error ignored (variant i)
pts = big + [p for p in [(0.1, 256), (0.15, 168)] if '%g_%d' % p in H]
D = np.array([p[0] for p in pts]); y = np.array([H['%g_%d' % p]['T'] for p in pts]) / T0 - 1; s = np.array([sdu(H['%g_%d' % p]) for p in pts]) / T0
w = 1 / s ** 2; a = (w * D ** 2 * y).sum() / (w * D ** 4).sum(); b = (w * D * y).sum() / (w * D * D).sum()
ax.plot(x, a * x, color=INK, lw=1.4, label='fit a·D² (a=%.2f)' % a); ax.plot(x, b * Dx, color=INK, lw=1.4, ls='-.', label='fit b·|D| (b=%.3f)' % b)
for pts_, filled in ((big, True), (small, False)):
    for (Dv, L) in pts_:
        r = H.get('%g_%d' % (Dv, L))
        if r is None: continue
        yv = r['T'] / T0 - 1
        e_out = np.hypot(r['tot'], r['T'] / T0 * r0['tot']) / T0; e_in = np.hypot(sdu(r), r['T'] / T0 * sdu(r0)) / T0
        xo = Dv ** 2 + (0.003 if filled else -0.003)
        ax.errorbar(xo, yv, yerr=e_out, color=DC[Dv], lw=0.9, capsize=0)
        ax.errorbar(xo, yv, yerr=e_in, color=DC[Dv], lw=2, capsize=3, marker=DM[Dv], ms=6, mfc=DC[Dv] if filled else 'white',
                    label=('D=%g' % Dv) if (filled or Dv < 0.2) else None)
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xlabel('D²  [J²]'); ax.set_ylabel('T*(D)/T*(0) − 1'); ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.set_title('C-peak shift vs D² (filled L/λ≈8, open L/λ≈4; inner bar dU/dT-only, outer total)', fontsize=8.5)
fig.savefig('figures/fig_t1_d2.png')
