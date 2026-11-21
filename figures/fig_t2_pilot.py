"""Task 2 pilot (1 seed, L/lambda~4, cooling 1.30->0.40): (a) curvature of S(k) at k=0, c1/r from the Landau-Brazovskii fit
(window 1.5 q), zero = disorder line T_L; (b) characteristic temperatures vs D: T_L (windows 1.2/1.5/2.0 as error bar),
T(q xi_q = 1), T* (task 1, dU/dT, L/lambda~8); (c) bond chiral correlation <C>/D vs T. Run from repo root."""
import sys, json; sys.path.insert(0, 'figures')
from _style import *
R = {r['D']: r for r in json.load(open('data/task2_pilot_summary.json'))}
H = json.load(open('data/task1_h1.json'))['rows']
DC = dict(DCOL); DC[0.1] = '#7a5af5'; DC[0.15] = '#008300'; DM = dict(DMRK); DM[0.1] = 'P'; DM[0.15] = 'X'
Ds = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 3.9))
for D in Ds:
    r = R[D]; T = np.array(r['T']); y = np.array(r['c1_w1.5'])
    a.plot(T, np.sign(y) * np.log10(1 + np.abs(y)), marker=DM[D], color=DC[D], ms=4, lw=1.2, label='D=%g' % D)
a.axhline(0, color=MUTED, lw=0.8); a.set_xlabel('T  [J/k_B]'); a.set_ylabel('sign(c₁/r)·log₁₀(1+|c₁/r|)'); a.legend(fontsize=7, ncol=2)
a.set_title('(a) curvature of S(k) at k=0 (>0: max at k=0)', fontsize=9)
for D in Ds:
    r = R[D]; tl = {w: min(r['TL_w%s' % w]) if r['TL_w%s' % w] else np.nan for w in ('1.2', '1.5', '2.0')}
    lo, hi = min(tl.values()), max(tl.values())
    b.errorbar(D, tl['1.5'], yerr=[[tl['1.5'] - lo], [hi - tl['1.5']]], marker=DM[D], color=DC[D], ms=7, capsize=3, label='T_L' if D == 0.1 else None)
    t1 = {w: (min(r['Txi1_w%s' % w]) if r['Txi1_w%s' % w] else np.nan) for w in ('1.2', '1.5', '2.0')}
    b.errorbar(D, t1['1.5'], yerr=[[t1['1.5'] - min(t1.values())], [max(t1.values()) - t1['1.5']]], marker=DM[D], mfc='white', color=DC[D], ms=7, capsize=3,
               label='T(q ξ_q = 1)' if D == 0.1 else None)
big = {0.1: 256, 0.15: 168, 0.2: 256, 0.3: 176, 0.4: 128, 0.5: 112}
b.errorbar(list(big), [H['%g_%d' % kv]['T'] for kv in big.items()], yerr=[H['%g_%d' % kv]['tot'] for kv in big.items()], color=INK, marker='_', ms=14, ls='--', lw=1, label='T* (task 1)')
b.axhline(H['0_256']['T'], color=MUTED, lw=0.8, ls=':'); b.text(0.42, H['0_256']['T'] - 0.02, 'T*(D=0)', fontsize=7, color=MUTED)
b.set_xlabel('D  [J]'); b.set_ylabel('T  [J/k_B]'); b.legend(fontsize=7.5); b.set_title('(b) characteristic temperatures (bar: fit window 1.2–2.0 q)', fontsize=9)
for D in Ds:
    r = R[D]; c.plot(r['T'], np.array(r['C']) / D, marker=DM[D], color=DC[D], ms=4, lw=1.2, label='D=%g' % D)
c.set_xlabel('T  [J/k_B]'); c.set_ylabel('⟨C⟩/D,  ⟨C⟩ = ⟨r̂_ij·(S_i×S_j)⟩ per bond'); c.legend(fontsize=7, ncol=2)
c.set_title('(c) bond chiral correlation (stat. err < 3e-4)', fontsize=9)
fig.savefig('figures/fig_t2_pilot.png')
