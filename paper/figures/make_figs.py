"""Figures 1-4 of the manuscript, from committed summary data only.  Run from the repository root:
    python3 paper/figures/make_figs.py
Output: paper/figures/fig{1,2,3,4}_*.pdf (vector, Type 42 fonts) and .png previews."""
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'in',
                     'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.0, 'savefig.dpi': 300})
W1 = 3.375                                     # single column (in)
BLUE, VERM, ORNG, GRAY, BLACK = '#0072B2', '#D55E00', '#E69F00', '#7a7a7a', 'k'   # Okabe-Ito
OUT = 'paper/figures/'
def save(fig, name):
    fig.savefig(OUT + name + '.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig(OUT + name + '.png', bbox_inches='tight', pad_inches=0.02); plt.close(fig)
def lab(ax, s, x=0.03, y=0.95, c='k'): ax.text(x, y, s, transform=ax.transAxes, ha='left', va='top', fontsize=8, color=c)
tab = np.genfromtxt(OUT + 'table1.csv', delimiter=',', names=True, skip_header=3)

# ---------------- Fig. 1: resolution of the two helical branches
fig = plt.figure(figsize=(W1, 3.0)); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.08)
ax = fig.add_subplot(gs[0, :]); k = np.linspace(-3.2, 3.2, 801)
for r, c, ls, name in ((2.3, GRAY, '--', r'$\kappa=2.3q$ (above $T_{\rm ring}$)'), (np.sqrt(3), BLACK, '-', r'$\kappa=\sqrt{3}\,q$ ($T_{\rm ring}$)'), (1.0, BLUE, '-', r'$\kappa=q$ (below $T_{\rm ring}$)')):
    s = 1 / ((k - 1) ** 2 + r * r) + 1 / ((k + 1) ** 2 + r * r); ax.plot(k, s / s.max(), color=c, ls=ls, label=name)
for x in (-1, 1): ax.axvline(x, color=GRAY, lw=0.5, ls=':')
ax.set_xlim(-3.2, 3.2); ax.set_ylim(0, 1.08); ax.set_xlabel(r'$k/q$', labelpad=1); ax.set_ylabel(r'$S_\perp(k)$ (norm.)')
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax.legend(loc='lower center', frameon=False, ncol=1, handlelength=1.8, borderaxespad=0.1)
lab(ax, '(a)')
fs = sorted(glob.glob('data/task2_extract/t2_main_D030_L176_s*.npz')); d0 = np.load(fs[0]); T = d0['T']; L = int(d0['L']); n = int(d0['kc_low'])
q = np.arctan(0.3); kk = 2 * np.pi * np.arange(-n, n + 1) / L / q
S = np.mean([np.load(f)['low_tr'] for f in fs], 0)
for j, (Tt, tag) in enumerate(((0.94, '(b)'), (0.76, '(c)'))):
    i = int(np.argmin(abs(T - Tt))); a = fig.add_subplot(gs[1, j]); M = S[i] / S[i].max()
    a.imshow(M.T, origin='lower', extent=[kk[0], kk[-1], kk[0], kk[-1]], cmap='cividis', vmin=0, vmax=1, interpolation='nearest')
    th = np.linspace(0, 2 * np.pi, 200); a.plot(np.cos(th), np.sin(th), color='w', lw=0.5, ls=':')
    a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_xlabel(r'$k_x/q$', labelpad=1); a.set_xticks([-2, 0, 2]); a.set_yticks([-2, 0, 2])
    if j == 0: a.set_ylabel(r'$k_y/q$', labelpad=0)
    else: a.set_yticklabels([])
    a.text(0.04, 0.95, '%s $T=%.2fJ$' % (tag, T[i]), transform=a.transAxes, ha='left', va='top', color='w', fontsize=7.5)
    a.tick_params(colors='w', labelcolor='k')
    print('Fig1: panel %s uses T = %.3f (T_ring = 0.854)' % (tag, T[i]))
save(fig, 'fig1_ring_onset')

# ---------------- Fig. 2: T* and T_ring versus D
fig, ax = plt.subplots(figsize=(W1, 2.35))
Dc = 0.105; ax.axvspan(0, Dc, color='0.92', lw=0)
ax.errorbar(tab['D'], tab['Tstar'], tab['Tstar_err'], fmt='o', ms=3.6, color=BLACK, mfc='w', mew=0.9, capsize=1.5, elinewidth=0.7, label=r'$T^*$ (specific-heat maximum)')
m = np.isfinite(tab['Tring'])
ax.errorbar(tab['D'][m], tab['Tring'][m], tab['Tring_err'][m], fmt='s', ms=3.4, color=BLUE, capsize=1.5, elinewidth=0.7, label=r'$T_{\rm ring}$ (ring onset)')
ax.set_xlim(-0.01, 0.53); ax.set_ylim(0.58, 1.12); ax.set_xlabel(r'$D/J$'); ax.set_ylabel(r'$T/J$')
ax.text(0.052, 1.085, 'inverted\norder', ha='center', va='top', fontsize=7, color='0.35')
ax.legend(loc='upper left', bbox_to_anchor=(0.2, 0.99), frameon=False, handletextpad=0.4)
save(fig, 'fig2_two_temperatures')

# ---------------- Fig. 3: alpha = xi_H(T_ring)/lambda
kw = json.load(open('review/data/t6_kappa_window.json')); key = ['005_1024', '008_384', '010_512', '015_336', '020_256', '030_176', '040_128', '050_112']
De = np.array([kw[k]['D'] for k in key]); ae = np.array([kw[k]['kappa_ex'] for k in key])
j3 = json.load(open('data/task6_judge.json'))['judge']['6-3']; Dw, aw, sw = map(np.array, (j3['D'], j3['kappa'], j3['err']))
ln = json.load(open('review/data/largeN_tri.json')); Dl = np.array([r['D'] for r in ln['sq']]); al = np.array([r['kappa'] for r in ln['sq']]); alt = np.array([r['kappa'] for r in ln['tri']])
tri = json.load(open('review/data/tri_analysis.json')); Dt, at, et = [], [], []
for D in (0.3, 0.5):
    v = [tri['tri_D%.1f' % D][w]['kappa'] for w in ('w1.0', 'w1.2', 'w1.5')]; Dt.append(D); at.append(np.mean(v)); et.append((max(v) - min(v)) / 2)
fig, ax = plt.subplots(figsize=(W1, 2.35))
ax.axhline(np.sqrt(3 / 8) / (2 * np.pi), color=GRAY, ls='--', lw=0.8); ax.text(0.02, 0.0985, 'Gaussian', fontsize=7, color=GRAY, va='bottom')
ax.plot(Dl, al, '-', color=BLACK, label=r'large $N$, square'); ax.plot(Dl, alt, ':', color=BLACK, lw=1.2, label=r'large $N$, triangular')
ax.errorbar(Dw, aw, sw, fmt='o', ms=3.4, color=GRAY, mfc='w', mew=0.8, capsize=1.3, elinewidth=0.6, label='MC square, window $1.5q$')
ax.plot(De, ae, 'o', ms=3.8, color=BLUE, label='MC square, window $\\to 0$')
ax.errorbar(np.array(Dt) + 0.008, at, et, fmt='D', ms=3.4, color=VERM, capsize=1.3, elinewidth=0.6, label='MC triangular')
ax.set_xlim(0, 0.72); ax.set_ylim(0.085, 0.195); ax.set_xlabel(r'$D/J$'); ax.set_ylabel(r'$\alpha=\xi_H(T_{\rm ring})/\lambda$')
ax.legend(loc='upper right', frameon=False, handlelength=1.6, handletextpad=0.4, labelspacing=0.25)
save(fig, 'fig3_alpha')
print('Fig3: alpha ext', np.round(ae, 3), ' window', np.round(aw, 3), ' tri', np.round(at, 3), '+-', np.round(et, 3))

# ---------------- Fig. 4: VAE order parameter
lv = json.load(open('review/data/linvae_eta.json')); pv = json.load(open('data/task3/pilot_D030_ch16_n250_e30.json'))['eta_test']
Tp, ep, sp = map(np.array, (pv['T'], pv['eta_mu'], pv['eta_mu_std']))
c = lv['D0.3_nz100_th0.5']; lt, le = np.array(c['T']), np.array(c['eta']); s = (Tp >= 0.2) & (Tp <= 0.7)
dev = float(np.mean(np.abs(np.interp(Tp[s], lt[::-1], le[::-1]) - ep[s])))
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.85), sharey=True); plt.subplots_adjust(wspace=0.08)
for a in axs:
    a.axvline(0.708, color=BLACK, lw=0.6, ls='--'); a.axvline(0.854, color=BLUE, lw=0.6, ls='--')
    a.set_xlim(0, 1.5); a.set_ylim(0, 1.02); a.set_xlabel(r'$T/J$', labelpad=1)
a = axs[0]
for D, col, ls in ((0.0, GRAY, '--'), (0.3, BLACK, '-'), (0.5, VERM, '-.')):
    c = lv['D%.1f_nz100_th0.5' % D]; a.plot(c['T'], c['eta'], ls=ls, color=col, label=r'$D=%.1fJ$' % D)
a.plot(Tp, ep, 'o', ms=2.6, color=BLUE, mfc='w', mew=0.7, label='trained, $D=0.3J$')
a.set_ylabel(r'$\eta$'); a.legend(loc='upper right', frameon=False, handlelength=1.5, labelspacing=0.2, borderaxespad=0.2); lab(a, '(a)', x=0.04, y=0.08)
a = axs[1]
for nz, col, ls in ((25, GRAY, '--'), (100, BLACK, '-'), (400, VERM, '-.')):
    c = lv['D0.3_nz%d_th0.5' % nz]; a.plot(c['T'], c['eta'], ls=ls, color=col, label=r'$n_z=%d$' % nz)
a.legend(loc='upper right', frameon=False, handlelength=1.5, labelspacing=0.2, borderaxespad=0.2); lab(a, '(b)', x=0.04, y=0.08)
axs[0].set_xticks([0, 0.5, 1.0]); axs[1].set_xticks([0.5, 1.0, 1.5])
axs[1].text(0.69, 0.02, r'$T^*$', fontsize=7, ha='right', va='bottom'); axs[1].text(0.90, 0.02, r'$T_{\rm ring}$', fontsize=7, color=BLUE, ha='left', va='bottom')
save(fig, 'fig4_vae')
print('Fig4: mean |eta_linear - eta_trained| for T = 0.2-0.7 (held-out set): %.4f' % dev)
print('Fig4: T_half n_z=25,100,400 at D=0.3:', [round(lv['D0.3_nz%d_th0.5' % z]['T_half'], 3) for z in (25, 100, 400)], ' D=0,0.3,0.5 at n_z=100:', [round(lv['D%.1f_nz100_th0.5' % D]['T_half'], 3) for D in (0, .3, .5)])
