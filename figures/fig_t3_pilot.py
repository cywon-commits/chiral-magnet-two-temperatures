"""Task 3 pilot: VAE (CPC architecture, channels 16-32-64-128, latent 100) trained on D=0.3, L=128 (250 samples per T, 80/20 split,
30 epochs). (a) eta(T) = <|S~|> on the test set (decoder(mu)), error bar = sample std; vertical lines: CPC T_c = 0.71, T* (task 1),
T_L (task 2). Baseline: local time-averaged eta_1 = <|<S_i>_W|> from task 2 (D=0.3, L=176, windows W=500, 2000 sweeps, block b=1).
(b) learning curves. Run from repo root."""
import sys, json, glob; sys.path.insert(0, 'figures')
from _style import *
r = json.load(open('data/task3/pilot_D030_ch16_n250_e30.json')); e = r['eta_test']; H = json.load(open('data/task2_h2.json'))
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.errorbar(e['T'], e['eta_mu'], yerr=e['eta_mu_std'], color=DCOL[0.0], marker='o', ms=4, lw=1.4, capsize=2, label='VAE η (pilot, 30 epochs)')
fs = sorted(glob.glob('data/task2_extract/t2_main_D030_L176_s*.npz')); E = np.mean([np.load(f)['eta1'] for f in fs], 0); T2 = np.load(fs[0])['T']
a.plot(T2, E[:, 0, 0], color=DCOL[0.2], ls='--', lw=1.3, label='η₁ local time average, W=500 sweeps')
a.plot(T2, E[:, 2, 0], color=DCOL[0.2], ls=':', lw=1.3, label='η₁ local time average, W=2000 sweeps')
for x, lab, col in ((0.71, 'CPC T_c=0.71', INK), (0.7083, None, INK), (H['per_D']['0.3']['T_L']['T'], 'T_L (task 2)', DCOL[0.3])):
    if lab: a.axvline(x, color=col, lw=0.9, ls='-.' if 'T_L' in lab else '-', label=lab)
a.axvspan(0.7083 - 0.0074, 0.7083 + 0.0074, color='#d9d7cf', alpha=0.7, label='T* = 0.708 (task 1)')
a.set_xlabel('T  [J/k_B]'); a.set_ylabel('η'); a.legend(fontsize=7); a.set_title('(a) D=0.3, L=128: VAE η vs local-time-average η₁', fontsize=9.5)
h = r['history']; ep = [x['epoch'] for x in h]
b.plot(ep, [x['train'][1] for x in h], color=DCOL[0.0], label='train reconstruction'); b.plot(ep, [x['test'][1] for x in h], color=DCOL[0.2], label='test reconstruction')
b.axhline(128 * 128, color=MUTED, lw=0.8, ls=':', label='predict zero (= N)'); b.set_xlabel('epoch'); b.set_ylabel('sum-MSE per sample'); b.legend(fontsize=7.5)
b.set_title('(b) learning curve (5600 train / 1400 test)', fontsize=9.5)
fig.savefig('figures/fig_t3_pilot.png')
