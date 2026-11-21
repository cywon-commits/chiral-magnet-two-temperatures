"""Task 2 runs: wide temperature range, structure factor and bond-resolved energies.

Protocol: random start -> cool from T_hi to T_lo in steps dT (measure at every step) [-> optional heating back].
Per T: NEQ equilibration + NMEAS measurement sweeps.  Per sweep (incl. equilibration): x = sum_bonds S_i.S_j and
c = sum_bonds r_ij.(S_i x S_j) per spin (2 bonds per spin), so e = -J x - D c, exchange per bond <S.S> = x/2 and bond
chiral correlation <C> = c/2 (review V-3).  Every 10 measurement sweeps: S(k) = sum_a |F_a(k)|^2 / N, accumulated as
(i) radial profile, bins 0.25 (units 2pi/L), (ii) unaveraged crop |kx|,|ky| <= kc (fftshifted), kc = max(8, ceil(2.5 nq)),
(iii) ring-weighted <cos4phi>, |<e^{2i phi}>| (ring 0.6..1.5 nq), nq = L sqrt2 arctan(D/sqrt2)/2pi.
Checkpoint after every temperature (resume reseeds numba RNG from (seed, stage)).
Output data/raw/t2_<tag>_D<D>_L<L>_s<seed>.npz (not committed).
"""
import numpy as np, sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from kernels import sweep, bond_sums, seed as nseed, random_state

p = argparse.ArgumentParser()
p.add_argument('tag'); p.add_argument('D', type=float); p.add_argument('L', type=int); p.add_argument('seed', type=int)
p.add_argument('--thi', type=float, default=1.2); p.add_argument('--tlo', type=float, default=0.4); p.add_argument('--dt', type=float, default=0.04)
p.add_argument('--neq', type=int, default=1000); p.add_argument('--nmeas', type=int, default=4000)
p.add_argument('--heat', action='store_true'); p.add_argument('--out', default='data/raw')
a = p.parse_args(); J = 1.0; D, L = a.D, a.L
os.makedirs(a.out, exist_ok=True)
fn = '%s/t2_%s_D%03d_L%d_s%d.npz' % (a.out, a.tag, round(D * 100), L, a.seed); ck = fn.replace('.npz', '.ckpt.npz')
kx = np.fft.fftfreq(L) * L; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KR = np.sqrt(KX ** 2 + KY ** 2); PHI = np.arctan2(KY, KX)
kbin = np.floor(KR / 0.25).astype(int).ravel(); nbin = int(L / 2 / 0.25); cnt = np.bincount(kbin, minlength=kbin.max() + 1)[:nbin]
nq = L * np.sqrt(2) * np.arctan(D / np.sqrt(2)) / (2 * np.pi) if D > 0 else 0.0
kc = int(min(L // 2 - 1, max(8, np.ceil(2.5 * nq)))); cidx = np.r_[0:kc + 1, L - kc:L]
ring = (KR > 0.6 * nq) & (KR < 1.5 * nq) if D > 0 else np.zeros_like(KR, bool)
grid = np.round(np.arange(a.thi, a.tlo - 1e-9, -a.dt), 4)
stages = [(T, 'cool') for T in grid] + ([(T, 'heat') for T in grid[::-1][1:]] if a.heat else [])
sig = lambda T: min(1.5, 1.2 * np.sqrt(T))
keys = ('T', 'dirn', 'X', 'Cb', 'Sk_rad', 'Sk_low', 'cos4', 'e2', 'acc')
rec = {k: [] for k in keys}; t0 = time.time()

def save():
    np.savez_compressed(fn, D=D, L=L, seed=a.seed, tag=a.tag, neq=a.neq, nmeas=a.nmeas, kc=kc, ring_nq=nq, kbin_width=0.25,
                        **{k: np.array(rec[k]) for k in keys})

def measure(S, T, dirn):
    s = sig(T); n = a.neq + a.nmeas; X = np.empty(n); Cb = np.empty(n); acc = 0.0; Sk = np.zeros((L, L)); m = 0
    for k in range(n):
        acc += sweep(S, T, J, D, s); X[k], Cb[k] = bond_sums(S)
        if k >= a.neq and (k - a.neq) % 10 == 0:
            F = np.fft.fft2(S, axes=(0, 1)); Sk += (np.abs(F) ** 2).sum(-1) / (L * L); m += 1
    Sk /= m
    rec['T'].append(T); rec['dirn'].append(dirn); rec['X'].append(X); rec['Cb'].append(Cb); rec['acc'].append(acc / n)
    rec['Sk_rad'].append(np.bincount(kbin, Sk.ravel(), minlength=len(cnt))[:nbin] / np.maximum(cnt, 1))
    rec['Sk_low'].append(np.fft.fftshift(Sk[np.ix_(cidx, cidx)]).astype(np.float32))
    if ring.any():
        w = Sk[ring]; ph = PHI[ring]
        rec['cos4'].append((w * np.cos(4 * ph)).sum() / w.sum()); rec['e2'].append(np.hypot((w * np.cos(2 * ph)).sum(), (w * np.sin(2 * ph)).sum()) / w.sum())
    else:
        rec['cos4'].append(np.nan); rec['e2'].append(np.nan)
    e = -J * X[a.neq:] - D * Cb[a.neq:]
    print('%s D=%.2f L=%d s=%d %s T=%.3f U=%.5f C=%.3f <C>=%.4f  t=%.0fs' % (a.tag, D, L, a.seed, dirn, T, e.mean(), L * L * e.var() / T ** 2, Cb[a.neq:].mean() / 2, time.time() - t0), flush=True)
    save()

start = 0; nseed(1000 * a.seed + 29); S = random_state(L, np.random.default_rng(1000 * a.seed + 29))
if os.path.exists(ck) and os.path.exists(fn):
    c = np.load(ck); start = int(c['next_stage']); S = c['S'].copy(); d = np.load(fn)
    for k in keys: rec[k] = list(d[k][:start])
    nseed(1000 * a.seed + 29 + 7919 * start); print('resume from stage %d/%d' % (start, len(stages)), flush=True)
else:
    for _ in range(a.neq): sweep(S, a.thi + 0.2, J, D, sig(a.thi + 0.2))   # pre-anneal above the grid
for k in range(start, len(stages)):
    measure(S, *stages[k]); np.savez(ck, next_stage=k + 1, S=S)
os.remove(ck); print('done', fn, '%.0fs' % (time.time() - t0))
