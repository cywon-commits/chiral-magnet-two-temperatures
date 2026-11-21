"""Task 1 production run: one (D, L, seed) per process.

Protocols
  anneal : random start -> cool 1.00..0.85 (step 0.05) -> measure 0.80..0.60 (step 0.01, cooling)
           -> cool 0.575..0.50 (no measurement) -> heat 0.55 -> measure 0.60..0.80 (step 0.01, heating)
  quench : random start -> T_q=0.1 for NQ sweeps (labyrinth) -> for each T in grid, start from the SAME
           quenched configuration and heat directly to T (CPC 2024 style).
Per measured T: NEQ sweeps equilibration + NMEAS sweeps measurement; energy per spin stored at EVERY
sweep (including equilibration) so that drift and hysteresis can be checked afterwards.
S(k) (sum over spin components, |F|^2/N) accumulated every 10 measurement sweeps; stored as a radial
profile in bins of width 0.25 (in units of 2pi/L) plus S(0); also (review V-1 #4) the unaveraged S(k) on
|kx|,|ky|<=kc (fftshifted, index kc = k=0) and, for D>0, ring-weighted <cos4phi> and |<e^{2i phi}>|
(ring 0.6..1.5 x nq, nq = L sqrt2 arctan(D/sqrt2)/2pi, as in prelim/aniso.py).
Output: data/raw/t1_<proto>_D<D>_L<L>_s<seed>.npz (not committed).
"""
import numpy as np, sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from kernels import sweep, energy, seed as nseed, random_state

p = argparse.ArgumentParser()
p.add_argument('proto'); p.add_argument('D', type=float); p.add_argument('L', type=int); p.add_argument('seed', type=int)
p.add_argument('--neq', type=int, default=1000); p.add_argument('--nmeas', type=int, default=6000)
p.add_argument('--tq', type=float, default=0.1); p.add_argument('--nq', type=int, default=2000)
p.add_argument('--out', default='data/raw')
a = p.parse_args()
J = 1.0; D, L = a.D, a.L
os.makedirs(a.out, exist_ok=True)
fn = '%s/t1_%s_D%03d_L%d_s%d.npz' % (a.out, a.proto, round(D * 100), L, a.seed)
nseed(1000 * a.seed + 17); rng = np.random.default_rng(1000 * a.seed + 17)
kx = np.fft.fftfreq(L) * L; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KR = np.sqrt(KX ** 2 + KY ** 2)
kbin = np.floor(KR / 0.25).astype(int).ravel(); nbin = int(L / 2 / 0.25)
cnt = np.bincount(kbin, minlength=kbin.max() + 1)[:nbin]
# [V-1 #4] unaveraged low-k S(k): crop |kx|,|ky| <= kc (kc >= 6, and >= 2 x ring radius for D>0)
PHI = np.arctan2(KY, KX)
nq = L * np.sqrt(2) * np.arctan(D / np.sqrt(2)) / (2 * np.pi) if D > 0 else 0.0   # ring radius (diagonal helix), units 2pi/L
kc = int(max(6, np.ceil(2 * nq)))
cidx = np.r_[0:kc + 1, L - kc:L]
ring = (KR > 0.6 * nq) & (KR < 1.5 * nq) if D > 0 else np.zeros_like(KR, bool)
grid = np.round(np.arange(0.80, 0.60 - 1e-9, -0.01), 3)
sig = lambda T: min(1.5, 1.2 * np.sqrt(T))
rec = dict(T=[], dirn=[], E=[], Sk_rad=[], S0=[], acc=[], Sk_low=[], cos4=[], e2=[])
t0 = time.time()

def measure(S, T, dirn):
    s = sig(T); E = np.empty(a.neq + a.nmeas, np.float64); acc = 0.0
    Sk = np.zeros((L, L)); n = 0
    for k in range(a.neq + a.nmeas):
        acc += sweep(S, T, J, D, s); E[k] = energy(S, J, D)
        if k >= a.neq and (k - a.neq) % 10 == 0:
            F = np.fft.fft2(S, axes=(0, 1)); Sk += (np.abs(F) ** 2).sum(-1) / (L * L); n += 1
    Sk /= n
    rec['T'].append(T); rec['dirn'].append(dirn); rec['E'].append(E); rec['acc'].append(acc / len(E))
    rec['Sk_rad'].append((np.bincount(kbin, Sk.ravel(), minlength=len(cnt))[:nbin] / np.maximum(cnt, 1)))
    rec['S0'].append(Sk[0, 0])
    rec['Sk_low'].append(np.fft.fftshift(Sk[np.ix_(cidx, cidx)]).astype(np.float32))
    if ring.any():
        w = Sk[ring]; ph = PHI[ring]
        rec['cos4'].append((w * np.cos(4 * ph)).sum() / w.sum())
        rec['e2'].append(np.hypot((w * np.cos(2 * ph)).sum(), (w * np.sin(2 * ph)).sum()) / w.sum())
    else:
        rec['cos4'].append(np.nan); rec['e2'].append(np.nan)
    m = E[a.neq:]
    print('%s D=%.2f L=%d s=%d %s T=%.3f U=%.5f C=%.3f  t=%.0fs' % (a.proto, D, L, a.seed, dirn, T, m.mean(), L * L * m.var() / T ** 2, time.time() - t0), flush=True)
    save()

def save():
    np.savez_compressed(fn, D=D, L=L, seed=a.seed, proto=a.proto, neq=a.neq, nmeas=a.nmeas,
                        T=np.array(rec['T']), dirn=np.array(rec['dirn']), E=np.array(rec['E'], np.float64),
                        Sk_rad=np.array(rec['Sk_rad']), S0=np.array(rec['S0']), acc=np.array(rec['acc']),
                        kbin_width=0.25, Sk_low=np.array(rec['Sk_low']), kc=kc, ring_nq=nq,
                        cos4=np.array(rec['cos4']), e2=np.array(rec['e2']))

S = random_state(L, rng)
if a.proto == 'anneal':
    for T in (1.0, 0.95, 0.9, 0.85):
        for _ in range(a.neq): sweep(S, T, J, D, sig(T))
    for T in grid: measure(S, T, 'cool')
    for T in (0.575, 0.55, 0.525, 0.50, 0.525, 0.55, 0.575):
        for _ in range(a.neq): sweep(S, T, J, D, sig(T))
    for T in grid[::-1]: measure(S, T, 'heat')
elif a.proto == 'quench':
    for _ in range(a.nq): sweep(S, a.tq, J, D, sig(a.tq))
    Sq = S.copy()
    for T in grid[::-1]:
        S = Sq.copy(); measure(S, T, 'quench')
print('done', fn, '%.0fs' % (time.time() - t0))
