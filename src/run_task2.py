"""Task 2 runs (v2, data-retention design of review V-4 / CLAUDE.md "데이터 보존 원칙").

Protocol: random start -> pre-anneal at T_hi+0.2 (NEQ sweeps) -> cool T_hi -> T_lo in steps dT; per T: NEQ equilibration +
NMEAS measurement sweeps.  One sweep = 1 Metropolis + 2 over-relaxation.

Per temperature a file <run_dir>/T<idx>_<T>.npz is written (atomically), containing
  (4) per-sweep bond sums by direction, incl. equilibration: xx, xy (S_i.S_{i+x}, S_i.S_{i+y}), cx, cy (x.(S_i x S_{i+x}),
      y.(S_i x S_{i+y})), per spin, float64.  e = -J(xx+xy) - D(cx+cy).
  (3) full spin-component structure-factor tensor S_ab(k) = <F_a(k) F_b(k)*>/N on the half plane (rfft layout: kx = 0..L-1,
      ky = 0..L/2), averaged every 10 measurement sweeps: Sdiag (3, L, L/2+1) float32 (aa = xx, yy, zz),
      Soff (3, L, L/2+1) complex64 (ab = xy, xz, yz).  Im S_ab is the antisymmetric (chiral) part.
  (1) final configuration after the measurement, float32 (L, L, 3)  -> restart point for any later extension.
  (2) NSAMP configuration samples, float16 (NSAMP, L, L, 3), equally spaced in the measurement (every NMEAS/NSAMP sweeps).
  (5) local time-averaged moments over consecutive windows of WIN sweeps (NMEAS/WIN windows): float16 (nwin, L, L, 8):
      m = <S> (3), traceless Q = <S S> - 1/3: (Qxx, Qyy, Qxy, Qxz, Qyz) (Qzz = -Qxx-Qyy).  Longer windows (e.g. 2000) are
      exact averages of consecutive WIN windows.
  plus derived summaries (radial S(k), low-k crop, ring cos4/e2, acceptance) for the existing analysis.
A small summary file <run_dir>.npz (same keys as v1: T, dirn, X, Cb, Sk_rad, Sk_low, cos4, e2, acc) is rebuilt after each T.
Restart: completed T files are kept; the run resumes from the final configuration of the last completed T
(numba RNG reseeded from (seed, index)).
"""
import numpy as np, sys, os, time, argparse, glob
sys.path.insert(0, os.path.dirname(__file__))
from kernels import sweep, bond_sums_dir, accum_moments, seed as nseed, random_state

p = argparse.ArgumentParser()
p.add_argument('tag'); p.add_argument('D', type=float); p.add_argument('L', type=int); p.add_argument('seed', type=int)
p.add_argument('--thi', type=float, default=1.3); p.add_argument('--tlo', type=float, default=0.31); p.add_argument('--dt', type=float, default=0.03)
p.add_argument('--neq', type=int, default=1000); p.add_argument('--nmeas', type=int, default=4000)
p.add_argument('--nsamp', type=int, default=20); p.add_argument('--win', type=int, default=500)
p.add_argument('--out', default='data/raw')
a = p.parse_args(); J = 1.0; D, L = a.D, a.L
base = '%s/t2_%s_D%03d_L%d_s%d' % (a.out, a.tag, round(D * 100), L, a.seed); os.makedirs(base, exist_ok=True)
kx = np.fft.fftfreq(L) * L; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KR = np.sqrt(KX ** 2 + KY ** 2); PHI = np.arctan2(KY, KX)
kbin = np.floor(KR / 0.25).astype(int).ravel(); nbin = int(L / 2 / 0.25); cnt = np.bincount(kbin, minlength=kbin.max() + 1)[:nbin]
nq = L * np.sqrt(2) * np.arctan(D / np.sqrt(2)) / (2 * np.pi) if D > 0 else 0.0
kc = int(min(L // 2 - 1, max(8, np.ceil(2.5 * nq)))); cidx = np.r_[0:kc + 1, L - kc:L]
ring = (KR > 0.6 * nq) & (KR < 1.5 * nq) if D > 0 else np.zeros_like(KR, bool)
grid = np.round(np.arange(a.thi, a.tlo - 1e-9, -a.dt), 4)
sig = lambda T: min(1.5, 1.2 * np.sqrt(T))
H = L // 2 + 1; jj = np.arange(H, L); mkx = (-np.arange(L)) % L

def full_from_half(h):
    f = np.empty((L, L)); f[:, :H] = h; f[:, H:] = h[mkx][:, L - jj]; return f

def measure(S, T, idx):
    s = sig(T); n = a.neq + a.nmeas; E4 = np.empty((n, 4)); acc = 0.0
    Sd = np.zeros((3, L, H)); So = np.zeros((3, L, H), complex); m = 0
    nwin = a.nmeas // a.win; mom = np.zeros((nwin, L, L, 8), np.float16); macc = np.zeros((L, L, 9))
    samp = np.zeros((a.nsamp, L, L, 3), np.float16); every = a.nmeas // a.nsamp
    for k in range(n):
        acc += sweep(S, T, J, D, s); E4[k] = bond_sums_dir(S)
        if k < a.neq: continue
        km = k - a.neq
        accum_moments(S, macc)
        if (km + 1) % a.win == 0:
            w = km // a.win; mm = macc / a.win
            mom[w, ..., 0:3] = mm[..., 0:3]; mom[w, ..., 3] = mm[..., 3] - 1 / 3; mom[w, ..., 4] = mm[..., 4] - 1 / 3
            mom[w, ..., 5] = mm[..., 6]; mom[w, ..., 6] = mm[..., 7]; mom[w, ..., 7] = mm[..., 8]; macc[:] = 0
        if (km + 1) % every == 0: samp[(km + 1) // every - 1] = S
        if km % 10 == 0:
            F = np.fft.rfft2(S, axes=(0, 1))
            for c in range(3): Sd[c] += np.abs(F[..., c]) ** 2
            So[0] += F[..., 0] * np.conj(F[..., 1]); So[1] += F[..., 0] * np.conj(F[..., 2]); So[2] += F[..., 1] * np.conj(F[..., 2]); m += 1
    Sd /= m * L * L; So /= m * L * L
    Sk = full_from_half(Sd.sum(0))
    rad = np.bincount(kbin, Sk.ravel(), minlength=len(cnt))[:nbin] / np.maximum(cnt, 1)
    low = np.fft.fftshift(Sk[np.ix_(cidx, cidx)]).astype(np.float32)
    if ring.any():
        w = Sk[ring]; ph = PHI[ring]; c4 = (w * np.cos(4 * ph)).sum() / w.sum(); e2 = np.hypot((w * np.cos(2 * ph)).sum(), (w * np.sin(2 * ph)).sum()) / w.sum()
    else: c4 = e2 = np.nan
    fn = '%s/T%02d_%.4f.npz' % (base, idx, T); tmp = fn + '.tmp.npz'
    np.savez(tmp, T=T, idx=idx, D=D, L=L, seed=a.seed, neq=a.neq, nmeas=a.nmeas, win=a.win, E4=E4, Sdiag=Sd.astype(np.float32),
             Soff=So.astype(np.complex64), last=S.astype(np.float32), samples=samp, moments=mom, Sk_rad=rad, Sk_low=low, cos4=c4, e2=e2,
             acc=acc / n, kc=kc, ring_nq=nq)
    os.replace(tmp, fn)
    e = -J * (E4[a.neq:, 0] + E4[a.neq:, 1]) - D * (E4[a.neq:, 2] + E4[a.neq:, 3])
    print('%s D=%.2f L=%d s=%d T=%.3f U=%.5f C=%.3f <C>=%.4f  t=%.0fs' % (a.tag, D, L, a.seed, T, e.mean(), L * L * e.var() / T ** 2,
          (E4[a.neq:, 2] + E4[a.neq:, 3]).mean() / 2, time.time() - t0), flush=True)

def rebuild_summary():
    fs = sorted(glob.glob(base + '/T??_*.npz')); keys = ('T', 'X', 'Cb', 'Sk_rad', 'Sk_low', 'cos4', 'e2', 'acc'); r = {k: [] for k in keys}
    for f in fs:
        d = np.load(f)
        r['T'].append(float(d['T'])); r['X'].append(d['E4'][:, 0] + d['E4'][:, 1]); r['Cb'].append(d['E4'][:, 2] + d['E4'][:, 3])
        for k in ('Sk_rad', 'Sk_low', 'cos4', 'e2', 'acc'): r[k].append(d[k])
    np.savez_compressed(base + '.npz', D=D, L=L, seed=a.seed, tag=a.tag, neq=a.neq, nmeas=a.nmeas, kc=kc, ring_nq=nq, kbin_width=0.25,
                        dirn=np.array(['cool'] * len(fs)), **{k: np.array(v) for k, v in r.items()})

t0 = time.time()
done = sorted(glob.glob(base + '/T??_*.npz')); start = len(done)
if start:
    S = np.load(done[-1])['last'].astype(np.float64); S /= np.linalg.norm(S, axis=-1, keepdims=True)
    nseed(1000 * a.seed + 29 + 7919 * start); print('resume after %d temperatures' % start, flush=True)
else:
    nseed(1000 * a.seed + 29); S = random_state(L, np.random.default_rng(1000 * a.seed + 29))
    for _ in range(a.neq): sweep(S, a.thi + 0.2, J, D, sig(a.thi + 0.2))
for idx in range(start, len(grid)):
    measure(S, grid[idx], idx); rebuild_summary()
open(base + '/DONE', 'w').write('%d temperatures\n' % len(grid))
print('done', base, '%.0fs' % (time.time() - t0))
