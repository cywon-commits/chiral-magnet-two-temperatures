"""Task 3 training data (CPC 2024 temperature grid), one independent cooling chain per process.

Grid (descending): 1.5, 1.4, ..., 1.0 ; 0.9, 0.875, ..., 0.5 ; 0.4, 0.3, 0.2, 0.1 ; and T = 0.0 (= the T=0.1 final state relaxed
by ZERO_T_SWEEPS sweeps of local-field alignment, i.e. a quench, as the T -> 0 limit).  28 temperatures.
Per T: NEQ equilibration sweeps, then NPER samples every GAP sweeps (float16).  Several chains (seeds) are mixed in the dataset
so that low-T samples are not confined to one topological sector (review V-5).
Retention (CLAUDE.md): samples, final configuration per T (float32), per-sweep bond sums by direction (E4).
Output: data/raw/t3_<D>_L<L>_c<chain>.npz
"""
import numpy as np, sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from kernels import sweep, bond_sums_dir, zero_T_relax, seed as nseed, random_state
p = argparse.ArgumentParser(); p.add_argument('D', type=float); p.add_argument('L', type=int); p.add_argument('chain', type=int)
p.add_argument('--neq', type=int, default=1000); p.add_argument('--nper', type=int, default=125); p.add_argument('--gap', type=int, default=40)
p.add_argument('--zero_t_sweeps', type=int, default=500); p.add_argument('--out', default='data/raw')
a = p.parse_args(); J = 1.0; D, L = a.D, a.L
grid = list(np.round(np.arange(1.5, 0.95, -0.1), 3)) + list(np.round(np.arange(0.9, 0.4999, -0.025), 3)) + [0.4, 0.3, 0.2, 0.1]
fn = '%s/t3_D%03d_L%d_c%d.npz' % (a.out, round(D * 100), L, a.chain); os.makedirs(a.out, exist_ok=True)
nseed(7000 + 97 * a.chain + round(D * 1000)); S = random_state(L, np.random.default_rng(7000 + 97 * a.chain + round(D * 1000)))
sig = lambda T: min(1.5, 1.2 * np.sqrt(T)); t0 = time.time()
Ts, samples, lasts, E4s = [], [], [], []
for _ in range(a.neq): sweep(S, 1.7, J, D, sig(1.7))
for T in grid:
    n = a.neq + a.nper * a.gap; E4 = np.empty((n, 4)); smp = np.empty((a.nper, L, L, 3), np.float16)
    for k in range(n):
        sweep(S, T, J, D, sig(T)); E4[k] = bond_sums_dir(S)
        if k >= a.neq and (k - a.neq + 1) % a.gap == 0: smp[(k - a.neq + 1) // a.gap - 1] = S
    Ts.append(T); samples.append(smp); lasts.append(S.astype(np.float32)); E4s.append(E4)
    print('D=%.2f L=%d c=%d T=%.3f e=%.4f t=%.0fs' % (D, L, a.chain, T, (-(E4[a.neq:, 0] + E4[a.neq:, 1]) - D * (E4[a.neq:, 2] + E4[a.neq:, 3])).mean(), time.time() - t0), flush=True)
# T = 0: quench each T=0.1 sample (and the final state) by local-field alignment
S0 = samples[-1].astype(np.float64); S0 /= np.linalg.norm(S0, axis=-1, keepdims=True); smp0 = np.empty_like(samples[-1])
for i in range(len(S0)):
    s = S0[i].copy(); zero_T_relax(s, J, D, a.zero_t_sweeps); smp0[i] = s
Ts.append(0.0); samples.append(smp0); lasts.append(smp0[-1].astype(np.float32)); E4s.append(np.full_like(E4s[-1], np.nan))
np.savez(fn, D=D, L=L, chain=a.chain, T=np.array(Ts), samples=np.array(samples), last=np.array(lasts), E4=np.array(E4s),
         neq=a.neq, nper=a.nper, gap=a.gap, zero_t_sweeps=a.zero_t_sweeps)
print('done', fn, '%.0fs' % (time.time() - t0))
