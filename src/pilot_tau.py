"""Pilot: integrated autocorrelation time of the energy near T*, to fix sweep counts."""
import numpy as np, sys, time
sys.path.insert(0, 'src')
from kernels import sweep, energy, seed, random_state

def tau_int(x, c=6.0):
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]
    t = 0.5
    for w in range(1, n):
        t += acf[w]
        if w >= c * t: break
    return t

D = float(sys.argv[1]); L = int(sys.argv[2]); T = float(sys.argv[3]); n = int(sys.argv[4])
seed(5); S = random_state(L, np.random.default_rng(5)); sig = min(1.5, 1.2 * np.sqrt(T))
t0 = time.time()
for Tc in np.arange(1.0, T - 1e-9, -0.05):
    for _ in range(300): sweep(S, Tc, 1.0, D, sig)
for _ in range(1000): sweep(S, T, 1.0, D, sig)
e = np.empty(n)
for k in range(n): sweep(S, T, 1.0, D, sig); e[k] = energy(S, 1.0, D)
h = n // 2
print('D=%.2f L=%d T=%.2f tau_int(E)=%.1f (1st half %.1f, 2nd half %.1f)  U=%.5f  C=%.3f  %.0fs' % (D, L, T, tau_int(e), tau_int(e[:h]), tau_int(e[h:]), e.mean(), L * L * e.var() / T ** 2, time.time() - t0), flush=True)
