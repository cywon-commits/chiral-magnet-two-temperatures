"""Task 6b post-hoc checks (not part of the pre-registered verdicts).
 (a) tau_int of the energy per sweep (E4 -> e) for every temperature of every task-6 run (measurement part);
     n_meas / tau_int reported (criterion of [R-10]: >= 100 tau_int).
 (b) |S(k)|^2 of the 20 stored samples at the smallest-k and helix shells: lag-1 autocorrelation between samples
     (spacing n_meas/20 sweeps) as a crude check of the slow modes.
 (c) T_L robustness: straight-line fit of c1/r vs T over the 9 points nearest the lowest-T crossing (per seed and seed-averaged).
Output data/task6_posthoc.json"""
import numpy as np, glob, json, os
def tau_int(x, c=6.0):
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]; t = 0.5
    for w in range(1, n):
        t += acf[w]
        if w >= c * t: break
    return t
out = json.load(open('data/task6_posthoc.json'))['tau'] if os.path.exists('data/task6_posthoc.json') else {}   # keep runs whose raw data were deleted
for d in sorted(glob.glob('data/raw/t2_small_*_s*/')):
    run = os.path.basename(d.rstrip('/')); rows = []
    for f in sorted(glob.glob(d + 'T*.npz')):
        z = np.load(f); D = float(z['D']); L = int(z['L']); neq = int(z['neq'])
        E4 = z['E4'][neq:]; e = -(E4[:, 0] + E4[:, 1]) - D * (E4[:, 2] + E4[:, 3])
        s = z['samples'].astype(np.float32); F = np.fft.fft2(s, axes=(1, 2)); P = (np.abs(F) ** 2).sum(-1) / L ** 2
        nq = max(1, int(round(L * np.arctan(D) / (2 * np.pi))))
        p1 = 0.5 * (P[:, 1, 0] + P[:, 0, 1]); pq = 0.5 * (P[:, nq, 0] + P[:, 0, nq])
        ac = lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1])
        rows.append(dict(T=float(z['T']), tau_e=float(tau_int(e)), nmeas=int(len(e)), ac1_kmin=ac(p1), ac1_kq=ac(pq)))
    out[run] = rows
    te = [r['tau_e'] for r in rows]
    print('%-26s n=%d  tau_e max %.1f (T=%.2f)  min nmeas/tau %.0f | lag-1 corr of samples: kmin mean %+.2f, kq mean %+.2f' % (run, len(rows), max(te), rows[int(np.argmax(te))]['T'],
          min(r['nmeas'] / r['tau_e'] for r in rows), np.mean([r['ac1_kmin'] for r in rows]), np.mean([r['ac1_kq'] for r in rows])))
R = json.load(open('data/raw/task2_small_runs.json')); lin = {}
def linfit_cross(T, y, n=9):
    o = np.argsort(T); T = np.asarray(T)[o]; y = np.asarray(y, float)[o]
    i = [k for k in range(len(T) - 1) if y[k] * y[k + 1] < 0]
    if not i: return np.nan
    i = i[0]; sel = np.argsort(np.abs(T - T[i]))[:n]; c = np.polyfit(T[sel], y[sel], 1); return float(-c[1] / c[0])
for r in R:
    if 'c1_w1.5' not in r or not len(r['T']): continue
    lin[r['file']] = {w: linfit_cross(r['T'], r['c1_w' + w]) for w in ('1.2', '1.5', '2.0')}
    print('%-26s linear-fit T_L (w 1.2,1.5,2.0): %s' % (r['file'], ' '.join('%.4f' % v for v in lin[r['file']].values())))
json.dump(dict(tau=out, TL_linfit=lin), open('data/task6_posthoc.json', 'w'), indent=1)
