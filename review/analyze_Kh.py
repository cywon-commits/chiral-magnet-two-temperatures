"""Analysis of review/mc_Kh.py runs: heat-capacity peak T*, component-resolved disorder lines, M/H criterion, ring anisotropy.
Run from the repository root."""
import numpy as np, glob, json

W = 6; n1 = np.r_[0:W + 1, -W:0]; NX, NY = np.meshgrid(n1, n1, indexing='ij'); N2 = NX ** 2 + NY ** 2
PHI = np.arctan2(NY, NX)


def load(tag):
    fs = sorted(glob.glob('review/data/Kh_%s_s*.npz' % tag)); d = [np.load(f) for f in fs]
    T = d[0]['T']; o = np.argsort(T)
    return T[o], np.array([x['E'][o] for x in d]), np.array([x['M'][o] for x in d]), np.array([x['P'][o] for x in d]), d[0]['par']


def tstar_one(T, U):
    p = np.polyfit(T, U, 6); tf = np.linspace(T[3], T[-4], 3001); c = np.polyval(np.polyder(p), tf); i = int(np.argmax(c))
    return tf[i], c[i]


def jack(fn, n):
    full = fn(np.arange(n)); sub = np.array([fn(np.delete(np.arange(n), i)) for i in range(n)])
    return full, np.sqrt((n - 1) * np.nanvar(sub, axis=0))


def comps(P, L):
    """S_long, S_trans(in-plane), S_zz on the window, from the Hermitian matrix P_ab(k)."""
    nrm = np.sqrt(np.where(N2 > 0, N2, 1)); kx = NX / nrm; ky = NY / nrm
    Pxx, Pyy, Pzz, Pxy = P[..., 0, 0].real, P[..., 1, 1].real, P[..., 2, 2].real, P[..., 0, 1].real
    SL = kx * kx * Pxx + ky * ky * Pyy + 2 * kx * ky * Pxy
    ST = ky * ky * Pxx + kx * kx * Pyy - 2 * kx * ky * Pxy
    return SL, ST, Pzz


def slope(S, L, mask):               # d S / d k^2 from the selected small-k points
    k2 = (2 * np.pi / L) ** 2 * N2[mask]; return np.polyfit(k2, S[..., mask].T, 1)[0]


def crossing(T, y, tmin=0.70):     # root of a quadratic fit to y(T) = (dS/dk^2)/S on T >= tmin (y > 0 cold, < 0 hot)
    m = T >= tmin; c = np.polyfit(T[m], y[m], 2); r = np.roots(c); r = r[np.isreal(r)].real
    r = [x for x in r if 0.6 < x < 1.3 and np.polyval(np.polyder(c), x) < 0]
    return min(r) if r else np.nan


ALL = (N2 > 0) & (N2 <= 5)
SECX = ALL & (np.abs(NY) <= 0.6 * np.abs(NX)); SECY = ALL & (np.abs(NX) <= 0.6 * np.abs(NY))
RING = (N2 >= 9) & (N2 <= 36)
out = {}
for tag in ['base', 'Kp05', 'Km05', 'hz05', 'hz10', 'hx10']:
    if not glob.glob('review/data/Kh_%s_s*.npz' % tag): continue
    T, E, M, P, par = load(tag); D, K, hx, hz, L = par; L = int(L); ns = len(E); U = E.mean(-1)
    r = dict(K=float(K), hx=float(hx), hz=float(hz), seeds=ns)
    (ts, cm), (es, ec) = jack(lambda s: np.array(tstar_one(T, U[s].mean(0))), ns); r['Tstar'] = [float(ts), float(es)]; r['Cmax'] = [float(cm), float(ec)]

    def lines(s, mask):
        SL, ST, SZ = comps(P[s].mean(0), L)
        return np.array([crossing(T, slope(X, L, mask) / X[..., mask].mean(-1)) for X in (SZ, ST, SL, SL + ST + SZ)])
    for name, mask in (('all', ALL), ('k||x', SECX), ('k||y', SECY)):
        v, e = jack(lambda s: lines(s, mask), ns); r['TL_' + name] = dict(zz=[float(v[0]), float(e[0])], trans=[float(v[1]), float(e[1])], long=[float(v[2]), float(e[2])], total=[float(v[3]), float(e[3])])
    tot = sum(comps(P.mean(0), L)); c2 = (tot[:, RING] * np.cos(2 * PHI[RING])).sum(1) / tot[:, RING].sum(1); r['cos2phi'] = {('%.3f' % t): float(c) for t, c in zip(T, c2)}
    # susceptibilities: chi_a(k=0) = N <m_a^2>/T at zero field (per component); m/h in a field
    chi0 = (P.mean(0)[:, 0, 0].real.diagonal(axis1=-2, axis2=-1)) / T[:, None]          # [T, a]
    r['chi0'] = chi0.tolist(); r['m'] = M.mean(0).tolist(); r['T'] = T.tolist()
    out[tag] = r
    print('\n%s  K=%+.2f hx=%.2f hz=%.2f (%d seeds)  T* = %.4f +- %.4f  Cmax = %.3f' % (tag, K, hx, hz, ns, ts, es, cm))
    for name in ('all', 'k||x', 'k||y'):
        q = r['TL_' + name]; print('   T_L[%-4s]  zz %.3f(%.0f)  in-plane trans %.3f(%.0f)  long %.3f(%.0f)  total %.3f(%.0f)' % ((name,) + tuple(x for key in ('zz', 'trans', 'long', 'total') for x in (q[key][0], 1e3 * q[key][1]))))
    print('   ring <cos 2phi> at T = 0.9, 0.8, 0.7, 0.6:', ' '.join('%+.3f' % r['cos2phi']['%.3f' % t] for t in (0.9, 0.8, 0.7, 0.6)))

b = out['base']; T = np.array(b['T'])
print('\nshifts of T* relative to base:')
for tag in out:
    if tag != 'base': print('   %s: dT*/T* = %+.4f +- %.4f' % (tag, out[tag]['Tstar'][0] / b['Tstar'][0] - 1, np.hypot(out[tag]['Tstar'][1], b['Tstar'][1]) / b['Tstar'][0]))
print('\nM/H criterion on the S_zz disorder line (prediction: equal values, Gaussian J/4D^2 = %.2f):' % (1 / (4 * 0.09)))
t0 = b['TL_all']['zz'][0]; chi = np.array(b['chi0']); print('   h = 0   : T_L^zz = %.3f, chi_perp (mean of x, y) = %.2f, chi_zz = %.2f' % (t0, np.interp(t0, T, chi[:, :2].mean(1)), np.interp(t0, T, chi[:, 2])))
for tag, h in (('hz05', 0.05), ('hz10', 0.10)):
    if tag in out:
        t = out[tag]['TL_all']['zz'][0]; mz = np.interp(t, T, np.array(out[tag]['m'])[:, 2]); print('   h = %.2f: T_L^zz = %.3f, m_z/h = %.2f   (m_z = %.3f)' % (h, t, mz / h, mz))
if 'hx10' in out:
    t = out['hx10']['TL_k||x']['zz'][0]; mx = np.interp(t, T, np.array(out['hx10']['m'])[:, 0]); print('   in-plane h = 0.10, k || h: T_L^zz = %.3f, m_x/h = %.2f' % (t, mx / 0.1))
json.dump(out, open('review/data/Kh_analysis.json', 'w'))
