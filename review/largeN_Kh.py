"""Large-N (spherical) model with uniaxial anisotropy -K (S^z)^2 and a field h along z, exact on the square lattice.
Eigenvalues of the coupling matrix:  lam_0 = J0,  lam_pm = J0 + K +- sqrt(K^2 + 4 D^2 |s|^2).
Constraint: m^2 + (T/N) sum_k sum_a 1/(mu - lam_a) = 1,  m = h/(mu - 4J - 2K);  U = (3T - mu)/2 - h m/2.
Component-resolved disorder lines (k along x; DMI couples y and z):
  S_zz: mu - 4J = 4D^2/J          S_yy: mu - 4J - 2K = 4D^2/J
  total: 3J A^2 - (8D^2 + 8JK) A + 8K(D^2 + JK) = 0,  A = mu - 4J   (larger root)
Run from the repository root."""
import numpy as np, json
from scipy.optimize import brentq

J = 1.0; L = 768
k = 2 * np.pi * (np.arange(L // 2) + 0.5) / L
KX, KY = np.meshgrid(k, k, indexing='ij'); J0 = 2 * J * (np.cos(KX) + np.cos(KY)); S2 = np.sin(KX) ** 2 + np.sin(KY) ** 2


def lams(D, K):
    r = np.sqrt(K * K + 4 * D * D * S2); return J0, J0 + K + r, J0 + K - r


def g(mu, ls): return sum(np.mean(1.0 / (mu - l)) for l in ls)


def comps(mu, kk, D, K):          # S_xx, S_yy, S_zz (over T) at k = (kk, 0)
    j0 = 2 * J * (np.cos(kk) + 1); ay = mu - j0; az = mu - j0 - 2 * K; det = ay * az - 4 * D * D * np.sin(kk) ** 2
    return 1 / ay, az / det, ay / det


def curv_root(which, D, K, lmax):  # mu at which the k^2 coefficient of the chosen component (or the total) vanishes
    e = 1e-3
    def f(mu):
        a = comps(mu, e, D, K); b = comps(mu, 0.0, D, K)
        return (sum(a) - sum(b)) if which == 3 else (a[which] - b[which])
    grid = lmax + np.geomspace(1e-6, 3, 4000); v = np.array([f(m) for m in grid]); i = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return brentq(f, grid[i[-1]], grid[i[-1] + 1]) if len(i) else np.nan


out = dict(K=[], h=[], star=[])
D = 0.3
print('D = %.1f : D^2/J = %.3f, 2D^2/J = %.3f' % (D, D * D, 2 * D * D))
print('  K      A_zz(num) 4D^2   A_yy(num) 4D^2+2K  A_tot(num) quad   | T_L^zz   T_L^yy   T_L^tot')
for K in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.17]:
    ls = lams(D, K); lmax = max(l.max() for l in ls); lmax = max(lmax, 4 * J + 2 * K)
    A = [curv_root(w, D, K, lmax) - 4 * J for w in (2, 1, 3)]
    disc = (8 * D * D + 8 * J * K) ** 2 - 96 * J * K * (D * D + J * K); Aq = ((8 * D * D + 8 * J * K) + np.sqrt(disc)) / (6 * J) if disc >= 0 else np.nan
    TL = [1 / g(4 * J + a, ls) if np.isfinite(a) and 4 * J + a > lmax else np.nan for a in A]
    out['K'].append(dict(K=K, A=A, TL=TL))
    print(' %+.2f   %.4f   %.4f  %.4f   %.4f   %.4f    %.4f | %.4f   %.4f   %.4f' % (K, A[0], 4 * D * D, A[1], 4 * D * D + 2 * K, A[2], Aq, TL[0], TL[1], TL[2]))

print('\nfield along z (K = 0): T_L(h) = (1 - h^2/A_L^2)/g(A_L)')
ls = lams(D, 0.0)
for name, AL in (('zz/yy component', 4 * D * D), ('total', 8 * D * D / 3)):
    T0 = 1 / g(4 * J + AL, ls); row = []
    for h in [0.0, 0.05, 0.1, 0.2, 0.3]:
        m = h / AL; row.append((h, (1 - m * m) * T0 if m < 1 else np.nan, m))
    out['h'].append(dict(line=name, AL=AL, rows=row))
    print('  %-16s h_L = A_L = %.3f :' % (name, AL), '  '.join('h=%.2f T_L=%.3f (m=%.2f)' % r for r in row))

# short-range-order marker T(C = 0.75) versus K and h
Ts = np.linspace(0.5, 1.5, 81)


def marker(D, K, h):
    ls = lams(D, K); lmax = max(max(l.max() for l in ls), 4 * J + 2 * K)
    def mu_of(T):
        f = lambda mu: (h / (mu - 4 * J - 2 * K)) ** 2 + T * g(mu, ls) - 1
        return brentq(f, lmax + 1e-12, lmax + 3 * T + 10 + 10 * h, xtol=1e-13)
    mu = np.array([mu_of(T) for T in Ts]); U = (3 * Ts - mu) / 2 - h * (h / (mu - 4 * J - 2 * K)) / 2; C = np.gradient(U, Ts)
    return float(np.interp(-0.75, -C, Ts)), float(np.interp(1.0, Ts, np.sqrt(J / (mu - lmax))))


for D_ in (0.0, 0.3):
    T0, _ = marker(D_, 0.0, 0.0); print('\nD = %.1f  marker T(C=0.75) at K = h = 0: %.4f' % (D_, T0))
    for K in (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05):
        t, _ = marker(D_, K, 0.0); out['star'].append(dict(D=D_, K=K, h=0.0, shift=t / T0 - 1)); print('   K=%+.3f shift %+.4f  (shift/K = %+.2f)' % (K, t / T0 - 1, (t / T0 - 1) / K))
    for h in (0.01, 0.02, 0.05, 0.1):
        t, _ = marker(D_, 0.0, h); out['star'].append(dict(D=D_, K=0.0, h=h, shift=t / T0 - 1)); print('   h=%.3f  shift %+.5f  (shift/h^2 = %+.2f)' % (h, t / T0 - 1, (t / T0 - 1) / h ** 2))
json.dump(out, open('review/data/largeN_Kh.json', 'w'), indent=1)
