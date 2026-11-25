"""Large-N (spherical) solution of the square-lattice Heisenberg + Bloch DMI model, exact on the lattice.
Constraint (unit spins, 3 channels):  1 = (T/N) sum_k sum_a 1/(mu - lam_a(k)),
lam_pm = J0 +- 2 D |s(k)|, lam_0 = J0, J0 = 2J(cos kx + cos ky), |s| = sqrt(sin^2 kx + sin^2 ky).
High-T limit mu = 3T reproduces the RPA.  U = (3T - mu)/2, C = (3 - dmu/dT)/2.
Disorder line (k^2 coefficient of S(k) at k = 0 vanishes):  mu - 4J = 8 D^2 / (3J)."""
import numpy as np, json
from scipy.optimize import brentq

J = 1.0; L = 1024
k = 2 * np.pi * (np.arange(L) + 0.5) / L            # shifted grid avoids the k = 0 point
KX, KY = np.meshgrid(k[:L // 2], k[:L // 2], indexing='ij')     # one quadrant suffices (even in kx, ky)
J0 = 2 * J * (np.cos(KX) + np.cos(KY)); SN = np.hypot(np.sin(KX), np.sin(KY))


def lam(D): return J0 + 2 * D * SN, J0 - 2 * D * SN, J0


def G(mu, T, D): return T * sum(np.mean(1.0 / (mu - l)) for l in lam(D))


def mu_of(T, D):
    lmax = lam(D)[0].max()
    return brentq(lambda m: G(m, T, D) - 1, lmax + 1e-13, lmax + 3 * T + 10, xtol=1e-14)


def lam_max_true(D):          # diagonal helix is the true maximum: 4J sqrt(1 + D^2/2J^2)
    return 4 * np.sqrt(J * J + D * D / 2)


Ts = np.linspace(0.25, 1.6, 271)
res = {}
for D in [0.0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    mu = np.array([mu_of(T, D) for T in Ts]); U = (3 * Ts - mu) / 2; C = np.gradient(U, Ts)
    res[D] = dict(mu=mu, C=C)
lam0max = lam(0.0)[2].max()
mu0 = res[0.0]['mu']; xi0 = np.sqrt(J / (mu0 - lam0max))                  # D = 0 correlation length (OZ form)
print('large-N has no heat-capacity maximum: C rises monotonically to 3/2 at low T (C at T=0.25, 0.7, 1.6: %.3f %.3f %.3f)' % (res[0.0]['C'][0], np.interp(0.7, Ts, res[0.0]['C']), res[0.0]['C'][-1]))


def markers(C):           # analogues of T*: steepest rise of C, and C = 0.75 (half of the low-T value)
    dC = -np.gradient(C, Ts); i = int(np.argmax(dC)); p = np.polyfit(Ts[i - 3:i + 4], dC[i - 3:i + 4], 2)
    return -p[1] / (2 * p[0]), float(np.interp(-0.75, -C, Ts))


Ts0a, Ts0b = markers(res[0.0]['C']); print('D=0 markers: T(max -dC/dT)=%.4f  T(C=0.75)=%.4f  xi0 there: %.2f %.2f' % (Ts0a, Ts0b, np.interp(Ts0a, Ts, xi0), np.interp(Ts0b, Ts, xi0)))
rows = []
for D in [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    mu, C = res[D]['mu'], res[D]['C']; q = np.arctan(D / J); lam_ = 2 * np.pi / q
    TL = float(np.interp(0.0, (mu - 4 * J) - 8 * D * D / (3 * J), Ts))
    xi0_TL = float(np.interp(TL, Ts, xi0)); Ta, Tb = markers(C)
    delta = mu - lam(D)[0].max(); xiq = np.sqrt(J / delta)               # helical (ring) correlation length
    Txi1 = float(np.interp(1.0, (q * xiq)[::-1], Ts[::-1])); Txil = float(np.interp(1.0, (xiq / lam_)[::-1], Ts[::-1]))
    rows.append(dict(D=D, lam=lam_, TL=TL, kappa_H=xi0_TL / lam_, Ta=float(Ta), Tb=float(Tb), sa=float(Ta / Ts0a - 1), sb=float(Tb / Ts0b - 1), T_qxi1=Txi1, T_xilam=Txil))
    print('D=%.3f lam=%6.1f  T_L=%.4f  xi0(T_L)/lam=%.4f | T(max -dC/dT)=%.4f (%+.5f)  T(C=.75)=%.4f (%+.5f) | T[q xi_q=1]=%.3f  T[xi_q=lam]=%.3f' % (D, lam_, TL, xi0_TL / lam_, Ta, Ta / Ts0a - 1, Tb, Tb / Ts0b - 1, Txi1, Txil))
x = np.array([r['D'] for r in rows])
for key in ('sa', 'sb'):
    y = np.array([r[key] for r in rows])
    print(key, 'local log-log slopes d ln(shift)/d ln D:', np.round(np.diff(np.log(np.abs(y))) / np.diff(np.log(x)), 2))
for r in rows: r['gapL_a'] = r['TL'] - r['Ta']
print('T_L - T(max -dC/dT):', [round(r['gapL_a'], 3) for r in rows])
# RPA check: mu = 3T  ->  T_L = 4J/3 + 8D^2/(9J)
print('RPA T_L check (D=0.3): %.4f' % (4 / 3 + 8 * 0.09 / 9))
json.dump(dict(rows=rows), open('review/data/largeN.json', 'w'), indent=1)
