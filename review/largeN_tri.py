"""Large-N (spherical) solution on the square and triangular lattices with the same code, to separate lattice effects.
Bonds delta (|delta| = 1): J(k) = J sum_delta cos(k.delta), d(k) = D sum_delta delta sin(k.delta); lam_pm = J(k) +- |d(k)|, lam_0 = J(k).
Small k: J(k) = zJ - c_J k^2, |d| = c_D k with c_J = zJ/4, c_D = zD/2  ->  q = c_D/(2 c_J) = D/J on both lattices.
Continuum stiffness per area rho_J = zJ/(4 A_c): J (square), sqrt(3) J (triangular).
Ring onset of the total S(k):  mu - zJ = 2 c_D^2/(3 c_J) = (2/3) z D^2/J  (exact on the lattice).  xi_H^2 = c_J/(mu_0 - zJ) at D = 0."""
import numpy as np, json
from scipy.optimize import brentq
J = 1.0; L = 512
u = (np.arange(L) + 0.5) / L; U1, U2 = np.meshgrid(u, u, indexing='ij')
def spectrum(lat, D):
    if lat == 'sq':
        a, b = 2 * np.pi * U1, 2 * np.pi * U2
        J0 = 2 * J * (np.cos(a) + np.cos(b)); dn = 2 * D * np.hypot(np.sin(a), np.sin(b)); z = 4
    else:
        a, b = 2 * np.pi * U1, 2 * np.pi * U2; c = b - a          # k.delta for delta1=(1,0), delta2=(1/2,r3/2), delta3=delta2-delta1
        J0 = 2 * J * (np.cos(a) + np.cos(b) + np.cos(c))
        dx = 2 * D * (np.sin(a) + 0.5 * np.sin(b) - 0.5 * np.sin(c)); dy = 2 * D * (np.sqrt(3) / 2) * (np.sin(b) + np.sin(c)); dn = np.hypot(dx, dy); z = 6
    return J0, dn, z
def pitch(lat, D):                                  # ring radius along a bond direction from the exact dispersion
    k = np.linspace(1e-4, 1.5, 200001)
    e = 2 * J * np.cos(k) + 2 * D * np.sin(k) if lat == 'sq' else 2 * J * (np.cos(k) + 2 * np.cos(k / 2)) + 2 * D * (np.sin(k) + np.sin(k / 2))
    return k[np.argmax(e)]
out = {}
for lat, Ts in (('sq', np.linspace(0.3, 1.7, 141)), ('tri', np.linspace(0.5, 3.0, 141))):
    def mu_of(T, D):
        J0, dn, z = spectrum(lat, D); lm = (J0 + dn).max()
        g = lambda m: T * (np.mean(1 / (m - J0 - dn)) + np.mean(1 / (m - J0 + dn)) + np.mean(1 / (m - J0))) - 1
        return brentq(g, lm + 1e-12, lm + 3 * T + 10, xtol=1e-13)
    z = 4 if lat == 'sq' else 6; cJ = z * J / 4; rho = z * J / (4 * (1.0 if lat == 'sq' else np.sqrt(3) / 2))
    mu0 = np.array([mu_of(T, 0.0) for T in Ts]); xi0 = np.sqrt(cJ / (mu0 - z * J))
    rows = []
    for D in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        mu = np.array([mu_of(T, D) for T in Ts]); q = pitch(lat, D); lam = 2 * np.pi / q
        Tr = float(np.interp(0.0, (mu - z * J) - (2 / 3) * z * D * D / J, Ts)); xi = float(np.exp(np.interp(Tr, Ts, np.log(xi0))))
        rows.append(dict(D=D, q=float(q), lam=float(lam), T_ring=Tr, T_ring_over_rho=Tr / rho, xiH=xi, kappa=xi / lam, kappa_cont=xi * D / (2 * np.pi)))
        print('%-3s D=%.2f q=%.4f (D/J=%.2f) lam=%6.2f  T_ring=%.4f  T_ring/rho_J=%.4f  xi_H=%.2f  kappa=%.4f' % (lat, D, q, D, lam, Tr, Tr / rho, xi, xi / lam))
    out[lat] = rows
json.dump(out, open('review/data/largeN_tri.json', 'w'), indent=1)
print('\nratio kappa_tri/kappa_sq:', [round(a['kappa'] / b['kappa'], 3) for a, b in zip(out['tri'], out['sq'])])
print('ratio (T_ring/rho)_tri/(T_ring/rho)_sq:', [round(a['T_ring_over_rho'] / b['T_ring_over_rho'], 3) for a, b in zip(out['tri'], out['sq'])])
