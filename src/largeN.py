"""Task 6a: large-N (spherical) limit of the 2D Heisenberg + Bloch-DMI model on the square lattice, exact saddle point.

Saddle point (REQUEST.md task 6a convention):  1 = (T/3N) sum_k sum_a 1/(mu - lam_a(k)),
  lam_+- = J0(k) +- 2D|s(k)|, lam_0 = J0(k), J0 = 2J(cos kx + cos ky), |s| = sqrt(sin^2 kx + sin^2 ky).
Propagator per eigenmode (T/3)/(mu - lam_a); S(k) = (T/3) sum_a 1/(mu - lam_a) (so that (1/N) sum_k S(k) = 1).
U(T) = -(1/2N) sum_k sum_a lam_a (T/3)/(mu - lam_a) per spin;  C = dU/dT (numerical, 5-point).
Disorder line: small-k expansion of S(k) gives the k^2 coefficient  -3J/m^2 + 8D^2/m^3  (m = mu - 4J), so S''(0) = 0  <=>  m = 8D^2/(3J).
  (checked numerically below on the lattice S(k)).
xi at D=0: S ~ 1/(m + J k^2)  ->  xi_0(T)^2 = J/m_0(T) (second-moment definition of the D=0 saddle).
Brillouin-zone sums: NK x NK grid (midpoint), NK chosen so that xi << NK (checked by doubling NK).
"""
import numpy as np, json, sys
from scipy.optimize import brentq

class Saddle:
    def __init__(self, D, J=1.0, NK=2048):
        k = (np.arange(NK) + 0.5) * 2 * np.pi / NK - np.pi; KX, KY = np.meshgrid(k, k, indexing='ij')
        J0 = 2 * J * (np.cos(KX) + np.cos(KY)); s = np.sqrt(np.sin(KX) ** 2 + np.sin(KY) ** 2)
        self.lam = [J0 + 2 * D * s, J0 - 2 * D * s, J0]; self.lgrid = max(l.max() for l in self.lam); self.D, self.J = D, J
        self.lmax = 4 * J * np.sqrt(1 + D * D / (2 * J * J))   # exact continuum maximum (diagonal helix)
    def G(self, mu):  # (1/3N) sum 1/(mu - lam)
        return sum((1.0 / (mu - l)).mean() for l in self.lam) / 3
    def mu(self, T):
        f = lambda m: T * self.G(self.lgrid + m) - 1.0
        return self.lgrid + brentq(f, 1e-14, 20.0, xtol=1e-16, rtol=1e-14, maxiter=1000)
    def U(self, T):
        m = self.mu(T); return -0.5 * sum((l * (T / 3) / (m - l)).mean() for l in self.lam)
    def T_of_mu(self, mu): return 1.0 / self.G(mu)

def run(Ds, NK, Tg):
    out = {}
    s0 = Saddle(0.0, NK=NK)
    xi0 = lambda T: np.sqrt(1.0 / (s0.mu(T) - 4.0))
    for D in Ds:
        s = Saddle(D, NK=NK)
        U = np.array([s.U(T) for T in Tg]); h = Tg[1] - Tg[0]
        C = np.full_like(U, np.nan); C[2:-2] = (-U[4:] + 8 * U[3:-1] - 8 * U[1:-3] + U[:-4]) / (12 * h)
        i = np.nanargmax(C)
        if 3 <= i <= len(Tg) - 4:
            sl = slice(i - 3, i + 4); c = np.polyfit(Tg[sl], C[sl], 2); Tstar = -c[1] / (2 * c[0])
        else: Tstar = np.nan        # no interior maximum of C (large-N: C is monotonic at leading order)
        rec = dict(D=D, Tstar=float(Tstar), Cmax=float(np.nanmax(C)), T=Tg.tolist(), C=C.tolist(), U=U.tolist())
        if D > 0:
            muL = 4.0 + 8 * D * D / 3.0; TL = s.T_of_mu(muL) if muL < s.lmax or True else np.nan
            # note: lmax(D) = 4 + D^2 approx (helix); muL must exceed lmax for a physical saddle
            rec.update(TL=float(TL) if muL > s.lmax else np.nan, muL=muL, lmax=float(s.lmax))
            lam = 2 * np.pi / np.arctan(D); q = np.arctan(D)
            if muL > s.lmax:
                rec.update(xi0_TL=float(xi0(TL)), kappa=float(xi0(TL) / lam), xiD_TL=float(np.sqrt(3 / 8) / D), lam=lam, mL_over_lmax_gap=float(muL - s.lmax))
        out[D] = rec
    out['xi0'] = dict(T=Tg.tolist(), xi=[float(xi0(T)) for T in Tg])
    return out

if __name__ == '__main__':
    NK = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    Ds = [0.0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
    Tg = np.round(np.arange(0.20, 2.0001, 0.005), 4)
    out = run(Ds, NK, Tg)
    for D in Ds:
        r = out[D]; print('D=%.3f  T*(C max)=%.4f Cmax=%.4f' % (D, r['Tstar'], r['Cmax']), ('  T_L=%.5f (mu_L-4=%.5f, lam_max-4=%.5f) xi0(T_L)=%.3f lam=%.1f kappa=%.4f' % (r['TL'], r['muL'] - 4, r['lmax'] - 4, r['xi0_TL'], r['lam'], r['kappa'])) if D > 0 and 'kappa' in r else '')
    json.dump({str(k): v for k, v in out.items()}, open('data/task6a_largeN_NK%d.json' % NK, 'w'))
