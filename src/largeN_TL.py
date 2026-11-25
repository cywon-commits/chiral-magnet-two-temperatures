"""Task 6a, fast part: T_L(D), xi_0(T_L), kappa = xi_0(T_L)/lambda for many D, and Brillouin-zone grid convergence (NK = 512, 1024, 2048).
Uses src/largeN.Saddle.  Output data/task6a_largeN_TL.json"""
import numpy as np, json, sys
sys.path.insert(0, 'src'); from largeN import Saddle
Ds = [0.025, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]; out = {}
for NK in (512, 1024, 2048):
    s0 = Saddle(0.0, NK=NK); res = []
    for D in Ds:
        s = Saddle(D, NK=NK); muL = 4 + 8 * D * D / 3; TL = s.T_of_mu(muL); xi0 = np.sqrt(1 / (s0.mu(TL) - 4)); lam = 2 * np.pi / np.arctan(D)
        res.append(dict(D=D, TL=TL, xi0=xi0, lam=lam, kappa=xi0 / lam, m0_over_mL=(s0.mu(TL) - 4) / (muL - 4)))
    out[NK] = res
    print('NK=%d' % NK); [print('  D=%.3f T_L=%.5f xi0=%.3f kappa=%.4f m0/mL=%.3f' % (r['D'], r['TL'], r['xi0'], r['kappa'], r['m0_over_mL'])) for r in res]
json.dump({str(k): v for k, v in out.items()}, open('data/task6a_largeN_TL.json', 'w'), indent=1)
