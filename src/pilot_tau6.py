"""Task 6b pilot: integrated autocorrelation times at L=512, T=0.60 for the energy and for |S(k_min)|^2 (k_min = 2pi/L along x and y,
summed over spin components), and for the ring mode |S(k_q)|^2 (nearest lattice k to the helix wavevector along the diagonal).
Anneal 0.80 -> 0.60 in steps 0.02 (NANN sweeps each), NEQ equilibration, NMEAS measurement sweeps."""
import numpy as np, sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from kernels import sweep, energy, seed as nseed, random_state
from pilot_tau import tau_int
D = float(sys.argv[1]); L = int(sys.argv[2]); T = float(sys.argv[3]); NANN, NEQ, NMEAS = (int(x) for x in sys.argv[4:7])
nseed(11); S = random_state(L, np.random.default_rng(11)); sig = lambda t: min(1.5, 1.2 * np.sqrt(t)); t0 = time.time()
for t in np.arange(0.80, T + 1e-9, -0.02):
    for _ in range(NANN): sweep(S, t, 1.0, D, sig(t))
for _ in range(NEQ): sweep(S, T, 1.0, D, sig(T))
x = np.arange(L); ex = np.exp(2j * np.pi * x / L)
nq = max(1, int(round(L * np.arctan(D / np.sqrt(2)) / (2 * np.pi))))      # diagonal helix: k = (nq, nq) 2pi/L
eq = np.exp(2j * np.pi * nq * x / L)
E = np.empty(NMEAS); K = np.empty(NMEAS); Q = np.empty(NMEAS)
for n in range(NMEAS):
    sweep(S, T, 1.0, D, sig(T)); E[n] = energy(S, 1.0, D)
    a = np.einsum('i,ijc->c', ex, S); b = np.einsum('j,ijc->c', ex, S); K[n] = (np.abs(a) ** 2 + np.abs(b) ** 2).sum() / (2 * L * L)
    c1 = np.einsum('i,j,ijc->c', eq, eq, S); c2 = np.einsum('i,j,ijc->c', eq, np.conj(eq), S); Q[n] = (np.abs(c1) ** 2 + np.abs(c2) ** 2).sum() / (2 * L * L)
r = dict(D=D, L=L, T=T, nann=NANN, neq=NEQ, nmeas=NMEAS, nq_diag=nq, tau_E=float(tau_int(E)), tau_Skmin=float(tau_int(K)), tau_Sq=float(tau_int(Q)),
         tau_E_halves=[float(tau_int(E[:NMEAS // 2])), float(tau_int(E[NMEAS // 2:]))], tau_Skmin_halves=[float(tau_int(K[:NMEAS // 2])), float(tau_int(K[NMEAS // 2:]))],
         tau_Sq_halves=[float(tau_int(Q[:NMEAS // 2])), float(tau_int(Q[NMEAS // 2:]))], Skmin_mean=float(K.mean()), Sq_mean=float(Q.mean()), time_s=time.time() - t0)
print(json.dumps(r)); json.dump(r, open('data/task6b_pilot_tau_D%03d_L%d_T%.2f.json' % (round(D * 1000), L, T), 'w'))
