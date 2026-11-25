"""Task 6a check of review V-8: temperature where the large-N C reaches half its T->0 value (C = 3/2 per spin in the corrected
convention, = 1/2 in the src/largeN.py convention where T_mine = 3 T).  Reports T(C half) in corrected units and its relative D shift.
Output data/task6a_largeN_Chalf.json"""
import numpy as np, json, sys
sys.path.insert(0, 'src'); from largeN import Saddle
Tg = np.round(np.arange(1.2, 4.0001, 0.004), 4); out = {}
for D in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    s = Saddle(D, NK=512); U = np.array([s.U(T) for T in Tg]); C = np.gradient(U, Tg)
    o = np.argsort(C); Th = float(np.interp(0.25, C[o], Tg[o]))   # C decreasing in T
    out[D] = Th / 3
T0 = out[0.0]
for D, t in out.items(): print('D=%.2f  T(C=3/4 of... C=0.75) = %.4f   shift = %.2f%%' % (D, t, 100 * (t / T0 - 1)))
json.dump({str(k): v for k, v in out.items()}, open('data/task6a_largeN_Chalf.json', 'w'))
