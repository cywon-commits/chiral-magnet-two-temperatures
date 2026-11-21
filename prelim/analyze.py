import json, numpy as np, sys, os

def load(tag):
    d = json.load(open('data/ts_%s.json' % tag)); r = d['data']
    T = np.array([x['T'] for x in r]); eb = np.array([x['eb'] for x in r]); e2b = np.array([x['e2b'] for x in r])
    rad = [np.array(x['rad']) for x in r]; xi = np.array([x['xi'] for x in r])
    o = np.argsort(T)
    return d['D'], d['L'], T[o], eb[o], e2b[o], [rad[i] for i in o], xi[o]

def peak(T, y, half=3):
    i = int(np.argmax(y)); lo = max(0, i - half); hi = min(len(T), i + half + 1)
    c = np.polyfit(T[lo:hi], y[lo:hi], 2)
    return (-c[1] / (2 * c[0]) if c[0] < 0 else np.nan), np.polyval(c, -c[1] / (2 * c[0]))

def tstar(T, eb, e2b, N):
    U = eb.mean(1); Cf = N * (e2b.mean(1) - U ** 2) / T ** 2
    # derivative of a smooth polynomial fit of U(T)
    p = np.polyfit(T, U, 6); Cd = np.polyval(np.polyder(p), T)
    Tf = np.linspace(T[2], T[-3], 2001); Cdf = np.polyval(np.polyder(p), Tf)
    return peak(T, Cf)[0], Tf[np.argmax(Cdf)], Cf, Cd, U

def jack(T, eb, e2b, N):
    nb = eb.shape[1]; a = []; b = []
    for k in range(nb):
        m = np.ones(nb, bool); m[k] = False
        t1, t2, *_ = tstar(T, eb[:, m], e2b[:, m], N); a.append(t1); b.append(t2)
    a = np.array(a); b = np.array(b)
    return np.sqrt((nb - 1) * a.var()), np.sqrt((nb - 1) * b.var())

res = {}
for tag in ['D020', 'D030', 'D040', 'D050']:
    if not os.path.exists('data/ts_%s.json' % tag): continue
    D, L, T, eb, e2b, rad, xi = load(tag)
    if len(T) < 12: print(tag, 'incomplete', len(T)); continue
    N = L * L; t1, t2, Cf, Cd, U = tstar(T, eb, e2b, N); s1, s2 = jack(T, eb, e2b, N)
    lam = 2 * np.pi / np.arctan(D)
    res[D] = dict(L=L, lam=lam, Tf=t1, sf=s1, Td=t2, sd=s2)
    print('D=%.1f L=%d lambda=%.1f  T*(C_fluct)=%.3f+-%.3f  T*(dU/dT)=%.3f+-%.3f  Cmax=%.2f' % (D, L, lam, t1, s1, t2, s2, Cf.max()))
    print('   T :', ' '.join('%.3f' % t for t in T)); print('   Cf:', ' '.join('%.2f' % c for c in Cf)); print('   Cd:', ' '.join('%.2f' % c for c in Cd))
    # ring peak position / height
    print('   ring peak n (T lowest, mid, highest):', [int(np.argmax(r[1:]) + 1) for r in (rad[0], rad[len(rad) // 2], rad[-1])], 'expected L/lambda=%.1f' % (L / lam))

if os.path.exists('data/ts_D000.json'):
    D, L, T, eb, e2b, rad, xi = load('D000')
    kx = np.fft.fftfreq(L) * L; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KR = np.rint(np.sqrt(KX ** 2 + KY ** 2)).astype(int)
    k2 = (4 * np.sin(np.pi * KX / L) ** 2 + 4 * np.sin(np.pi * KY / L) ** 2)
    k2m = (np.bincount(KR.ravel(), k2.ravel()) / np.bincount(KR.ravel()))[:L // 2]
    xis = []
    for t, r, x in zip(T, rad, xi):
        best = None
        xg = 5.0
        for it in range(6):  # self-consistent fit window k*xi < 2
            sel = (np.arange(len(r)) >= 1) & (np.sqrt(k2m) * xg < 2.0)
            if sel.sum() < 3: sel = (np.arange(len(r)) >= 1) & (np.arange(len(r)) <= 3)
            c = np.polyfit(k2m[sel], 1 / r[sel], 1); xg = np.sqrt(max(c[0] / c[1], 1e-6))
        xis.append(xg)
    xis = np.array(xis)
    print('D=0  T :', ' '.join('%.3f' % t for t in T)); print('  xi_OZ :', ' '.join('%.2f' % x for x in xis)); print('  xi_2nd:', ' '.join('%.2f' % x for x in xi))
    lx = np.log(xis)
    for D, r in sorted(res.items()):
        for key in ('Tf', 'Td'):
            xiT = np.exp(np.interp(r[key], T, lx)) if T.min() <= r[key] <= T.max() else np.nan
            print('D=%.1f  %s=%.3f  xi_H(T*)=%.2f  lambda=%.1f  xi_H/lambda=%.3f' % (D, key, r[key], xiT, r['lam'], xiT / r['lam']))
    # predicted T* for constant kappa fixed at D=0.3
    if 0.3 in res:
        for key in ('Tf', 'Td'):
            kap = np.exp(np.interp(res[0.3][key], T, lx)) / res[0.3]['lam']
            print('kappa(%s) fixed at D=0.3: %.3f ; predicted T*:' % (key, kap), {D: round(float(np.interp(np.log(kap * r['lam']), lx[::-1], T[::-1])), 3) for D, r in sorted(res.items())})

if len(res) >= 3:
    Ds = np.array(sorted(res)); x = np.log(1 / Ds)
    for key, sk in (('Tf', 'sf'), ('Td', 'sd')):
        y = 1 / np.array([res[d][key] for d in Ds]); c = np.polyfit(x, y, 1)
        print('%s: 1/T* vs ln(J/D): slope=%.3f (one-loop 1/2pi=0.159, D-independent=0) intercept=%.3f' % (key, c[0], c[1]))
        yt = y * np.sqrt(1 + Ds ** 2); c = np.polyfit(x, yt, 1)
        print('    with J~=sqrt(J^2+D^2): slope=%.3f' % c[0])
