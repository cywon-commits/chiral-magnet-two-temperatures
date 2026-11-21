"""H1 judgement exactly as fixed in REPORT.md [R-2] before looking at results.
Input data/task1_summary.json -> prints table, writes data/task1_h1.json."""
import json, numpy as np

S = json.load(open('data/task1_summary.json'))
def get(proto, D, L, dirn):
    for e in S:
        if e['proto'] == proto and abs(e['D'] - D) < 1e-9 and e['L'] == L and e['dirn'] == dirn: return e

def tstar(D, L, proto='anneal'):
    c, h = get(proto, D, L, 'cool'), get(proto, D, L, 'heat')
    if c is None or h is None: return None
    mc, sc = c['Tstar_du'][0], c['Tstar_du'][1]; mh_, sh = h['Tstar_du'][0], h['Tstar_du'][1]
    T = 0.5 * (mc + mh_); stat = 0.5 * np.hypot(sc, sh)
    est = {'du': T, 'fl': 0.5 * (c['Tstar_fl'][0] + h['Tstar_fl'][0])}
    mh_ok = min(c['ess_min'], h['ess_min']) >= 50
    if mh_ok: est['mh'] = 0.5 * (c['Tstar_mh'][0] + h['Tstar_mh'][0])
    ev = [v for v in est.values() if np.isfinite(v)]
    s_est = (max(ev) - min(ev)) / 2
    wv = [0.5 * (c['Tstar_du_by_window'][w][0] + h['Tstar_du_by_window'][w][0]) for w in c['Tstar_du_by_window']]
    wv = [v for v in wv if np.isfinite(v)]; s_win = (max(wv) - min(wv)) / 2
    s_hys = abs(mc - mh_) / 2
    syst = np.sqrt(s_est ** 2 + s_win ** 2 + s_hys ** 2)
    return dict(D=D, L=L, T=T, stat=stat, syst=syst, tot=float(np.hypot(stat, syst)), s_est=s_est, s_win=s_win, s_hys=s_hys,
                est=est, mh_used=mh_ok, cool=mc, heat=mh_)

big = {0.2: 256, 0.3: 176, 0.4: 128, 0.5: 112}; small = {0.2: 128, 0.3: 88, 0.4: 64, 0.5: 56}
rows = {}
print('%-5s %-4s %-7s %-7s %-7s %-7s | est %-6s win %-6s hys %-6s | cool    heat    | estimators' % ('D', 'L', 'T*', 'stat', 'syst', 'tot', '', '', ''))
for D, L in [(0.0, 64), (0.0, 128), (0.0, 256)] + sorted(small.items()) + sorted(big.items()):
    r = tstar(D, L)
    if r is None: continue
    rows[(D, L)] = r
    print('%-5.2f %-4d %.4f  %.4f  %.4f  %.4f  |     %.4f     %.4f     %.4f | %.4f  %.4f  | %s' % (D, L, r['T'], r['stat'], r['syst'], r['tot'], r['s_est'], r['s_win'], r['s_hys'], r['cool'], r['heat'],
          ' '.join('%s=%.4f' % kv for kv in r['est'].items())))

def judge(sizes, label):
    r128, r256 = rows.get((0.0, 128)), rows.get((0.0, 256))
    same_L = abs(r128['T'] - r256['T']) > np.hypot(r128['tot'], r256['tot'])
    print('\n[%s]  D=0: L=128 vs 256 differ by %.4f (comb. err %.4f) -> %s' % (label, r128['T'] - r256['T'], np.hypot(r128['tot'], r256['tot']),
          'compare at same L where possible' if same_L else 'use D=0 L=256 as reference'))
    res = []; verdict = 'support'
    for D, L in sorted(sizes.items()):
        r = rows.get((D, L))
        if r is None: continue
        ref = rows.get((0.0, L)) if (same_L and (0.0, L) in rows) else (rows.get((0.0, L)) if label.startswith('aux') and (0.0, L) in rows else r256)
        diff = r['T'] - ref['T']; comb = np.hypot(r['tot'], ref['tot']); z = diff / comb
        res.append(dict(D=D, L=L, ref_L=ref['L'], diff=diff, comb=comb, z=z))
        print('  D=%.1f L=%d vs D=0 L=%d: dT*=%+.4f  comb.err=%.4f  z=%+.2f' % (D, L, ref['L'], diff, comb, z))
        if abs(z) > 3: verdict = 'reject'
        elif abs(z) > 2 and verdict == 'support': verdict = 'undecided'
    Ds = np.array([x['D'] for x in res]); Ts = np.array([rows[(x['D'], x['L'])]['T'] for x in res]); sg = np.array([rows[(x['D'], x['L'])]['tot'] for x in res])
    out = {}
    for tag, X, Y, W in (('D>0', Ds, Ts, sg), ('incl. D=0', np.r_[0, Ds], np.r_[r256['T'], Ts], np.r_[r256['tot'], sg])):
        A = np.vstack([X, np.ones_like(X)]).T / W[:, None]; cov = np.linalg.inv(A.T @ A); p = cov @ A.T @ (Y / W)
        chi2 = (((Y - (p[0] * X + p[1])) / W) ** 2).sum(); zs = p[0] / np.sqrt(cov[0, 0])
        print('  slope dT*/dD (%s) = %.4f +- %.4f  (z=%.2f, chi2/dof=%.2f/%d)' % (tag, p[0], np.sqrt(cov[0, 0]), zs, chi2, len(X) - 2))
        out[tag] = dict(slope=float(p[0]), err=float(np.sqrt(cov[0, 0])), z=float(zs), chi2=float(chi2), dof=len(X) - 2)
    zsl = out['D>0']['z']
    if abs(zsl) > 3: verdict = 'reject'
    elif abs(zsl) > 2 and verdict == 'support': verdict = 'undecided'
    print('  => H1 verdict (%s): %s' % (label, verdict))
    return dict(pairs=res, slope=out, verdict=verdict, ref_same_L=bool(same_L))

J = dict(rows={'%g_%d' % k: v for k, v in rows.items()})
if (0.0, 128) in rows and (0.0, 256) in rows:
    J['main'] = judge(big, 'main L/lambda~8'); J['aux'] = judge(small, 'aux L/lambda~4')
q = get('quench', 0.3, 128, 'quench')
if q:
    ests = {'du': q['Tstar_du'][0], 'fl': q['Tstar_fl'][0]}
    if q['ess_min'] >= 50: ests['mh'] = q['Tstar_mh'][0]
    ev = [v for v in ests.values() if np.isfinite(v)]
    J['quench'] = dict(T=q['Tstar_du'][0], stat=q['Tstar_du'][1], s_est=(max(ev) - min(ev)) / 2, s_win=q['Tstar_du'][3], est=ests)
    a = rows.get((0.3, 128))
    print('\nquench D=0.3 L=128: T*(du)=%.4f +- %.4f (stat)  s_est=%.4f s_win=%.4f  %s' % (q['Tstar_du'][0], q['Tstar_du'][1], J['quench']['s_est'], q['Tstar_du'][3], ests))
    for dirn in ('cool', 'heat'):
        e = get('anneal', 0.3, 176, dirn)
        if e: print('   anneal D=0.3 L=176 %s: T*(du)=%.4f +- %.4f' % (dirn, e['Tstar_du'][0], e['Tstar_du'][1]))
json.dump(J, open('data/task1_h1.json', 'w'), indent=1, default=float)

# ---------------- post-hoc supplementary analyses (NOT part of the pre-registered criterion) ----------------
print('\n[post hoc 1] dU/dT only: sigma = stat (+) window (+) cool/heat  (estimator spread excluded)')
print('[post hoc 2] rescaled T*/sqrt(1+D^2) (J_eff = sqrt(J^2+D^2), exact for a 1D chain under the local spin rotation)')
ph = []
for label, sizes in (('main', big), ('aux', small)):
    ref = rows.get((0.0, 256)) if label == 'main' else rows.get((0.0, 128))
    if ref is None: continue
    s0 = np.sqrt(ref['stat'] ** 2 + ref['s_win'] ** 2 + ref['s_hys'] ** 2)
    for D, L in sorted(sizes.items()):
        r = rows.get((D, L))
        if r is None: continue
        s = np.sqrt(r['stat'] ** 2 + r['s_win'] ** 2 + r['s_hys'] ** 2); comb = np.hypot(s, s0)
        Tr = r['T'] / np.sqrt(1 + D * D); sr = s / np.sqrt(1 + D * D); combr = np.hypot(sr, s0)
        z1 = (r['T'] - ref['T']) / comb; z2 = (Tr - ref['T']) / combr
        ph.append(dict(set=label, D=D, L=L, ref_L=ref['L'], dT=r['T'] - ref['T'], sig=comb, z=z1, T_rescaled=Tr, dT_rescaled=Tr - ref['T'], z_rescaled=z2))
        print('  %-4s D=%.1f L=%-3d vs D=0 L=%d: dT*=%+.4f +- %.4f (z=%+.1f) | T*/sqrt(1+D^2)=%.4f, diff %+.4f (z=%+.1f)' % (label, D, L, ref['L'], r['T'] - ref['T'], comb, z1, Tr, Tr - ref['T'], z2))
J['posthoc'] = ph
json.dump(J, open('data/task1_h1.json', 'w'), indent=1, default=float)
