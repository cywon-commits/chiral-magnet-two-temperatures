"""Aggregate data/raw/task1_runs.json (per run) into data/task1_summary.json (committed).

For every (proto, D, L, dirn) and every estimator: seed mean, statistical error = seed std/sqrt(n_seed).
Systematic error of an estimator = half range over peak-finder windows (fl, du).
Seed-averaged curves U(T), C_fl(T), C_du(T), C_mh(T) with seed-scatter errors are stored for figures.
"""
import json, numpy as np
from collections import defaultdict

runs = json.load(open('data/raw/task1_runs.json'))
g = defaultdict(list)
for r in runs: g[(r['proto'], r['D'], r['L'], r['dirn'])].append(r)

def ms(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0: return [np.nan, np.nan, 0]
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan, int(len(x))]

out = []
for (proto, D, L, dirn), rs in sorted(g.items()):
    lam = 2 * np.pi / np.arctan(D) if D > 0 else None
    e = dict(proto=proto, D=D, L=L, L_over_lambda=(L / lam if lam else None), dirn=dirn, seeds=[r['seed'] for r in rs])
    e['Tstar_mh'] = ms([r['Tstar_mh'] for r in rs])
    for est in ('fl', 'du'):
        per_w = {w: ms([r['Tstar_' + est][w] for r in rs]) for w in rs[0]['Tstar_' + est]}
        vals = [v[0] for v in per_w.values() if np.isfinite(v[0])]
        e['Tstar_' + est] = per_w['0.04'] + [float((max(vals) - min(vals)) / 2) if vals else np.nan]  # [mean, stat, n, syst_window]
        e['Tstar_%s_by_window' % est] = per_w
    e['Cmax_fl'] = ms([r['Cmax_fl'] for r in rs])
    e['ess_min'] = float(min(r['ess_min'] for r in rs))
    e['max_abs_drift_sigma'] = float(max(np.max(np.abs(r['drift_sigma'])) for r in rs))
    e['tau_mean'] = float(np.mean([np.mean(r['tau']) for r in rs]))
    for key, tkey in (('U', 'T'), ('Cfl', 'T'), ('Cdu', 'Tdu'), ('Cmh', 'Tmh_grid'), ('S0', 'T'), ('cos4', 'T'), ('e2', 'T'), ('xi_oz', 'T')):
        if key not in rs[0]: continue
        A = np.array([r[key] for r in rs]); e['curve_' + key] = dict(T=rs[0][tkey], mean=A.mean(0).tolist(),
                                                                     err=(A.std(0, ddof=1) / np.sqrt(len(rs))).tolist() if len(rs) > 1 else None)
    out.append(e)
json.dump(out, open('data/task1_summary.json', 'w'), indent=1)
print('%-7s %-5s %-4s %-6s %-22s %-28s %-28s %-6s %s' % ('proto', 'D', 'L', 'dir', 'T*_mh', 'T*_fl (stat, syst_w)', 'T*_du (stat, syst_w)', 'Cmax', 'n'))
for e in out:
    print('%-7s %-5.2f %-4d %-6s %.4f±%.4f        %.4f±%.4f (±%.4f)     %.4f±%.4f (±%.4f)     %.3f  %d' % (
        e['proto'], e['D'], e['L'], e['dirn'], e['Tstar_mh'][0], e['Tstar_mh'][1], e['Tstar_fl'][0], e['Tstar_fl'][1], e['Tstar_fl'][3],
        e['Tstar_du'][0], e['Tstar_du'][1], e['Tstar_du'][3], e['Cmax_fl'][0], e['Tstar_mh'][2]))
