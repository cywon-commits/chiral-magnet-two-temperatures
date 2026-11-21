import numpy as np, sys, json
rng = np.random.default_rng(1)

def make(L):
    ii, jj = np.indices((L, L))
    return [(ii + jj) % 2 == p for p in (0, 1)]

def field(S, J, D):
    # neighbours at +x,-x,+y,-y ; axis0 = x, axis1 = y
    B = np.zeros_like(S)
    for ax, sgn, r in ((0, -1, (1, 0, 0)), (0, 1, (-1, 0, 0)), (1, -1, (0, 1, 0)), (1, 1, (0, -1, 0))):
        Sj = np.roll(S, sgn, axis=ax)          # sgn=-1 -> neighbour at +direction
        B += J * Sj
        if D:
            B += D * np.cross(Sj, np.array(r, float))
    return B

def sweep(S, T, J, D, K, masks, sig):
    acc = 0
    for m in masks:
        B = field(S, J, D)
        new = S + sig * rng.standard_normal(S.shape)
        new /= np.linalg.norm(new, axis=-1, keepdims=True)
        dE = -np.sum((new - S) * B, -1) - K * (new[..., 2] ** 2 - S[..., 2] ** 2)
        ok = m & (rng.random(dE.shape) < np.exp(-np.clip(dE / T, -50, 50)))
        S[ok] = new[ok]
        acc += ok.sum()
    return acc / S.shape[0] / S.shape[1]

def energy(S, J, D, K):
    return float((-0.5 * np.sum(S * field(S, J, D), -1) - K * S[..., 2] ** 2).mean())

def run(L, J, D, K, Ts, S0, neq, nwin, nsamp, gap):
    """window-averaged local moments: m_i=<S_i>, Q_i=<S_i S_i>; several windows"""
    masks = make(L); out = []
    for T in Ts:
        S = S0.copy(); sig = min(1.0, 0.9 * np.sqrt(T))
        for _ in range(neq): sweep(S, T, J, D, K, masks, sig)
        e1 = []; e2 = []; U = []
        for w in range(nwin):
            m = np.zeros_like(S); Q = np.zeros(S.shape + (3,))
            for s in range(nsamp):
                for _ in range(gap): sweep(S, T, J, D, K, masks, sig)
                m += S; Q += S[..., :, None] * S[..., None, :]; U.append(energy(S, J, D, K))
            m /= nsamp; Q /= nsamp
            mm = np.linalg.norm(m, axis=-1); n = m / mm[..., None]
            p2 = 1.5 * np.einsum('...a,...ab,...b', n, Q, n) - 0.5
            e1.append(mm.mean()); e2.append(p2.mean())
        out.append(dict(T=float(T), eta1=float(np.mean(e1)), eta2=float(np.mean(e2)), U=float(np.mean(U))))
        print(out[-1], flush=True)
    return out

if __name__ == '__main__':
    case = sys.argv[1]
    L = 48
    Ts = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    if case == 'A':      # uniform easy-axis FM: J=1, K=0.2
        S0 = np.zeros((L, L, 3)); S0[..., 2] = 1
        res = run(L, 1.0, 0.0, 0.2, Ts, S0, 1500, 4, 100, 2)
    else:                # chiral maze: J=1, D=0.3 (+K)
        K = float(sys.argv[2]); J, D = 1.0, 0.3
        S = rng.standard_normal((L, L, 3)); S /= np.linalg.norm(S, axis=-1, keepdims=True)
        masks = make(L)
        for T in np.geomspace(1.5, 0.01, 60):
            for _ in range(150): sweep(S, T, J, D, K, masks, min(1.0, 0.9 * np.sqrt(T)))
        print('annealed U=', energy(S, J, D, K), flush=True)
        res = run(L, J, D, K, Ts, S, 1500, 4, 100, 2)
    json.dump(res, open(f'data/res_{"_".join(sys.argv[1:])}.json', 'w'))
