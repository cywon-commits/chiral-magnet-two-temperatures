"""Extract compact, committed summaries from the task-2 raw run directories (data/raw/t2_<tag>_..._sN/T??_*.npz),
so that the raw data can be deleted afterwards (PI instruction 2026-11-22).  Output: data/task2_extract/<run>.npz (float32).

Per temperature (index t):
 S(k) from the S_ab tensor (400 measurements):
   rad_k               radial bin centres (bin 0.5 in units of 2pi/L) up to |k| = pi
   rad_tr[t], rad_aa[t,3], rad_reoff[t,3]     radial averages of tr S, S_xx/yy/zz, Re S_xy/xz/yz
   rad_chi_par[t], rad_chi_perp[t]            radial averages of khat.chi(k) and |khat x chi(k)|, chi = (Im S_yz, Im S_zx, Im S_xy)
                                              (antisymmetric = chiral part; Bloch helix: chi parallel to k)
   low_tr[t]           unaveraged tr S on |kx|,|ky| <= KC_LOW (fftshifted)  -> Landau-Brazovskii / OZ refits
   ang_tr[t, nphi]     ring-angle profile of tr S in the shell 0.6..1.5 nq (24 angle bins), D>0
   G_x[t], G_d[t]      real-space correlation <S_0.S_r> along x (avg over x,y) and along the diagonal (avg of both), r <= L/2
 configuration samples (20, float16 -> renormalised):
   Qtop[t, 20]         topological charge (Berg-Luscher, 2 triangles per plaquette)
   Qabs_density[t]     <|q|> per plaquette ; chi_loc[t] <|S_i.(S_{i+x} x S_{i+y})|>
   cos4_s[t,20]        ring <cos 4phi> per sample (anisotropy fluctuations, task 4)
 local moments (8 windows x 500 sweeps):
   for window lengths W = 500, 1000, 2000, 4000 (exact averages of consecutive windows) and block sizes b = 1,2,4,8,16,32:
   eta1[t, iW, ib] = < |block-average of m| >,  eta2[t, iW, ib] = < |block-average of Q| > (Frobenius, traceless),
   m2[t, iW, ib]   = < |m_block|^2 >;  hist_m1[t, iW, :] histogram of |m_i| (b=1), 50 bins on [0,1]
 per-sweep bond sums E4 (measurement part):
   E4_mean[t,4], E4_cov[t,4,4], E4_blk[t,20,4] (20 block means)  -> anisotropy and errors
   e_hist[t, 200], e_edges[t, 201]: histogram of e = -J(xx+xy) - D(cx+cy); e_bin_E4[t,200,4]: mean E4 per e-bin  -> single-histogram reweighting
   E4_eq[t, 20, 4]  equilibration part in 20 blocks (drift check)
Run: python3 src/extract_task2.py <tag> [run-dir ...]   (default: all completed runs of the tag)
"""
import numpy as np, glob, os, sys
KC_LOW = 24; NPHI = 24; BLOCKS = (1, 2, 4, 8, 16, 32); WINS = (1, 2, 4, 8)   # in units of 500 sweeps

def full_from_half(h, L):
    H = L // 2 + 1; jj = np.arange(H, L); mkx = (-np.arange(L)) % L
    f = np.empty((L, L), h.dtype); f[:, :H] = h; f[:, H:] = np.conj(h[mkx][:, L - jj]) if np.iscomplexobj(h) else h[mkx][:, L - jj]
    return f

def topo(S):
    """Berg-Luscher topological charge of one configuration (L, L, 3) and local chirality."""
    a = S; b = np.roll(S, -1, 0); c = np.roll(np.roll(S, -1, 0), -1, 1); d = np.roll(S, -1, 1)
    def omega(s1, s2, s3):
        num = np.einsum('...i,...i', s1, np.cross(s2, s3)); den = 1 + np.einsum('...i,...i', s1, s2) + np.einsum('...i,...i', s2, s3) + np.einsum('...i,...i', s3, s1)
        return 2 * np.arctan2(num, den)
    q = (omega(a, b, c) + omega(a, c, d)) / (4 * np.pi)
    chi = np.abs(np.einsum('...i,...i', a, np.cross(b, d)))
    return q.sum(), np.abs(q).mean(), chi.mean()

def extract(rdir):
    fs = sorted(glob.glob(rdir + '/T??_*.npz')); d0 = np.load(fs[0]); L = int(d0['L']); D = float(d0['D']); nq = float(d0['ring_nq'])
    kx = np.fft.fftfreq(L) * L; KX, KY = np.meshgrid(kx, kx, indexing='ij'); KR = np.sqrt(KX ** 2 + KY ** 2); PHI = np.arctan2(KY, KX)
    bw = 0.5; kb = np.floor(KR / bw).astype(int); nb = int((L / 2) / bw); m = kb < nb; cnt = np.bincount(kb[m], minlength=nb)
    radial = lambda A: (np.bincount(kb[m], A[m], minlength=nb) / np.maximum(cnt, 1)).astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        khx, khy = np.where(KR > 0, KX / KR, 0), np.where(KR > 0, KY / KR, 0)
    kc = min(KC_LOW, L // 2 - 1); cidx = np.r_[0:kc + 1, L - kc:L]
    ring = (KR > 0.6 * nq) & (KR < 1.5 * nq) if D > 0 else None
    out = {k: [] for k in ('T', 'rad_tr', 'rad_aa', 'rad_reoff', 'rad_chi_par', 'rad_chi_perp', 'low_tr', 'ang_tr', 'G_x', 'G_d', 'Qtop', 'Qabs_density',
                           'chi_loc', 'cos4_s', 'eta1', 'eta2', 'm2', 'hist_m1', 'E4_mean', 'E4_cov', 'E4_blk', 'e_hist', 'e_edges', 'e_bin_E4', 'E4_eq', 'acc')}
    for f in fs:
        d = np.load(f); T = float(d['T']); neq = int(d['neq'])
        Sd = [full_from_half(d['Sdiag'][c].astype(np.float64), L) for c in range(3)]
        So = [full_from_half(d['Soff'][c].astype(np.complex128), L) for c in range(3)]      # xy, xz, yz
        tr = Sd[0] + Sd[1] + Sd[2]
        chi = (So[2].imag, -So[1].imag, So[0].imag)                                          # (Im S_yz, Im S_zx, Im S_xy)
        par = khx * chi[0] + khy * chi[1]; perp = np.sqrt((khy * chi[2]) ** 2 + (khx * chi[2]) ** 2 + (khx * chi[1] - khy * chi[0]) ** 2)
        out['T'].append(T); out['rad_tr'].append(radial(tr)); out['rad_aa'].append(np.array([radial(s) for s in Sd]))
        out['rad_reoff'].append(np.array([radial(s.real) for s in So])); out['rad_chi_par'].append(radial(par)); out['rad_chi_perp'].append(radial(perp))
        out['low_tr'].append(np.fft.fftshift(tr[np.ix_(cidx, cidx)]).astype(np.float32))
        if ring is not None:
            pb = np.floor((PHI[ring] + np.pi) / (2 * np.pi) * NPHI).astype(int) % NPHI
            out['ang_tr'].append((np.bincount(pb, tr[ring], minlength=NPHI) / np.maximum(np.bincount(pb, minlength=NPHI), 1)).astype(np.float32))
        else: out['ang_tr'].append(np.zeros(NPHI, np.float32))
        G = np.fft.ifft2(tr).real                                                            # <S_0.S_r>, since tr S(k) = sum_r G(r) e^{-ikr}
        r = np.arange(L // 2 + 1); out['G_x'].append(((G[r, 0] + G[0, r]) / 2).astype(np.float32)); out['G_d'].append(((G[r, r] + G[r, (-r) % L]) / 2).astype(np.float32))
        # samples
        smp = d['samples'].astype(np.float32); smp /= np.linalg.norm(smp, axis=-1, keepdims=True)
        tq = np.array([topo(s) for s in smp]); out['Qtop'].append(tq[:, 0].astype(np.float32)); out['Qabs_density'].append(np.float32(tq[:, 1].mean())); out['chi_loc'].append(np.float32(tq[:, 2].mean()))
        if ring is not None:
            c4 = []
            for s in smp:
                F = np.fft.fft2(s, axes=(0, 1)); sk = (np.abs(F) ** 2).sum(-1); w = sk[ring]; c4.append((w * np.cos(4 * PHI[ring])).sum() / w.sum())
            out['cos4_s'].append(np.array(c4, np.float32))
        else: out['cos4_s'].append(np.full(len(smp), np.nan, np.float32))
        # local moments
        mo = d['moments'].astype(np.float32); nw = mo.shape[0]
        e1 = np.full((len(WINS), len(BLOCKS)), np.nan, np.float32); e2 = e1.copy(); mm2 = e1.copy(); hm = np.zeros((len(WINS), 50), np.float32)
        for iw, w in enumerate(WINS):
            if nw % w: continue
            mw = mo.reshape(nw // w, w, L, L, 8).mean(1)                                     # (nseg, L, L, 8)
            for ib, b in enumerate(BLOCKS):
                if L % b: continue
                mb = mw.reshape(mw.shape[0], L // b, b, L // b, b, 8).mean((2, 4))
                m_ = mb[..., :3]; q = mb[..., 3:]
                qn = np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + (q[..., 0] + q[..., 1]) ** 2 + 2 * (q[..., 2] ** 2 + q[..., 3] ** 2 + q[..., 4] ** 2))
                mn = np.linalg.norm(m_, axis=-1); e1[iw, ib] = mn.mean(); e2[iw, ib] = qn.mean(); mm2[iw, ib] = (mn ** 2).mean()
                if b == 1: hm[iw] = np.histogram(mn, bins=50, range=(0, 1))[0] / mn.size
        out['eta1'].append(e1); out['eta2'].append(e2); out['m2'].append(mm2); out['hist_m1'].append(hm)
        # bond sums
        E4 = d['E4']; Em = E4[neq:]; Eq = E4[:neq]
        out['E4_mean'].append(Em.mean(0)); out['E4_cov'].append(np.cov(Em.T)); n20 = len(Em) // 20 * 20
        out['E4_blk'].append(Em[:n20].reshape(20, -1, 4).mean(1)); out['E4_eq'].append(Eq[:len(Eq) // 20 * 20].reshape(20, -1, 4).mean(1))
        e = -(Em[:, 0] + Em[:, 1]) - D * (Em[:, 2] + Em[:, 3]); h, ed = np.histogram(e, bins=200); idx = np.clip(np.digitize(e, ed) - 1, 0, 199)
        out['e_hist'].append(h); out['e_edges'].append(ed); out['e_bin_E4'].append(np.array([np.bincount(idx, Em[:, c], minlength=200) / np.maximum(h, 1) for c in range(4)]).T)
        out['acc'].append(float(d['acc']))
    res = {k: np.array(v, np.float64 if k in ('T', 'e_edges', 'E4_mean', 'E4_cov', 'E4_blk', 'E4_eq', 'e_bin_E4') else np.float32) for k, v in out.items()}
    res.update(D=D, L=L, seed=int(d0['seed']), nq=nq, rad_k=((np.arange(nb) + 0.5) * bw * 2 * np.pi / L).astype(np.float32), kc_low=kc,
               blocks=np.array(BLOCKS), wins_sweeps=np.array(WINS) * 500, n_temps=len(fs))
    return res

if __name__ == '__main__':
    tag = sys.argv[1]; dirs = sys.argv[2:] or sorted(os.path.dirname(f) for f in glob.glob('data/raw/t2_%s_*/DONE' % tag))
    os.makedirs('data/task2_extract', exist_ok=True)
    for rd in dirs:
        name = os.path.basename(rd); fo = 'data/task2_extract/%s.npz' % name
        if os.path.exists(fo): print('exists', fo); continue
        r = extract(rd); np.savez_compressed(fo, **r); print('wrote', fo, '%.1f MB' % (os.path.getsize(fo) / 1e6), flush=True)
