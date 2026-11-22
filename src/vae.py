"""VAE of Yoon et al., CPC 305, 109329 (2024), as specified in the paper, with the unspecified parts as explicit options.

Paper: input L x L x 3; encoder (Conv + LeakyReLU + MaxPool) x 4 -> FC 512 -> mu, sigma (latent 100 each);
decoder FC 512 -> (Upsample + Conv + LeakyReLU) x 4 + Conv; loss MSE + KL; Adam lr 1e-3; batch 300; 150 epochs; 80/20 split.
Not in the paper (our choices, recorded in every output): channels per conv block CH, kernel K (padding 'same'), LeakyReLU slope 0.2,
decoder FC 512 -> FC (C4 x L/16 x L/16) reshape, nearest-neighbour upsampling x2, linear output layer (no activation, so
|S~| is unconstrained), loss reduction: MSE summed over pixels and components + beta * KL summed over latent dims, averaged over
the batch (beta = 1; standard Gaussian-likelihood VAE), log-variance parametrisation of sigma.
Order parameter: eta = < |S~_i| > over sites and samples, S~ = decoder(mu) (deterministic) and, for comparison, decoder(z), z ~ q.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time, glob, os, argparse

class VAE(nn.Module):
    def __init__(self, L=128, ch=(16, 32, 64, 128), k=3, latent=100, fc=512, slope=0.2):
        super().__init__(); self.L, self.ch, self.slope = L, ch, slope; p = k // 2
        cin = 3; enc = []
        for c in ch: enc += [nn.Conv2d(cin, c, k, padding=p), nn.LeakyReLU(slope), nn.MaxPool2d(2)]; cin = c
        self.enc = nn.Sequential(*enc); self.s = L // 16; nflat = ch[-1] * self.s * self.s
        self.fc_e = nn.Linear(nflat, fc); self.mu = nn.Linear(fc, latent); self.lv = nn.Linear(fc, latent)
        self.fc_d1 = nn.Linear(latent, fc); self.fc_d2 = nn.Linear(fc, nflat)
        dec = []; chs = list(ch[::-1])
        for i in range(4):
            cout = chs[i + 1] if i < 3 else ch[0]
            dec += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(chs[i], cout, k, padding=p), nn.LeakyReLU(slope)]
        self.dec = nn.Sequential(*dec); self.out = nn.Conv2d(ch[0], 3, k, padding=p)
    def encode(self, x):
        h = F.leaky_relu(self.fc_e(self.enc(x).flatten(1)), self.slope); return self.mu(h), self.lv(h)
    def decode(self, z):
        h = F.leaky_relu(self.fc_d1(z), self.slope); h = F.leaky_relu(self.fc_d2(h), self.slope)
        return self.out(self.dec(h.view(-1, self.ch[-1], self.s, self.s)))
    def forward(self, x):
        mu, lv = self.encode(x); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv); return self.decode(z), mu, lv

def loss_fn(xr, x, mu, lv, beta=1.0):
    rec = ((xr - x) ** 2).sum((1, 2, 3)).mean(); kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
    return rec + beta * kl, rec, kl

def load_data(D, L, files=None):
    files = files or sorted(glob.glob('data/raw/t3_D%03d_L%d_c*.npz' % (round(D * 100), L)))
    Xs, Ts, Cs = [], [], []
    for f in files:
        d = np.load(f); s = d['samples']                     # (nT, nper, L, L, 3) float16
        Xs.append(s.reshape(-1, L, L, 3)); Ts.append(np.repeat(d['T'], s.shape[1])); Cs.append(np.full(s.shape[0] * s.shape[1], int(d['chain'])))
    X = np.concatenate(Xs); T = np.concatenate(Ts); C = np.concatenate(Cs)
    return X, T, C, files

def split(T, frac=0.8, seed=0):
    rng = np.random.default_rng(seed); tr = np.zeros(len(T), bool)
    for t in np.unique(T):
        idx = np.where(T == t)[0]; rng.shuffle(idx); tr[idx[:int(round(frac * len(idx)))]] = True
    return tr

def to_t(X): return torch.from_numpy(np.ascontiguousarray(X.transpose(0, 3, 1, 2)).astype(np.float32))

@torch.no_grad()
def evaluate(model, X, T, bs=500):
    model.eval(); eta_mu, eta_z, mu_all, lv_all = [], [], [], []
    for i in range(0, len(X), bs):
        x = to_t(X[i:i + bs]); mu, lv = model.encode(x)
        eta_mu.append(model.decode(mu).norm(dim=1).mean((1, 2)).numpy())
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv); eta_z.append(model.decode(z).norm(dim=1).mean((1, 2)).numpy())
        mu_all.append(mu.numpy()); lv_all.append(lv.numpy())
    eta_mu = np.concatenate(eta_mu); eta_z = np.concatenate(eta_z); mu_all = np.concatenate(mu_all); lv_all = np.concatenate(lv_all)
    Tu = np.unique(T)
    per_T = dict(T=Tu.tolist(),
                 eta_mu=[float(eta_mu[T == t].mean()) for t in Tu], eta_mu_std=[float(eta_mu[T == t].std()) for t in Tu],
                 eta_z=[float(eta_z[T == t].mean()) for t in Tu], eta_z_std=[float(eta_z[T == t].std()) for t in Tu],
                 kl_per_dim=[(-0.5 * (1 + lv_all[T == t] - mu_all[T == t] ** 2 - np.exp(lv_all[T == t]))).mean(0).tolist() for t in Tu])
    return per_T, eta_mu, eta_z, mu_all, lv_all

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=float, required=True); ap.add_argument('--L', type=int, default=128); ap.add_argument('--tag', required=True)
    ap.add_argument('--ch', default='16,32,64,128'); ap.add_argument('--k', type=int, default=3); ap.add_argument('--latent', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--bs', type=int, default=300); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--beta', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max_per_T', type=int, default=0, help='subsample (pilot)'); ap.add_argument('--out', default='data/raw/vae')
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, T, C, files = load_data(a.D, a.L)
    if a.max_per_T:
        rng = np.random.default_rng(a.seed); keep = np.concatenate([rng.permutation(np.where(T == t)[0])[:a.max_per_T] for t in np.unique(T)]); keep.sort()
        X, T, C = X[keep], T[keep], C[keep]
    tr = split(T, 0.8, a.seed); Xtr, Xte = X[tr], X[~tr]
    model = VAE(a.L, tuple(int(c) for c in a.ch.split(',')), a.k, a.latent); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True); base = '%s/%s' % (a.out, a.tag)
    cfg = vars(a) | dict(files=files, n_train=int(tr.sum()), n_test=int((~tr).sum()), n_params=sum(p.numel() for p in model.parameters()),
                         slope=0.2, upsample='nearest', output='linear', loss='sum-MSE + beta*KL, batch mean', sigma='log-variance')
    hist = []; t0 = time.time(); rng = np.random.default_rng(a.seed + 1)
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(len(Xtr)); s = np.zeros(3); nb = 0
        for i in range(0, len(perm), a.bs):
            x = to_t(Xtr[np.sort(perm[i:i + a.bs])]); xr, mu, lv = model(x); l, r, k = loss_fn(xr, x, mu, lv, a.beta)
            opt.zero_grad(); l.backward(); opt.step(); s += [l.item(), r.item(), k.item()]; nb += 1
        model.eval()
        with torch.no_grad():
            te = np.zeros(3); nt = 0
            for i in range(0, len(Xte), 500):
                x = to_t(Xte[i:i + 500]); xr, mu, lv = model(x); l, r, k = loss_fn(xr, x, mu, lv, a.beta); te += np.array([l.item(), r.item(), k.item()]) * len(x); nt += len(x)
        hist.append(dict(epoch=ep + 1, train=(s / nb).tolist(), test=(te / nt).tolist(), t=time.time() - t0))
        print('ep %3d  train loss %.1f (rec %.1f, KL %.1f)  test %.1f (rec %.1f, KL %.1f)  %.0fs' % (ep + 1, *(s / nb), *(te / nt), time.time() - t0), flush=True)
    per_T, eta_mu, eta_z, mu_all, lv_all = evaluate(model, Xte, T[~tr])
    per_T_all, *_ = evaluate(model, X, T)
    torch.save(model.state_dict(), base + '.pt')
    np.savez_compressed(base + '_eval.npz', T_test=T[~tr], chain_test=C[~tr], eta_mu=eta_mu, eta_z=eta_z, mu=mu_all.astype(np.float32), logvar=lv_all.astype(np.float32))
    json.dump(dict(config=cfg, history=hist, eta_test=per_T, eta_all=per_T_all, train_time_s=time.time() - t0), open(base + '.json', 'w'))
    print('done', base, '%.0fs' % (time.time() - t0))
