"""Window dependence of T_L and kappa for all D with the review estimator (2D unbinned fit, smooth crossing)."""
import numpy as np, json, glob, sys
sys.path.insert(0,'review')
import importlib.util
spec=importlib.util.spec_from_file_location('c','review/check_t6.py')
src=open('review/check_t6.py').read().split('res={}')[0]; exec(src)
j=json.load(open('data/task6_judge.json')); xT=np.array(j['xiH']['T']); xx=np.array(j['xiH']['xi'])
o=np.argsort(xT); xi=lambda t: np.exp(np.interp(t,xT[o],np.log(xx[o])))
runs=[('small','005',1024,0.622),('small','005',512,0.62),('small','008',384,0.655),('main','010',512,0.68),('main','015',336,0.73),('main','020',256,0.77),('main','030',176,0.85),('main','040',128,0.95),('main','050',112,1.05)]
out={}
print('D     L    T_L(w=0.8,1.0,1.2,1.5,2.0)                 extrap(w->0)  kappa(1.5)  kappa(extrap)')
for kind,D,L,t0 in runs:
    ds=[np.load(f) for f in sorted(glob.glob(E+'t2_%s_D%s_L%d_s*.npz'%(kind,D,L)))]
    Dv=float(ds[0]['D']); lam=2*np.pi/np.arctan(Dv); T=ds[0]['T']; row=[]; W=(0.8,1.0,1.2,1.5,2.0)
    for w in W:
        y=np.mean([c1r(d,w)[0] for d in ds],0); row.append(cross_smooth(T,y,t0-0.08,t0+0.08))
    row=np.array(row); ok=np.isfinite(row[:4])
    p=np.polyfit(np.array(W[:4])[ok]**2,row[:4][ok],1); ex=p[1]   # T_L linear in window^2 (k^6 truncation), windows <= 1.5 q
    out[D+'_%d'%L]=dict(D=Dv,L=L,TL=row.tolist(),extrap=float(ex),kappa15=float(xi(row[3])/lam),kappa_ex=float(xi(ex)/lam))
    print('%.3f %4d '%(Dv,L),np.round(row,4),'  %.4f     %.3f      %.3f'%(ex,xi(row[3])/lam,xi(ex)/lam))
json.dump(out,open('review/data/t6_kappa_window.json','w'),indent=1)
