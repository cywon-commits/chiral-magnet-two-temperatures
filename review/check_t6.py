"""Review-session independent check of task 6 (R-11) from the committed extracts.
T*: (a) U(T) 6th-order polynomial derivative peak, (b) fluctuation C = N var(e)/T^2 peak (quadratic, +-0.04).
T_L: 1/S(k) = r + c1 k^2 + c2 k^4 fitted on the 2D low-k window (no radial binning), windows 0.8-2.0 q;
     crossing from a smooth quadratic fit of c1/r vs T (not first sign change)."""
import numpy as np, json, glob, sys
E='data/task2_extract/'
def load(D,L):
    fs=sorted(glob.glob(E+'t2_small_D%s_L%d_s*.npz'%(D,L))); return [np.load(f) for f in fs]
def U_C(d):
    Dv=float(d['D']); L=int(d['L']); T=d['T']; w=np.array([-1,-1,-Dv,-Dv])
    U=d['E4_mean']@w; blk=d['E4_blk']@w; sU=blk.std(1,ddof=1)/np.sqrt(blk.shape[1])
    var=np.einsum('i,tij,j->t',w,d['E4_cov'],w); C=L*L*var/T**2
    return T,U,sU,C
def tstar_poly(T,U):
    c=np.polyfit(T-0.7,U,6); tt=np.linspace(T.min()+0.02,T.max()-0.02,2001); dc=np.polyval(np.polyder(c),tt-0.7); return tt[np.argmax(dc)]
def tstar_fluc(T,C,w=0.04):
    i=np.argmax(C); s=np.abs(T-T[i])<=w+1e-9; c=np.polyfit(T[s]-T[i],C[s],2); return T[i]-c[1]/(2*c[0])
def c1r(d,win):
    Dv=float(d['D']); L=int(d['L']); q=np.arctan(Dv); S=d['low_tr']; n=(S.shape[1]-1)//2
    k=2*np.pi*np.arange(-n,n+1)/L; kx,ky=np.meshgrid(k,k,indexing='ij'); k2=4*np.sin(kx/2)**2+4*np.sin(ky/2)**2
    m=(k2<=(win*q)**2)&(k2>0); out=[]
    for t in range(S.shape[0]):
        A=np.vstack([np.ones(m.sum()),k2[m],k2[m]**2]).T; y=1/S[t][m]
        c=np.linalg.lstsq(A,y,rcond=None)[0]; out.append(c[1]/c[0])
    return np.array(out),int(m.sum())
def cross_smooth(T,y,lo,hi):
    s=(T>=lo)&(T<=hi); c=np.polyfit(T[s],y[s],2); r=np.roots(c); r=r[np.isreal(r)].real; r=r[(r>lo-0.03)&(r<hi+0.03)]
    return float(r.min()) if len(r) else np.nan
def cross_first(T,y):
    o=np.argsort(T); T=T[o]; y=y[o]
    for i in range(len(T)-1):
        if y[i]*y[i+1]<0: return T[i]-y[i]*(T[i+1]-T[i])/(y[i+1]-y[i])
    return np.nan
res={}
for D,L in (('005',512),('005',1024),('008',384)):
    ds=load(D,L); r={}
    ts=[];tf=[]
    for d in ds:
        T,U,sU,C=U_C(d); ts.append(tstar_poly(T,U)); tf.append(tstar_fluc(T,C))
    Um=np.mean([U_C(d)[1] for d in ds],0); Cm=np.mean([U_C(d)[3] for d in ds],0)
    r['Tstar_poly']=tstar_poly(T,Um); r['Tstar_poly_seeds']=ts; r['Tstar_fluc']=tstar_fluc(T,Cm); r['Tstar_fluc_seeds']=tf
    r['Cmax']=float(Cm.max()); r['U_at_0.68']=float(np.interp(0.68,T[::-1],Um[::-1]))
    r['TL']={}
    for w in (0.8,1.0,1.2,1.5,2.0):
        ys=[c1r(d,w) for d in ds]; y=np.mean([a for a,_ in ys],0)
        f=cross_first(T,y); lo,hi=(0.56,0.70)
        r['TL']['%.1f'%w]=dict(npts=ys[0][1],first=float(f),smooth=cross_smooth(T,y,lo,hi),seeds=[cross_smooth(T,a,lo,hi) for a,_ in ys])
        if w==1.5: r['c1r_1.5']=dict(T=T.tolist(),y=y.tolist())
    res['D%s_L%d'%(D,L)]=r
    print(D,L,'T*poly %.4f'%r['Tstar_poly'],np.round(ts,4),'T*fluc %.4f'%r['Tstar_fluc'],np.round(tf,4),'Cmax %.3f'%r['Cmax'])
    for w,v in r['TL'].items(): print('   win',w,'n',v['npts'],'first %.4f smooth %.4f'%(v['first'],v['smooth']),np.round(v['seeds'],4))
json.dump(res,open('review/data/t6_check.json','w'),indent=1)
