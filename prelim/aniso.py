import numpy as np, sys, time
from tstar import metropolis, overrelax, seed
D=float(sys.argv[1]); L=int(sys.argv[2]); J=1.0; seed(7); rng=np.random.default_rng(7)
S=rng.standard_normal((L,L,3)); S/=np.linalg.norm(S,axis=-1,keepdims=True)
kx=np.fft.fftfreq(L)*L; KX,KY=np.meshgrid(kx,kx,indexing='ij'); KR=np.sqrt(KX**2+KY**2); PHI=np.arctan2(KY,KX)
q=np.sqrt(2)*np.arctan(D/np.sqrt(2)); nq=L*q/(2*np.pi); ring=(KR>0.6*nq)&(KR<1.5*nq)
print('D=%.1f L=%d nq=%.2f  E_axis=%.5f E_diag=%.5f diff=%.2e (D^4/16=%.2e)'%(D,L,nq,-1-np.sqrt(1+D*D),-2*np.sqrt(1+D*D/2),-1-np.sqrt(1+D*D)+2*np.sqrt(1+D*D/2),D**4/16))
for T in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]:
    sig=min(1.5,1.2*np.sqrt(T))
    for _ in range(3000): metropolis(S,T,J,D,sig); overrelax(S,J,D); overrelax(S,J,D)
    Sk=np.zeros((L,L)); n=0
    for s in range(4000):
        metropolis(S,T,J,D,sig); overrelax(S,J,D); overrelax(S,J,D)
        if s%10==0: Sk+=(np.abs(np.fft.fft2(S,axes=(0,1)))**2).sum(-1)/(L*L); n+=1
    Sk/=n; w=Sk[ring]
    a4=(w*np.cos(4*PHI[ring])).sum()/w.sum(); a2=np.hypot((w*np.cos(2*PHI[ring])).sum(),(w*np.sin(2*PHI[ring])).sum())/w.sum()
    print('T=%.1f  <cos4phi>=%+.3f  |<e^{2i phi}>|=%.3f'%(T,a4,a2),flush=True)
