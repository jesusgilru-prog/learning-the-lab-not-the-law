import json, time, numpy as np, pandas as pd
import audit_v2 as A
R='/home/jesus/paper8_hipergravedad/release_repo/data/processed_checkpoints/'
df=pd.read_csv(R+'cross_rotor_dataset_v3.csv'); sr=json.load(open(R+'class_sr_results.json'))
ly=np.log(df.Cp.values); lre=np.log(df.Re_Omega.values); M=df.M_tip.values
geom=df.geometry_id.values; fac=df.source.values
print('n',len(df),'facilities',dict(zip(*np.unique(fac,return_counts=True))))
print('published q (class_sr_results):', sr['global_exponents'])
g,gi=np.unique(geom,return_inverse=True)
s1=A.fit_s1(ly,lre,gi,len(g)); print('S1 q',s1['q'])
s5=A.fit_s5_free(ly,lre,M,gi,len(g)); print('S5',{k:s5[k] for k in ('q1','q2','m')},'BF',np.exp(s5['lap']-s1['lap']))
t=time.time(); out=A.run_audit(ly,lre,M,geom,fac,0.30,np.random.default_rng(1),n_perm=199,n_boot=200)
print({k:v for k,v in out.items() if k!='t1_grid'}); print('t1_grid',out.get('t1_grid')); print('secs',time.time()-t)
