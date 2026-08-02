import os
import pandas as pd, numpy as np
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_checkpoints", "cross_rotor_dataset_v3.csv"))
liu = df[df.source=='Liu2024']

print("== 1. Pares con Re casi identico por caminos distintos (falsacion sin modelo) ==")
L=liu[['omega_rad_s','p_Pa','Re_Omega','Cp','M_tip','data_origin']].reset_index(drop=True)
found=[]
for i in range(len(L)):
    for j in range(i+1,len(L)):
        a,b=L.loc[i],L.loc[j]
        if a.omega_rad_s==b.omega_rad_s: continue
        dRe=abs(a.Re_Omega/b.Re_Omega-1)
        if dRe<0.05:
            dCp=a.Cp/b.Cp-1
            found.append((a.p_Pa,a.omega_rad_s,a.Cp,b.p_Pa,b.omega_rad_s,b.Cp,dRe,dCp,a.data_origin,b.data_origin))
for f in sorted(found,key=lambda x:x[6]):
    print(f"  ({f[0]/1000:.0f}kPa, {f[1]:.1f} rad/s) Cp={f[2]:.5f}  vs  ({f[3]/1000:.0f}kPa, {f[4]:.1f} rad/s) Cp={f[5]:.5f}"
          f"   dRe={f[6]*100:.1f}%  dCp={f[7]*100:+.1f}%   [{f[8][:9]}/{f[9][:9]}]")

print()
print("== 2. Numero de Knudsen a cada presion (aire, T=293.15K, longitud caracteristica = gap 0.15 m) ==")
kB=1.380649e-23; d=3.7e-10; T=293.15; Lc=0.15
for p in sorted(liu.p_Pa.unique()):
    lam=kB*T/(np.sqrt(2)*np.pi*d**2*p)
    print(f"   p={p/1000:6.1f} kPa   camino libre medio={lam:.3e} m   Kn={lam/Lc:.3e}")

print()
print("== 3. El monomio Omega^2 R / g en las fuentes etiquetadas 'g=1' ==")
for s,g in df.groupby('source'):
    mono=g.omega_rad_s**2*g.R_m/9.80665
    print(f"   {s:12s} g_level tabulado=[{g.g_level.min():.0f}, {g.g_level.max():.0f}]   "
          f"Omega^2R/g REAL=[{mono.min():.0f}, {mono.max():.0f}]")

print()
print("== 4. Bug T=288 vs 293.15 en la densidad de Liu ==")
for p in sorted(liu.p_Pa.unique()):
    r_tab=liu[liu.p_Pa==p].rho_kgm3.iloc[0]
    r288=p/(287.05*288.0); r293=p/(287.05*293.15)
    print(f"   p={p/1000:6.1f} kPa  rho tabulado={r_tab:.6f}  rho(288K)={r288:.6f}  rho(293.15K)={r293:.6f}"
          f"   -> tabulado usa {'288K' if abs(r_tab-r288)<abs(r_tab-r293) else '293.15K'}")
