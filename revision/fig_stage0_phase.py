import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=pd.read_csv("stage0_phase_trials.csv"); S=pd.read_csv("stage0_threshold_sensitivity.csv")
R["rec"]=R.rel_err<0.5
kb=[-np.inf,1,np.log10(30),2,3,np.inf]; kl=["<10","10–30","30–100","100–10³",">10³"]
cb=[0,0.9,0.99,0.999,1.0001]; cl=["<0.9","0.9–0.99","0.99–0.999",">0.999"]
R["lk"]=pd.cut(np.log10(R.kappa.replace(np.inf,1e12)),kb,labels=kl)
R["lc"]=pd.cut(R.cmax,cb,labels=cl)
P=R.pivot_table(index="lk",columns="lc",values="rec",aggfunc="mean",observed=False)
N=R.pivot_table(index="lk",columns="lc",values="rec",aggfunc="size",observed=False)
fig,ax=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw=dict(width_ratios=[1.1,1]))
im=ax[0].imshow(P.values,cmap="viridis",vmin=0,vmax=1,aspect="auto")
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        n=N.values[i,j]
        if n>0: ax[0].text(j,i,f"{P.values[i,j]:.2f}\n(n={int(n)})",ha="center",va="center",fontsize=8,color="w" if P.values[i,j]<0.6 else "k")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(cl); ax[0].set_yticks(range(5)); ax[0].set_yticklabels(kl)
ax[0].set_xlabel("max pairwise |r| (raw log-groups)"); ax[0].set_ylabel("κ(Z), nuisance-adjusted")
ax[0].axhline(1.5,color="r",lw=2,ls="--"); ax[0].axvline(1.5,color="orange",lw=2,ls=":")
ax[0].set_title("(a) Recovery rate of θ_Ma (error < 50%)")
plt.colorbar(im,ax=ax[0],fraction=0.046)
for c,m in zip([0.2,0.3,0.5,0.7],["o","s","^","D"]):
    sub=S[(S.c==c)&S.rule.str.startswith("Stage0")].copy()
    sub["k"]=sub.rule.str.extract(r"<=(\d+)").astype(int)
    sub=sub.sort_values("k"); ax[1].plot(sub.k,sub.accuracy,marker=m,label=f"Stage 0, c={c}")
    pw=S[(S.c==c)&(S.rule=="pairwise |r|<0.99")].accuracy.iloc[0]
    ax[1].axhline(pw,ls=":",lw=1,color=ax[1].lines[-1].get_color())
ax[1].axvline(30,color="r",ls="--",lw=1); ax[1].set_xscale("log"); ax[1].set_xticks([10,20,30,50,100]); ax[1].set_xticklabels([10,20,30,50,100])
ax[1].set_xlabel("κ threshold"); ax[1].set_ylabel("accuracy of the screen"); ax[1].set_ylim(0.45,0.9)
ax[1].set_title("(b) Threshold sensitivity (dotted: pairwise |r|<0.99)"); ax[1].legend(fontsize=8,loc="lower right")
plt.tight_layout(); plt.savefig("../manuscript_R1/figures/stage0_phase_map.png",dpi=200)
print("ok")
