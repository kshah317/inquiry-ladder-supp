"""QUICK PREVIEW (unreviewed) of the planned contrasts. Superseded by the analysis module once built.
Run: python3 scripts/90_preview_primary.py"""
import sys, json, random, numpy as np
sys.path.insert(0, "src")
from ddx.game import _vote_info, norm_conf, _seed
def load(P, sg): return {json.loads(l)["pid"]: json.loads(l) for l in open(f"results/main/{P}/{sg}.jsonl", encoding="utf-8")}
def boot(d, B=3000, seed=0, alpha=0.05):
    r=np.random.default_rng(seed); n=len(d); m=np.array([d[r.integers(0,n,n)].mean() for _ in range(B)])
    return np.percentile(m,[100*alpha/2,100*(1-alpha/2)])
out={}
for P in ["P0","P1"]:
    full=load(P,"ABCD"); solos=[load(P,s) for s in "ABCD"]
    pids=sorted(full)
    team=np.array([full[p]["correct"] for p in pids],float)
    solo=np.array([[s[p]["correct"] for p in pids] for s in solos],float)
    best=solo.mean(1).argmax()
    # independent vote of the 4 solo games (same tie rule)
    iv=[]
    for p in pids:
        ch=[(s[p]["final"][0]["dx"], norm_conf(s[p]["agents"][0]["model"], s[p]["final"][0]["confidence"], "final")) for s in solos if s[p]["final"][0]["dx"]]
        w,_=_vote_info(ch, random.Random(_seed(20261120,p,"indep")))
        iv.append(w==full[p]["truth"])
    iv=np.array(iv,float)
    out[P]=team
    print(f"{P}: team {team.mean():.3f} {np.round(boot(team),3)} | solos {np.round(solo.mean(1),3)} | best-member synergy {team.mean()-solo[best].mean():+.3f} {np.round(boot(team-solo[best]),3)} | indep vote {iv.mean():.3f}, team-vote {team.mean()-iv.mean():+.3f} {np.round(boot(team-iv),3)}")
d=out["P1"]-out["P0"]
print(f"C1 P1-P0: {d.mean():+.3f}, 90% CI {np.round(boot(d,alpha=0.10),3)} (TOST margin +/-0.05), 95% CI {np.round(boot(d),3)}")
b=((out['P1']==1)&(out['P0']==0)).sum(); c=((out['P1']==0)&(out['P0']==1)).sum()
from scipy.stats import binomtest
print("McNemar discordant", b, c, "p=%.3f"%binomtest(b,b+c).pvalue)
from ddx.shapley import shapley
from ddx.panels import subgroups, sg_name
for P in ["P2","P1Q","CHECK"]:
    full=load(P,"ABCD"); solos=[load(P,s) for s in "ABCD"]; pids=sorted(full)
    team=np.array([full[p]["correct"] for p in pids],float)
    solo=np.array([[s[p]["correct"] for p in pids] for s in solos],float); best=solo.mean(1).argmax()
    iv=[]
    for p in pids:
        ch=[(s[p]["final"][0]["dx"], norm_conf(s[p]["agents"][0]["model"], s[p]["final"][0]["confidence"], "final")) for s in solos if s[p]["final"][0]["dx"]]
        w,_=_vote_info(ch, random.Random(_seed(20261120,p,"indep"))); iv.append(w==full[p]["truth"])
    iv=np.array(iv,float)
    L={sg:load(P,sg_name(sg)) for sg in subgroups()}
    v={frozenset(sg):np.array([float(L[sg][p]["correct"]) for p in pids]) for sg in subgroups()}
    v[frozenset()]=np.full(len(pids),1/49)
    phi=shapley(v).mean(1)
    print(f"{P}: team {team.mean():.3f} | solos {np.round(solo.mean(1),3)} | team-best {team.mean()-solo[best].mean():+.3f} {np.round(boot(team-solo[best]),3)} | indep {iv.mean():.3f} team-indep {team.mean()-iv.mean():+.3f} {np.round(boot(team-iv),3)} | Shapley {np.round(phi,3)}")
    if P=="P2":
        d=team-out["P0"]; print(f"C2a P2-P0 {d.mean():+.3f} 95% {np.round(boot(d),3)}; C2b P2-indep {team.mean()-iv.mean():+.3f}")
