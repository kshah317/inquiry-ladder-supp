import sys, itertools; sys.path.insert(0,'results/analysis/verify')
from v_load import *
import numpy as np
G,_=load_all()
rng=np.random.default_rng(2027); B=10000
def boot(d):
    d=np.asarray(d,float); m=d[rng.integers(0,len(d),(B,len(d)))].mean(1); return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]
R={}
for p in PANELS:
    E={}; odd={}
    for s in SUBS:
        d,dup,n=load(f'results/replay/{p}/{s}.jsonl'); E[s]=d
        mp=set(G[p][s]); rp=set(d)
        odd[s]={'lines':n,'dups':dup,'missing_vs_main':len(mp-rp),'extra':len(rp-mp),
          'null_dx':sum(g['dx'] is None for g in d.values()),
          'correct_ne_dx_eq_truth':sum(g['correct']!=(g['dx']==g['truth']) for g in d.values()),
          'truth_mismatch':sum(g['truth']!=G[p][s][q]['truth'] for q,g in d.items() if q in G[p][s]),
          'asked_mismatch_main':sum(list(g['asked'])!=list(G[p][s][q]['asked']) for q,g in d.items() if q in G[p][s]),
          'asked_ne_rounds_chosen':sum(list(G[p][s][q]['asked'])!=[r['chosen'] for r in G[p][s][q]['rounds']] for q in G[p][s]),
          'subgroup_label_bad':sum(g.get('subgroup')!=s or g.get('panel_id')!=p for g in d.values())}
    pids=sorted(G[p]['ABCD'])
    ev={s:np.array([E[s][q]['correct'] for q in pids]) for s in SUBS}
    e={s:float(ev[s].mean()) for s in SUBS}
    best=max('ABCD',key=lambda x:e[x])
    d1=ev['ABCD'].astype(float)-ev[best]
    v=np.array([G[p]['ABCD'][q]['correct'] for q in pids]); d2=v.astype(float)-ev['ABCD']
    phi={x:0.0 for x in 'ABCD'}
    val=lambda S: 1/49 if not S else e[''.join(sorted(S))]
    for perm in itertools.permutations('ABCD'):
        S=set()
        for x in perm: phi[x]+=val(S|{x})-val(S); S|={x}
    phi={x:phi[x]/24 for x in phi}
    agree=None
    ag={'P0':'A','P1':'D','P2':'A'}.get(p)
    if ag:
        a=t=0; per={}
        for s in SUBS:
            if ag not in s: continue
            i=s.index(ag); aa=sum(E[s][q]['dx']==G[p][s][q]['final'][i]['dx'] for q in pids); per[s]=aa; a+=aa; t+=len(pids)
        agree={'agent':ag,'agree':a,'total':t,'rate':a/t,'by_sub':per}
    nd=[len({tuple(G[p][s][q]['asked']) for s in SUBS}) for q in pids]
    ndr=[len({tuple(E[s][q]['asked']) for s in SUBS}) for q in pids]
    R[p]={'e':e,'best_solo':best,'eABCD_minus_best':float(d1.mean()),'ci1':boot(d1),'v_ABCD':float(v.mean()),'v_minus_e':float(d2.mean()),'ci2':boot(d2),
          'shapley_e':phi,'shap_sum':sum(phi.values()),'eABCD_minus_v0':e['ABCD']-1/49,'agree':agree,'distinct_asked_mean':float(np.mean(nd)),'distinct_asked_replay_mean':float(np.mean(ndr)),
          'distinct_hist':dict(collections.Counter(nd)),'odd':{s:o for s,o in odd.items() if any(v for k,v in o.items() if k!='lines') or o['lines']!=len(pids)}}
json.dump(R,open('results/analysis/verify/verify_replay.json','w'),indent=1,default=float)
for p,o in R.items():
    print('==',p,{k:round(o['e'][k],4) for k in ['A','B','C','D','ABCD']},'best',o['best_solo'],round(o['eABCD_minus_best'],4),[round(x,4) for x in o['ci1']],'v',round(o['v_ABCD'],4),'v-e',round(o['v_minus_e'],4),[round(x,4) for x in o['ci2']])
    print('  shap',{k:round(x,4) for k,x in o['shapley_e'].items()},'sum',round(o['shap_sum'],4),round(o['eABCD_minus_v0'],4))
    print('  agree',o['agree']); print('  distinct',o['distinct_asked_mean'],o['distinct_asked_replay_mean'],o['distinct_hist']); print('  odd',o['odd'])
