import sys, itertools, math, csv; sys.path.insert(0,'results/analysis/verify')
from v_load import *
import numpy as np
from scipy import stats
G,_=load_all()
SC=json.load(open('prereg/confidence_scales.json'))
rng=np.random.default_rng(12345)
B=10000
R={}
def pct(model,c):
    L=SC[model]['final']; lt=sum(x<c for x in L); le=sum(x<=c for x in L); return (lt+le)/2/len(L)
def boot(d,lvl=0.95):
    d=np.asarray(d,float); n=len(d); idx=rng.integers(0,n,(B,n)); m=d[idx].mean(1); a=(1-lvl)/2
    return [float(np.quantile(m,a)),float(np.quantile(m,1-a))]
def mcn(x,y):  # x,y bool arrays; b=x right y wrong
    b=int(np.sum(x&~y)); c=int(np.sum(~x&y)); p=float(stats.binomtest(b,b+c,0.5).pvalue) if b+c else 1.0
    return {'x_right_y_wrong':b,'x_wrong_y_right':c,'p_exact':p}
oo={m:{(o:=json.loads(l))['pid']:o for l in open(f'results/main/opening_only/{m}.jsonl') if l.strip()} for m in ['gemma4_31b','gemma4_26b_moe','qwen38_27b','qwen36_27b']}
markers=collections.defaultdict(set)
for r in csv.DictReader(open('prereg/marker_questions_draft.csv')):
    if str(r['is_marker']).strip() in ('1','1.0','True'): markers[r['top_condition']].add(r['code'])
VEC={}
for p in PANELS:
    pids=sorted(G[p]['ABCD']); n=len(pids); out={'n':n}
    acc={s:np.array([G[p][s][q]['correct'] for q in pids]) for s in SUBS}
    out['acc']={s:float(acc[s].mean()) for s in SUBS}
    team=acc['ABCD']; solo={x:acc[x].mean() for x in 'ABCD'}; best=max(solo,key=solo.get)
    d=team.astype(float)-acc[best]; out['team_minus_best']={'best':best,'diff':float(d.mean()),'ci95':boot(d)}
    # independent vote
    iv=[]; nties=0; ncoin=0; nallnull=0
    for q in pids:
        sup=collections.defaultdict(list)
        for x in 'ABCD':
            g=G[p][x][q]; f=g['final'][0]
            if f['dx'] is None: continue
            sup[f['dx']].append(pct(g['agents'][0]['model'],f['confidence'] if f['confidence'] is not None else 0))
        if not sup: iv.append(False); nallnull+=1; continue
        mx=max(len(v) for v in sup.values()); cand=[k for k,v in sup.items() if len(v)==mx]
        if len(cand)>1:
            nties+=1; top=max(max(sup[k]) for k in cand); cand=[k for k in cand if max(sup[k])==top]
            if len(cand)>1: ncoin+=1
        win=cand[rng.integers(len(cand))] if len(cand)>1 else cand[0]
        iv.append(win==G[p]['ABCD'][q]['truth'])
    iv=np.array(iv); d=team.astype(float)-iv
    out['indep_vote']={'acc':float(iv.mean()),'team_minus_iv':float(d.mean()),'ci95':boot(d),'n_ties':nties,'n_coin':ncoin,'n_all_null':nallnull,'mcnemar_team_vs_iv':mcn(team,iv)}
    # Shapley via orderings
    def v(S,v0):
        k=''.join(sorted(S)); return v0 if not k else out['acc'][k]
    if p in('P0','P1'): v0s={'gemma4_31b_all294':np.mean([oo['gemma4_31b'][q]['correct'] for q in pids])}
    elif p=='CHECK': v0s={'gemma4_31b_check98':np.mean([oo['gemma4_31b'][q]['correct'] for q in pids]),'gemma4_31b_all294':np.mean([o['correct'] for o in oo['gemma4_31b'].values()])}
    elif p=='P1Q': v0s={'qwen38_27b':np.mean([oo['qwen38_27b'][q]['correct'] for q in pids])}
    else: v0s={'mean4':np.mean([np.mean([oo[m][q]['correct'] for q in pids]) for m in oo]), **{m:np.mean([oo[m][q]['correct'] for q in pids]) for m in oo}}
    sh={}
    for name,v0 in [('uniform_1/49',1/49)]+[('opening_'+k,float(val)) for k,val in v0s.items() if k in('gemma4_31b_all294','gemma4_31b_check98','qwen38_27b','mean4')]:
        phi={x:0.0 for x in 'ABCD'}
        for perm in itertools.permutations('ABCD'):
            S=set()
            for x in perm: phi[x]+=v(S|{x},v0)-v(S,v0); S|={x}
        phi={x:phi[x]/24 for x in phi}
        sh[name]={'v0':v0,'phi':phi,'sum':sum(phi.values()),'team_minus_v0':out['acc']['ABCD']-v0}
    out['opening_only_acc']={k:float(x) for k,x in v0s.items()}
    out['shapley']=sh
    out['loo']={x:out['acc']['ABCD']-out['acc'][''.join(c for c in 'ABCD' if c!=x)] for x in 'ABCD'}
    # error overlap on solo
    truth=np.array([G[p]['ABCD'][q]['truth'] for q in pids])
    dx={x:np.array([G[p][x][q]['final'][0]['dx'] for q in pids],dtype=object) for x in 'ABCD'}
    strata=[np.where(truth==t)[0] for t in np.unique(truth)]
    def ov(a,b):
        wa=np.array([u!=t for u,t in zip(a,truth)]); wb=np.array([u!=t for u,t in zip(b,truth)])
        bw=wa&wb; sm=bw&np.array([u is not None and u==w for u,w in zip(a,b)])
        return bw.mean(), sm.mean(), (sm.sum()/bw.sum() if bw.sum() else float('nan')), bw.sum(), sm.sum()
    NP=2000 if p!='CHECK' else 2000
    perms={x:[] for x in 'ABCD'}
    prng=np.random.default_rng(777)
    for _ in range(NP):
        for x in 'ABCD':
            idx=np.arange(n)
            for st in strata: idx[st]=prng.permutation(st)
            perms[x].append(dx[x][idx])
    pairs={}
    for a,b in itertools.combinations('ABCD',2):
        o=ov(dx[a],dx[b]); nul=np.array([ov(perms[a][k],perms[b][k]) for k in range(NP)])
        typ={'P0':'same_model','P1':'persona','P1Q':'persona','P2':('same_family' if a+b in('AB','CD') else 'cross_family')}.get(p,'n/a')
        pairs[a+b]={'type':typ,'both_wrong':o[0],'same_wrong':o[1],'p_same_given_bw':o[2],'nbw':int(o[3]),'nsame':int(o[4]),
            'null_both_wrong':float(nul[:,0].mean()),'null_same_wrong':float(nul[:,1].mean()),'null_p_same_given_bw':float(np.nanmean(nul[:,2])),
            'null_nbw_sum':nul[:,3],'null_nsame_sum':nul[:,4]}
    pooled={}
    for typ in sorted({v['type'] for v in pairs.values()}):
        ps=[v for v in pairs.values() if v['type']==typ]
        pooled[typ]={'n_pairs':len(ps),'both_wrong':np.mean([v['both_wrong'] for v in ps]),'same_wrong':np.mean([v['same_wrong'] for v in ps]),
          'p_same_given_bw':sum(v['nsame'] for v in ps)/sum(v['nbw'] for v in ps),
          'null_both_wrong':np.mean([v['null_both_wrong'] for v in ps]),'null_same_wrong':np.mean([v['null_same_wrong'] for v in ps]),
          'null_p_same_given_bw':float(np.mean(sum(v['null_nsame_sum'] for v in ps)/sum(v['null_nbw_sum'] for v in ps)))}
    for v_ in pairs.values(): del v_['null_nbw_sum'],v_['null_nsame_sum']
    out['overlap']={'pairs':pairs,'pooled':pooled}
    # deliberation
    rw=wr=sw=tot=0
    for q in pids:
        g=G[p]['ABCD'][q]
        for f,r in zip(g['final'],g['revised']):
            tot+=1; a=f['dx']==g['truth']; b=r['dx']==g['truth']
            rw+=a and not b; wr+=(not a) and b; sw+=f['dx']!=r['dx']
    pre=np.array([G[p]['ABCD'][q]['correct_pre'] for q in pids])
    out['deliberation']={'agent_answers':tot,'right_to_wrong':rw,'wrong_to_right':wr,'switch_rate':sw/tot,'team_pre':float(pre.mean()),'team_post':float(team.mean()),'mcnemar_post_vs_pre':mcn(team,pre)}
    # markers (approximation)
    def hit(g): return any(r['chosen'] in markers.get(g['truth'],()) and r['positive'] is True for r in g['rounds'])
    hs={s:float(np.mean([hit(G[p][s][q]) for q in pids])) for s in 'ABCD'}
    out['marker']={'ABCD':float(np.mean([hit(G[p]['ABCD'][q]) for q in pids])),'solo_mean':float(np.mean(list(hs.values()))),'solo':hs,'n_conditions_with_markers':len(markers),'n_marker_codes':sum(len(v) for v in markers.values())}
    # cost
    cA=[sum(u.get('cost') or 0 for u in G[p]['ABCD'][q]['usage']) for q in pids]
    cAll=[sum(sum(u.get('cost') or 0 for u in G[p][s][q]['usage']) for s in SUBS) for q in pids]
    nocost=sum(1 for s in SUBS for q in pids for u in G[p][s][q]['usage'] if u.get('cost') is None)
    out['cost']={'ABCD_mean':float(np.mean(cA)),'all15_mean_per_patient':float(np.mean(cAll)),'usage_entries_missing_cost':nocost}
    R[p]=out; VEC[p]={'team':team,'iv':iv,'pids':pids}
# primary contrasts
P0,P1,P2=VEC['P0'],VEC['P1'],VEC['P2']
assert P0['pids']==P1['pids']==P2['pids']
d=P1['team'].astype(float)-P0['team']; n=len(d); m=d.mean(); se=d.std(ddof=1)/math.sqrt(n)
pL=1-stats.t.cdf((m+0.05)/se,n-1); pU=stats.t.cdf((m-0.05)/se,n-1)
R['C1']={'diff':float(m),'ci90':boot(d,0.90),'ci95':boot(d),'tost_p':float(max(pL,pU)),'p_lower':float(pL),'p_upper':float(pU),'se':float(se),'mcnemar':mcn(P1['team'],P0['team'])}
d=P2['team'].astype(float)-P0['team']; R['C2a']={'diff':float(d.mean()),'ci95':boot(d),'mcnemar':mcn(P2['team'],P0['team'])}
d=P2['team'].astype(float)-P2['iv']; R['C2b']={'diff':float(d.mean()),'ci95':boot(d),'mcnemar':mcn(P2['team'],P2['iv'])}
def cv(o):
    if isinstance(o,dict): return {str(k):cv(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [cv(v) for v in o]
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,np.bool_): return bool(o)
    return o
json.dump(cv(R),open('results/analysis/verify/verify_results.json','w'),indent=1)
print('done')
