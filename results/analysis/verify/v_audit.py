import sys; sys.path.insert(0,'results/analysis/verify')
from v_load import *
G,audit=load_all()
for p in PANELS:
    base=set(G[p]['ABCD'])
    print(p, 'lines/dups/unique', {s:audit[(p,s)] for s in SUBS if audit[(p,s)][1] or audit[(p,s)][2]!=len(base)}, 'ABCD n',len(base))
    for s in SUBS:
        ps=set(G[p][s])
        if ps!=base: print('  pid mismatch',s,len(ps-base),len(base-ps))
    samp=collections.Counter(); pidmismatch=0; badc=0; badcpre=0; nulls=0; nullrev=0; sg=0; pnl=0; trmis=0; nfinal=0; rnd=collections.Counter()
    for s in SUBS:
        for pid,g in G[p][s].items():
            samp[(g.get('sample'))]+=1
            if g.get('subgroup')!=s: sg+=1
            if g.get('panel_id')!=p: pnl+=1
            if g['correct']!=(g['panel']==g['truth']): badc+=1
            if g['correct_pre']!=(g['panel_pre']==g['truth']): badcpre+=1
            if len(g['final'])!=len(s) or len(g['revised'])!=len(s): nfinal+=1
            nulls+=sum(f['dx'] is None for f in g['final']); nullrev+=sum(f['dx'] is None for f in g['revised'])
            rnd[len(g['rounds'])]+=1
    # truth consistency across subs
    for pid in base:
        ts={G[p][s][pid]['truth'] for s in SUBS if pid in G[p][s]}
        if len(ts)>1: trmis+=1
    print('  sample',dict(samp),'subgroup-mislabel',sg,'panel_id-mislabel',pnl,'correct!=(panel==truth)',badc,'correct_pre!=',badcpre,'final/revised len bad',nfinal,'null final dx',nulls,'null revised dx',nullrev,'rounds len',dict(rnd),'truth inconsistent pids',trmis)
    # solo: correct vs final dx / revised dx
    for s in 'ABCD':
        a=sum(g['correct']!=(g['final'][0]['dx']==g['truth']) for g in G[p][s].values())
        b=sum(g['correct']!=(g['revised'][0]['dx']==g['truth']) for g in G[p][s].values())
        c=sum(g['final'][0]['dx']!=g['revised'][0]['dx'] for g in G[p][s].values())
        print('  solo',s,'correct!=final==truth',a,'correct!=revised==truth',b,'final!=revised',c, 'models',set(g['agents'][0]['model'] for g in G[p][s].values()), 'roles', set(g['agents'][0]['role'] for g in G[p][s].values()))
    print('  vote types ABCD', collections.Counter(g['vote'] for g in G[p]['ABCD'].values()), collections.Counter(g['vote_pre'] for g in G[p]['ABCD'].values()))
    print('  n truths', len({g['truth'] for g in G[p]['ABCD'].values()}))
