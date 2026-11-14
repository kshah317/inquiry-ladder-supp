import json, os, collections
SUBS=['A','B','C','D','AB','AC','AD','BC','BD','CD','ABC','ABD','ACD','BCD','ABCD']
PANELS=['P0','P1','P1Q','P2','CHECK']
def load(path):
    d={}; dup=0; n=0
    for line in open(path):
        line=line.strip()
        if not line: continue
        o=json.loads(line); n+=1
        if o['pid'] in d: dup+=1
        d[o['pid']]=o
    return d,dup,n
def load_all():
    G={}; audit={}
    for p in PANELS:
        G[p]={}
        for s in SUBS:
            d,dup,n=load(f'results/main/{p}/{s}.jsonl'); G[p][s]=d
            audit[(p,s)]=(n,dup,len(d))
    return G,audit
