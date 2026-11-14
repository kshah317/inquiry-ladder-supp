"""Check 7 coin sensitivity: exact set of achievable (ge2, wrong) pairs over all coin outcomes."""
import sys; from pathlib import Path; sys.argv=[""]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import v_review as V
from collections import Counter
for pn in V.PANELS:
    base_w = base_g = 0; opts = []
    for p in V.PIDS:
        chs = [V.solo_choice(V.L[(pn, s)][p]) for s in V.SOLO]
        tw = V.tied_winners(chs); t = V.PAT[p]["truth"]
        sup = Counter(d for d, _, _ in chs if d)[tw[0]] if tw[0] else 0
        if len(tw) == 1:
            if tw[0] != t: base_w += 1; base_g += sup >= 2
        else:
            o = set()
            for w in tw: o.add((int(w != t and sup >= 2), int(w != t)))
            opts.append(o)
    S = {(base_g, base_w)}
    for o in opts: S = {(a + x, b + y) for a, b in S for x, y in o}
    print(pn, "coin patients", len(opts), "achievable (ge2, wrong):", sorted(S))
