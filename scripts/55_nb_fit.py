"""Reader-free evidence value: fit a naive-Bayes model of DDXPlus on the VALIDATION split (no LLM, no API).

A robustness check: the replay's evidence value depends on one LLM reader that is
also a team member. This model scores any transcript (opening + asked questions + scripted answers) by the posterior
of the true condition. Fitted on the validation split only, so it never sees a test (main-run) patient.

Likelihood factors per condition c (Laplace smoothing, alpha = 1):
  sex | c ; age bin | c ; opening complaint (initial evidence) = x | c ;
  binary item present | c ; categorical item value | c ; multi-choice: each value present | c (independent).
  Follow-up items are counted only among patients whose parent item is present (they are only revealed then).
Output: results/analysis/_nb/nb_params.npz
"""
from __future__ import annotations
import ast, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddx.data import kb, load_split, parse_evidences

OUT = Path(__file__).resolve().parents[1] / "results" / "analysis" / "_nb"
AGE_BINS = [0, 1, 5, 15, 30, 45, 60, 75, 200]


def age_bin(a: int) -> int:
    return int(np.searchsorted(AGE_BINS, a, side="right") - 1)


def feature_index():
    """Map every observable outcome to a column. Returns (cols dict, n_cols)."""
    K = kb()
    cols = {}
    def add(key):
        cols[key] = len(cols)
    for s in ("M", "F"):
        add(("sex", s))
    for b in range(len(AGE_BINS) - 1):
        add(("age", b))
    for code in K.evidence:
        add(("init", code))
    for code, e in K.evidence.items():
        if e.dtype == "B":
            add(("B", code))
        elif e.dtype == "C":
            for v in e.values:
                add(("C", code, v))
        else:
            for v in e.values:
                if v != "V_11":
                    add(("M", code, v))
    return cols


def main():
    t0 = time.time()
    K = kb()
    conds = K.condition_names
    ci = {c: i for i, c in enumerate(conds)}
    cols = feature_index()
    df = load_split("validate")
    n_c = np.zeros(len(conds))
    counts = np.zeros((len(conds), len(cols)))
    for r in df.itertuples(index=False):
        c = ci[r.PATHOLOGY]
        n_c[c] += 1
        rec = parse_evidences(ast.literal_eval(r.EVIDENCES), K)
        counts[c, cols[("sex", r.SEX)]] += 1
        counts[c, cols[("age", age_bin(int(r.AGE)))]] += 1
        counts[c, cols[("init", r.INITIAL_EVIDENCE)]] += 1
        for code, e in K.evidence.items():
            if e.parent is not None and e.parent not in rec:
                continue                      # follow-ups are only revealed when the parent is "yes"
            if e.dtype == "B":
                if code in rec:
                    counts[c, cols[("B", code)]] += 1
            elif e.dtype == "C":
                v = rec.get(code, e.default)
                counts[c, cols[("C", code, v)]] += 1
            else:
                for v in rec.get(code, set()):
                    if v != "V_11":
                        counts[c, cols[("M", code, v)]] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    keys = np.array([repr(k) for k in cols], dtype=object)
    np.savez(OUT / "nb_params.npz", counts=counts, n_c=n_c, conds=np.array(conds, dtype=object), keys=keys)
    print(f"fitted on {int(n_c.sum())} validation patients, {len(cols)} columns, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
