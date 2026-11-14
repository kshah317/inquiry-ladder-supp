"""Independent NB (reader-free evidence model) for the review checks. Own code; reads only data/raw."""
import json, re, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent / "_v_review_nb.pkl"
AGEB = [0, 1, 5, 15, 30, 45, 60, 75]
E = json.load(open(RAW / "release_evidences.json"))
CODES = list(E)
PARENT = {c: e["code_question"] for c, e in E.items() if e["code_question"] != c and e["code_question"] in E}
CHILDREN = {}
for c, p in PARENT.items(): CHILDREN.setdefault(p, []).append(c)
TOP = [c for c in CODES if c not in PARENT]
TOK = re.compile(r"'([^']*)'")

def agebin(a):
    b = 0
    for i, lo in enumerate(AGEB):
        if a >= lo: b = i
    return b

def parse(evs):
    """record: code -> True (B) / value (C) / set (M)"""
    rec = {}
    for t in TOK.findall(evs):
        if "_@_" in t:
            c, v = t.split("_@_")
            if E[c]["data_type"] == "M":
                rec.setdefault(c, set()).add(v)
            else:
                rec[c] = int(v) if re.fullmatch(r"-?\d+", v) else v
        else:
            rec[t] = True
    return rec

def answer(rec, c):
    e = E[c]; t = e["data_type"]
    if t == "B":
        v = rec.get(c) is True; return v, v
    if t == "C":
        v = rec.get(c, e["default_value"])
        return v, (v != e["default_value"] and v not in ("V_10", "V_11"))
    v = frozenset(rec.get(c, set())) - {"V_11"}
    return v, bool(v)

def fit():
    t0 = time.time()
    df = pd.read_csv(RAW / "release_validate_patients", usecols=["AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE"])
    conds = sorted(df.PATHOLOGY.unique()); ci = {c: i for i, c in enumerate(conds)}; C = len(conds)
    n_c = np.zeros(C); sex = {}; age = np.zeros((C, 8)); init = {}
    B = {c: np.zeros(C) for c in CODES if E[c]["data_type"] == "B"}
    Cat = {c: {} for c in CODES if E[c]["data_type"] == "C"}
    M = {c: {v: np.zeros(C) for v in E[c]["possible-values"] if v != "V_11"} for c in CODES if E[c]["data_type"] == "M"}
    den = {c: np.zeros(C) for c in PARENT}
    for a, s, pa, ev, ie in zip(df.AGE, df.SEX, df.PATHOLOGY, df.EVIDENCES, df.INITIAL_EVIDENCE):
        k = ci[pa]; n_c[k] += 1
        sex.setdefault(s, np.zeros(C))[k] += 1
        age[k, agebin(a)] += 1
        init.setdefault(ie, np.zeros(C))[k] += 1
        rec = parse(ev)
        for c in CODES:
            if c in PARENT:
                if rec.get(PARENT[c]) is not True: continue   # parents are all binary
                den[c][k] += 1
            t = E[c]["data_type"]
            if t == "B":
                if rec.get(c) is True: B[c][k] += 1
            elif t == "C":
                v = rec.get(c, E[c]["default_value"])
                Cat[c].setdefault(v, np.zeros(C))[k] += 1
            else:
                for v in rec.get(c, ()):
                    if v in M[c]: M[c][v][k] += 1
    pickle.dump(dict(conds=conds, n_c=n_c, sex=sex, age=age, init=init, B=B, Cat=Cat, M=M, den=den), open(OUT, "wb"))
    print("fit", len(df), "patients", C, "conds", round(time.time() - t0, 1), "s")

class NB:
    def __init__(self):
        d = pickle.load(open(OUT, "rb")); self.__dict__.update(d)
        self.C = len(self.conds); self.ci = {c: i for i, c in enumerate(self.conds)}
    def _d(self, c):
        return self.den[c] if c in PARENT else self.n_c
    def ll_item(self, c, val):
        t = E[c]["data_type"]; d = self._d(c); z = np.zeros(self.C)
        if t == "B":
            p = (self.B[c] + 1) / (d + 2); return np.log(p if val else 1 - p)
        if t == "C":
            return np.log((self.Cat[c].get(val, z) + 1) / (d + len(E[c]["possible-values"])))
        out = np.zeros(self.C)
        for v, cnt in self.M[c].items():
            p = (cnt + 1) / (d + 2); out += np.log(p if v in val else 1 - p)
        return out
    def opening(self, age, sex, ie):
        z = np.zeros(self.C)
        ll = np.log((self.sex.get(sex, z) + 1) / (self.n_c + 2))
        ll += np.log((self.age[:, agebin(age)] + 1) / (self.n_c + 8))
        ll += np.log((self.init.get(ie, z) + 1) / (self.n_c + len(CODES)))
        return ll
    def code_ll(self, rec, ie, c):
        val, pos = answer(rec, c)
        ll = np.zeros(self.C) if c == ie else self.ll_item(c, val)
        if pos:
            for ch in CHILDREN.get(c, []):
                ll = ll + self.ll_item(ch, answer(rec, ch)[0])
        return ll

if __name__ == "__main__":
    fit()
