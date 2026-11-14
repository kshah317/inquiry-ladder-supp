"""Reader-free evidence value (naive Bayes fitted on the DDXPlus validation split; scripts/55_nb_fit.py).

score(pid, asked) -> (top1_correct, p_truth): the posterior of the true condition given exactly what a team saw:
sex, age, the opening complaint, and the scripted answers to the asked questions (follow-ups included when revealed).
Uniform prior over the 49 conditions (the main sample has 6 patients per condition). No LLM involved.
"""
from __future__ import annotations
import ast
from functools import lru_cache
from pathlib import Path
import numpy as np
from .data import kb, answer, get_patient

PARAMS = Path(__file__).resolve().parents[2] / "results" / "analysis" / "_nb" / "nb_params.npz"
AGE_BINS = [0, 1, 5, 15, 30, 45, 60, 75, 200]


@lru_cache(maxsize=1)
def _model():
    z = np.load(PARAMS, allow_pickle=True)
    counts, n_c = z["counts"], z["n_c"]
    conds = list(z["conds"])
    cols = {ast.literal_eval(k): i for i, k in enumerate(z["keys"])}
    K = kb()
    n_codes = len(K.evidence)
    def denom(code):            # patients of each condition for whom this item is observable
        e = K.evidence[code]
        return counts[:, cols[("B", e.parent)]] if e.parent else n_c
    return counts, n_c, conds, cols, n_codes, denom


def _ll_answer(code: str, ans) -> np.ndarray:
    counts, n_c, conds, cols, n_codes, denom = _model()
    e = kb().evidence[code]
    d = denom(code)
    if e.dtype == "B":
        p = (counts[:, cols[("B", code)]] + 1) / (d + 2)
        return np.log(p if ans.value else 1 - p)
    if e.dtype == "C":
        return np.log((counts[:, cols[("C", code, ans.value)]] + 1) / (d + len(e.values)))
    out = np.zeros(len(conds))
    present = set(ans.value)
    for v in e.values:
        if v == "V_11":
            continue
        p = (counts[:, cols[("M", code, v)]] + 1) / (d + 2)
        out += np.log(p if v in present else 1 - p)
    return out


@lru_cache(maxsize=None)
def _opening_ll(pid: str) -> np.ndarray:
    counts, n_c, conds, cols, n_codes, _ = _model()
    split, row = pid.split(":")
    p = get_patient(split, int(row))
    b = int(np.searchsorted(AGE_BINS, p.age, side="right") - 1)
    ll = np.log((counts[:, cols[("sex", p.sex)]] + 1) / (n_c + 2))
    ll += np.log((counts[:, cols[("age", b)]] + 1) / (n_c + len(AGE_BINS) - 1))
    ll += np.log((counts[:, cols[("init", p.initial_evidence)]] + 1) / (n_c + n_codes))
    return ll


@lru_cache(maxsize=None)
def _code_ll(pid: str, code: str) -> np.ndarray:
    """Log-likelihood of what the patient says when `code` is asked (parent + revealed follow-ups)."""
    split, row = pid.split(":")
    p = get_patient(split, int(row))
    K = kb()
    first = answer(p, code)
    ll = np.zeros(len(_model()[2])) if code == p.initial_evidence else _ll_answer(code, first)
    if first.positive:
        for ch in K.children.get(code, []):
            ll = ll + _ll_answer(ch, answer(p, ch))
    return ll


def posterior(pid: str, asked) -> np.ndarray:
    ll = _opening_ll(pid).copy()
    for code in dict.fromkeys(c for c in asked if c):     # each question counted once, order irrelevant
        ll += _code_ll(pid, code)
    ll -= ll.max()
    w = np.exp(ll)
    return w / w.sum()


def score(pid: str, truth: str, asked) -> tuple[float, float]:
    post = posterior(pid, asked)
    conds = _model()[2]
    t = conds.index(truth)
    return float(post.argmax() == t), float(post[t])


def full_record_codes(pid: str) -> list[str]:
    split, row = pid.split(":")
    return list(kb().top_level)
