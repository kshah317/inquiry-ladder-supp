"""Statistics over patients: paired bootstrap CIs, exact McNemar, paired TOST.

Every test here is PAIRED: two arrays of per-patient values (1 = right, 0 = wrong, or any
per-patient score such as a Shapley value) measured on the same patients. Bootstrap draws
resample patients, and the same draw indices are reused for every statistic in one analysis,
so CIs of related quantities are computed on the same resamples.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

B_DEFAULT = 10_000
SEED_DEFAULT = 20261120


def boot_idx(n: int, B: int = B_DEFAULT, seed: int = SEED_DEFAULT) -> np.ndarray:
    """B x n matrix of patient indices (resampling with replacement)."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Bootstrap distribution of the mean of x (last axis = patients). Returns shape [..., B]."""
    x = np.asarray(x, float)
    return x[..., idx].mean(-1)


def ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)], axis=-1)
    return float(lo), float(hi)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact (binomial) McNemar for paired 0/1 outcomes. b10 = a right & b wrong; b01 = the reverse."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(_st.binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p": p}


def tost_paired(d: np.ndarray, margin: float, draws: np.ndarray | None = None) -> dict:
    """Equivalence test for a paired mean difference d (per patient), margin in the same units.
    Two versions: (1) the usual two one-sided t-tests (p = max of the two one-sided p values);
    (2) the bootstrap version: equivalent at alpha = .05 if the 90% percentile CI lies inside +/- margin."""
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    se = max(se, 1e-12)                                # all-equal pairs: avoid division by zero
    p_low = 1 - _st.t.cdf((m + margin) / se, n - 1)    # H0: mean <= -margin
    p_high = _st.t.cdf((m - margin) / se, n - 1)       # H0: mean >= +margin
    out = {"mean": float(m), "se": float(se), "margin": margin,
           "p_tost_t": float(max(p_low, p_high)),
           "ci90_t": (float(m - _st.t.ppf(0.95, n - 1) * se), float(m + _st.t.ppf(0.95, n - 1) * se))}
    if draws is not None:
        lo, hi = ci(draws, 0.90)
        out["ci90_boot"] = (lo, hi)
        out["equivalent_boot"] = bool(lo > -margin and hi < margin)
    out["equivalent_t"] = out["p_tost_t"] < 0.05
    return out


def paired_contrast(a: np.ndarray, b: np.ndarray, idx: np.ndarray, margin: float | None = None,
                    binary: bool = True) -> dict:
    """a - b on the same patients: mean, 95% and 90% bootstrap CIs, McNemar (if 0/1), TOST (if margin)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    dr = boot_means(d, idx)
    out = {"a": float(np.mean(a)), "b": float(np.mean(b)), "diff": float(d.mean()),
           "ci95": ci(dr, 0.95), "ci90": ci(dr, 0.90), "n": int(len(d))}
    if binary:
        out["mcnemar"] = mcnemar(a, b)
    if margin is not None:
        out["tost"] = tost_paired(d, margin, dr)
    return out


def mean_ci(x: np.ndarray, idx: np.ndarray, level: float = 0.95) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "ci95" if level == 0.95 else f"ci{int(level*100)}": ci(boot_means(x, idx), level)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values."""
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def _selftest():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 300)
    idx = boot_idx(300, 2000, 0)
    c = paired_contrast(a, a, idx, margin=0.05)
    assert c["diff"] == 0 and c["ci95"] == (0.0, 0.0) and c["mcnemar"]["p"] == 1.0
    # known McNemar: 15 vs 14 discordant -> p = 1.0
    x = np.r_[np.ones(15), np.zeros(14)]; y = 1 - x
    assert abs(mcnemar(x, y)["p"] - 1.0) < 1e-9
    # TOST: tiny mean difference with small se is equivalent
    d = np.r_[np.zeros(280), np.ones(10), -np.ones(10)]
    t = tost_paired(d, 0.05, boot_means(d, idx))
    assert t["equivalent_t"] and t["equivalent_boot"], t
    assert holm({"x": 0.01, "y": 0.04}) == {"x": 0.02, "y": 0.04}
    print("stats selftest ok")


if __name__ == "__main__":
    _selftest()
