"""Exact Shapley values for a 4-agent panel from the scores of its 15 subgroups.

Shapley is linear in the value function, so we compute it per patient (value = 1 if the subgroup's
final answer is right, else 0) and average; this lets us bootstrap over patients.
v(empty) is a fixed baseline (chance, or the opening-only accuracy). It shifts every agent's value
by v(empty)/n, so it never changes the differences between agents.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np


def shapley(v: dict[frozenset, np.ndarray], n: int = 4) -> np.ndarray:
    """v maps every subset (frozenset of agent indices, including the empty set) to per-patient
    values (array of shape [n_patients]). Returns array [n, n_patients] of Shapley values."""
    players = range(n)
    out = np.zeros((n, len(next(iter(v.values())))))
    for i in players:
        others = [j for j in players if j != i]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                S = frozenset(S)
                out[i] += w * (v[S | {i}] - v[S])
    return out


def bootstrap_ci(per_patient: np.ndarray, stat=np.mean, B: int = 5000, seed: int = 0, axis=-1):
    """Percentile 95% CI by resampling patients (last axis)."""
    rng = np.random.default_rng(seed)
    n = per_patient.shape[axis]
    draws = np.stack([stat(np.take(per_patient, rng.integers(0, n, n), axis=axis), axis=axis) for _ in range(B)])
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def _selftest():
    # glove game sanity check: v=1 iff agent 0 and at least one of {1,2,3} present -> phi0 = 0.75, others 1/12
    n = 4
    v = {}
    for k in range(n + 1):
        for S in combinations(range(n), k):
            S = frozenset(S)
            v[S] = np.array([1.0 if (0 in S and len(S) > 1) else 0.0])
    phi = shapley(v)[:, 0]
    assert np.allclose(phi, [0.75, 1 / 12, 1 / 12, 1 / 12]), phi
    assert np.isclose(phi.sum(), 1.0)   # efficiency


if __name__ == "__main__":
    _selftest()
    print("shapley selftest ok")
