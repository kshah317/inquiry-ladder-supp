"""Fixed, stratified case samples.

  main  : test split, equal number of patients per condition (default 6 x 49 = 294).
  pilot : validation split, so tuning in the pilot never touches a main-run patient.
Equal allocation (not proportional) so rare but severe conditions (Ebola, Bronchiolitis)
and every specialty group are represented. Sampling is seeded and saved to CSV;
the CSV, not this code, is the source of truth for later runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ROOT, load_split

SAMPLES = ROOT / "data" / "samples"


def stratified(split: str, per_condition: int, seed: int) -> pd.DataFrame:
    df = load_split(split)
    rng = np.random.default_rng(seed)
    rows = []
    for cond in sorted(df.PATHOLOGY.unique()):
        idx = df.index[df.PATHOLOGY == cond].to_numpy()
        pick = rng.choice(idx, size=per_condition, replace=False)
        rows += sorted(int(i) for i in pick)
    out = df.loc[rows, ["AGE", "SEX", "PATHOLOGY", "INITIAL_EVIDENCE"]].copy()
    out.insert(0, "row", rows)
    out.insert(0, "split", split)
    return out.reset_index(drop=True)


def load_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLES / f"{name}.csv")
