"""Subsets of the main sample (no new patients, so their games are reused by the full runs).

  check.csv : 2 patients per condition (98), seeded - the CHECK positive-control panel runs here.
              CHECK agents A, B are P1's Cardiologist and Pulmonologist, so subgroups A, B, AB are shared with P1.
  trial.csv : 20 patients, one from each of 20 seeded-random conditions, all inside check.csv,
              so the trial run of P1/P2 is reused by both the main run and the CHECK panel.
Seed: study seed + 1 (check), + 2 (trial). Rows keep main.csv order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.llm import config  # noqa: E402

seed = config()["seed"]
main = pd.read_csv(ROOT / "data" / "samples" / "main.csv")
rng = np.random.default_rng(seed + 1)
keep = []
for cond, g in main.groupby("PATHOLOGY", sort=True):
    keep += sorted(rng.choice(g.index.to_numpy(), size=2, replace=False).tolist())
check = main.loc[sorted(keep)].reset_index(drop=True)
rng = np.random.default_rng(seed + 2)
conds = sorted(rng.choice(sorted(check.PATHOLOGY.unique()), size=20, replace=False).tolist())
trial = check[check.PATHOLOGY.isin(conds)].groupby("PATHOLOGY", sort=True).head(1).reset_index(drop=True)
assert len(check) == 98 and check.PATHOLOGY.value_counts().eq(2).all()
assert len(trial) == 20 and trial.PATHOLOGY.nunique() == 20
assert set(trial.row) <= set(check.row) <= set(main.row)
check.to_csv(ROOT / "data" / "samples" / "check.csv", index=False)
trial.to_csv(ROOT / "data" / "samples" / "trial.csv", index=False)
print("check:", len(check), "trial:", len(trial))
print(trial[["row", "PATHOLOGY", "INITIAL_EVIDENCE"]].to_string())
