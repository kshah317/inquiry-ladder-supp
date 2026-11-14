"""Draw the fixed case samples (main = test split, pilot = validation split) and save them."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.sample import SAMPLES, stratified  # noqa: E402

cfg = yaml.safe_load((ROOT / "config" / "study.yaml").read_text())
SAMPLES.mkdir(parents=True, exist_ok=True)
for name, s in cfg["samples"].items():
    df = stratified(s["split"], s["per_condition"], cfg["seed"])
    # pilot: put one case per condition first, so "first 49 rows" is itself a balanced mini-pilot
    if name == "pilot":
        df["k"] = df.groupby("PATHOLOGY").cumcount()
        df = df.sort_values(["k", "PATHOLOGY"]).drop(columns="k").reset_index(drop=True)
    df.to_csv(SAMPLES / f"{name}.csv", index=False)
    print(name, len(df), "patients from", s["split"], "| ages", df.AGE.min(), "-", df.AGE.max(),
          "| female share", round((df.SEX == "F").mean(), 2))
