"""Freeze each model's confidence scale from the PILOT (validation split) runs.

Why: confidence breaks ties in votes. Models report confidence on different scales
(pilot medians: Qwen3.8 45, Gemma 4 26B 50, Gemma 4 31B 60, Qwen3.6 70, Mistral 85), although
within each model higher confidence does go with being right (AUROC 0.71-0.80 for the P2 models).
Raw numbers would let the model that talks most confidently win every mixed-panel tie. So ties are
broken on each agent's percentile within its own model's pilot distribution. In a same-model panel
this gives exactly the same order as raw confidence, so the rule is identical across all panels.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = {}
for f in sorted((ROOT / "results" / "pilot").glob("*_generalist_r5_n*.jsonl")):
    model = f.name.split("_generalist")[0]
    if "_full" in f.name:
        continue
    prop, fin = [], []
    for line in f.open(encoding="utf-8"):
        r = json.loads(line)
        prop += [p["confidence"] for rd in r["rounds"] for p in rd["proposals"] if p["valid"]]
        fin += [x["confidence"] for x in r["final"] if x["dx"]]
    prev = out.get(model, {"propose": [], "final": []})
    if len(prop) + len(fin) >= len(prev["propose"]) + len(prev["final"]):
        out[model] = {"propose": sorted(prop), "final": sorted(fin)}
(ROOT / "prereg" / "confidence_scales.json").write_text(json.dumps(out))
print({m: (len(v["propose"]), len(v["final"])) for m, v in out.items()})
