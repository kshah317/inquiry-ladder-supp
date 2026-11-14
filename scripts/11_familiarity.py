"""Familiarity checks (pilot): does a model know DDXPlus itself, beyond medicine?

Probe 1 (recall): name the conditions in DDXPlus without seeing the list -> overlap with the real 49.
Probe 2 (odd markers): 8 questions whose "yes" points to one condition only because of how the
synthetic knowledge base was built (no clinical logic, e.g. Asian descent -> Bronchiectasis).
A model that picks the DDXPlus condition far above chance (1/49 ~ 2%) is likely remembering the dataset.
5 samples per probe, temperature 0.7. Opening-only accuracy (check a) comes from 10_pilot.py --rounds 0.
"""
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddx.game import match_condition  # noqa: E402
from ddx.data import kb  # noqa: E402
from ddx.llm import call, content, parse_json  # noqa: E402
from ddx.text import question_text, render_condition_list  # noqa: E402

ODD = {"E_17": "Bronchiectasis", "E_163": "Sarcoidosis", "E_195": "Bronchiolitis", "E_183": "Bronchiolitis",
       "E_185": "Panic attack", "E_99": "Panic attack", "E_11": "SLE", "E_141": "SLE"}
MODELS = sys.argv[1:] or ["gemma4_31b", "gemma3_27b", "qwen38_27b", "qwen36_27b", "gemma4_26b_moe"]
K = 5


async def main():
    sem = asyncio.Semaphore(24)
    out = {}
    names = {n.lower() for n in kb().condition_names}
    async with httpx.AsyncClient() as c:
        for m in MODELS:
            recall_msgs = [{"role": "user", "content":
                            "DDXPlus is a public dataset of synthetic patients for differential diagnosis (NeurIPS 2022). "
                            "List as many of the conditions (pathologies) it contains as you can. "
                            'Return JSON: {"conditions": ["...", "..."]}'}]
            rec = await asyncio.gather(*(call(c, sem, m, recall_msgs, temperature=0.7, seed=s, max_tokens=1200) for s in range(K)))
            overlaps = []
            for r in rec:
                obj = parse_json(content(r)) or {}
                got = {match_condition(x) for x in obj.get("conditions", []) if isinstance(x, str)} - {None}
                overlaps.append(len(got))
            hits = 0
            jobs = []
            for code, target in ODD.items():
                msg = [{"role": "user", "content":
                        f"A patient has exactly one of these conditions:\n{render_condition_list()}\n\n"
                        f'Asked "{question_text(code)}", the patient answered Yes. Nothing else is known. '
                        'Which single condition is most likely? Return JSON: {"diagnosis": "..."}'}]
                jobs += [(target, call(c, sem, m, msg, temperature=0.7, seed=s, max_tokens=200)) for s in range(K)]
            res = await asyncio.gather(*(j for _, j in jobs))
            for (target, _), r in zip(jobs, res):
                hits += match_condition((parse_json(content(r)) or {}).get("diagnosis")) == target
            out[m] = {"recall_overlap_mean_of_49": sum(overlaps) / K, "odd_marker_hits": f"{hits}/{len(jobs)}",
                      "odd_marker_rate": round(hits / len(jobs), 3), "chance": round(1 / 49, 3)}
            print(m, out[m], flush=True)
    (ROOT / "results" / "pilot" / "familiarity.json").write_text(json.dumps(out, indent=1))

asyncio.run(main())
