"""Run manifest: everything needed to audit or reproduce one panel run, rebuilt from the game logs.

Written to results/<out>/<panel>/MANIFEST.json after every runner call (so it is always current).
Retry, repair and abstain rules are identical for every agent and panel (game.py); the manifest
reports how often each fired, per agent and per subgroup size, so differences are visible.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import prompts
from .data import ROOT
from .llm import config


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def fingerprints() -> dict:
    src = sorted((ROOT / "src" / "ddx").glob("*.py"))
    roles = "\n".join(prompts.role_block(r) for r in sorted(prompts.ROLES))
    return {
        "config_sha256": _sha((ROOT / "config" / "study.yaml").read_bytes()),
        "system_prompt_sha256": _sha(prompts.system_prompt().encode()),
        "role_blocks_sha256": _sha(roles.encode()),
        "code_sha256": _sha(b"".join(f.read_bytes() for f in src)),
        "prereg_sha256": {f.name: _sha(f.read_bytes()) for f in sorted((ROOT / "prereg").glob("*")) if f.is_file()},
    }


def _rate(a, b):
    return round(a / b, 4) if b else None


def build(run_dir: Path, agents, sample_name: str, extra: dict | None = None) -> dict:
    cfg = config()
    recs = []
    for f in sorted(run_dir.glob("[A-D]*.jsonl")):
        recs += [json.loads(line) for line in f.open(encoding="utf-8")]
    m = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
         "python": platform.python_version(), "run_dir": str(run_dir.relative_to(ROOT)),
         "sample": sample_name, **fingerprints(), "seed": cfg["seed"], "game": cfg["game"]}
    m["agents"] = []
    for i, a in enumerate(agents):
        mc = cfg["models"][a.model]
        m["agents"].append({"slot": "ABCD"[i], "name": a.name, "model_key": a.model, "openrouter_id": mc["id"],
                            "provider_pin": mc["provider"], "price_per_M": mc["price"],
                            "reasoning": mc.get("reasoning", "disabled"), "role": a.role,
                            "seed_id": a.seed_id, "replica": a.replica, "full_record": a.full_record})
    m["games"] = {"total": len(recs), "patients": len({r["pid"] for r in recs}),
                  "per_subgroup": dict(sorted(Counter(r["subgroup"] for r in recs).items(), key=lambda t: (len(t[0]), t[0])))}
    ts = [r["ts"] for r in recs if r.get("ts")]
    m["dates"] = {"first_game": min(ts) if ts else None, "last_game": max(ts) if ts else None}

    # calls, tokens, cost, providers, truncation - per model
    per_model = defaultdict(lambda: Counter())
    providers = defaultdict(Counter)
    finish = defaultdict(Counter)
    for r in recs:
        for u in r["usage"]:
            mk = u.get("model", "?")
            c = per_model[mk]
            fresh = not u.get("cached")
            c["calls"] += 1
            c["calls_fresh"] += fresh
            c["tokens_in"] += u.get("prompt_tokens") or 0
            c["tokens_out"] += u.get("completion_tokens") or 0
            c["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            c["cost_all_usd"] += u.get("cost") or 0
            c["cost_fresh_usd"] += (u.get("cost") or 0) if fresh else 0
            providers[mk][u.get("provider")] += 1
            finish[mk][u.get("finish_reason")] += 1
    m["usage_by_model"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in per_model.items()}
    m["providers_served"] = {k: dict(v) for k, v in providers.items()}
    m["finish_reasons"] = {k: dict(v) for k, v in finish.items()}
    tot = sum((v for v in per_model.values()), Counter())
    m["totals"] = {"calls": tot["calls"], "calls_fresh": tot["calls_fresh"],
                   "cost_fresh_usd": round(tot["cost_fresh_usd"], 4),
                   "cost_if_uncached_usd": round(tot["cost_all_usd"], 4),
                   "cost_per_game_usd": _rate(tot["cost_all_usd"], len(recs)),
                   "truncated_calls": sum(v.get("length", 0) for v in finish.values())}

    # validity, retries, abstentions - per agent
    ag = defaultdict(Counter)
    for r in recs:
        for name, k in (r.get("retries_by_agent") or {}).items():
            ag[name]["retries"] += k
        for rd in r["rounds"]:
            for x in rd["proposals"]:
                ag[x["agent"]]["proposals"] += 1
                ag[x["agent"]]["invalid_proposals"] += not x["valid"]
        for f in r["final"]:
            ag[f["agent"]]["finals"] += 1
            ag[f["agent"]]["abstain_final"] += f["dx"] is None
        if len(r["agents"]) > 1:
            for f in r["revised"]:
                ag[f["agent"]]["revisions"] += 1
                ag[f["agent"]]["invalid_revision_kept_old"] += bool(f.get("kept_after_invalid"))
    m["agent_validity"] = {n: {**c, "invalid_proposal_rate": _rate(c["invalid_proposals"], c["proposals"]),
                               "abstain_rate": _rate(c["abstain_final"], c["finals"]),
                               "retries_per_game": _rate(c["retries"], c["finals"])}
                           for n, c in ag.items()}

    # vote resolution (ties) per subgroup size
    ties = defaultdict(Counter)
    for r in recs:
        size = len(r["agents"])
        for rd in r["rounds"]:
            ties[size][f"question:{rd.get('vote')}"] += 1
        ties[size][f"final:{r.get('vote')}"] += 1
        ties[size]["rounds_no_question"] += sum(rd.get("chosen") is None for rd in r["rounds"])
    m["vote_resolution_by_size"] = {str(k): dict(v) for k, v in sorted(ties.items())}
    m["accuracy_by_subgroup"] = {sg: _rate(sum(r["correct"] for r in recs if r["subgroup"] == sg), n)
                                 for sg, n in m["games"]["per_subgroup"].items()}
    ff = run_dir / "failures.log"
    m["failed_game_attempts"] = sum(1 for _ in ff.open(encoding="utf-8")) if ff.exists() else 0
    if extra:
        m["last_invocation"] = extra
    (run_dir / "MANIFEST.json").write_text(json.dumps(m, indent=1, default=str))
    return m
