"""The consultation game: N agents jointly gather information, then diagnose.

Per patient:
  start      agents see sex, age, opening complaint
  rounds     each agent proposes one question (+ top-3 and confidence); majority vote picks it;
             ties -> highest proposer confidence -> seeded coin. The script answers from the record.
  final      each agent names one condition from the list of 49
  deliberate (panels of 2+ only) each agent sees the others' finals + reasoning, may revise once
  panel      majority vote of revised diagnoses (same tie rule)

Invalid output (bad JSON, unknown or repeated question, diagnosis not in the list) gets one
retry with the error explained; if still invalid, that agent abstains for that step.
Seeds depend on the agent's identity, the patient and the step - never on which subgroup the
agent is in - so the same agent behaves the same way wherever randomness allows.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass

import httpx

from . import prompts
from .data import Patient, kb, reveal
from .llm import LLMError, call, content, parse_json
from .text import (code_for_qid, condition_display, qid_map, render_full_record,
                   render_opening, render_reveal)


@dataclass(frozen=True)
class Agent:
    name: str            # label shown to peers, e.g. "Cardiologist" or "Physician B"
    model: str           # key in config/study.yaml
    role: str            # cardiology | pulmonology | infectious | generalist
    seed_id: str         # stable identity for seeding (e.g. "P1.card")
    full_record: bool = False   # check panel only: sees the whole record up front
    replica: int = 0            # 0 = main run; r > 0 = independent repeat (fresh seeds, no cache reuse)

    @property
    def sid(self) -> str:
        """Seed identity. Replica 0 keeps the original seed_id, so all cached main-run calls stay valid;
        replicas append '#r<k>', so every request (and its cache key) is new."""
        return self.seed_id if self.replica == 0 else f"{self.seed_id}#r{self.replica}"


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- diagnosis parsing
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


_DX = None


def match_condition(text) -> str | None:
    """Map a model's diagnosis string to a dataset condition name (None if no match)."""
    global _DX
    if _DX is None:
        _DX = {}
        for name in kb().condition_names:
            _DX[_norm(name)] = name
            _DX[_norm(condition_display(name))] = name
    if not text:
        return None
    n = _norm(text)
    if n in _DX:
        return _DX[n]
    close = difflib.get_close_matches(n, list(_DX), n=1, cutoff=0.88)
    return _DX[close[0]] if close else None


# ----------------------------------------------------------------------------- one agent step
def _usage(resp: dict) -> dict:
    """Token usage + the facts needed to audit a call: provider that served it, and whether the
    output was cut off by the token limit (finish_reason == 'length')."""
    u = dict(resp.get("usage") or {})
    u["provider"] = resp.get("provider")
    u["finish_reason"] = (resp.get("choices") or [{}])[0].get("finish_reason")
    u["cached"] = bool(resp.get("_cached"))      # True = served from our cache, no new spend
    return u


async def _ask(client, sem, agent, messages, temperature, seed, validate):
    """Call once; on invalid output retry once with the reason. Returns (parsed|None, raw, usage list)."""
    usages = []
    resp = await call(client, sem, agent.model, messages, temperature=temperature, seed=seed)
    usages.append(_usage(resp) | {"agent": agent.name, "model": agent.model})
    raw = content(resp)
    obj = parse_json(raw)
    err = validate(obj)
    if err is None:
        return obj, raw, usages, 0
    retry = messages + [{"role": "assistant", "content": raw},
                        {"role": "user", "content": f"Your answer was invalid: {err}. Reply again with one valid JSON object only."}]
    resp = await call(client, sem, agent.model, retry, temperature=temperature, seed=seed + 1)
    usages.append(_usage(resp) | {"agent": agent.name, "model": agent.model})
    raw2 = content(resp)
    obj2 = parse_json(raw2)
    return (obj2 if validate(obj2) is None else None), raw2, usages, 1


_SCALES = None


def norm_conf(model: str, conf: float, kind: str) -> float:
    """Percentile of `conf` within this model's frozen pilot distribution (prereg/confidence_scales.json).
    kind = 'propose' or 'final'. Falls back to conf/100 for a model without pilot data."""
    global _SCALES
    if _SCALES is None:
        import json
        from .data import ROOT
        _SCALES = json.loads((ROOT / "prereg" / "confidence_scales.json").read_text())
    dist = (_SCALES.get(model) or {}).get(kind) or []
    if not dist:
        return conf / 100
    import bisect
    lo, hi = bisect.bisect_left(dist, conf), bisect.bisect_right(dist, conf)
    return (lo + hi) / 2 / len(dist)    # mid-rank percentile, ties handled


def _vote(choices: list[tuple[str, float]], rng: random.Random):
    return _vote_info(choices, rng)[0]


def _vote_info(choices: list[tuple[str, float]], rng: random.Random):
    """choices = [(option, confidence)] from valid agents. Majority, then max confidence, then coin.
    Returns (winner, how) with how in {None (no valid votes), 'single', 'majority', 'confidence', 'coin'}.
    'single' = only one valid vote (always the case for a one-agent team)."""
    if not choices:
        return None, None
    counts = Counter(c for c, _ in choices)
    top = max(counts.values())
    tied = sorted(c for c, k in counts.items() if k == top)
    if len(tied) == 1:
        return tied[0], ("single" if len(choices) == 1 else "majority")
    best_conf = {c: max(conf for o, conf in choices if o == c) for c in tied}
    m = max(best_conf.values())
    tied = sorted(c for c in tied if best_conf[c] == m)
    if len(tied) == 1:
        return tied[0], "confidence"
    return rng.choice(tied), "coin"


def _conf(obj) -> float:
    try:
        return float(obj.get("confidence", 0))
    except (TypeError, ValueError):
        return 0.0


# ----------------------------------------------------------------------------- the game
async def play(client: httpx.AsyncClient, sem: asyncio.Semaphore, p: Patient, agents: list[Agent],
               *, rounds: int, temperature: float, base_seed: int, deliberate: bool = True,
               question_rule: str = "plurality") -> dict:
    replica = {a.replica for a in agents}
    assert len(replica) == 1, "all agents in one game must share a replica id"
    replica = replica.pop()
    opening = render_opening(p)
    full = render_full_record(p) if any(a.full_record for a in agents) else None
    # The opening complaint counts as asked - unless it is a parent question (pain, skin, swelling),
    # whose details the panel can still ask for (19% of patients open with pain).
    asked = set() if kb().children.get(p.initial_evidence) else {p.initial_evidence}
    transcript: list[str] = []
    log = {"pid": p.pid, "truth": p.pathology, "agents": [asdict(a) for a in agents], "replica": replica,
           "rounds": [], "usage": [], "retries": 0, "retries_by_agent": {}}
    rng = random.Random(_seed(base_seed, p.pid, "tie") if replica == 0 else _seed(base_seed, p.pid, "tie", f"r{replica}"))

    assert question_rule in ("plurality", "rotate")
    rot_start = _seed(base_seed, p.pid, "rotate") % len(agents)
    if question_rule == "rotate":
        log["question_rule"], log["rotate_start"] = "rotate", rot_start

    def valid_q(obj):
        if obj is None:
            return "not valid JSON"
        qid = str(obj.get("question_id", "")).strip().upper()
        if qid not in code_for_qid():
            return f"question_id {qid!r} is not in the QUESTION LIST"
        if code_for_qid()[qid] in asked:
            return f"{qid} has already been asked"
        return None

    def valid_dx(obj):
        if obj is None:
            return "not valid JSON"
        if match_condition(obj.get("diagnosis")) is None:
            return f"diagnosis {obj.get('diagnosis')!r} is not in the CONDITION LIST"
        return None

    # ---- information-gathering rounds
    for k in range(1, rounds + 1):
        async def propose(a: Agent):
            msgs = prompts.propose_messages(a.role, opening, transcript, k, rounds, len(agents),
                                            full if a.full_record else None)
            return await _ask(client, sem, a, msgs, temperature, _seed(base_seed, a.sid, p.pid, "q", k), valid_q)
        res = await asyncio.gather(*(propose(a) for a in agents))
        props = []
        for a, (obj, raw, us, rt) in zip(agents, res):
            log["usage"] += us
            log["retries"] += rt
            log["retries_by_agent"][a.name] = log["retries_by_agent"].get(a.name, 0) + rt
            qid = str(obj["question_id"]).strip().upper() if obj else None
            props.append({"agent": a.name, "qid": qid, "code": code_for_qid().get(qid) if qid else None,
                          "top3": [match_condition(x) for x in (obj or {}).get("top3", [])][:3] if obj else None,
                          "confidence": _conf(obj) if obj else None,
                          "reasoning": (obj or {}).get("reasoning"), "valid": obj is not None})
        if question_rule == "rotate":
            # members take turns. The proposer of round k is fixed by a seeded start and the
            # round number, and if its proposal is invalid the turn passes to the next member in order.
            order = [(rot_start + k - 1 + j) % len(agents) for j in range(len(agents))]
            pick = next((i for i in order if props[i]["valid"]), None)
            chosen, how = (props[pick]["code"], "rotate") if pick is not None else (None, None)
        else:
            pick = None
            chosen, how = _vote_info([(x["code"], norm_conf(a.model, x["confidence"], "propose"))
                                      for a, x in zip(agents, props) if x["valid"]], rng)
        entry = {"k": k, "proposals": props, "chosen": chosen, "vote": how}
        if pick is not None:
            entry["proposer"] = agents[pick].name
        if chosen:
            asked.add(chosen)
            ans = reveal(p, chosen)
            entry["positive"] = ans[0].positive
            transcript.append(f"Round {k} ({qid_map()[chosen]}): " + render_reveal(ans))
        log["rounds"].append(entry)

    # ---- final diagnoses
    async def final(a: Agent):
        msgs = prompts.final_messages(a.role, opening, transcript, rounds, full if a.full_record else None)
        obj, raw, us, rt = await _ask(client, sem, a, msgs, temperature, _seed(base_seed, a.sid, p.pid, "final"), valid_dx)
        return msgs, obj, raw, us, rt
    fin = await asyncio.gather(*(final(a) for a in agents))
    finals = []
    for a, (msgs, obj, raw, us, rt) in zip(agents, fin):
        log["usage"] += us
        log["retries"] += rt
        log["retries_by_agent"][a.name] = log["retries_by_agent"].get(a.name, 0) + rt
        finals.append(_dx_record(a, obj))
    log["final"] = finals
    log["panel_pre"], log["vote_pre"] = _vote_info([(f["dx"], norm_conf(a.model, f["confidence"], "final"))
                                                    for a, f in zip(agents, finals) if f["dx"]], rng)

    # ---- one round of deliberation (panels of 2+)
    if deliberate and len(agents) > 1:
        async def revise(i, a):
            others = [f for j, f in enumerate(finals) if j != i]
            peers = "\n".join(
                f"- {f['agent']}: {condition_display(f['dx']) if f['dx'] else 'no valid answer'}"
                f" (confidence {f['confidence']:.0f}). {f['reasoning'] or ''}".rstrip() for f in others)
            msgs = prompts.delib_messages(fin[i][0], fin[i][2], peers)
            return await _ask(client, sem, a, msgs, temperature, _seed(base_seed, a.sid, p.pid, "delib"), valid_dx)
        rev = await asyncio.gather(*(revise(i, a) for i, a in enumerate(agents)))
        revised = []
        for a, f, (obj, raw, us, rt) in zip(agents, finals, rev):
            log["usage"] += us
            log["retries"] += rt
            log["retries_by_agent"][a.name] = log["retries_by_agent"].get(a.name, 0) + rt
            r = _dx_record(a, obj)
            if r["dx"] is None:            # invalid revision -> keep the pre-deliberation answer
                r = dict(f, kept_after_invalid=True)
            revised.append(r)
        log["revised"] = revised
        log["panel"], log["vote"] = _vote_info([(f["dx"], norm_conf(a.model, f["confidence"], "final"))
                                                for a, f in zip(agents, revised) if f["dx"]], rng)
    else:
        log["revised"] = finals
        log["panel"], log["vote"] = log["panel_pre"], log["vote_pre"]

    log["correct"] = log["panel"] == p.pathology
    log["correct_pre"] = log["panel_pre"] == p.pathology
    log["asked"] = [r["chosen"] for r in log["rounds"]]
    return log


def _dx_record(a: Agent, obj) -> dict:
    if obj is None:
        return {"agent": a.name, "dx": None, "top3": [], "confidence": 0.0, "reasoning": None}
    top3 = [match_condition(x) for x in (obj.get("top3") or [])][:3]
    return {"agent": a.name, "dx": match_condition(obj.get("diagnosis")), "top3": top3,
            "confidence": _conf(obj), "reasoning": obj.get("reasoning")}
