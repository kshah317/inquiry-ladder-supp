"""Measures computed from the game logs (results/main/<panel>/<subgroup>.jsonl). No model calls.

All measures are per patient first (arrays over the panel's patients, sorted by pid), so the
statistics in stats.py can resample patients. Agents are indexed 0-3 = A-D as in panels.py.

  load_panel           slim, cached copy of the logs
  shapley_values       exact Shapley from the 15 re-executed sub-teams, v(empty) = chance or opening-only
  loo                  leave-one-out: v(N) - v(N minus i)
  indep_vote           majority vote of the members' independent solo games (same tie rule as the game)
  error_overlap        pairs' shared errors vs a within-condition permutation null
  deliberation         answer switches in the deliberation round
  persona_check        do role prompts change which questions are asked / which diagnoses are given?
  marker_*             did the panel ask (and get a yes to) a marker question of the true condition?
  coin_vote            final vote recomputed with a pure seeded-coin tie-break (sensitivity)
  masking_shapley      Shapley from the full-panel game only, by masking votes (no re-execution)
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, get_patient, reveal
from .game import _seed, _vote_info, norm_conf
from .panels import PAIR_TYPE, subgroups, sg_name
from .shapley import shapley

RESULTS = ROOT / "results" / "main"
SLIM = ROOT / "results" / "analysis" / "_slim"
BASE_SEED = 20261120
CHANCE = 1 / 49
SUBGROUPS = subgroups()                       # 15 tuples of agent indices
FULL = (0, 1, 2, 3)
ROLE_FIELD = {"cardiology": "Cardiology", "pulmonology": "Pulmonology", "infectious": "Infectious disease"}


# ----------------------------------------------------------------------------- loading
def _slim(r: dict) -> dict:
    us = r.get("usage", [])
    return {
        "pid": r["pid"], "truth": r["truth"],
        "models": [a["model"] for a in r["agents"]], "roles": [a["role"] for a in r["agents"]],
        "rounds": [{"chosen": e.get("chosen"), "positive": e.get("positive"), "vote": e.get("vote"),
                    "props": [(x["code"] if x["valid"] else None) for x in e["proposals"]]} for e in r["rounds"]],
        "final": [(f["dx"], f["confidence"]) for f in r["final"]],
        "revised": [(f["dx"], f["confidence"]) for f in r["revised"]],
        "panel_pre": r["panel_pre"], "panel": r["panel"], "vote": r["vote"],
        "correct": bool(r["correct"]), "correct_pre": bool(r["correct_pre"]),
        "cost": sum((u.get("cost") or 0) for u in us), "calls": len(us),
        "tokens_in": sum((u.get("prompt_tokens") or 0) for u in us),
        "tokens_out": sum((u.get("completion_tokens") or 0) for u in us),
        "retries": r.get("retries", 0),
        "truncated": sum(u.get("finish_reason") == "length" for u in us),
    }


class Panel:
    def __init__(self, name: str, games: dict[str, dict[str, dict]]):
        self.name = name
        self.games = games                                   # sg name -> pid -> slim record
        self.pids = sorted(games["ABCD"])
        for sg in SUBGROUPS:
            assert sorted(games[sg_name(sg)]) == self.pids, (name, sg_name(sg), "patient sets differ")
        g0 = games["ABCD"][self.pids[0]]
        self.models, self.roles = g0["models"], g0["roles"]
        self.truth = np.array([games["ABCD"][p]["truth"] for p in self.pids])
        self.n = len(self.pids)

    def g(self, sg) -> list[dict]:
        key = sg if isinstance(sg, str) else sg_name(sg)
        return [self.games[key][p] for p in self.pids]

    def correct(self, sg, pre: bool = False) -> np.ndarray:
        return np.array([x["correct_pre" if pre else "correct"] for x in self.g(sg)], float)

    def v(self, v0: float) -> dict[frozenset, np.ndarray]:
        out = {frozenset(sg): self.correct(sg) for sg in SUBGROUPS}
        out[frozenset()] = np.full(self.n, v0)
        return out

    def solo_dx(self, i: int) -> list[tuple[str | None, float]]:
        return [x["final"][0] for x in self.g((i,))]

    def subset(self, mask: np.ndarray) -> "Panel":
        keep = {p for p, m in zip(self.pids, mask) if m}
        return Panel(self.name, {sg: {p: r for p, r in d.items() if p in keep} for sg, d in self.games.items()})


def load_panel(name: str, root: Path = RESULTS) -> Panel:
    """Slim copy cached in results/analysis/_slim/<panel>.json, rebuilt when any log is newer."""
    src = root / name
    files = {sg_name(sg): src / f"{sg_name(sg)}.jsonl" for sg in SUBGROUPS}
    cache = SLIM / f"{root.name}_{name}.json"
    newest = max(f.stat().st_mtime for f in files.values())
    if cache.exists() and cache.stat().st_mtime > newest:
        games = json.loads(cache.read_text(encoding="utf-8"))
    else:
        games = {}
        for sg, f in files.items():
            d = {}
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                d[r["pid"]] = _slim(r)          # a re-run game (after a failure) replaces the earlier line
            games[sg] = d
        SLIM.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(games), encoding="utf-8")
    for d in games.values():
        for r in d.values():
            r["final"] = [tuple(x) for x in r["final"]]
            r["revised"] = [tuple(x) for x in r["revised"]]
    return Panel(name, games)


def opening_only(model: str) -> float:
    f = RESULTS / "opening_only" / f"{model}.jsonl"
    y = [json.loads(line)["correct"] for line in f.open(encoding="utf-8")]
    return float(np.mean(y))


def v0_opening(panel: Panel) -> float:
    """Primary baseline (b): base model's opening-only accuracy; P2 = mean of its four members.
    CHECK has no planned (b); we use Gemma 4 31B's (its base model)."""
    if panel.name == "P2":
        return float(np.mean([opening_only(m) for m in panel.models]))
    base = {"P0": "gemma4_31b", "P1": "gemma4_31b", "P1Q": "qwen38_27b", "CHECK": "gemma4_31b"}[panel.name]
    return opening_only(base)


# ----------------------------------------------------------------------------- credit
def shapley_values(panel: Panel, v0: float) -> np.ndarray:
    """[4, n_patients] per-patient Shapley values."""
    return shapley(panel.v(v0))


def loo(panel: Panel) -> np.ndarray:
    full = panel.correct(FULL)
    return np.stack([full - panel.correct(tuple(j for j in FULL if j != i)) for i in FULL])


def masking_shapley(panel: Panel, v0: float = CHANCE) -> np.ndarray:
    """Shapley WITHOUT re-execution: every coalition's answer = vote of its members' revised answers in
    the full-panel game (others' votes masked). Same tie rule; seeded coin per patient and coalition."""
    games = panel.g(FULL)
    v = {frozenset(): np.full(panel.n, v0)}
    for sg in SUBGROUPS:
        y = []
        for p, g in zip(panel.pids, games):
            ch = [(g["revised"][i][0], norm_conf(g["models"][i], g["revised"][i][1], "final"))
                  for i in sg if g["revised"][i][0]]
            w, _ = _vote_info(ch, random.Random(_seed(BASE_SEED, p, "mask", sg_name(sg))))
            y.append(w == g["truth"])
        v[frozenset(sg)] = np.array(y, float)
    return shapley(v)


# ----------------------------------------------------------------------------- votes
def indep_vote(panel: Panel, members: tuple[int, ...], coin_only: bool = False) -> tuple[np.ndarray, list]:
    """Majority vote of the members' INDEPENDENT solo games (their solo final answers).
    Tie rule = the game's (model-percentile confidence, then seeded coin); coin_only skips confidence.
    The coin seed depends on the patient only, as in the preview."""
    solos = {i: panel.solo_dx(i) for i in members}
    ok, winners = [], []
    for k, p in enumerate(panel.pids):
        ch = [(solos[i][k][0], 0.0 if coin_only else norm_conf(panel.models[i], solos[i][k][1], "final"))
              for i in members if solos[i][k][0]]
        w, _ = _vote_info(ch, random.Random(_seed(BASE_SEED, p, "indep_coin" if coin_only else "indep")))
        winners.append(w)
        ok.append(w == panel.truth[k])
    return np.array(ok, float), winners


def coin_vote(panel: Panel, sg) -> np.ndarray:
    """Team's final vote recomputed from its revised answers with a pure seeded-coin tie-break.
    (Question votes cannot be recomputed without replaying the game; this covers the diagnosis vote.)"""
    out = []
    name = sg if isinstance(sg, str) else sg_name(sg)
    for p, g in zip(panel.pids, panel.g(sg)):
        ch = [(dx, 0.0) for dx, _ in g["revised"] if dx]
        w, _ = _vote_info(ch, random.Random(_seed(BASE_SEED, p, "coin_sens", name)))
        out.append(w == g["truth"])
    return np.array(out, float)


def vote_resolution(panel: Panel, sg) -> dict:
    """How the final diagnosis vote and the question votes were resolved (shares)."""
    gs = panel.g(sg)
    fin = pd.Series([g["vote"] for g in gs]).value_counts(normalize=True).to_dict()
    q = pd.Series([e["vote"] for g in gs for e in g["rounds"]]).value_counts(normalize=True).to_dict()
    return {"final": fin, "questions": q}


# ----------------------------------------------------------------------------- error overlap
def _strata_perm(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation of patient indices that only swaps patients with the same true condition."""
    perm = np.arange(len(truth))
    for c in np.unique(truth):
        ix = np.flatnonzero(truth == c)
        perm[ix] = rng.permutation(ix)
    return perm


def error_overlap(panel: Panel, B: int = 2000, seed: int = BASE_SEED) -> dict:
    """For every pair of agents (solo games): both wrong, both wrong with the SAME wrong answer, and
    P(same wrong | both wrong). Null: each agent's answers are permuted among patients with the same
    true condition, independently per agent. This keeps every agent's accuracy and wrong-answer habits
    per condition (its confusion row) and only breaks which patient it errs on."""
    dx = np.array([[d or "<none>" for d, _ in panel.solo_dx(i)] for i in FULL], dtype=object)
    truth = panel.truth
    wrong = dx != truth[None, :]
    pairs = list(combinations(FULL, 2))

    def stats(D, W):
        out = {}
        for i, j in pairs:
            both = W[i] & W[j]
            same = both & (D[i] == D[j]) & (D[i] != "<none>")
            out[(i, j)] = (both.mean(), same.mean(), same.sum() / max(both.sum(), 1))
        return out

    obs = stats(dx, wrong)
    rng = np.random.default_rng(seed)
    null = {pr: [] for pr in pairs}
    for _ in range(B):
        perms = [_strata_perm(truth, rng) for _ in FULL]
        D = np.stack([dx[i, perms[i]] for i in FULL])
        W = np.stack([wrong[i, perms[i]] for i in FULL])
        for pr, s in stats(D, W).items():
            null[pr].append(s)
    res = {"pairs": {}, "by_type": {}}
    types = PAIR_TYPE.get(panel.name, {pr: "check" for pr in pairs})
    for pr in pairs:
        nl = np.array(null[pr])
        res["pairs"][sg_name(pr)] = {
            "type": types[pr],
            **{k: {"obs": float(obs[pr][m]), "null_mean": float(nl[:, m].mean()),
                   "null_95": [float(x) for x in np.percentile(nl[:, m], [2.5, 97.5])],
                   "excess": float(obs[pr][m] - nl[:, m].mean()),
                   "p_upper": float((1 + (nl[:, m] >= obs[pr][m] - 1e-12).sum()) / (B + 1))}
               for m, k in enumerate(["both_wrong", "same_wrong", "same_given_both"])}}
    for t in sorted(set(types.values())):
        prs = [pr for pr in pairs if types[pr] == t]
        o = np.mean([obs[pr] for pr in prs], axis=0)
        nl = np.mean([np.array(null[pr]) for pr in prs], axis=0)
        res["by_type"][t] = {k: {"obs": float(o[m]), "null_mean": float(nl[:, m].mean()),
                                 "null_95": [float(x) for x in np.percentile(nl[:, m], [2.5, 97.5])],
                                 "excess": float(o[m] - nl[:, m].mean()),
                                 "p_upper": float((1 + (nl[:, m] >= o[m] - 1e-12).sum()) / (B + 1))}
                             for m, k in enumerate(["both_wrong", "same_wrong", "same_given_both"])} | {"n_pairs": len(prs)}
    return res


# ----------------------------------------------------------------------------- deliberation
def deliberation(panel: Panel, sgs=None) -> dict:
    """Per agent-answer in teams of 2+: right->right, right->wrong, wrong->right, wrong->other wrong,
    wrong kept; how many switches moved to an answer a peer had given. Plus the team-level
    effect (team vote before vs after deliberation) per sub-team size."""
    sgs = sgs or [sg for sg in SUBGROUPS if len(sg) > 1]
    rows = []
    for sg in sgs:
        for g in panel.g(sg):
            for i in range(len(sg)):
                pre, post = g["final"][i][0], g["revised"][i][0]
                peers = {g["final"][j][0] for j in range(len(sg)) if j != i}
                rows.append({"size": len(sg), "sg": sg_name(sg), "pre_ok": pre == g["truth"],
                             "post_ok": post == g["truth"], "switched": pre != post,
                             "to_peer": pre != post and post in peers})
    df = pd.DataFrame(rows)

    def summ(d):
        return {"n_answers": int(len(d)), "switch_rate": float(d.switched.mean()),
                "right_to_wrong": int((d.pre_ok & ~d.post_ok).sum()),
                "wrong_to_right": int((~d.pre_ok & d.post_ok).sum()),
                "wrong_to_wrong_switch": int((~d.pre_ok & ~d.post_ok & d.switched).sum()),
                "to_peer_share_of_switches": float(d.to_peer.sum() / max(d.switched.sum(), 1)),
                "acc_pre": float(d.pre_ok.mean()), "acc_post": float(d.post_ok.mean())}
    out = {"all": summ(df), "by_size": {int(k): summ(d) for k, d in df.groupby("size")}}
    out["team_effect"] = {}
    for k in sorted({len(sg) for sg in sgs}):
        pre = np.mean([panel.correct(sg, pre=True) for sg in sgs if len(sg) == k], axis=0)
        post = np.mean([panel.correct(sg) for sg in sgs if len(sg) == k], axis=0)
        out["team_effect"][k] = {"pre": float(pre.mean()), "post": float(post.mean())}
    return out


# ----------------------------------------------------------------------------- persona check
@lru_cache(maxsize=1)
def evidence_field() -> dict[str, str]:
    df = pd.read_csv(ROOT / "prereg" / "evidence_specialty.csv")
    return dict(zip(df.code, df.specialty))


@lru_cache(maxsize=1)
def condition_field() -> dict[str, tuple[str, bool]]:
    df = pd.read_csv(ROOT / "prereg" / "condition_specialty.csv")
    return {c: (s, bool(a)) for c, s, a in zip(df.condition, df.specialty, df.ambiguous)}


def field_share(panel: Panel, i: int, field: str, rounds: tuple[int, ...], source: str = "solo") -> np.ndarray:
    """Per patient: share of agent i's valid question proposals (in the given rounds) whose evidence
    item is tagged `field`. source='solo' = agent i's own solo game; 'full' = its proposals in the 4-agent game.
    NaN where the agent made no valid proposal in those rounds."""
    ef = evidence_field()
    gs = panel.g((i,)) if source == "solo" else panel.g(FULL)
    k = 0 if source == "solo" else i
    out = []
    for g in gs:
        codes = [g["rounds"][r - 1]["props"][k] for r in rounds if r <= len(g["rounds"])]
        codes = [c for c in codes if c]
        out.append(np.mean([ef.get(c) == field for c in codes]) if codes else np.nan)
    return np.array(out, float)


def dx_field_share(panel: Panel, i: int, field: str, source: str = "solo") -> np.ndarray:
    cf = condition_field()
    gs = panel.g((i,)) if source == "solo" else panel.g(FULL)
    k = 0 if source == "solo" else i
    return np.array([float(cf.get(g["final"][k][0], ("", 0))[0] == field) if g["final"][k][0] else np.nan
                     for g in gs])


# ----------------------------------------------------------------------------- markers
@lru_cache(maxsize=1)
def marker_codes() -> dict[str, frozenset]:
    df = pd.read_csv(ROOT / "prereg" / "marker_questions_draft.csv")
    df = df[df.is_marker == 1]
    return {c: frozenset(d.code) for c, d in df.groupby("top_condition")}


@lru_cache(maxsize=None)
def _patient(pid: str):
    split, row = pid.split(":")
    return get_patient(split, int(row))


@lru_cache(maxsize=None)
def _revealed_markers(pid: str, code: str) -> bool:
    p = _patient(pid)
    m = marker_codes().get(p.pathology, frozenset())
    return any(a.positive and a.code in m for a in reveal(p, code))


@lru_cache(maxsize=None)
def marker_available(pid: str) -> bool:
    """The patient answers yes to at least one marker question of their own condition."""
    p = _patient(pid)
    m = marker_codes().get(p.pathology, frozenset())
    return any(c in p.record for c in m)


def marker_hit(panel: Panel, sg) -> np.ndarray:
    """1 if the team asked a question that revealed a 'yes' to a marker of the true condition."""
    return np.array([float(any(_revealed_markers(g["pid"], e["chosen"]) for e in g["rounds"] if e["chosen"]))
                     for g in panel.g(sg)])


# ----------------------------------------------------------------------------- cost
def cost(panel: Panel) -> dict:
    """Nominal cost (as billed when first made; cached calls keep their original cost) per consultation."""
    full = panel.g(FULL)
    per_sg = {sg_name(sg): float(np.mean([g["cost"] for g in panel.g(sg)])) for sg in SUBGROUPS}
    return {"per_consultation_usd": float(np.mean([g["cost"] for g in full])),
            "calls_per_consultation": float(np.mean([g["calls"] for g in full])),
            "tokens_in": float(np.mean([g["tokens_in"] for g in full])),
            "tokens_out": float(np.mean([g["tokens_out"] for g in full])),
            "solo_mean_usd": float(np.mean([per_sg[s] for s in "ABCD"])),
            "campaign_per_patient_usd": float(sum(per_sg.values())),
            "per_subgroup_usd": per_sg}


# ----------------------------------------------------------------------------- coin-free (expected) votes
def _vote_expected(choices: list[tuple[str, float]], truth: str) -> float:
    """Exact expected correctness of the game's vote rule, averaging over the coin instead of drawing it."""
    if not choices:
        return 0.0
    cnt = {}
    for c, _ in choices:
        cnt[c] = cnt.get(c, 0) + 1
    top = max(cnt.values())
    tied = [c for c in cnt if cnt[c] == top]
    best = {c: max(x for o, x in choices if o == c) for c in tied}
    m = max(best.values())
    tied = [c for c in tied if best[c] == m]
    return (truth in tied) / len(tied)


def indep_vote_expected(panel: Panel, members: tuple[int, ...]) -> np.ndarray:
    """Independent vote with the coin averaged out (per-patient expected correctness). Sensitivity:
    with few voters, one seeded coin can move the independent-vote baseline by about 1 point."""
    solos = {i: panel.solo_dx(i) for i in members}
    return np.array([_vote_expected([(solos[i][k][0], norm_conf(panel.models[i], solos[i][k][1], "final"))
                                     for i in members if solos[i][k][0]], panel.truth[k]) for k in range(panel.n)])


def team_vote_expected(panel: Panel, sg) -> np.ndarray:
    """Team's final diagnosis vote with the coin averaged out (question votes stay as played)."""
    return np.array([_vote_expected([(dx, norm_conf(g["models"][i], c, "final")) for i, (dx, c) in enumerate(g["revised"]) if dx],
                                    g["truth"]) for g in panel.g(sg)])


# ----------------------------------------------------------------------------- replay (asking vs reading)
REPLAY = ROOT / "results" / "replay"


def replay_correct(panel: Panel) -> dict[str, np.ndarray]:
    """e(S): fixed reader (Gemma 4 31B generalist) right on sub-team S's transcript, per patient."""
    out = {}
    for sg in SUBGROUPS:
        f = REPLAY / panel.name / f"{sg_name(sg)}.jsonl"
        d = {}
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            d[r["pid"]] = float(r["correct"])
        assert sorted(d) == panel.pids, (panel.name, sg_name(sg), "replay incomplete")
        out[sg_name(sg)] = np.array([d[p] for p in panel.pids])
    return out


def evidence_shapley(panel: Panel, v0: float) -> np.ndarray:
    e = replay_correct(panel)
    v = {frozenset(sg): e[sg_name(sg)] for sg in SUBGROUPS}
    v[frozenset()] = np.full(panel.n, v0)
    return shapley(v)


def transcript_diversity(panel: Panel) -> dict:
    """How many distinct 5-question transcripts the 15 sub-teams produced per patient (1 = all identical)."""
    k = [len({tuple(e["chosen"] for e in self_g["rounds"]) for self_g in (panel.games[sg_name(sg)][p] for sg in SUBGROUPS)})
         for p in panel.pids]
    sets = [len({frozenset(e["chosen"] for e in panel.games[sg_name(sg)][p]["rounds"]) for sg in SUBGROUPS}) for p in panel.pids]
    return {"distinct_transcripts_per_patient": float(np.mean(k)), "distinct_question_sets_per_patient": float(np.mean(sets))}
