"""Rotating-proposer analyses fixed before its calls. Existing logs only, no API calls.
Each rotate team is paired with the primary-run plurality team on the same patients.
Output: results/analysis/rotate.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddx import metrics as M, stats as S, nb
from ddx.metrics import load_panel, FULL

ROOT = Path(__file__).resolve().parents[1]
r1 = lambda x: round(100 * float(x), 1)
r3 = lambda x: round(float(x), 3)


def load_rotate(name, pids):
    d = {}
    for line in open(ROOT / "results/rotate" / name / "ABCD.jsonl", encoding="utf-8"):
        r = json.loads(line)
        assert r.get("question_rule") == "rotate"
        d[r["pid"]] = r                                   # a re-run episode replaces the earlier line
    assert sorted(d) == list(pids), (name, len(d))
    return [d[p] for p in pids]


def asked(rounds):
    return [e["chosen"] for e in rounds if e["chosen"]]


def cont(a, b, idx, binary=True):
    c = S.paired_contrast(a, b, idx, binary=binary)
    out = {"a": r3(c["a"]), "b": r3(c["b"]), "diff": r1(c["diff"]), "ci95": [r1(x) for x in c["ci95"]]}
    if binary:
        out["mcnemar_p"] = round(c["mcnemar"]["p"], 4)
    return out


def team(name, idx):
    P = load_panel(name)
    R = load_rotate(name, P.pids)
    ceil = lambda rounds_list: np.array([nb.score(p, t, asked(r))[0] for p, t, r in zip(P.pids, P.truth, rounds_list)])
    e_rot = ceil([r["rounds"] for r in R])
    e_plu = ceil([g["rounds"] for g in P.g(FULL)])
    e_solo = np.mean([ceil([g["rounds"] for g in P.g((i,))]) for i in FULL], 0)
    e_pool = np.array([nb.score(p, t, sum((asked(P.games[k][p]["rounds"]) for k in "ABCD"), []))[0]
                       for p, t in zip(P.pids, P.truth)])
    acc_rot = np.array([float(bool(r["correct"])) for r in R])
    acc_pre = np.array([float(bool(r["correct_pre"])) for r in R])
    vote, _ = M.indep_vote(P, FULL)
    solo_acc = np.mean([P.correct((i,)) for i in FULL], 0)

    names = [a["name"] for a in R[0]["agents"]]
    per_member, also, shared, n_q = np.zeros(4), 0, [], 0
    for r, g in zip(R, P.g(FULL)):
        for e in r["rounds"]:
            if not e["chosen"]:
                continue
            n_q += 1
            per_member[names.index(e["proposer"])] += 1
            also += sum(x["valid"] and x["code"] == e["chosen"] for x in e["proposals"]) >= 2
        shared.append(len(set(asked(r["rounds"])) & set(asked(g["rounds"]))))
    marker = lambda rounds_list: float(np.mean([any(M._revealed_markers(p, c) for c in asked(r))
                                                for p, r in zip(P.pids, rounds_list) if M.marker_available(p)]))
    return {
        "n": P.n,
        "1_ceiling_rotate_minus_plurality": cont(e_rot, e_plu, idx),
        "2_ceiling_rotate_minus_mean_member": cont(e_rot, e_solo, idx, binary=False),
        "2_ceiling_pooled_minus_rotate": cont(e_pool, e_rot, idx),
        "3_accuracy_rotate_minus_plurality": cont(acc_rot, P.correct(FULL), idx),
        "3_accuracy_rotate_minus_indep_vote": cont(acc_rot, vote, idx),
        "3_accuracy_rotate_minus_mean_member": cont(acc_rot, solo_acc, idx, binary=False),
        "3_accuracy_before_deliberation": {"rotate": r3(acc_pre.mean()), "plurality": r3(P.correct(FULL, pre=True).mean())},
        "4_asked_questions": n_q,
        "4_asked_per_member_mean": [round(float(x) / P.n, 2) for x in per_member],
        "4_asked_also_proposed_by_another_member": r3(also / n_q),
        "4_asked_questions_shared_with_plurality_mean": round(float(np.mean(shared)), 2),
        "4_marker_asked_when_available": {"rotate": r3(marker([r["rounds"] for r in R])),
                                          "plurality": r3(marker([g["rounds"] for g in P.g(FULL)]))},
        "run": {"truncated_calls": int(sum(u.get("finish_reason") == "length" for r in R for u in r["usage"])),
                "final_answers_missing": int(sum(f["dx"] is None for r in R for f in r["final"])),
                "passed_turns": int(sum(e["proposer"] != names[(r["rotate_start"] + e["k"] - 1) % 4]
                                        for r in R for e in r["rounds"] if e["chosen"])),
                "cost_per_episode_usd": round(float(np.mean([sum(u.get("cost") or 0 for u in r["usage"]) for r in R])), 4)},
    }


def main():
    idx = S.boot_idx(294)
    out = {k: team(k, idx) for k in ["P1", "P2"]}
    (ROOT / "results/analysis/rotate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
