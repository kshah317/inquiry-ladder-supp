"""Turn results/analysis/analysis.json into results/analysis/ANALYSIS_REPORT.md (tables only, no model calls)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results/analysis/analysis.json").read_text())
L = []
pt = lambda x: f"{100*x:+.1f}"                                      # noqa: E731
ci = lambda c: f"[{100*c[0]:+.1f}, {100*c[1]:+.1f}]"                # noqa: E731
acc = lambda x: f"{x:.3f}"                                          # noqa: E731
NAMES = {"P0": "P0 same model", "P1": "P1 roles (Gemma)", "P1Q": "P1-Q roles (Qwen)", "P2": "P2 mixed models", "CHECK": "CHECK (98)"}

L += ["# Analysis report (scripts/40_analyse.py → 41_report.py)", "",
      f"Bootstrap B = {R['settings']['B']} over patients (seed {R['settings']['seed']}); permutation null {R['settings']['perm']} draws; TOST margin ±5 pts.",
      "Differences are in percentage points. Independently re-computed by a separate verification script (results/analysis/verify/).", ""]

p = R["primary"]
L += ["## 1. Primary contrasts (294 patients, paired)", "",
      "| Contrast | Diff | 95% CI | 90% CI | McNemar (a only / b only, p) | TOST p (±5) |", "|---|---|---|---|---|---|"]
for k, lab in [("C1_P1_vs_P0", "C1: P1 − P0"), ("C2a_P2_vs_P0", "C2a: P2 − P0"), ("C2b_P2_vs_indep_vote", "C2b: P2 − its members' independent vote")]:
    c = p[k]; m = c["mcnemar"]
    L.append(f"| {lab} | {pt(c['diff'])} | {ci(c['ci95'])} | {ci(c['ci90'])} | {m['a_only']} / {m['b_only']}, p = {m['p']:.2f} | {c['tost']['p_tost_t']:.3f} |")
L += ["", "TOST was planned for C1 only; for C2 it is shown for interpretation. Holm-adjusted McNemar p (C2a, C2b): "
      f"{p['holm_C2_mcnemar']['C2a']:.2f}, {p['holm_C2_mcnemar']['C2b']:.2f}.",
      f"Secondary: P1-Q − P0 = {pt(p['P1Q_vs_P0']['diff'])} {ci(p['P1Q_vs_P0']['ci95'])}.", ""]
cf = p["coin_free"]
L += ["Coin averaged out of the final vote (sensitivity): " + "; ".join(f"{k} {pt(v['diff'])} {ci(v['ci95'])}" for k, v in cf.items()) + ".", ""]
nm = R["primary_no_marker_subset"]
L += [f"No-marker subset (n = {nm['n']} patients with no marker question available; sanity check, low power): " +
      "; ".join(f"{k.split('_')[0]} {pt(nm[k]['diff'])} {ci(nm[k]['ci95'])}" for k in ("C1_P1_vs_P0", "C2a_P2_vs_P0", "C2b_P2_vs_indep_vote")) + ".", ""]

L += ["## 2. Is the team worth it? (full 4-agent panels)", "",
      "| Panel | Team | Before delib. | Solo A/B/C/D | Team − best member [95% CI] | (best re-chosen per draw) | Indep. vote | Team − indep. vote [95% CI] | coin-free | Any solo right |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for k, b in R["panels"].items():
    L.append(f"| {NAMES[k]} | {acc(b['team']['mean'])} | {acc(b['team_pre_delib'])} | {'/'.join(f'{x:.3f}' for x in b['solo'])} | "
             f"{pt(b['team_minus_best']['diff'])} {ci(b['team_minus_best']['ci95'])} | {ci(b['team_minus_best_reselected_ci95'])} | "
             f"{acc(b['indep_vote'])} | {pt(b['team_minus_indep_vote']['diff'])} {ci(b['team_minus_indep_vote']['ci95'])} | "
             f"{pt(b['coin_free']['team_minus_indep_vote']['diff'])} | {acc(b['oracle_any_solo_right'])} |")
L += ["", "Sub-teams by size (mean over sub-teams): team − best member / team − independent vote", "",
      "| Panel | pairs | triples | full |", "|---|---|---|---|"]
for k, b in R["panels"].items():
    s = b["subteams_by_size"]
    L.append(f"| {k} | " + " | ".join(f"{pt(s[n]['team_minus_best']['mean'])} {ci(s[n]['team_minus_best']['ci95'])} / {pt(s[n]['team_minus_indep_vote']['mean'])}" for n in ("2", "3", "4")) + " |")
L += ["", "Pair synergy by pair type (pair − better member; pair − the two members' independent vote)", "", "| Panel | Pair type | n pairs | Synergy [95% CI] | − indep. vote [95% CI] |", "|---|---|---|---|---|"]
for k, b in R["panels"].items():
    for t, v in b["pair_types"].items():
        L.append(f"| {k} | {t} | {v['n_pairs']} | {pt(v['synergy']['mean'])} {ci(v['synergy']['ci95'])} | {pt(v['minus_indep_vote']['mean'])} {ci(v['minus_indep_vote']['ci95'])} |")

L += ["", "## 3. Who gets credit (Shapley, per agent A/B/C/D)", "",
      "| Panel | Roles/models | Shapley v0 = 1/49 | Shapley v0 = opening-only | LOO | Masking Shapley (no re-execution) |", "|---|---|---|---|---|---|"]
for k, b in R["panels"].items():
    f = lambda key: " / ".join(f"{x['mean']:.3f}" for x in b[key])  # noqa: E731
    L.append(f"| {k} | v0(b) = {b['v0']['opening_only']:.3f} | {f('shapley_chance')} | {f('shapley_opening_only')} | {f('loo')} | {f('masking_shapley_chance')} |")
L += ["", "95% CIs per agent are in analysis.json. CHECK: strong (full record) agent D vs weak agent C separates clearly (positive control passes).", ""]

L += ["## 4. Why: error overlap on solo games vs within-condition permutation null", "",
      "| Panel | Pair type | Both wrong (obs / null) | Same wrong answer (obs / null) | Same answer given both wrong (obs / null) |", "|---|---|---|---|---|"]
for k, v in R["error_overlap"].items():
    for t, x in v["by_type"].items():
        L.append(f"| {k} | {t} | " + " | ".join(f"{x[m]['obs']:.3f} / {x[m]['null_mean']:.3f} (p {x[m]['p_upper']:.4f})" for m in ("both_wrong", "same_wrong", "same_given_both")) + " |")
L += ["", "p = one-sided permutation p (smallest possible = 1/(perm+1)).", ""]

L += ["## 5. Deliberation (full 4-agent games; no-peer control still to run)", "",
      "| Panel | Switch rate | Right→wrong | Wrong→right | Wrong→other wrong | Switches to a peer's answer | Team before → after |", "|---|---|---|---|---|---|---|"]
for k, v in R["deliberation_full_panel"].items():
    a = v["all"]; t = v["team_effect"]["4"]
    L.append(f"| {k} | {a['switch_rate']:.3f} | {a['right_to_wrong']} | {a['wrong_to_right']} | {a['wrong_to_wrong_switch']} | {a['to_peer_share_of_switches']:.2f} | {t['pre']:.3f} → {t['post']:.3f} |")

L += ["", "## 6. Persona manipulation check (share of proposed questions in the role's field; specialist − generalist, same panel)", "",
      "| Panel | Role | Solo, rounds 1-2 | Solo, rounds 4-5 | Full panel, all rounds | Final dx in field (solo) |", "|---|---|---|---|---|---|"]
for k, roles in R["persona_check"].items():
    if not k.startswith("P1"):
        continue
    for role, r in roles.items():
        g = lambda key: f"{r[key]['specialist']:.2f} vs {r[key]['generalist']:.2f} ({pt(r[key]['diff'])} {ci(r[key]['ci95'])})"  # noqa: E731
        L.append(f"| {k} | {role} | {g('solo_early_r1_2')} | {g('solo_late_r4_5')} | {g('full_all')} | {g('final_dx_in_field_solo')} |")

L += ["", "## 7. Marker questions (a 'yes' to a question unique to the true condition)", "",
      "| Panel | Hit rate solo / pair / triple / full | Accuracy given hit (solo) | Accuracy, marker available but not hit (solo) | Full − mean solo hit rate |", "|---|---|---|---|---|"]
for k, v in R["markers"].items():
    L.append(f"| {k} | " + " / ".join(f"{v[f'size{n}']['hit_rate']:.3f}" for n in (1, 2, 3, 4)) +
             f" | {v['size1']['acc_given_hit']:.2f} | {v['size1']['acc_given_no_hit_but_available']:.2f} | {pt(v['full_minus_mean_solo_hit']['diff'])} {ci(v['full_minus_mean_solo_hit']['ci95'])} |")
L += ["", f"Patients with at least one marker available: {R['markers']['P0']['patients_with_marker_available']}/294.", ""]

c = R["coin_tiebreak_sensitivity"]
L += ["## 8. Coin-flip tie-break sensitivity (final diagnosis vote recomputed with a pure seeded coin)", "",
      "Largest change in any sub-team's accuracy: " + ", ".join(f"{k} {pt(v['max_abs_change'])}" for k, v in c.items() if k != "primary_coin_only") +
      " (CHECK pairs with the weak agent depend on the confidence rule).",
      "Primary contrasts with coin-only votes: " + "; ".join(f"{k} {pt(v['diff'])} {ci(v['ci95'])}" for k, v in c["primary_coin_only"].items()) + ".", ""]

L += ["## 9. Cost (nominal; cached calls keep their original cost)", "",
      "| Panel | $ per 4-agent consultation | Calls | $ per solo consultation | $ per patient, all 15 sub-teams (attribution campaign) |", "|---|---|---|---|---|"]
for k, v in R["cost"].items():
    L.append(f"| {k} | {v['per_consultation_usd']:.4f} | {v['calls_per_consultation']:.1f} | {v['solo_mean_usd']:.4f} | {v['campaign_per_patient_usd']:.3f} |")

L += ["", "## 10. Exploratory: specialists' Shapley on in-field vs out-of-field patients (v0 = 1/49)", "", "| Panel | Role | n in field | Shapley in / out | Solo accuracy in / out |", "|---|---|---|---|---|"]
for k, roles in R["in_field_exploratory"].items():
    for role, r in roles.items():
        L.append(f"| {k} | {role} | {r['n_in']} | {r['shapley_in']:.3f} / {r['shapley_out']:.3f} | {r['solo_acc_in']:.3f} / {r['solo_acc_out']:.3f} |")
if "replay" in R:
    X = R["replay"]
    L += ["", "## 11. Replay test: asking vs reading (fixed reader = Gemma 4 31B generalist on each sub-team's transcript)", "",
          "e = reader accuracy on the team's 5-question transcript (evidence value); v = the team's own answer.", "",
          "| Panel | Distinct transcripts / patient (of 15) | e solo A/B/C/D | e full | v full | e full − best solo e [95% CI] | v full − e full (reading+delib+vote) [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for k in ("P0", "P1", "P1Q", "P2", "CHECK"):
        x = X[k]
        L.append(f"| {k} | {x['diversity']['distinct_transcripts_per_patient']:.1f} | {'/'.join('%.3f' % x['e'][c] for c in 'ABCD')} | "
                 f"{x['e']['ABCD']:.3f} | {x['v']['ABCD']:.3f} | {pt(x['e_full_minus_best_solo_e']['diff'])} {ci(x['e_full_minus_best_solo_e']['ci95'])} | "
                 f"{pt(x['v_full_minus_e_full']['diff'])} {ci(x['v_full_minus_e_full']['ci95'])} |")
    L += ["", "Exact decomposition against the Gemma generalist's own solo game (reader = that agent, so e = v there):", "",
          "| Panel | Total: team − Gemma generalist solo | Asking: e(team) − e(solo) | Reading + deliberation + vote: v(team) − e(team) |", "|---|---|---|---|"]
    for k in ("P0", "P1", "P2"):
        d = X[k]["decomposition_vs_gemma_generalist_solo"]
        L.append(f"| {k} (ref {d['ref_agent']}) | " + " | ".join(f"{pt(d[m]['diff'])} {ci(d[m]['ci95'])}" for m in ("total", "asking", "reading_deliberation_vote")) + " |")
    L += ["", "| Panel | Evidence Shapley A/B/C/D (v0 = 1/49) | Answer − evidence Shapley A/B/C/D [95% CI] |", "|---|---|---|"]
    for k in ("P0", "P1", "P1Q", "P2", "CHECK"):
        x = X[k]
        L.append(f"| {k} | {' / '.join('%.3f' % y['mean'] for y in x['evidence_shapley_chance'])} | " +
                 " / ".join(f"{pt(y['mean'])} {ci(y['ci95'])}" for y in x["answer_minus_evidence_shapley_chance"]) + " |")
    a = X["asking_contrasts_full_panel"]
    L += ["", "Do diverse panels ask better? e(full panel) − e(P0 full panel): " + "; ".join(f"{k.split('_')[0]} {pt(v['diff'])} {ci(v['ci95'])}" for k, v in a.items()) + ".",
          "", "Exploratory (post hoc): " + "; ".join(f"{k}: {pt(v['diff'])} {ci(v['ci95'])}" for k, v in X["exploratory_qwen_asks_gemma_reads"].items()) + "."]
(ROOT / "results/analysis/ANALYSIS_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote results/analysis/ANALYSIS_REPORT.md")
