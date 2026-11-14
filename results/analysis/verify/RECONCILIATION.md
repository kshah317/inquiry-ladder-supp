# Verification pass — analysis module vs independent re-computation

The independent check was written from scratch as separate code (results/analysis/verify/v_*.py). It did not read the analysis code. It used its own vote, Shapley (average over all 24 orderings), bootstrap, McNemar, TOST and permutation code.

**Matches (to 4 decimals):**
- Team and solo accuracies for all 5 panels.
- Team − best member, with CIs.
- Shapley under both v(empty) baselines, efficiency, and LOO.
- C1, C2a and C2b: diff, CIs, TOST p 0.0058 and McNemar 15/14, 21/22, 21/20.
- Independent vote for P0, P2 and CHECK.
- Error overlap by pair type.
- Deliberation switch counts.
- Marker hit rates for the full panel.
- Cost per consultation and per attribution campaign.

**One difference, explained:** the independent vote for P1 (ours 0.391, check 0.381) and P1-Q (0.364 vs 0.354). Both votes use the same rule; only the final seeded coin differs. That coin decides 12 patients in P1 and 20 in P1-Q (0 in P2). Over 2,000 coin seeds the P1 independent vote ranges over [0.378, 0.391] and P1-Q over [0.347, 0.367]. Both values fall inside these ranges.
- Fix adopted: report a coin-free version as a sensitivity check. This is the exact expectation over the coin, for both the team's final vote and the independent vote (metrics.indep_vote_expected / team_vote_expected).
- With the coin removed: P1 team − independent vote = +2.9 [−0.5, +6.4]; P1-Q = −2.0 [−5.8, +1.9]. The primary C2b is unaffected because P2 has no coin ties.

**Data audit (from the check):**
- No duplicate patients and no missing patients.
- `correct` always equals panel == truth, and `correct_pre` always equals panel_pre == truth.
- Abstentions: P1-Q 1, P2 2, CHECK 315 (all from Gemma 3 4B).
- 20 patients tagged `sample: trial` in P1 and P2. This is by design: trial games are part of the main run.

## No-peer control

Independent check (results/analysis/verify/v_nopeer.py -> verify_nopeer.json). Own vote, switch, bootstrap (10,000 joint patient resamples, seed 777) and exact McNemar code; only `game.norm_conf` and `game.match_condition` reused. Cache opened read-only; no API calls.

- **Consistency:** 294 patients in every main and no-peer file, same pid set in all 8. `final`, `revised`, `panel_pre`, `panel_real`, truth and models in the no-peer files match the main logs for every pid; `correct_np` always equals panel_np == truth.
- **Raw records:** 4,601 no-peer requests in the cache. For 50 random (panel, pid, agent) rows, each matched exactly one cached no-peer request. The parsed diagnosis and confidence equal the stored `nopeer` answer in 50/50. All 50 also link to a real-deliberation request with the same first three messages, this pid's peer answers, the same seed, and a response that maps to the logged revised answer (so only the last message differs).
- **Recompute:** team accuracy (pre / real / no-peer, logged coin and coin-free expectation), switch counts, wrong→right, right→wrong, wrong→other-wrong, share to a peer's answer, unanimity, member accuracy, all effects and McNemar counts/p all match nopeer.json within 0.1 pt. All CIs are within 0.5 pt. Every stored team answer is among my tied winners.
- **Coin ties (patients, pre / real / no-peer):** P0 2/2/1, P1 14/6/15, P1-Q 15/2/9, P2 0/1/0.
- **Pooled (mean of 4):** deliberation +2.1 [+1.0, +3.3] = second look +1.4 [+0.4, +2.5] + peer effect +0.8 [−0.3, +1.8]; coin-free peer effect +0.7 [−0.3, +1.7]. The peer-effect CI includes 0, so the reading rule's conclusion holds.
- **Differences:** none.

## Anchor

Independent check (results/analysis/verify/v_anchor.py -> verify_anchor.json). Own code; reads only the jsonl logs and analysis.json. Bootstrap 10,000 patient resamples, seed 4242, paired on the same pids. Exact McNemar via scipy binomtest. Cache not opened; no API calls.

- **Consistency:** 294 unique pids in ANCHOR/A.jsonl and in ABCD plus the best-member file for P0, P1, P2 and P1-Q; all pid sets identical. `correct` always equals final dx == truth.
- **Recompute (all match anchor.json, 86 checks, no flags):** top-1 0.463 [0.405, 0.520]; cost $0.1825 per consultation; 2,092 calls, 422 truncated (20.2%); retries 328; abstained proposals 84; no final answer 13; games with < 5 chosen questions 61 (the 84 abstentions are exactly the missing rounds in those games). Accuracy by truncated calls: 0: 0.607 (n=117), 1: 0.494 (n=77), 2+: 0.270 (n=100).
- **Comparisons:** all point estimates, McNemar counts and p values match exactly. CIs within 0.5 pt (largest gap 0.4 pt: P1-Q best member upper bound, 19.4 vs 19.0).
  - P0 team +5.4 [+0.3, +10.9] (claimed [+0.0, +10.5]); 40 vs 24, p=0.060
  - P0 best (B) +5.4 [−0.3, +11.2]; 46 vs 30, p=0.085
  - P1 team +5.1 [−0.3, +10.5]; 41 vs 26, p=0.086
  - P1 best (A) +6.8 [+1.7, +12.2]; 43 vs 23, p=0.019
  - P2 team +5.8 [+1.4, +10.5]; 34 vs 17, p=0.024
  - P2 best (A) +9.5 [+4.4, +14.6]; 44 vs 16, p=0.0004
  - P1-Q team +12.2 [+7.1, +17.7]; 53 vs 17, p=2e-5
  - P1-Q best (D) +13.6 [+8.2, +19.4]; 58 vs 18, p=5e-6
- **Cost ratio anchor ÷ team:** 6.4× (P0), 6.4× (P1), 5.9× (P2), 9.3× (P1-Q); same from raw usage.
- **Notes (not numeric mismatches):**
  - P0 team CI lower bound sits on 0 (claimed +0.0, mine +0.3). Whether it "excludes 0" depends on the seed, so it should be read as borderline, not as a clear win. McNemar p=0.060 agrees.
  - The report's "best member" cost column is the panel's mean solo cost (solo_mean_usd), not the named member's own cost. Only P2 differs in a visible way: member A costs $0.0061 (anchor ÷ A = 30×), against the $0.0067 / 27× shown. The others are within rounding (P0 30×, P1 29.5×, P1-Q 43×). Fix: relabel the column as "mean solo cost", or use each member's own cost.

## Repeats (replica 1 + pooled)

Independent check (results/analysis/verify/v_repeats.py -> v_repeats_out.json). Own loader (last line per pid wins), own vote (plurality of solo `final` dx -> max within-model confidence percentile, re-implemented from prereg/confidence_scales.json -> own seeded coin), own coin-free expectation, bootstrap (10,000 patient resamples, seed 20261102), paired-t TOST (±5 pt) and exact McNemar (scipy binomtest). repeats.json was read only after my numbers were computed. No cache, no API calls.

- **Integrity (replica 1):** every agent in every main_r1 game, and every game record, has replica == 1. 294 lines = 294 unique pids in all 30 files (r0 + r1, P0/P1/P2 × A/B/C/D/ABCD); same pid set everywhere; truth labels identical to replica 0; `correct` always equals panel == truth; solo `panel` always equals its `final[0].dx`; solo model equals the matching team slot's model. The 6 lines in main_r1/P2/failures.log are ConnectErrors that were re-run (no pid missing).
- **Spot checks (replica 1, one random pid per panel):** P0 test:101562, P1 test:77940, P2 test:122358. In all three the logged team answer is the unanimous/majority answer of `revised`, `correct` = (answer == truth) (all three False), and the solo finals I used are the solo files' `final[0]` (P1: A GERD, B/C Inguinal hernia, D Pancreatic neoplasm = truth; P2: B and D Epiglottitis = truth). Across all 1,764 team games the logged team answer is always inside my tied set.
- **Matches (218 of 228 checks):** team, pre-deliberation and solo accuracies, best member and team − best (diff, CI, McNemar), coin-free team and independent vote and their difference, C1/C2a/C2b and their coin-free versions (diff, 95% and 90% CI, TOST p to 4 decimals, McNemar counts and p) for r0, r1 and pooled; stability (r1 − r0, McNemar, agreement 0.915/0.929/0.867, solo r1 − r0, r1 team − r1 solo of the r0 best member). All point estimates match to 4 decimals. CIs agree within one patient step (max gap 0.34 pt = 1/294; the bootstrap percentiles fall on that grid, so this is seed noise).
- **Only difference, explained:** the coin-decided independent vote (and team − independent vote) for P0 and P1 (r0 P0 0.412 vs 0.408, r0 P1 0.384 vs 0.391, r1 P1 0.405 vs 0.408, and the pooled values). Coin ties for the independent vote: P0 7/5, P1 12/11, P2 0/0 (r0/r1). The main values are inside my exact coin range in every case (e.g. r0 P1 [0.378, 0.395], r1 P1 [0.398, 0.408]). The coin-free values match exactly. P2 has no coin ties, so C2b is unaffected.
- **Headline (my numbers):** C1 r0 +0.3 [−3.1, +4.1] TOST p 0.006; r1 +3.1 [−0.3, +6.5] p 0.136; pooled +1.7 [−1.0, +4.4] p 0.010. C2a r0 −0.3, p 0.019; r1 +3.1, p 0.229; pooled +1.4, p 0.036. C2b r0 +0.3, p 0.017; r1 +2.4, p 0.141; pooled +1.4, p 0.024. Equivalence holds in r0 and pooled, not in r1 alone.

## Review analyses

Independent check (results/analysis/verify/v_review_nb.py, v_review.py -> v_review_out.json, v_review_coin7.py; table in V_REVIEW.md). Own naive Bayes fitted from raw validation data, own loader, vote, bootstrap (10,000, seed 20261103) and McNemar. No src/ddx or scripts/ imports, no cache, no API calls.

- **All 11 checked items match** (item 12 skipped as instructed): NB sanity (0.367 / 0.283 / 0.993), NB top-1 of every solo and ABCD transcript in P0/P1/P1Q/P2, union of solos, ABCD − mean solo (P2 −4.76 [−8.0, −1.7]), anchor NB 0.850 with 61 short games, question-vote support shares and P2 vote types, SC8 (0.398, $0.0487, +6.46 [+1.0, +11.9], p = .025), LLM-reader decomposition for P1 and P2, r1-chosen best member (P0 D +2.04, P1 D +4.42, P2 A +3.74), and billed qwen rates and cache shares.
- **Coin-dependent items:** check 7 (P0 7, P1 12, P1Q 20 coin patients, P2 none) and SC8 (3 coin patients). Every project value can be reached by some coin outcome. SC8 range is 0.391–0.398, and anchor − SC8 ranges from +6.46 to +7.14.
- **CIs:** all within 0.5 pt; the largest gap is the P1 best-member upper end (+8.5 vs +8.2).
- **Differences:** none.

## P0-Q

Independent check (results/analysis/verify/v_p0q.py -> v_p0q_out.json; table in V_P0Q.md). Own loader, vote with tie rule, bootstrap (10,000, seed 202611032), paired-t TOST and exact McNemar. NB reused from v_review_nb.py. No src/ddx or scripts/ imports, no cache, no API calls.

- **All 11 checks match.** Integrity: 294/294 pids in every P0Q file. P0Q/A is the same game as P2/C for all 294 pids (asked, chosen codes, final dx and confidence, correct). The only differences are the agent label and the usage `cached` flag.
- **Coin-dependent:** independent vote (item 3). 14 tied patients, 7 decisive. Range .361 to .384; team − vote ranges from +0.34 to +2.72. The project's .374, +1.36 and McNemar 23 vs 19 can all be reached by a coin outcome. Mine was 24 vs 20, which gives the same difference.
- **CIs:** all within 0.5 pt. The largest gaps are the item 5 upper end (+4.08 vs +3.7) and the item 4 90% CI (0.3 pt at each end).
- **Note:** P0Q/A usage entries are cache replays (`cached: True`). Any P0Q cost figure should say whether slot A's replayed calls are counted as billed.
- **Differences:** none.

## Analyses of the submitted draft

`scripts/58_v10_analyses.py` recomputes the matched baselines, the Shapley values with the chance baseline, accuracy by coalition size, the question-vote retention measures, the vote-failure split and the reference-model contrast against P0-Q, and writes `results/analysis/v10.json`. A separate script written without access to that code (`v_v10.py`, report `V_V10.md`) checked 68 values against an earlier draft of the paper. 67 matched and 1 did not, and the paper was corrected as follows.

- The share of asked P2 questions that share a proposer with the previous asked question is 923 of 1176, which is 78%. The earlier draft said 79%, a double rounding of 0.785.
- The interval for the P0-Q team minus its mean member is [+1.8, +8.9] with the study seed. The earlier draft said [+2.0, +8.8]. Other seeds give +1.9 to +2.0 and +8.8, so the difference is bootstrap noise of one grid step.
- Three team-minus-vote intervals (P1, P1-Q, P2) have an endpoint that moves by one grid step of 0.34 pp between bootstrap seeds. The paper reports the values from the study seed. No conclusion depends on these endpoints.

## Rotating-proposer follow-up

`scripts/59_rotate_analyse.py` writes `results/analysis/rotate.json`. A separate script (`v_rotate.py`, report `V_ROTATE.md`) recomputed 48 values from the raw logs with its own code and its own naive Bayes implementation, and checked the protocol in all 2,940 rounds. All values agree. The one row marked MISMATCH in `V_ROTATE.md` (P2 truncated calls) is a transcription slip in the list of claims given to the verifier, which read 0 where the analysis has 60, and the verifier's 60 agrees with `rotate.json`.

- Several interval endpoints move by one grid step of 0.34 pp between bootstrap seeds. The paper reports the study seed. For the P2 information ceiling, rotate minus plurality, the study seed gives [+1.7, +8.8] and three other seeds give [+1.4, +8.8]. The interval excludes zero in every case.
- The seeded start of the rotation is not evenly spread over the four members (52, 77, 94 and 71 patients), but each member selects between 1.18 and 1.33 questions per patient.
