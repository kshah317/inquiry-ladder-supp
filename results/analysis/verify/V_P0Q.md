# V_P0Q: independent check of the P0-Q panel

Own code: `v_p0q.py` -> `v_p0q_out.json`. Reuses only the earlier verifier's independent NB (`v_review_nb.py`, `_v_review_nb.pkl`). No src/ddx or scripts/ imports, no cache, no API calls. Loader: last line per pid wins, restricted to the 294 pids in data/samples/main.csv. Bootstrap: paired over patients, B = 10,000, seed 202611032, percentile CIs. TOST: paired t, margin ±5 pt. McNemar: exact (scipy binomtest). Points = percentage points.

| # | Check | Project | Mine | Match |
|---|---|---|---|---|
| 1 | P0Q files have all 294 pids | 294 each | 294 lines, 294/294 pids in A, B, C, D, ABCD | MATCH |
| 1 | P0Q/A = P2/C (seed Q38.gen.a) | identical asked + final dx | 0/294 differ in asked, rounds[].chosen, final dx, confidence, correct. Rounds and final are identical apart from the agent label (Physician A vs C). Usage differs only in the `cached` flag (P0Q/A calls were cache replays). | MATCH |
| 2 | Team (ABCD correct) | .388 | .388 (114/294) | MATCH |
| 2 | Pre-deliberation team | .354 | .354 | MATCH |
| 2 | Solos A/B/C/D | .327/.354/.344/.313 | .327/.354/.344/.313 | MATCH |
| 3 | Independent vote | .374 | 14 tied patients; 7 can go either way. Exact range .361 to .384, expected .372; my coin gives .374 | MATCH (inside coin range) |
| 3 | Team − vote | +1.36 [−3.1, +5.8] | +1.36 [−3.06, +5.78] (my coin); range +0.34 to +2.72 | MATCH |
| 3 | McNemar | 23 vs 19 | 24 vs 20 (my coin). Possible: 23–25 vs 17–22. The project's 23 vs 19 is the outcome where both team-correct coin patients and 2 of the 5 team-wrong ones go to the truth (vote 110/294 = .374) | MATCH (coin) |
| 4 | C1-Q = P1Q team − P0Q team | −4.76 [−9.5, 0.0] | −4.76 [−9.52, 0.00] | MATCH |
| 4 | 90% CI | [−8.8, −0.7] | [−8.50, −1.02] | MATCH (within 0.34 pt, bootstrap noise) |
| 4 | TOST p | ≈ .46 | .460 | MATCH |
| 4 | McNemar | 18 vs 32, p ≈ .065 | 18 vs 32, p = .0649 | MATCH |
| 5 | P0Q team − P0 team | −2.04 [−8.2, +3.7] | −2.04 [−8.16, +4.08] | MATCH (upper end 0.38 pt off, within 0.5) |
| 6 | (P1Q − P0Q) − (P1 − P0) | −5.10 [−11.2, +1.0] | −5.10 [−11.22, +1.02] | MATCH |
| 7 | Team − best member (B) | +3.40 [−1.4, +8.2] | best = B; +3.40 [−1.36, +8.16]; McNemar 31 vs 21, p = .21 | MATCH |
| 8 | Same wrong given both wrong, pooled over 6 pairs | 0.428 | 0.428 (434/1,015 pooled counts; mean of pair ratios also 0.428; pairs .428/.400/.440/.469/.419/.411) | MATCH |
| 8 | Permutation null mean | 0.231 | 0.231 (2,000 draws, within-condition shuffle per agent; 95% [0.209, 0.254]; p_upper = 0.0005) | MATCH |
| 9 | NB top-1 solos A/B/C/D | .830/.830/.816/.840 | .830/.830/.816/.840 | MATCH |
| 9 | NB top-1 ABCD | .847 | .847 | MATCH |
| 9 | NB top-1 union of 4 solos | .929 | .929 | MATCH |
| 10 | Switch rate (n answers) | .258 of 1,176 | .258 of 1,176 | MATCH |
| 10 | Wrong→right / right→wrong | 67 / 21 | 67 / 21 | MATCH |
| 10 | Member accuracy pre → post | .341 → .380 | .341 → .380 | MATCH |
| 11 | Chosen code with 1/2/3/4 valid proposers | .180/.397/.216/.207 | .180/.397/.216/.207 (1,470 rounds, no round without a chosen code, no round with 0 support) | MATCH |

All 11 checks match. Coin-dependent: item 3 only (14 tied patients, 7 decisive). No differences found.
