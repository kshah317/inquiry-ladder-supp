# V_REVIEW: independent check of the review analyses

Own code: v_review_nb.py (naive Bayes fit from data/raw/release_validate_patients, 132,448 patients, 49 conditions, 209 top-level codes, 14 follow-ups under E_53/E_129/E_151; the 5 items whose parent code is missing from the file are treated as top-level), v_review.py (all checks -> v_review_out.json), v_review_coin7.py (exact coin sets for check 7). Nothing imported from src/ddx or scripts/. No cache, no API calls. Logs: last line per pid wins, only the 294 main pids. Bootstrap: 10,000 paired patient resamples, seed 20261103. McNemar: exact (scipy binomtest).

Integrity: no follow-up code was ever asked directly; `asked` matches rounds[].chosen in every game; every final dx is a condition name; replay truth labels match the raw data; no unknown codes.

| # | Check | Project | Mine | Match |
|---|---|---|---|---|
| 1 | NB opening only, top-1 / mean p_truth | 0.367 / 0.283 | 0.367 / 0.283 | MATCH |
| 1 | NB full record (209 top-level codes), top-1 | 0.993 | 0.993 (p_truth 0.993) | MATCH |
| 2 | P0 A/B/C/D, ABCD | .799/.793/.779/.769, .793 | .799/.793/.779/.769, .793 | MATCH |
| 2 | P1 | .854/.759/.765/.799, .806 | .854/.759/.765/.799, .806 | MATCH |
| 2 | P1Q | .830/.769/.782/.830, .782 | .830/.769/.782/.830, .782 | MATCH |
| 2 | P2 | .799/.793/.830/.816, .762 | .799/.793/.830/.816, .762 | MATCH |
| 3 | Union of solos P0/P1/P1Q/P2 | .816/.901/.898/.895 | .816/.901/.898/.895 | MATCH |
| 4 | P2 ABCD − mean solo | −4.76 [−8.0, −1.7] | −4.76 [−7.99, −1.70] | MATCH |
| 4 | P0 / P1 / P1Q | +0.77 / +1.19 / −2.04 | +0.77 [−0.9, +2.6] / +1.19 [−1.5, +3.8] / −2.04 [−5.2, +1.1] | MATCH |
| 5 | Anchor NB top-1; games < 5 questions | 0.850; 61 | 0.850; 61 (43×4, 13×3, 5×2) | MATCH |
| 6 | P0 chosen code backed by 1/2/3/4 valid agents | .005/.101/.171/.722 | .005/.101/.171/.722 (1,470 rounds) | MATCH |
| 6 | P2 | .241/.454/.210/.095 | .241/.454/.210/.095 | MATCH |
| 6 | P2 vote majority / confidence | .683 / .317 | .683 / .317 (no coin) | MATCH |
| 7 | Wrong independent vote, winner given by ≥2 (P0) | 174/174 | coin range 173/173 to 175/175 (7 coin patients) | MATCH (coin) |
| 7 | P1 | 175/179 | achievable; range 174/178 to 178/183 (12 coin patients) | MATCH (coin) |
| 7 | P1Q | 158/187 | achievable; range 158/184 to 162/193 (20 coin patients) | MATCH (coin) |
| 7 | P2 | 149/176 | 149/176 (0 coin patients) | MATCH |
| 8 | SC8 (8 × gemma4_31b P0 solos, r0+r1) top-1 | 0.398 | 0.398 with my coin; range [0.391, 0.398] (3 coin patients) | MATCH |
| 8 | SC8 mean summed cost per patient | $0.0487 | $0.0487 | MATCH |
| 8 | Anchor − SC8 | +6.46 [+1.0, +11.9], p ≈ .025 | +6.46 [+1.02, +11.90]; 42 vs 23, p = 0.0248 (coin range +6.46 to +7.14) | MATCH |
| 9 | P2 total / asking / read+vote / deliberation / residual | +1.70 / −1.11 / −0.34 / +3.74 / −0.60 | +1.70 / −1.11 / −0.34 / +3.74 / −0.60 | MATCH |
| 9 | P1 | +2.38 / +0.26 / +0.68 / +2.04 / −0.60 | +2.38 / +0.26 / +0.68 / +2.04 / −0.60 | MATCH |
| 10 | P0 best on r1, team − it on r0 | D, +2.04 | D, +2.04 [−1.4, +5.4] | MATCH |
| 10 | P1 | D, +4.42 [+0.7, +8.2] | D, +4.42 [+0.68, +8.50] | MATCH (upper end 0.3 pt off, bootstrap noise) |
| 10 | P2 | A, +3.74 | A, +3.74 [−0.7, +8.2] | MATCH |
| 11 | qwen38_27b input $/M; cached share | 0.073; 0.686 | 0.0728; 0.686 (1,900 calls) | MATCH |
| 11 | qwen38_max input / output $/M; cached share | 0.482 / 6.0; 0.868 | 0.4818 / 6.000; 0.868 (2,092 calls) | MATCH |
| 12 | Marker-available subset | — | skipped as instructed | — |

Notes
- Check 7: the support count (≥2) is the same for every tied winner, so the coin only changes whether a patient counts as wrong. Every project pair falls in my exact set of achievable pairs.
- Check 8: the 3 coin patients only move SC8 down (range 0.391–0.398). If all 3 coins go against the truth, anchor − SC8 becomes +7.14 (p = 0.0125), so the project's value is the conservative end.
- Extra (not claimed): P1Q decomposition comes out as total −4.76, asking +0.94, read+vote −7.14, deliberation +2.04, residual −0.60.
