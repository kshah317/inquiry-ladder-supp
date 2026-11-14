# V_ROTATE: independent check of the rotating-proposer numbers

Own code: v_rotate.py, from the raw episode logs (last line per pid wins). Naive Bayes ceiling: the earlier independent implementation in v_review_nb.py with its fitted table (not ddx.nb). Nothing imported from src/ddx or scripts/. No API calls.
294 patients per file. Bootstrap: paired over patients, 10,000 resamples, percentile 95% interval, seeds 11 / 2027 / 990131 (three intervals shown in that order). McNemar: exact binomial. CI rule: MATCH if every endpoint is within 0.2 pp; seed-sensitive if within one grid step (0.34 pp for single-transcript differences, 0.09 pp for differences against a four-member mean).

| Claim | Claimed | Mine | Verdict |
|---|---|---|---|
| 1 P1 ceiling, rotate transcript | 0.806 | 0.806 | MATCH |
| 1 P1 ceiling, plurality transcript | 0.806 | 0.806 | MATCH |
| 1 P1 ceiling rotate - plurality | +0.0 [-3.4, +3.4] | +0.0 [-3.7, +3.4] / [-3.4, +3.7] / [-3.4, +3.4] | MATCH (seed-sensitive, one grid step) |
| 1 P1 exact McNemar p | 1.000 | 1.000 (14 vs 14) | MATCH |
| 2 P1 mean member ceiling | 0.794 | 0.794 | MATCH |
| 2 P1 ceiling rotate - mean member | +1.2 [-1.8, +4.2] | +1.2 [-1.8, +4.2] / [-1.8, +4.2] / [-1.8, +4.2] | MATCH |
| 2 P1 pooled ceiling | 0.901 | 0.901 | MATCH |
| 2 P1 ceiling pooled - rotate | +9.5 [+5.4, +13.9] | +9.5 [+5.4, +13.9] / [+5.4, +13.6] / [+5.4, +13.6] | MATCH (seed-sensitive, one grid step) |
| 3 P1 team accuracy, rotate | 0.405 | 0.405 | MATCH |
| 3 P1 team accuracy, plurality | 0.412 | 0.412 | MATCH |
| 3 P1 accuracy rotate - plurality | -0.7 [-4.1, +3.1] | -0.7 [-4.4, +3.1] / [-4.1, +3.1] / [-4.4, +3.1] | MATCH (seed-sensitive, one grid step) |
| 3 P1 independent vote accuracy | 0.391 | 0.391 | MATCH |
| 3 P1 accuracy rotate - independent vote | +1.4 [-2.7, +5.4] | +1.4 [-2.7, +5.4] / [-2.7, +5.4] / [-2.4, +5.4] | MATCH (seed-sensitive, one grid step) |
| 3 P1 mean member accuracy | 0.364 | 0.364 | MATCH |
| 3 P1 accuracy rotate - mean member | +4.1 [+0.5, +7.7] | +4.1 [+0.5, +7.8] / [+0.5, +7.7] / [+0.5, +7.7] | MATCH |
| 3 P1 rotate accuracy before deliberation | 0.374 | 0.374 | MATCH |
| 4 P1 asked questions | 1470 | 1470 | MATCH |
| 4 P1 asked questions per member per patient | 1.18/1.26/1.32/1.24 | 1.18/1.26/1.32/1.24 | MATCH |
| 4 P1 % asked also validly proposed by another member | 63.1 | 63.1 | MATCH |
| 4 P1 distinct questions shared with plurality transcript | 3.91 | 3.91 | MATCH |
| 5 P1 truncated calls | 0 | 0 | MATCH |
| 5 P1 missing final answers | 0 | 0 | MATCH |
| 5 P1 turns passed to another member | 0 | 0 | MATCH |
| 5 P1 mean cost per episode ($) | 0.0283 | 0.0283 | MATCH |
| 1 P2 ceiling, rotate transcript | 0.813 | 0.813 | MATCH |
| 1 P2 ceiling, plurality transcript | 0.762 | 0.762 | MATCH |
| 1 P2 ceiling rotate - plurality | +5.1 [+1.7, +8.8] | +5.1 [+1.4, +8.8] / [+1.4, +8.8] / [+1.4, +8.8] | MATCH (seed-sensitive, one grid step) |
| 1 P2 exact McNemar p | 0.011 | 0.011 (23 vs 8) | MATCH |
| 2 P2 mean member ceiling | 0.810 | 0.810 | MATCH |
| 2 P2 ceiling rotate - mean member | +0.3 [-2.7, +3.2] | +0.3 [-2.6, +3.3] / [-2.6, +3.2] / [-2.6, +3.2] | MATCH |
| 2 P2 pooled ceiling | 0.895 | 0.895 | MATCH |
| 2 P2 ceiling pooled - rotate | +8.2 [+4.4, +12.2] | +8.2 [+4.4, +12.2] / [+4.4, +12.2] / [+4.4, +11.9] | MATCH (seed-sensitive, one grid step) |
| 3 P2 team accuracy, rotate | 0.405 | 0.405 | MATCH |
| 3 P2 team accuracy, plurality | 0.405 | 0.405 | MATCH |
| 3 P2 accuracy rotate - plurality | +0.0 [-4.1, +4.1] | +0.0 [-4.1, +4.1] / [-4.1, +4.1] / [-4.1, +4.1] | MATCH |
| 3 P2 independent vote accuracy | 0.401 | 0.401 | MATCH |
| 3 P2 accuracy rotate - independent vote | +0.3 [-4.4, +4.8] | +0.3 [-4.4, +4.8] / [-4.1, +4.8] / [-4.1, +5.1] | MATCH (seed-sensitive, one grid step) |
| 3 P2 mean member accuracy | 0.349 | 0.349 | MATCH |
| 3 P2 accuracy rotate - mean member | +5.6 [+1.9, +9.4] | +5.6 [+1.9, +9.4] / [+2.0, +9.4] / [+1.9, +9.4] | MATCH |
| 3 P2 rotate accuracy before deliberation | 0.371 | 0.371 | MATCH |
| 4 P2 asked questions | 1470 | 1470 | MATCH |
| 4 P2 asked questions per member per patient | 1.18/1.25/1.33/1.25 | 1.18/1.25/1.33/1.25 | MATCH |
| 4 P2 % asked also validly proposed by another member | 56.6 | 56.6 | MATCH |
| 4 P2 distinct questions shared with plurality transcript | 3.56 | 3.56 | MATCH |
| 5 P2 truncated calls | 0 | 60 | MISMATCH |
| 5 P2 missing final answers | 0 | 0 | MATCH |
| 5 P2 turns passed to another member | 6 | 6 | MATCH |
| 5 P2 mean cost per episode ($) | 0.0305 | 0.0305 | MATCH |

## Protocol check

- P1: PASS. In all 1,470 rounds the asked code is the valid proposal of the named proposer; the proposer is the designated member (rotate_start + k - 1 mod 4) unless that member's proposal was invalid, and a passed turn went to the next member in order with a valid proposal.
- P2: PASS. In all 1,470 rounds the asked code is the valid proposal of the named proposer; the proposer is the designated member (rotate_start + k - 1 mod 4) unless that member's proposal was invalid, and a passed turn went to the next member in order with a valid proposal.

## Log notes

- Truth labels in the rotate logs vs raw data: 0 mismatches. Unknown question codes: 0. File/pid problems: none.
- P1: 8453 calls, finish reasons {'stop': 8453}; truncated by member {}; invalid proposals 0; episodes with final list not of length 4: 0; empty revised answers 0; calls without cost 0; calls flagged cached 5561; rotate_start counts {0: 52, 1: 77, 2: 94, 3: 71}; episodes with fewer than 5 distinct asked questions 0; independent-vote coin flips 12.
- P2: 8931 calls, finish reasons {'stop': 8871, 'length': 60}; truncated by member {'Physician C': 14, 'Physician D': 45, 'Physician B': 1}; invalid proposals 30; episodes with final list not of length 4: 0; empty revised answers 0; calls without cost 0; calls flagged cached 5307; rotate_start counts {0: 52, 1: 77, 2: 94, 3: 71}; episodes with fewer than 5 distinct asked questions 0; independent-vote coin flips 0.
