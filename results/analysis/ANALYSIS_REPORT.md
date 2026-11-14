# Analysis report (scripts/40_analyse.py → 41_report.py)

Bootstrap B = 10000 over patients (seed 20261120); permutation null 2000 draws; TOST margin ±5 pts.
Differences are in percentage points. Independently re-computed by a separate verification script (results/analysis/verify/).

## 1. Primary contrasts (294 patients, paired)

| Contrast | Diff | 95% CI | 90% CI | McNemar (a only / b only, p) | TOST p (±5) |
|---|---|---|---|---|---|
| C1: P1 − P0 | +0.3 | [-3.4, +3.7] | [-2.7, +3.4] | 15 / 14, p = 1.00 | 0.006 |
| C2a: P2 − P0 | -0.3 | [-4.8, +4.1] | [-4.1, +3.4] | 21 / 22, p = 1.00 | 0.019 |
| C2b: P2 − its members' independent vote | +0.3 | [-4.1, +4.8] | [-3.4, +4.1] | 21 / 20, p = 1.00 | 0.017 |

TOST was planned for C1 only; for C2 it is shown for interpretation. Holm-adjusted McNemar p (C2a, C2b): 1.00, 1.00.
Secondary: P1-Q − P0 = -6.8 [-12.2, -1.0].

Coin averaged out of the final vote (sensitivity): C1 +0.7 [-2.9, +4.2]; C2a -0.3 [-4.8, +4.1]; C2b +0.3 [-4.1, +4.8].

No-marker subset (n = 82 patients with no marker question available; sanity check, low power): C1 +4.9 [-2.4, +12.2]; C2a +1.2 [-7.3, +9.8]; C2b +2.4 [-6.1, +11.0].

## 2. Is the team worth it? (full 4-agent panels)

| Panel | Team | Before delib. | Solo A/B/C/D | Team − best member [95% CI] | (best re-chosen per draw) | Indep. vote | Team − indep. vote [95% CI] | coin-free | Any solo right |
|---|---|---|---|---|---|---|---|---|---|
| P0 same model | 0.408 | 0.401 | 0.367/0.408/0.388/0.388 | +0.0 [-3.4, +3.4] | [-3.4, +2.7] | 0.408 | +0.0 [-2.7, +2.7] | +0.0 | 0.469 |
| P1 roles (Gemma) | 0.412 | 0.391 | 0.395/0.344/0.350/0.367 | +1.7 [-2.7, +6.1] | [-2.7, +5.4] | 0.391 | +2.0 [-1.4, +5.4] | +2.9 | 0.531 |
| P1-Q roles (Qwen) | 0.340 | 0.320 | 0.313/0.309/0.320/0.327 | +1.4 [-3.1, +5.8] | [-3.7, +4.1] | 0.364 | -2.4 [-6.5, +1.7] | -2.0 | 0.510 |
| P2 mixed models | 0.405 | 0.367 | 0.367/0.344/0.327/0.357 | +3.7 [-0.7, +8.2] | [-1.0, +6.8] | 0.401 | +0.3 [-4.1, +4.8] | +0.3 | 0.568 |
| CHECK (98) | 0.510 | 0.469 | 0.388/0.337/0.061/0.745 | -23.5 [-32.6, -14.3] | [-32.6, -14.3] | 0.561 | -5.1 [-15.3, +4.1] | -5.1 | 0.775 |

Sub-teams by size (mean over sub-teams): team − best member / team − independent vote

| Panel | pairs | triples | full |
|---|---|---|---|
| P0 | +1.2 [-0.8, +3.2] / -0.2 | +0.3 [-2.0, +2.5] / +1.6 | +0.0 [-3.4, +3.4] / +0.0 |
| P1 | +2.0 [-0.4, +4.4] / -0.8 | +0.9 [-2.5, +4.3] / +1.3 | +1.7 [-2.7, +6.1] / +2.0 |
| P1Q | +1.7 [-0.9, +4.3] / -2.1 | +5.8 [+2.5, +9.2] / +2.3 | +1.4 [-3.1, +5.8] / -2.4 |
| P2 | +0.7 [-1.6, +3.1] / -2.8 | +4.1 [+0.9, +7.4] / +0.3 | +3.7 [-0.7, +8.2] / +0.3 |
| CHECK | -2.9 [-6.8, +1.0] / -1.2 | -11.2 [-17.3, -5.6] / -4.1 | -23.5 [-32.6, -14.3] / -5.1 |

Pair synergy by pair type (pair − better member; pair − the two members' independent vote)

| Panel | Pair type | n pairs | Synergy [95% CI] | − indep. vote [95% CI] |
|---|---|---|---|---|
| P0 | same model | 6 | +1.2 [-0.8, +3.2] | -0.2 [-2.0, +1.7] |
| P1 | persona | 6 | +2.0 [-0.4, +4.4] | -0.8 [-3.0, +1.4] |
| P1Q | persona | 6 | +1.7 [-0.9, +4.3] | -2.1 [-4.6, +0.3] |
| P2 | cross family | 4 | +0.4 [-2.0, +2.8] | -3.2 [-5.8, -0.8] |
| P2 | same family | 2 | +1.2 [-1.9, +4.2] | -1.9 [-4.9, +1.2] |
| CHECK | check | 6 | -2.9 [-6.8, +1.0] | -1.2 [-5.1, +2.5] |

## 3. Who gets credit (Shapley, per agent A/B/C/D)

| Panel | Roles/models | Shapley v0 = 1/49 | Shapley v0 = opening-only | LOO | Masking Shapley (no re-execution) |
|---|---|---|---|---|---|
| P0 | v0(b) = 0.225 | 0.083 / 0.107 / 0.099 / 0.098 | 0.032 / 0.056 / 0.048 / 0.047 | -0.007 / 0.010 / 0.000 / 0.007 | 0.097 / 0.097 / 0.097 / 0.097 |
| P1 | v0(b) = 0.225 | 0.125 / 0.079 / 0.088 / 0.100 | 0.074 / 0.028 / 0.037 / 0.049 | 0.041 / -0.003 / 0.000 / 0.020 | 0.095 / 0.101 / 0.099 / 0.097 |
| P1Q | v0(b) = 0.191 | 0.073 / 0.084 / 0.082 / 0.081 | 0.030 / 0.042 / 0.039 / 0.038 | -0.058 / -0.024 / -0.031 / -0.058 | 0.090 / 0.077 / 0.069 / 0.084 |
| P2 | v0(b) = 0.196 | 0.105 / 0.093 / 0.084 / 0.102 | 0.061 / 0.049 / 0.040 / 0.058 | -0.010 / 0.010 / -0.010 / 0.007 | 0.103 / 0.090 / 0.093 / 0.098 |
| CHECK | v0(b) = 0.225 | 0.088 / 0.048 / -0.032 / 0.386 | 0.037 / -0.003 / -0.083 / 0.335 | -0.071 / -0.143 / -0.051 / 0.133 | 0.072 / 0.072 / 0.030 / 0.316 |

95% CIs per agent are in analysis.json. CHECK: strong (full record) agent D vs weak agent C separates clearly (positive control passes).

## 4. Why: error overlap on solo games vs within-condition permutation null

| Panel | Pair type | Both wrong (obs / null) | Same wrong answer (obs / null) | Same answer given both wrong (obs / null) |
|---|---|---|---|---|
| P0 | same model | 0.568 / 0.449 (p 0.0005) | 0.489 / 0.160 (p 0.0005) | 0.860 / 0.356 (p 0.0005) |
| P1 | persona | 0.546 / 0.469 (p 0.0005) | 0.351 / 0.141 (p 0.0005) | 0.640 / 0.300 (p 0.0005) |
| P1Q | persona | 0.585 / 0.526 (p 0.0005) | 0.196 / 0.097 (p 0.0005) | 0.335 / 0.184 (p 0.0005) |
| P2 | cross family | 0.538 / 0.485 (p 0.0005) | 0.181 / 0.086 (p 0.0005) | 0.337 / 0.176 (p 0.0005) |
| P2 | same family | 0.542 / 0.486 (p 0.0005) | 0.213 / 0.102 (p 0.0005) | 0.393 / 0.211 (p 0.0005) |
| CHECK | check | 0.415 / 0.401 (p 0.0005) | 0.104 / 0.077 (p 0.0005) | 0.285 / 0.238 (p 0.0050) |

p = one-sided permutation p (smallest possible = 1/(perm+1)).

## 5. Deliberation (full 4-agent games; no-peer control still to run)

| Panel | Switch rate | Right→wrong | Wrong→right | Wrong→other wrong | Switches to a peer's answer | Team before → after |
|---|---|---|---|---|---|---|
| P0 | 0.038 | 3 | 14 | 28 | 0.98 | 0.401 → 0.408 |
| P1 | 0.146 | 11 | 51 | 110 | 0.98 | 0.391 → 0.412 |
| P1Q | 0.250 | 25 | 57 | 212 | 0.97 | 0.320 → 0.340 |
| P2 | 0.322 | 38 | 101 | 240 | 0.96 | 0.367 → 0.405 |
| CHECK | 0.148 | 7 | 23 | 28 | 0.81 | 0.469 → 0.510 |

## 6. Persona manipulation check (share of proposed questions in the role's field; specialist − generalist, same panel)

| Panel | Role | Solo, rounds 1-2 | Solo, rounds 4-5 | Full panel, all rounds | Final dx in field (solo) |
|---|---|---|---|---|---|
| P1 | cardiology | 0.11 vs 0.03 (+7.5 [+5.4, +9.7]) | 0.24 vs 0.14 (+9.9 [+6.5, +13.4]) | 0.27 vs 0.09 (+18.2 [+15.6, +20.9]) | 0.15 vs 0.13 (+2.7 [-0.3, +5.8]) |
| P1 | pulmonology | 0.68 vs 0.37 (+31.3 [+27.7, +34.9]) | 0.36 vs 0.30 (+6.1 [+1.9, +10.4]) | 0.69 vs 0.32 (+37.5 [+34.5, +40.7]) | 0.30 vs 0.24 (+5.4 [+1.7, +9.2]) |
| P1 | infectious | 0.43 vs 0.16 (+27.0 [+24.1, +29.9]) | 0.17 vs 0.13 (+4.6 [+1.0, +8.3]) | 0.66 vs 0.14 (+51.7 [+48.6, +54.9]) | 0.11 vs 0.09 (+2.0 [-0.7, +4.8]) |
| P1Q | cardiology | 0.24 vs 0.10 (+13.8 [+10.7, +16.8]) | 0.25 vs 0.13 (+12.6 [+8.8, +16.3]) | 0.31 vs 0.12 (+19.8 [+17.0, +22.6]) | 0.18 vs 0.11 (+7.1 [+3.4, +11.2]) |
| P1Q | pulmonology | 0.53 vs 0.26 (+27.4 [+23.0, +31.8]) | 0.35 vs 0.25 (+10.5 [+5.8, +15.1]) | 0.54 vs 0.25 (+29.1 [+25.8, +32.4]) | 0.36 vs 0.22 (+14.0 [+8.5, +19.4]) |
| P1Q | infectious | 0.45 vs 0.17 (+27.6 [+24.3, +30.9]) | 0.19 vs 0.11 (+7.5 [+3.7, +11.1]) | 0.47 vs 0.12 (+35.0 [+31.8, +38.1]) | 0.09 vs 0.04 (+4.1 [+1.0, +7.1]) |

## 7. Marker questions (a 'yes' to a question unique to the true condition)

| Panel | Hit rate solo / pair / triple / full | Accuracy given hit (solo) | Accuracy, marker available but not hit (solo) | Full − mean solo hit rate |
|---|---|---|---|---|
| P0 | 0.101 / 0.102 / 0.097 / 0.099 | 0.97 | 0.33 | -0.3 [-1.5, +0.9] |
| P1 | 0.094 / 0.095 / 0.095 / 0.085 | 0.98 | 0.31 | -0.9 [-2.3, +0.3] |
| P1Q | 0.094 / 0.102 / 0.118 / 0.119 | 0.94 | 0.25 | +2.5 [+0.2, +4.9] |
| P2 | 0.093 / 0.105 / 0.105 / 0.109 | 0.90 | 0.31 | +1.5 [-0.6, +3.8] |
| CHECK | 0.089 / 0.124 / 0.158 / 0.133 | 0.67 | 0.39 | +4.3 [-0.5, +9.4] |

Patients with at least one marker available: 212/294.

## 8. Coin-flip tie-break sensitivity (final diagnosis vote recomputed with a pure seeded coin)

Largest change in any sub-team's accuracy: P0 +0.3, P1 +0.7, P1Q +1.4, P2 +1.0, CHECK +22.4 (CHECK pairs with the weak agent depend on the confidence rule).
Primary contrasts with coin-only votes: C1 +0.7 [-3.1, +4.4]; C2a +0.0 [-4.4, +4.4]; C2b +2.7 [-1.4, +6.8].

## 9. Cost (nominal; cached calls keep their original cost)

| Panel | $ per 4-agent consultation | Calls | $ per solo consultation | $ per patient, all 15 sub-teams (attribution campaign) |
|---|---|---|---|---|
| P0 | 0.0286 | 29.1 | 0.0061 | 0.223 |
| P1 | 0.0283 | 28.8 | 0.0061 | 0.222 |
| P1Q | 0.0197 | 29.7 | 0.0042 | 0.153 |
| P2 | 0.0310 | 30.6 | 0.0067 | 0.244 |
| CHECK | 0.0243 | 32.3 | 0.0052 | 0.190 |

## 10. Exploratory: specialists' Shapley on in-field vs out-of-field patients (v0 = 1/49)

| Panel | Role | n in field | Shapley in / out | Solo accuracy in / out |
|---|---|---|---|---|
| P1 | cardiology | 48 | 0.170 / 0.108 | 0.396 / 0.382 |
| P1 | pulmonology | 42 | 0.056 / 0.072 | 0.238 / 0.333 |
| P1 | infectious | 36 | 0.099 / 0.077 | 0.278 / 0.333 |
| P1Q | cardiology | 48 | 0.116 / 0.062 | 0.312 / 0.303 |
| P1Q | pulmonology | 42 | 0.108 / 0.079 | 0.333 / 0.308 |
| P1Q | infectious | 36 | 0.104 / 0.068 | 0.222 / 0.325 |

## 11. Replay test: asking vs reading (fixed reader = Gemma 4 31B generalist on each sub-team's transcript)

e = reader accuracy on the team's 5-question transcript (evidence value); v = the team's own answer.

| Panel | Distinct transcripts / patient (of 15) | e solo A/B/C/D | e full | v full | e full − best solo e [95% CI] | v full − e full (reading+delib+vote) [95% CI] |
|---|---|---|---|---|---|---|
| P0 | 3.8 | 0.367/0.398/0.378/0.384 | 0.398 | 0.408 | +0.0 [-3.1, +3.1] | +1.0 [-0.3, +2.7] |
| P1 | 10.1 | 0.408/0.367/0.354/0.367 | 0.384 | 0.412 | -2.4 [-6.8, +2.0] | +2.7 [+0.7, +5.1] |
| P1Q | 13.0 | 0.398/0.360/0.374/0.432 | 0.391 | 0.340 | -4.1 [-8.8, +0.3] | -5.1 [-10.2, -0.3] |
| P2 | 12.1 | 0.367/0.405/0.432/0.398 | 0.371 | 0.405 | -6.1 [-10.5, -1.7] | +3.4 [-0.3, +7.1] |
| CHECK | 11.1 | 0.408/0.337/0.296/0.357 | 0.408 | 0.510 | +0.0 [-10.2, +10.2] | +10.2 [+3.1, +18.4] |

Exact decomposition against the Gemma generalist's own solo game (reader = that agent, so e = v there):

| Panel | Total: team − Gemma generalist solo | Asking: e(team) − e(solo) | Reading + deliberation + vote: v(team) − e(team) |
|---|---|---|---|
| P0 (ref A) | +4.1 [+0.7, +7.8] | +3.1 [-0.3, +6.5] | +1.0 [-0.3, +2.7] |
| P1 (ref D) | +4.4 [+0.7, +8.2] | +1.7 [-2.0, +5.4] | +2.7 [+0.7, +5.1] |
| P2 (ref A) | +3.7 [-0.7, +8.2] | +0.3 [-3.7, +4.4] | +3.4 [-0.3, +7.1] |

| Panel | Evidence Shapley A/B/C/D (v0 = 1/49) | Answer − evidence Shapley A/B/C/D [95% CI] |
|---|---|---|
| P0 | 0.083 / 0.103 / 0.094 / 0.097 | +0.1 [-0.5, +0.7] / +0.4 [-0.4, +1.2] / +0.5 [-0.3, +1.3] / +0.1 [-0.7, +0.9] |
| P1 | 0.115 / 0.086 / 0.077 / 0.086 | +0.9 [-0.6, +2.5] / -0.7 [-2.0, +0.6] / +1.1 [-0.5, +2.8] / +1.4 [+0.2, +2.7] |
| P1Q | 0.100 / 0.085 / 0.082 / 0.103 | -2.8 [-5.3, -0.2] / -0.1 [-2.4, +2.2] / -0.0 [-2.4, +2.5] / -2.2 [-4.7, +0.2] |
| P2 | 0.060 / 0.098 / 0.105 / 0.088 | +4.5 [+2.6, +6.5] / -0.4 [-2.6, +1.7] / -2.1 [-4.5, +0.3] / +1.4 [-0.9, +3.6] |
| CHECK | 0.114 / 0.095 / 0.075 / 0.104 | -2.5 [-5.9, +0.7] / -4.8 [-8.7, -0.8] / -10.7 [-14.5, -7.1] / +28.2 [+21.5, +35.0] |

Do diverse panels ask better? e(full panel) − e(P0 full panel): P1 -1.4 [-4.8, +2.0]; P2 -2.7 [-7.1, +2.0]; P1Q -0.7 [-6.1, +5.1].

Exploratory (post hoc): qwen_gen_transcript_vs_P0_team: +2.4 [-2.7, +7.5]; qwen_gen_transcript_vs_P2_team: +2.7 [-2.4, +7.8]; qwen_gen_transcript_vs_P2_joint_transcript: +6.1 [+1.7, +10.5]; mean_4_qwen_solo_transcripts_vs_mean_4_gemma_solo_transcripts: +0.9 [-2.6, +4.7]; qwen_reads_own_vs_gemma_reads_qwen_mean_4: -7.4 [-11.1, -3.7].
