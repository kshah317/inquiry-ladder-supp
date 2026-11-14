# No-peer reconsideration control

Full panels (ABCD), 294 patients each. Team top-1 accuracy; differences in pts [95% bootstrap CI].

| Panel | Pre | Real delib | No-peer | Real − pre | No-peer − pre (second look) | Real − no-peer (peer effect) | McNemar real vs np | Coin-free peer effect |
|---|---|---|---|---|---|---|---|---|
| P0 | 0.401 | 0.408 | 0.415 | +0.7 [+0.0, +1.7] | +1.4 [+0.3, +2.7] | -0.7 [-1.7, +0.0] | 0 vs 2, p=0.500 | -0.7 [-1.7, +0.0] |
| P1 | 0.391 | 0.412 | 0.398 | +2.0 [+0.3, +4.1] | +0.7 [-0.7, +2.0] | +1.4 [-0.3, +3.4] | 6 vs 2, p=0.289 | +1.2 [-0.7, +3.1] |
| P1-Q | 0.320 | 0.340 | 0.337 | +2.0 [+0.0, +4.4] | +1.7 [+0.0, +3.4] | +0.3 [-1.7, +2.4] | 6 vs 5, p=1.000 | +0.3 [-1.7, +2.4] |
| P2 | 0.367 | 0.405 | 0.384 | +3.7 [+1.0, +6.8] | +1.7 [-0.7, +4.1] | +2.0 [-0.3, +4.4] | 9 vs 3, p=0.146 | +2.0 [-0.3, +4.4] |

**Pooled (mean of 4 panels):** deliberation +2.1 [+1.0, +3.3] = second look +1.4 [+0.4, +2.4] + peer effect +0.8 [-0.3, +1.8]; coin-free peer effect +0.7 [-0.3, +1.8].
Reading rule: pooled peer-effect CI includes 0: not attributable to peers.

Per agent answer (1,176 answers per panel):

| Panel | Condition | Switch rate | Wrong→right | Right→wrong | Wrong→other wrong | Switches to a peer's answer | Unanimous after |
|---|---|---|---|---|---|---|---|
| P0 | real | 3.8% | 14 | 3 | 28 | 98% | 99% (pre 90%) |
| P0 | no-peer | 3.0% | 17 | 1 | 17 | 31% | 89% (pre 90%) |
| P1 | real | 14.6% | 51 | 11 | 110 | 98% | 94% (pre 68%) |
| P1 | no-peer | 3.3% | 24 | 0 | 15 | 36% | 67% (pre 68%) |
| P1-Q | real | 25.0% | 57 | 25 | 212 | 97% | 77% (pre 40%) |
| P1-Q | no-peer | 6.5% | 18 | 2 | 56 | 63% | 42% (pre 40%) |
| P2 | real | 32.2% | 101 | 38 | 240 | 96% | 73% (pre 32%) |
| P2 | no-peer | 9.4% | 26 | 5 | 79 | 62% | 33% (pre 32%) |

Mean member accuracy (pre / real / no-peer): P0 0.399 / 0.408 / 0.412; P1 0.380 / 0.414 / 0.401; P1-Q 0.311 / 0.338 / 0.325; P2 0.345 / 0.399 / 0.363

Run health: P0 retries 0, kept-after-invalid 0, truncated 0, $1.20; P1 retries 0, kept-after-invalid 0, truncated 0, $1.20; P1-Q retries 1, kept-after-invalid 1, truncated 0, $0.84; P2 retries 1, kept-after-invalid 0, truncated 1, $1.25
