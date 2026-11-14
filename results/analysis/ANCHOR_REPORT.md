# Anchor: Qwen3.8-Max alone

Top-1 0.463 [0.405, 0.520], n = 294. Cost $0.182 per consultation.
Truncated at 8,000 tokens: 422 of 2092 calls (20%); retries 328; abstained proposals 84; no final answer 13; games with < 5 questions 61.
Accuracy by truncated calls in the game (exploratory, confounded by case difficulty): 0: 0.607 (n=117), 1: 0.494 (n=77), 2+: 0.270 (n=100)

| Comparison | Anchor − other, pts [95% CI] | Other's cost per consultation | Anchor cost ÷ other |
|---|---|---|---|
| P0 team | +5.4 [+0.0, +10.5], McNemar 40 vs 24, p=0.060 | $0.029 | 6.4× |
| P0 best member (B) | +5.4 [-0.3, +11.2], McNemar 46 vs 30, p=0.085 | $0.0061 | 30× |
| P1 team | +5.1 [-0.3, +10.5], McNemar 41 vs 26, p=0.086 | $0.028 | 6.4× |
| P1 best member (A) | +6.8 [+1.4, +12.2], McNemar 43 vs 23, p=0.019 | $0.0062 | 29× |
| P2 team | +5.8 [+1.4, +10.5], McNemar 34 vs 17, p=0.024 | $0.031 | 5.9× |
| P2 best member (A) | +9.5 [+4.4, +14.6], McNemar 44 vs 16, p=0.000 | $0.0061 | 30× |
| P1Q team | +12.2 [+6.8, +17.7], McNemar 53 vs 17, p=0.000 | $0.020 | 9.3× |
| P1Q best member (D) | +13.6 [+7.8, +19.0], McNemar 58 vs 18, p=0.000 | $0.0042 | 43× |
