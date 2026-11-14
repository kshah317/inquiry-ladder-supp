# Code for an anonymized submission

Code, prompts, analysis outputs and verification scripts for a study of persona and model diversity in four-agent medical diagnosis teams on DDXPlus.

> The episode logs under `results/main/`, `results/main_r1/`, `results/rotate/`, `results/nopeer/`, `results/replay/` and `results/pilot/` (about 480 MB) are in the supplementary zip submitted with the paper, so the scripts that read them need that zip.

## What is here

- `src/ddx/` is the library: the deterministic oracle over DDXPlus, the prompts, the consultation protocol, the cache, Shapley values, the measures and the statistics.
- `scripts/` holds one numbered script per step, from sampling to runs, analyses and figures.
- `config/study.yaml` holds every setting: seed, samples, rounds, temperature, model IDs, pinned providers and list prices.
- `prereg/` holds the files fixed before the runs: specialty maps, wording fixes, question IDs, marker questions and confidence scales.
- `results/main/` holds the episode logs of the primary run, one JSON line per episode and one file per coalition, with a `MANIFEST.json` per team. `results/main_r1/` is the replication, `results/rotate/` the rotating-proposer follow-up, `results/nopeer/` the no-peer control, `results/replay/` the fixed-reader transcripts and `results/pilot/` the pilot.
- `results/analysis/` holds every analysis output as JSON and short reports, and `results/analysis/verify/` holds the independent verification scripts and their reports.

## Reproducing the numbers without any model call

Python 3.10 or later is needed. These scripts run from the logs in this archive alone.

```
pip install -r requirements.txt
python3 scripts/58_v10_analyses.py     # Table 2, Shapley values, coalition size, retention, vote failures
python3 scripts/52_nopeer_analyse.py   # no-peer control (Table 3)
python3 scripts/53_anchor_analyse.py   # single-agent reference
python3 scripts/54_repeats_analyse.py  # replication and pooled contrasts (Appendix D)
python3 scripts/61_run_stats.py        # run statistics (Appendix A.3)
python3 scripts/62_tables.py           # cost table
python3 scripts/41_report.py           # tables from results/analysis/analysis.json
```

These scripts also read patient records, so they need the DDXPlus patient files, which `scripts/00_download_ddxplus.sh` downloads and checks against `data/raw/SHA256SUMS`.

```
bash scripts/00_download_ddxplus.sh
python3 scripts/40_analyse.py          # primary tests C1 and C2, error overlap, deliberation, persona check, markers
python3 scripts/55_nb_fit.py           # fits the information-ceiling model (parameters are included in results/analysis/_nb/)
python3 scripts/56_review_analyses.py  # information ceiling, decomposition, question vote, self-consistency
python3 scripts/57_p0q_analyse.py      # P0-Q
python3 scripts/59_rotate_analyse.py   # rotating-proposer follow-up (Section 5.3)
python3 tests/test_data.py
```

`scripts/65_figures_v4.py` draws the figures from the analysis files and writes them to `../paper/figures/`, so create that folder first. Every output of these scripts is already in `results/analysis/`.

## Re-running episodes

Runs call the models through OpenRouter and need a key in the environment variable `OPENROUTER_API_KEY`. `scripts/20_run_panel.py` plays all coalitions of one team and is resumable. Seeds depend on the agent, patient and round, and every request is cached by a hash of the full request.

## What is left out

- The DDXPlus patient files (about 200 MB, CC BY 4.0, downloaded by the script above).
- The SQLite cache of all requests and responses (1.3 GB). It will be released on publication. The episode logs here contain every parsed response, and the code rebuilds every request deterministically.
- Local run logs. Host names in the manifests are replaced by "redacted". Nothing else was edited.

## License

Code is under the MIT license (`LICENSE`). DDXPlus is by Fansi Tchango et al. (2022) under CC BY 4.0.

## Layout
| Path | What it does | Status |
|---|---|---|
| `config/study.yaml` | All settings: seed, samples, rounds, temperature, models | done |
| `prereg/` | Files fixed before the runs: specialty maps, wording fixes, question IDs | done |
| `data/raw/` | DDXPlus files + SHA256SUMS (`scripts/00_download_ddxplus.sh`) | done |
| `data/samples/` | `main.csv` (294 test patients), `pilot.csv` (98 validation patients) | done |
| `src/ddx/data.py` | Loads DDXPlus; `answer()` / `reveal()` look up the hidden record (no LLM) | done |
| `src/ddx/text.py` | Opening case, question menu (neutral IDs), answer text, condition list | done |
| `src/ddx/sample.py` | Seeded stratified sampling | done |
| `src/ddx/llm.py` | OpenRouter client: pinned provider, no fallbacks, thinking off, retries, JSON parsing | done |
| `src/ddx/cache.py` | SQLite cache of every request/response (key = hash of full request) | done |
| `src/ddx/prompts.py` | One shared template; role line + specialty checklist only | done |
| `src/ddx/game.py` | 5-round consultation: proposals, vote, tie-break, reveal, final, deliberation | done |
| `src/ddx/panels.py` | Panel definitions (P0, P1, P1-Q, P2, check) and the 15 subgroups | done |
| `src/ddx/shapley.py` | Exact Shapley from 15 subgroup scores + empty-set baseline (self-test passes) | done |
| `src/ddx/budget.py` | Spending stop: reads OpenRouter key/account counters before every 50 games | done |
| `src/ddx/manifest.py` | MANIFEST.json per run: hashes, models, providers served, tokens, cost, truncation, retry/abstain/tie rates | done |
| `src/ddx/metrics.py` | Shapley (both v0), LOO, masking Shapley, independent vote (+coin-free), error overlap + permutation null, deliberation, persona check, markers, coin sensitivity, cost, replay/evidence Shapley | done |
| `src/ddx/stats.py` | Paired bootstrap CIs, exact McNemar, paired TOST, Holm (self-test passes) | done |
| `scripts/40_analyse.py` | All analyses -> `results/analysis/analysis.json` | done |
| `scripts/41_report.py` | Tables -> `results/analysis/ANALYSIS_REPORT.md` | done |
| `scripts/50_replay.py` | Replay test: fixed Gemma reader on every sub-team transcript -> `results/replay/` (resumable, `--estimate`) | done |
| `scripts/51_nopeer.py` | No-peer reconsideration control: same deliberation turn without peer answers, full panels; `--check` reproduces every real deliberation call from cache for free -> `results/nopeer/` | done |
| `scripts/52_nopeer_analyse.py` | No-peer analysis -> `results/analysis/nopeer.json`, `NOPEER_REPORT.md` (independently verified: `results/analysis/verify/v_nopeer.py`) | done |
| `scripts/21_run_anchor.py` | Anchor run: Qwen3.8-Max solo game, chunk-safe, per-call spend cap | done |
| `scripts/53_anchor_analyse.py` | Anchor analysis -> `results/analysis/anchor.json`, `ANCHOR_REPORT.md` (verified: `verify/v_anchor.py`) | done |
| `scripts/58_v10_analyses.py` | Matched baselines (mean member, independent vote), Shapley with the chance baseline, accuracy by coalition size, question-vote retention, vote-failure split, reference model vs P0-Q -> `results/analysis/v10.json` (verified: `verify/v_v10.py`, `V_V10.md`) | done |
| `scripts/59_rotate_analyse.py` | Rotating-proposer follow-up, run with `scripts/20_run_panel.py --question-rule rotate --out rotate` -> `results/analysis/rotate.json` (verified: `verify/v_rotate.py`, `V_ROTATE.md`) | done |
| `scripts/65_figures_v4.py` | Figures of the submitted paper | done |
| `scripts/60_figures.py` | Paper figures 1-3 -> `../paper/figures/` | done |
| `scripts/61_run_stats.py` | Run statistics for appendix A.3 -> `results/analysis/run_stats.json` | done |
| `RUN_REPEATS.bat` | Repeats (replica 1) of P0/P1/P2, sub-teams A,B,C,D,ABCD -> `results/main_r1/` | done; analysis: `scripts/54_repeats_analyse.py` -> `results/analysis/REPEATS_REPORT.md`; tables: `scripts/62_tables.py` -> `paper/tables/` |
| `results/analysis/verify/` | Independent re-computation + RECONCILIATION.md | done |
| `scripts/01_build_specialty_maps.py` | Writes `prereg/condition_specialty.csv`, `evidence_specialty.csv`, `question_ids.csv` | done |
| `scripts/02_build_samples.py` | Writes `data/samples/*.csv` | done |
| `scripts/03_build_subsamples.py` | Writes `data/samples/check.csv` (98, 2/condition) and `trial.csv` (20), both subsets of main | done |
| `scripts/13_opening_only.py` | Opening-only (0-question) baseline on the main sample, 4 team models -> v(empty) baseline (b) | done |
| `scripts/10_pilot.py` | Single-agent pilot (also opening-only and full-record modes) | done |
| `scripts/11_familiarity.py` | Dataset-recall and odd-marker probes | done |
| `scripts/20_run_panel.py` | Runs all 15 subgroups of a panel; resumable, time-budgeted, spending stop, `--replica`, writes MANIFEST.json | done |
| `scripts/30_run_main.py` | Overnight job: P1 -> P0 -> P2 -> P1-Q -> CHECK with per-panel spending caps | done |
| `results/pilot/PILOT_REPORT.md` | Pilot numbers and the decisions they led to | done |
| `tests/test_data.py` | Loader / lookup / template / map checks (`python3 tests/test_data.py`) | 7/7 pass |
| `tests/test_evidence_types.py` | One test per evidence type (binary, categorical + default, multi-choice absent, follow-ups), votes, dx matching | 10/10 pass |
| `tests/test_game_cache.py` | Replica 0 replays pilot games exactly from cache; replica 1 makes new requests (no network) | 2/2 pass |
