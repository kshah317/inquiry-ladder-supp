"""Run the main panels one after another (overnight job). Resumable: just start it again.

Order and per-panel spending caps (USD of NEW spend; estimate x ~1.3). The global stop in
config/study.yaml (key spend $394 / balance $50) applies on top of these.
  P1 -> P0 -> P2 -> P1Q -> CHECK (98-patient subset)
usage (Windows):  py scripts\\30_run_main.py            (or: python scripts/30_run_main.py)
       one panel: py scripts\\30_run_main.py --only P2
Run ONE copy at a time, and never while another machine/session writes to the same cache file.
Progress: results/logs/main_<date>.log ; per-panel manifest: results/main/<panel>/MANIFEST.json
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = [("P1", "main", 85), ("P0", "main", 85), ("P2", "main", 90), ("P1Q", "main", 60), ("CHECK", "check", 25)]

ap = argparse.ArgumentParser()
ap.add_argument("--only", default=None)
ap.add_argument("--hours-per-panel", type=float, default=6)
ap.add_argument("--concurrency", type=int, default=48)
a = ap.parse_args()
log = ROOT / "results" / "logs" / f"main_{time.strftime('%Y%m%d_%H%M')}.log"
log.parent.mkdir(parents=True, exist_ok=True)
for panel, sample, cap in PLAN:
    if a.only and panel != a.only:
        continue
    cmd = [sys.executable, str(ROOT / "scripts" / "20_run_panel.py"), "--panel", panel, "--sample", sample,
           "--budget", str(int(a.hours_per_panel * 3600)), "--max-spend", str(cap),
           "--concurrency", str(a.concurrency), "--games-in-flight", "24"]
    print(f"=== {time.strftime('%H:%M')} {panel} ({sample}), cap ${cap}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M')} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    tail = log.read_text(encoding="utf-8").splitlines()[-3:]
    print("\n".join(tail), flush=True)
    if any("reached the stop" in t or "reserve floor" in t or "could not read" in t for t in tail):
        print("Global spending stop hit - not starting further panels.")
        break
print("done. Log:", log)
