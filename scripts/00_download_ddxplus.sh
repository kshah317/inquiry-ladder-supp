#!/usr/bin/env bash
# Downloads DDXPlus (Fansi Tchango et al., NeurIPS 2022 D&B; CC-BY) from figshare article 20043374.
# We use only the validation split (pilot + evidence->specialty map) and the test split (main runs).
# The train split is not needed: no model is trained in this study.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
base=https://ndownloader.figshare.com/files
curl -sSL -o README_ddxplus.md          $base/40495496
curl -sSL -o release_evidences.json     $base/40495562
curl -sSL -o release_conditions.json    $base/62657140
curl -sSL -o release_test_patients.zip  $base/40495565
curl -sSL -o release_validate_patients.zip $base/40495571
unzip -o -q release_test_patients.zip
unzip -o -q release_validate_patients.zip
sha256sum -c SHA256SUMS
