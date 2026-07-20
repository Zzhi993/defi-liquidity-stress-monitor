#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ "${REPRODUCE_SUBMITTED_RESULTS:-0}" != "1" ]]; then
  "${PYTHON_BIN}" src/01_fetch_defillama.py
  "${PYTHON_BIN}" src/02_build_features.py
  "${PYTHON_BIN}" src/02b_build_market_control.py
else
  echo "Using submitted staging and processed data; network extraction skipped."
fi
"${PYTHON_BIN}" src/03_model_contagion.py
"${PYTHON_BIN}" src/04_make_visuals_and_report.py
"${PYTHON_BIN}" src/05_make_academic_report.py
"${PYTHON_BIN}" src/06_validate_outputs.py

echo "Pipeline complete. Quality checks passed; open report/technical_report.pdf."
