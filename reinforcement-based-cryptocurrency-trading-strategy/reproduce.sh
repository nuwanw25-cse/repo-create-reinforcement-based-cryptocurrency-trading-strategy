#!/usr/bin/env bash
# Reproducibility test script for the RL Crypto Trading Strategy project.
# Run this from the repo root after a fresh clone to verify the full pipeline.
#
# Usage:
#   bash reproduce.sh                          # expects data/raw/1h/ already populated
#   bash reproduce.sh --data-source /path/1h   # copies raw CSVs from another location

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
info() { printf "${YELLOW}[..] %s${NC}\n" "$*"; }
warn() { printf "${YELLOW}[WARN] %s${NC}\n" "$*"; }
fail() { printf "${RED}[FAIL]${NC} %s\n" "$*"; exit 1; }

RAW_DATA_DIR="data/raw/1h"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
echo ""
echo "============================================================"
echo "  Reproducibility Test — RL Crypto Trading Strategy"
echo "  Repo: $REPO_DIR"
echo "  Date: $(date)"
echo "============================================================"
echo ""

# ── 0. Create data directories ────────────────────────────────────────────────
mkdir -p "$RAW_DATA_DIR"

# ── 0a. Optional: copy raw data from another location ────────────────────────
DATA_SOURCE=""
if [[ "${1:-}" == "--data-source" && -n "${2:-}" ]]; then
    DATA_SOURCE="$2"
fi

if [[ -n "$DATA_SOURCE" ]]; then
    info "Copying raw data from $DATA_SOURCE ..."
    cp -r "$DATA_SOURCE"/. "$RAW_DATA_DIR/"
    ok "Raw data copied ($(find "$RAW_DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ') CSV files)"
fi

# ── 1. Check raw data exists ──────────────────────────────────────────────────
info "Step 1/6 — Checking raw data ..."
# Use find instead of ls so the command returns 0 even when no files match
# (ls *.csv exits non-zero when empty, which trips set -e / pipefail)
CSV_COUNT=$(find "$RAW_DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')
if [[ "$CSV_COUNT" -eq 0 ]]; then
    echo ""
    printf "${RED}[FAIL]${NC} No CSV files found in %s\n" "$RAW_DATA_DIR"
    echo ""
    echo "  This project requires 36 monthly BTC/USDT 1-hour candle files"
    echo "  from Binance Vision (Jan 2022 – Dec 2024)."
    echo ""
    echo "  Download them from:"
    echo "    https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1h/"
    echo ""
    echo "  Files needed (BTCUSDT-1h-YYYY-MM.zip, unzip each):"
    echo "    BTCUSDT-1h-2022-01.csv  through  BTCUSDT-1h-2024-12.csv"
    echo ""
    echo "  Place all 36 CSV files in:"
    echo "    $REPO_DIR/$RAW_DATA_DIR/"
    echo ""
    echo "  Or re-run with --data-source if you have them elsewhere:"
    echo "    bash reproduce.sh --data-source /path/to/existing/1h"
    echo ""
    exit 1
fi

EXPECTED=36
if [[ "$CSV_COUNT" -lt "$EXPECTED" ]]; then
    warn "Found only $CSV_COUNT of $EXPECTED expected CSV files in $RAW_DATA_DIR/ — pipeline may fail."
else
    ok "Found $CSV_COUNT CSV files in $RAW_DATA_DIR/"
fi

# ── 2. Python environment ─────────────────────────────────────────────────────
info "Step 2/6 — Setting up Python environment ..."

PYTHON_BIN=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done
[[ -z "$PYTHON_BIN" ]] && fail "No Python interpreter found. Install Python 3.12."

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
ok "Using $PYTHON_VERSION ($PYTHON_BIN)"

if [[ ! -d ".venv" ]]; then
    info "Creating virtual environment ..."
    "$PYTHON_BIN" -m venv .venv
    ok "Virtual environment created at .venv/"
else
    ok "Virtual environment already exists at .venv/"
fi

source .venv/bin/activate
info "Installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# ── 3. Run unit tests ─────────────────────────────────────────────────────────
info "Step 3/6 — Running unit tests ..."
pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.txt
TEST_EXIT=${PIPESTATUS[0]}
if [[ "$TEST_EXIT" -ne 0 ]]; then
    fail "Unit tests failed. See output above."
fi
PASSED=$(grep -E "passed" /tmp/pytest_output.txt | tail -1)
ok "All tests passed — $PASSED"

# ── 4. Run data pipeline ──────────────────────────────────────────────────────
info "Step 4/6 — Running data pipeline ..."

# Explicitly return to repo root — pytest or venv activation can shift CWD
cd "$REPO_DIR"

# Sanity-check the first required file before handing off to Python
FIRST_FILE="$RAW_DATA_DIR/BTCUSDT-1h-2022-01.csv"
if [[ ! -f "$FIRST_FILE" ]]; then
    fail "Expected file not found: $FIRST_FILE
  Files must be directly inside $RAW_DATA_DIR/ (not in a sub-folder).
  Current contents of $RAW_DATA_DIR/:
$(ls "$RAW_DATA_DIR"/ | head -10)"
fi

python -m src.data.pipeline
ok "Data pipeline complete"

# Verify expected output directories
for dir in data/processed data/features data/normalized; do
    FILE_COUNT=$(find "$dir" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')
    if [[ "$FILE_COUNT" -eq 0 ]]; then
        fail "Pipeline did not produce files in $dir/"
    fi
    ok "$dir/ — $FILE_COUNT files"
done

# ── 5. Execute notebooks ──────────────────────────────────────────────────────
info "Step 5/6 — Executing notebooks (this may take several minutes) ..."

# Notebooks save plots and CSVs into results/ — create dirs so savefig doesn't fail
mkdir -p results/figures results/tables

NOTEBOOKS=(
    "notebooks/v2/01_data_exploration.ipynb"
    "notebooks/v2/02_feature_engineering.ipynb"
    "notebooks/v2/03_environment_testing.ipynb"
    "notebooks/v2/04_training_analysis.ipynb"
    "notebooks/v2/05_results_analysis.ipynb"
)

mkdir -p /tmp/nb_outputs

for NB in "${NOTEBOOKS[@]}"; do
    NB_NAME=$(basename "$NB")
    info "  Running $NB_NAME ..."
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        "$NB" 2>&1 | tail -3
    ok "  $NB_NAME — done"
done

# ── 6. Verify key outputs ─────────────────────────────────────────────────────
info "Step 6/6 — Verifying key outputs ..."

# Trained model
if [[ ! -f "models/best/best_model.zip" ]]; then
    fail "models/best/best_model.zip not found — training may not have completed"
fi
ok "models/best/best_model.zip exists ($(du -h models/best/best_model.zip | cut -f1))"

# Results CSV
if [[ ! -f "results/tables/test_metrics.csv" ]]; then
    warn "results/tables/test_metrics.csv not found — check notebook 05 output"
else
    ok "results/tables/test_metrics.csv exists"
    echo ""
    echo "── Test Metrics ──────────────────────────────────────────"
    cat results/tables/test_metrics.csv
    echo "──────────────────────────────────────────────────────────"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
printf "  ${GREEN}Reproducibility test PASSED${NC}\n"
echo "============================================================"
echo ""
