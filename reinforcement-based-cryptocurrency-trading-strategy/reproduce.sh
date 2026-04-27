#!/usr/bin/env bash
# Reproducibility test script for the RL Crypto Trading Strategy project.
#
# Run from the repo root on a fresh clone:
#
#   Step 1: Download 36 monthly BTC/USDT 1h CSV files from Binance Vision and
#           place them in  data/raw/1h/  (see Step 1 output for the exact URL).
#   Step 2: bash reproduce.sh

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
info() { printf "${YELLOW}[..] %s${NC}\n" "$*"; }
warn() { printf "${YELLOW}[WARN] %s${NC}\n" "$*"; }
fail() { printf "${RED}[FAIL]${NC} %s\n" "$*"; exit 1; }

RAW_DATA_DIR="data/raw/1h"
EXPECTED_FILES=36
FIRST_FILE="BTCUSDT-1h-2022-01.csv"
LAST_FILE="BTCUSDT-1h-2024-12.csv"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo ""
echo "============================================================"
echo "  Reproducibility Test — RL Crypto Trading Strategy"
echo "  Repo: $REPO_DIR"
echo "  Date: $(date)"
echo "============================================================"
echo ""

# ── 1. Check raw data ─────────────────────────────────────────────────────────
info "Step 1/6 — Checking raw data in $RAW_DATA_DIR/ ..."
mkdir -p "$RAW_DATA_DIR"

CSV_COUNT=$(find "$RAW_DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')

if [[ "$CSV_COUNT" -eq 0 ]]; then
    # Try to unzip from the bundled archive in data/zipped/
    BUNDLE="data/zipped/1h.zip"
    if [[ -f "$BUNDLE" ]]; then
        info "No CSVs found — extracting from bundled $BUNDLE ..."
        unzip -q "$BUNDLE" -d data/raw/
        # zip contains a 1h/ subfolder, so files land at data/raw/1h/*.csv
        CSV_COUNT=$(find "$RAW_DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')
        ok "Extracted $CSV_COUNT CSV files from $BUNDLE"
    else
        echo ""
        printf "${RED}[FAIL]${NC} No CSV files found in %s/ and no bundle at %s\n\n" "$RAW_DATA_DIR" "$BUNDLE"
        echo "  This project requires $EXPECTED_FILES monthly BTC/USDT 1-hour candle"
        echo "  CSV files from Binance Vision covering Jan 2022 – Dec 2024."
        echo ""
        echo "  Download from:"
        echo "    https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1h/"
        echo ""
        echo "  Download BTCUSDT-1h-2022-01.zip through BTCUSDT-1h-2024-12.zip,"
        echo "  unzip each, and place the 36 CSV files in:"
        echo "    $REPO_DIR/$RAW_DATA_DIR/"
        echo ""
        exit 1
    fi
fi

if [[ ! -f "$RAW_DATA_DIR/$FIRST_FILE" ]] || [[ ! -f "$RAW_DATA_DIR/$LAST_FILE" ]]; then
    fail "Found $CSV_COUNT CSV files but missing required range.
  Need: $FIRST_FILE  through  $LAST_FILE
  Found in $RAW_DATA_DIR/:
$(find "$RAW_DATA_DIR" -maxdepth 1 -name "*.csv" | sort | head -5)
  ..."
fi

if [[ "$CSV_COUNT" -lt "$EXPECTED_FILES" ]]; then
    warn "Found $CSV_COUNT of $EXPECTED_FILES expected CSV files — pipeline may fail on missing months."
else
    ok "Found $CSV_COUNT CSV files ($FIRST_FILE → $LAST_FILE)"
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
[[ -z "$PYTHON_BIN" ]] && fail "No Python interpreter found. Install Python 3.12+."

ok "Using $("$PYTHON_BIN" --version 2>&1) ($PYTHON_BIN)"

if [[ ! -d ".venv" ]]; then
    info "Creating virtual environment ..."
    "$PYTHON_BIN" -m venv .venv
    ok "Virtual environment created at .venv/"
else
    ok "Virtual environment already exists — reusing .venv/"
fi

source .venv/bin/activate
info "Installing dependencies from requirements.txt ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# ── 3. Unit tests ─────────────────────────────────────────────────────────────
info "Step 3/6 — Running unit tests ..."
pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest_output.txt
TEST_EXIT=${PIPESTATUS[0]}
if [[ "$TEST_EXIT" -ne 0 ]]; then
    fail "Unit tests failed. See output above."
fi
ok "All tests passed — $(grep -E 'passed' /tmp/pytest_output.txt | tail -1)"

# ── 4. Data pipeline ──────────────────────────────────────────────────────────
info "Step 4/6 — Running data pipeline ..."
cd "$REPO_DIR"   # ensure CWD is repo root for relative path resolution

python -m src.data.pipeline
ok "Data pipeline complete"

for dir in data/processed data/features data/normalized; do
    FILE_COUNT=$(find "$dir" -maxdepth 1 -name "*.csv" | wc -l | tr -d ' ')
    [[ "$FILE_COUNT" -eq 0 ]] && fail "Pipeline produced no files in $dir/"
    ok "$dir/ — $FILE_COUNT file(s)"
done

# ── 5. Execute notebooks ──────────────────────────────────────────────────────
info "Step 5/6 — Executing notebooks (this may take several minutes) ..."
mkdir -p results/figures results/tables

NOTEBOOKS=(
    "notebooks/v2/01_data_exploration.ipynb"
    "notebooks/v2/02_feature_engineering.ipynb"
    "notebooks/v2/03_environment_testing.ipynb"
    "notebooks/v2/04_training_analysis.ipynb"
    "notebooks/v2/05_results_analysis.ipynb"
)

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

# ── 6. Verify outputs ─────────────────────────────────────────────────────────
info "Step 6/6 — Verifying key outputs ..."

[[ ! -f "models/best/best_model.zip" ]] && \
    fail "models/best/best_model.zip not found — training may not have completed"
ok "models/best/best_model.zip — $(du -h models/best/best_model.zip | cut -f1)"

if [[ -f "results/tables/test_metrics.csv" ]]; then
    ok "results/tables/test_metrics.csv"
    echo ""
    echo "── Test Metrics ──────────────────────────────────────────────"
    cat results/tables/test_metrics.csv
    echo "──────────────────────────────────────────────────────────────"
else
    warn "results/tables/test_metrics.csv not found — check notebook 05 output"
fi

echo ""
echo "============================================================"
printf "  ${GREEN}Reproducibility test PASSED${NC}\n"
echo "============================================================"
echo ""
