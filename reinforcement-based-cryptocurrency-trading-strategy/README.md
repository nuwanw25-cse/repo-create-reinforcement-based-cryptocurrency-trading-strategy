# Reinforcement Learning-Based Cryptocurrency Trading Strategy

CS5998 Capstone Project — Index 258743R

A Deep Q-Network (DQN) agent trained to make buy/sell/hold decisions on BTC/USDT
daily candlestick data, benchmarked against Buy-and-Hold and Moving Average Crossover baselines.

---

## Project Structure

```
.
├── data/
│   ├── raw/          # Downloaded CSVs from Binance Vision
│   ├── processed/    # Cleaned and normalized data
│   └── features/     # Data with computed technical indicators
├── src/
│   ├── data/         # Data loading, preprocessing, feature engineering
│   ├── environment/  # Custom Gymnasium trading environment
│   ├── agents/       # DQN agent and rule-based baselines
│   ├── training/     # Training loop and hyperparameter search
│   ├── evaluation/   # Financial metrics and backtesting pipeline
│   └── visualization/# Plotting utilities
├── configs/          # Hyperparameter and experiment configuration
├── models/           # Saved model weights and checkpoints
├── logs/             # TensorBoard training logs
├── results/          # Metrics outputs and generated figures
├── notebooks/        # Jupyter notebooks for exploration and analysis
├── tests/            # Unit tests
└── reports/          # Final report figures
```

---

## Setup

```bash
# Create virtual environment (Python 3.12 via Homebrew)
/opt/homebrew/bin/python3.12 -m venv .venv

# Install dependencies
pip install -r requirements.txt
```

---

## Data

Download BTC/USDT daily OHLCV CSV from [Binance Vision](https://data.binance.vision/)
and place it in `data/raw/`.

---

## Usage

```bash
# 1. Preprocess data and compute features
python -m src.data.preprocessor

# 2. Train the DQN agent
python -m src.training.trainer

# 3. Run backtesting and evaluation
python -m src.evaluation.backtester

# 4. Run tests
pytest tests/
```

---

## Virtual Environment

```bash
# Activate (run this every time you open a new terminal)
source .venv/bin/activate

# Confirm it's active — prompt will show (.venv) and Python should be 3.12
which python       # → .../reinforcement-based-cryptocurrency-trading-strategy/.venv/bin/python
python --version   # → Python 3.12.x

# Deactivate (when you're done working)
deactivate
```

---

## Key Dependencies

- `stable-baselines3` — DQN implementation
- `gymnasium` — Trading environment interface
- `torch` — Neural network backend
- `pandas`, `numpy` — Data processing
- `ta` — Technical indicator computation
- `matplotlib`, `seaborn` — Visualizations
