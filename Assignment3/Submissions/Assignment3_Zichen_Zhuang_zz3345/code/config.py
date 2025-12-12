import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ASSIGNMENT1_PATH = PROJECT_ROOT.parent.parent.parent / "Assignment1" / "Submissions" / "Assignment1_Zichen_Zhuang_zz3345"
ASSIGNMENT2_PATH = PROJECT_ROOT.parent.parent.parent / "Assignment2" / "Submissions" / "Zichen_Zhuang_zz3345"
DATA_DIR = PROJECT_ROOT / "data"
NEWS_DIR = DATA_DIR / "news"
RESULTS_DIR = PROJECT_ROOT / "results"
BACKTEST_DIR = PROJECT_ROOT / "backtest"

for dir_path in [DATA_DIR, NEWS_DIR, RESULTS_DIR, BACKTEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths from Assignment 2
FINGPT_BASE_MODEL = ASSIGNMENT2_PATH / "DeepSeek-R1-Distill-Llama-8B"
FINGPT_FINETUNED_LOCAL = ASSIGNMENT2_PATH / "FinGPT_Forecaster" / "finetuned_models" / "deepseek"
FINGPT_FINETUNED_COLAB = ASSIGNMENT2_PATH / "finetuned_models_colab" / "deepseek_from_base"

# Stock selection from Assignment 1
STOCK_SELECTION_FILE = ASSIGNMENT1_PATH / "source_codes" / "results" / "stock_selected.csv"

# Backtest configuration
BACKTEST_START = "2024-12-01"
BACKTEST_END = "2025-11-30"
INITIAL_CAPITAL = 1000000  # $1M
REBALANCE_FREQUENCY = "monthly"  # Rebalance monthly
TOP_N_STOCKS = 10  # Select top 10 stocks each period

# News API configuration (Finnhub)
FINNHUB_API_KEY = ""

MODEL_TYPE = "base"  # Options: "base", "local_finetuned", "colab_finetuned"
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512

# Signal generation
SENTIMENT_THRESHOLD_BULLISH = 0.6  # Above this = bullish signal
SENTIMENT_THRESHOLD_BEARISH = 0.4  # Below this = bearish signal
USE_PREDICTED_RETURN = True  # Use model's predicted return for signals

# Benchmark
BENCHMARK_TICKER = "^GSPC"  # S&P 500
