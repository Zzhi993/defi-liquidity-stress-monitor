from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
MODELS_DIR = ROOT / "models"
SQL_DIR = ROOT / "sql"

RANDOM_SEED = 20260717
ROLLING_WINDOW_DAYS = 60
MIN_PAIR_OBSERVATIONS = 45
MIN_ACTIVE_PROTOCOLS = 6
EDGE_CORRELATION_THRESHOLD = 0.45
HIGH_RISK_THRESHOLD = 70.0
WATCH_RISK_THRESHOLD = 55.0
MONTE_CARLO_RUNS = 25000
TRAIN_SHARE = 0.70
FORECAST_HORIZON_DAYS = 7
BLOCK_BOOTSTRAP_REPLICATIONS = 1000
BLOCK_BOOTSTRAP_LENGTH = 14
MAX_TRAINING_ALERT_RATE = 0.30

SELECTED_PROTOCOLS = [
    {"slug": "aave-v3", "name": "Aave V3", "category": "Lending"},
    {"slug": "compound-v3", "name": "Compound V3", "category": "Lending"},
    {"slug": "morpho-blue", "name": "Morpho Blue", "category": "Lending"},
    {"slug": "sky-lending", "name": "Sky Lending", "category": "CDP"},
    {"slug": "curve-dex", "name": "Curve DEX", "category": "DEX"},
    {"slug": "uniswap-v3", "name": "Uniswap V3", "category": "DEX"},
    {"slug": "lido", "name": "Lido", "category": "Liquid Staking"},
    {"slug": "convex-finance", "name": "Convex Finance", "category": "Yield"},
]

SOURCE_URLS = {
    "protocol": "https://api.llama.fi/protocol/{slug}",
    "stablecoin_supply": "https://stablecoins.llama.fi/stablecoincharts/all",
    "eth_price": (
        "https://coins.llama.fi/chart/coingecko:ethereum"
        "?start={start}&span=500&period=1d"
    ),
}


def ensure_directories() -> None:
    for path in [RAW_DIR, STAGING_DIR, PROCESSED_DIR, FIGURES_DIR, REPORT_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
