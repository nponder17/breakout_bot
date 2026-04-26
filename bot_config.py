import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca ---
ALPACA_KEY          = os.getenv("ALPACA_KEY")
ALPACA_SECRET       = os.getenv("ALPACA_SECRET")
ALPACA_BASE_URL     = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE_URL= os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")

# --- Telegram ---
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID")

# --- APIs ---
POLYGON_API_KEY     = os.getenv("POLYGON_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL     = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# --- Triple-barrier exit params (must match backtest) ---
PROFIT_TARGET   = 0.07    # +7% profit target
STOP_LOSS       = 0.03    # -3% stop loss
MAX_HOLD_DAYS   = 10      # max trading days to hold

# --- Scanner params ---
MIN_PRICE               = 5.0       # minimum stock price to consider
MIN_PRICE_CHANGE_PCT    = 2.0       # min % move today to qualify as breakout
MIN_VOL_RATIO           = 1.3       # min volume vs 20-day avg
EARNINGS_LOOKBACK_DAYS  = 3         # how far back to look for earnings events
MODEL_SCORE_THRESHOLD   = None      # if None, loaded from model pickle (90th pct)
MAX_NEW_BUYS_PER_DAY    = 10        # cap on daily new entries

# --- Position sizing ---
# Fixed notional per position; scale by model score decile optionally
NOTIONAL_BASE   = 500.0   # base position size $
NOTIONAL_BOOST  = 750.0   # size when LLM says BOOST
NOTIONAL_TAKE   = 500.0   # size when LLM says TAKE

# --- Sector blacklist (matches backtest) ---
SECTOR_BLACKLIST = {"Energy", "Real Estate"}

# --- LLM gate ---
LLM_GATE_ENABLED        = os.getenv("LLM_GATE_ENABLED", "true").lower() == "true"
LLM_GATE_MAX_CANDIDATES = int(os.getenv("LLM_GATE_MAX_CANDIDATES", "15"))

# --- Storage ---
DB_PATH  = os.getenv("DB_PATH", "breakout_bot_state.sqlite")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(DATA_DIR, "breakout_model.pkl"))

# --- VIX ---
VIX_LOOKBACK_DAYS = 252
VIX_CSV_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# --- Telegram label ---
BOT_NAME = os.getenv("BOT_NAME", "Breakout Bot")


def require_env():
    missing = []
    for k in ["ALPACA_KEY", "ALPACA_SECRET", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]:
        if not globals().get(k):
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")
