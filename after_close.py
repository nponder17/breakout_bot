"""
after_close.py — Breakout Bot nightly scanner.

Run after market close each trading day. Produces a plan for the next open:
  - SELL: open lots that hit +7% target, -3% stop, or max hold days today
  - BUY:  post-earnings breakout candidates scored by the quant model,
          filtered by LLM continuation gate, top 10% by model score

Scanner pipeline:
  1. Load S&P 500 universe, fetch daily bars (past 60 days)
  2. Find stocks with notable moves today (>= MIN_PRICE_CHANGE_PCT + vol surge)
  3. Filter to stocks with recent earnings (last EARNINGS_LOOKBACK_DAYS days)
  4. Apply sector blacklist (Energy, Real Estate)
  5. Compute 10 breakout features, score with production model
  6. Keep top 10% by model score (above threshold from model pickle)
  7. Run LLM gate on top candidates (BOOST / TAKE / SKIP)
  8. Check exits for all open lots (triple barrier: target / stop / time)
  9. Write plan, send Telegram summary
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pytz
import requests
import yfinance as yf

from bot_config import (
    require_env, DATA_DIR, BOT_NAME,
    PROFIT_TARGET, STOP_LOSS, MAX_HOLD_DAYS,
    MIN_PRICE, MIN_PRICE_CHANGE_PCT, MIN_VOL_RATIO,
    EARNINGS_LOOKBACK_DAYS, MAX_NEW_BUYS_PER_DAY,
    NOTIONAL_BASE, NOTIONAL_BOOST,
    SECTOR_BLACKLIST,
    LLM_GATE_ENABLED, LLM_GATE_MAX_CANDIDATES,
    MODEL_PATH, VIX_CSV_URL, VIX_LOOKBACK_DAYS,
)
from alpaca_utils import get_trading_calendar, get_next_trading_day, get_daily_bars, add_trading_days
from indicators import compute_features, passes_breakout_filter, get_consecutive_beats
from llm_gate import run_llm_gate
from state_db import (
    init_db, upsert_plan, log_event, log_llm_gate_decision,
    open_lots, symbol_already_open,
)
from telegram_utils import tg_send

ET = pytz.timezone("America/New_York")

TEST_TODAY = '2026-04-24'   # override: e.g. "2024-01-05"


# ── VIX ───────────────────────────────────────────────────────────────────────

def fetch_vix() -> pd.Series:
    """Download VIX history from CBOE. Returns Series indexed by date."""
    try:
        resp = requests.get(VIX_CSV_URL, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"date": "date", "close": "vix_close"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        return df.set_index("date")["vix_close"]
    except Exception as e:
        print(f"  ⚠️  VIX fetch failed: {e}")
        return pd.Series(dtype=float)


# ── S&P 500 universe ──────────────────────────────────────────────────────────

def load_sp500_symbols() -> list[str]:
    """Load S&P 500 tickers. Falls back to Wikipedia scrape if local file missing."""
    universe_path = Path(DATA_DIR) / "universe.csv"
    if universe_path.exists():
        return pd.read_csv(universe_path)["symbol"].dropna().astype(str).str.upper().unique().tolist()

    # Fallback: scrape Wikipedia (no API key needed)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        symbols = df["Symbol"].str.replace(".", "-", regex=False).str.upper().tolist()
        # Save for future runs
        pd.DataFrame({"symbol": symbols}).to_csv(universe_path, index=False)
        print(f"  Saved {len(symbols)} symbols to {universe_path}")
        return symbols
    except Exception as e:
        print(f"  ⚠️  Could not load S&P 500 universe: {e}")
        return []


# ── Sector filter ─────────────────────────────────────────────────────────────

def load_sector_map() -> dict[str, str]:
    sector_cache = Path(DATA_DIR) / "sector_cache.json"
    if sector_cache.exists():
        with open(sector_cache) as f:
            return json.load(f)
    return {}


def is_blacklisted(symbol: str, sector_map: dict) -> bool:
    return sector_map.get(symbol, "Unknown") in SECTOR_BLACKLIST


# ── Earnings recency check ────────────────────────────────────────────────────

def had_recent_earnings(symbol: str, today: object, lookback_days: int = 3) -> bool:
    """
    Check if this stock reported earnings within the last lookback_days calendar days.
    Uses yfinance earnings_history — returns False on any error.
    """
    try:
        cutoff = today - timedelta(days=lookback_days)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tk = yf.Ticker(symbol)
            hist = tk.earnings_history
        if hist is None or hist.empty:
            return False
        # earnings_history index is the earnings date
        recent = hist[hist.index.date >= cutoff]
        return not recent.empty
    except Exception:
        return False


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model() -> dict | None:
    """Load the serialized production model from pickle."""
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"  ⚠️  Model not found at {model_path}. Run export_production_model.py first.")
        return None
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        print(f"  ✓  Model loaded: {payload.get('trained_on', '?')}  "
              f"threshold={payload.get('threshold', 0.5):.3f}")
        return payload
    except Exception as e:
        print(f"  ⚠️  Model load failed: {e}")
        return None


def score_candidate(features: dict, model_payload: dict) -> float:
    """Return model probability for a candidate. Returns 0.5 on error."""
    try:
        feat_order = model_payload["features"]
        model      = model_payload["model"]
        x = np.array([[features[f] for f in feat_order]], dtype=float)
        if np.any(np.isnan(x)):
            return 0.5
        return float(model.predict_proba(x)[0][1])
    except Exception:
        return 0.5


# ── Exit check ────────────────────────────────────────────────────────────────

def find_exits(open_df: pd.DataFrame, bars_by_sym: dict, today_str: str, cal: pd.DataFrame) -> list[dict]:
    """
    Check all open lots against today's bar for triple-barrier exits.

    Returns list of {symbol, exit_trigger, lot_info} for positions to sell tomorrow.
    """
    exits = []
    seen  = set()

    for _, lot in open_df.iterrows():
        sym         = str(lot["symbol"])
        if sym in seen:
            continue

        avg_entry   = lot.get("avg_entry_price")
        max_hold    = lot.get("max_hold_date")
        status      = lot.get("status", "OPEN")

        if status == "PENDING_EXIT":
            exits.append({"symbol": sym, "exit_trigger": lot.get("exit_trigger", "MANUAL")})
            seen.add(sym)
            continue

        if avg_entry is None or avg_entry <= 0:
            continue

        avg_entry = float(avg_entry)
        bars = bars_by_sym.get(sym)

        # ── Time exit: check max_hold_date ────────────────────────────────────
        if max_hold and today_str >= max_hold:
            exits.append({"symbol": sym, "exit_trigger": "TIME"})
            seen.add(sym)
            continue

        if bars is None or bars.empty:
            continue

        today_bar = bars[bars["date"].astype(str) == today_str]
        if today_bar.empty:
            continue

        today_high = float(today_bar["high"].values[0])
        today_low  = float(today_bar["low"].values[0])

        target_price = avg_entry * (1 + PROFIT_TARGET)
        stop_price   = avg_entry * (1 - STOP_LOSS)

        # ── Profit target ─────────────────────────────────────────────────────
        if today_high >= target_price:
            exits.append({"symbol": sym, "exit_trigger": "TARGET"})
            seen.add(sym)
            continue

        # ── Stop loss ─────────────────────────────────────────────────────────
        if today_low <= stop_price:
            exits.append({"symbol": sym, "exit_trigger": "STOP"})
            seen.add(sym)
            continue

    return exits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    require_env()
    init_db()
    os.makedirs(DATA_DIR, exist_ok=True)

    now_et = datetime.now(ET)
    today  = datetime.strptime(TEST_TODAY, "%Y-%m-%d").date() if TEST_TODAY else now_et.date()
    today_str = str(today)

    # Trading calendar
    cal = get_trading_calendar(start=str(today - timedelta(days=15)),
                               end=str(today + timedelta(days=30)))
    if cal.empty:
        raise RuntimeError("Trading calendar empty.")

    cal_dates = set(cal["date"].tolist())
    if today not in cal_dates:
        print(f"Not a trading day ({today}); exiting.")
        return

    next_td = get_next_trading_day(cal, today_date=today)

    print(f"\n{'='*60}")
    print(f"  BREAKOUT BOT — After-Close Scan  ({today_str})")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model_payload = load_model()
    threshold = model_payload.get("threshold", 0.5) if model_payload else 0.5

    # ── VIX ──────────────────────────────────────────────────────────────────
    print("Fetching VIX …")
    vix_series = fetch_vix()

    # ── Universe ──────────────────────────────────────────────────────────────
    print("Loading universe …")
    symbols = load_sp500_symbols()
    sector_map = load_sector_map()
    print(f"  {len(symbols)} symbols loaded")

    # ── Price data (past 65 days for all indicators) ──────────────────────────
    start_date = str(today - timedelta(days=90))
    end_date   = str(today + timedelta(days=1))

    print(f"Fetching daily bars for {len(symbols)} symbols …")
    all_bars = get_daily_bars(symbols, start=start_date, end=end_date)
    if all_bars.empty:
        raise RuntimeError("No bars returned from Alpaca.")

    all_bars["symbol"] = all_bars["symbol"].astype(str).str.upper()
    all_bars["date"]   = pd.to_datetime(all_bars["date"]).dt.date

    # Index bars by symbol for fast lookup
    bars_by_sym = {sym: grp.reset_index(drop=True) for sym, grp in all_bars.groupby("symbol")}

    # ── Exit check for open lots ──────────────────────────────────────────────
    print("\nChecking exits for open lots …")
    current_open = open_lots(include_pending_entry=False)
    open_symbols_exits = []

    if not current_open.empty:
        sell_candidates = find_exits(current_open, bars_by_sym, today_str, cal)
        open_symbols_exits = sell_candidates
        for ex in sell_candidates:
            print(f"  → SELL {ex['symbol']}  [{ex['exit_trigger']}]")
    else:
        print("  No open lots.")

    sell_symbols   = [e["symbol"] for e in open_symbols_exits]
    exit_triggers  = {e["symbol"]: e["exit_trigger"] for e in open_symbols_exits}

    # ── Breakout scan ─────────────────────────────────────────────────────────
    print(f"\nScanning for post-earnings breakouts (today = {today_str}) …")

    candidates = []
    open_held  = set(current_open["symbol"].tolist()) if not current_open.empty else set()

    for sym in symbols:
        # Skip blacklisted sectors
        if is_blacklisted(sym, sector_map):
            continue

        # Skip already-held positions
        if sym in open_held:
            continue

        bars = bars_by_sym.get(sym)
        if bars is None or len(bars) < 45:
            continue

        # Today's bar must exist
        today_bars = bars[bars["date"] == today]
        if today_bars.empty:
            continue

        # Quick breakout filter (price move + volume)
        if not passes_breakout_filter(bars, min_price=MIN_PRICE,
                                      min_price_change_pct=MIN_PRICE_CHANGE_PCT,
                                      min_vol_ratio=MIN_VOL_RATIO):
            continue

        # Must have had recent earnings
        if not had_recent_earnings(sym, today, lookback_days=EARNINGS_LOOKBACK_DAYS):
            continue

        # Compute features
        feats = compute_features(bars, vix_series=vix_series, vix_lookback=VIX_LOOKBACK_DAYS)
        if feats is None:
            continue

        # Get consecutive beats (yfinance call — only for breakout candidates)
        feats["consecutive_beats"] = float(get_consecutive_beats(sym))

        # Score with model
        if model_payload:
            score = score_candidate(feats, model_payload)
        else:
            score = 0.5

        today_row  = bars.iloc[-1]
        prev_row   = bars.iloc[-2] if len(bars) >= 2 else bars.iloc[-1]
        ret_1d     = float(today_row["close"] / prev_row["close"] - 1)
        gap_pct    = float(today_row["open"]  / prev_row["close"] - 1)
        vol_ratio  = float(today_row["volume"] / bars["volume"].iloc[-21:-1].mean())

        candidates.append({
            "symbol"      : sym,
            "model_score" : score,
            "ret_1d"      : ret_1d,
            "gap_pct"     : gap_pct,
            "vol_ratio"   : vol_ratio,
            **feats,
        })

    print(f"  {len(candidates)} candidates passed breakout filter + earnings check")

    # ── Model score threshold filter ──────────────────────────────────────────
    if not candidates:
        print("  No candidates to score.")
        buy_df = pd.DataFrame()
    else:
        cand_df = pd.DataFrame(candidates).sort_values("model_score", ascending=False)
        above_threshold = cand_df[cand_df["model_score"] >= threshold]
        print(f"  {len(above_threshold)} above model threshold ({threshold:.3f})")

        # Cap at MAX_NEW_BUYS_PER_DAY * 2 before LLM gate (gate will trim further)
        buy_df = above_threshold.head(min(len(above_threshold), MAX_NEW_BUYS_PER_DAY * 2))

    # ── LLM continuation gate ─────────────────────────────────────────────────
    llm_boosted    = []
    llm_skipped    = []
    llm_analyses   = {}
    llm_enabled    = LLM_GATE_ENABLED and not buy_df.empty

    if llm_enabled:
        gate_input = buy_df.head(LLM_GATE_MAX_CANDIDATES)
        print(f"\nRunning LLM gate on {len(gate_input)} candidates …")
        try:
            buy_df, llm_boosted, llm_skipped, llm_analyses = run_llm_gate(
                candidates  = gate_input,
                signal_date = today_str,
                verbose     = True,
            )

            # Log all decisions
            for sym, analysis in llm_analyses.items():
                cand_row = gate_input[gate_input["symbol"] == sym]
                score_val = float(cand_row["model_score"].values[0]) if not cand_row.empty else 0.0
                ret_val   = float(cand_row["ret_1d"].values[0])      if not cand_row.empty else 0.0
                try:
                    log_llm_gate_decision(
                        signal_date     = today_str,
                        symbol          = sym,
                        model_score     = score_val,
                        ret_1d          = ret_val,
                        action          = analysis.action,
                        sentiment_score = analysis.sentiment_score,
                        confidence      = analysis.confidence,
                        event_type      = analysis.event_type,
                        reason          = analysis.reason,
                        key_headline    = analysis.key_headline,
                        n_articles      = getattr(analysis, "_n_articles", 0),
                        n_filings       = getattr(analysis, "_n_filings", 0),
                    )
                except Exception as log_err:
                    print(f"  ⚠️  Failed to log gate decision for {sym}: {log_err}")

        except Exception as e:
            print(f"  ⚠️  LLM gate failed: {e} — proceeding without gate")
            llm_enabled = False

    # ── Final buy list (cap at MAX_NEW_BUYS_PER_DAY) ─────────────────────────
    if not buy_df.empty:
        buy_df = buy_df.head(MAX_NEW_BUYS_PER_DAY)

    buy_symbols   = []
    buy_notionals = {}
    boosted_set   = {e["symbol"] for e in llm_boosted}

    if not buy_df.empty:
        # Compute max_hold_date for each new position
        for _, row in buy_df.iterrows():
            sym = str(row["symbol"])
            try:
                max_hold_date = add_trading_days(cal, today, MAX_HOLD_DAYS)
            except Exception:
                max_hold_date = str(today + timedelta(days=MAX_HOLD_DAYS + 4))

            notional = NOTIONAL_BOOST if sym in boosted_set else NOTIONAL_BASE

            buy_symbols.append(sym)
            buy_notionals[sym] = {
                "notional"    : notional,
                "model_score" : float(row.get("model_score", 0.5)),
                "llm_action"  : "BOOST" if sym in boosted_set else "TAKE",
                "max_hold"    : max_hold_date,
            }

    # ── Save plan ─────────────────────────────────────────────────────────────
    upsert_plan(
        plan_date     = next_td,
        buy_symbols   = buy_symbols,
        sell_symbols  = sell_symbols,
        buy_notionals = buy_notionals,
    )

    # ── Telegram summary ──────────────────────────────────────────────────────
    gate_str = "OFF"
    if llm_enabled:
        gate_str = f"ON — skip={len(llm_skipped)}, boost={len(llm_boosted)}"

    msg = [
        f"📈 {BOT_NAME} After-Close Scan",
        f"Signal date : {today_str}",
        f"Plan date   : {next_td}",
        f"Model thresh: {threshold:.3f}",
        f"LLM gate    : {gate_str}",
        f"",
        f"Exits ({len(sell_symbols)}):",
    ]
    for sym in sell_symbols:
        msg.append(f"  🔴 {sym}  [{exit_triggers.get(sym,'?')}]")
    if not sell_symbols:
        msg.append("  None")

    msg.append(f"\nBuys ({len(buy_symbols)}):")
    for sym in buy_symbols:
        info = buy_notionals.get(sym, {})
        score_str = f"  score={info.get('model_score', 0):.3f}"
        action_str = f"  [{info.get('llm_action','TAKE')}]"
        notional_str = f"  ${info.get('notional', NOTIONAL_BASE):.0f}"
        msg.append(f"  🟢 {sym}{score_str}{action_str}{notional_str}")
    if not buy_symbols:
        msg.append("  None")

    if llm_skipped:
        msg.append(f"\n🚫 LLM skipped ({len(llm_skipped)}):")
        for s in llm_skipped:
            msg.append(f"  {s['symbol']}: [{s['event_type']}] {s['reason'][:60]}")

    tg_msg = "\n".join(msg)
    tg_send(tg_msg)
    log_event("AFTER_CLOSE", tg_msg.replace("\n", " | "))
    print("\n" + tg_msg)


if __name__ == "__main__":
    main()
