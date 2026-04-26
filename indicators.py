"""
indicators.py — Compute the 10 breakout model features from live price data.

Features (must match backtest exactly):
  ret_1              : 1-day return on the breakout day
  ret_3              : 3-day return ending on breakout day
  range_expansion    : today's H-L range / 20-day avg H-L range
  compression_ratio  : ATR(10d) / ATR(40d)  — lower = tighter base
  is_50d_bo          : 1 if close > 50-day SMA
  vol_accel          : 5-day avg volume / 20-day avg volume
  surprise_reaction_gap   : open / prior_close - 1  (the gap at open)
  earnings_reaction_pct   : close / open - 1  (intraday return)
  vix_pct_rank       : VIX percentile rank over last 252 trading days
  consecutive_beats  : count of consecutive prior EPS beats

All features computed from OHLCV history + VIX data + EPS data (yfinance).
Needs at least 60 days of daily bars for the stock.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional


# ── ATR helper ─────────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    """True range using OHLC columns."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).rolling(period, min_periods=max(1, period // 2)).mean()


# ── EPS data (yfinance) ───────────────────────────────────────────────────────

def get_consecutive_beats(symbol: str) -> int:
    """
    Count consecutive prior EPS beats using yfinance earnings_history.
    Returns 0 if data unavailable (safe fallback).
    """
    try:
        import yfinance as yf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tk = yf.Ticker(symbol)
            hist = tk.earnings_history
        if hist is None or hist.empty:
            return 0
        # earnings_history columns: epsEstimate, epsActual, epsDifference, surprisePercent
        # Drop rows with missing actuals
        hist = hist.dropna(subset=["epsActual", "epsEstimate"])
        # Sort ascending (oldest first)
        hist = hist.sort_index(ascending=True)
        # Count consecutive beats from the END (most recent first)
        beats = (hist["epsActual"] >= hist["epsEstimate"]).tolist()
        count = 0
        for beat in reversed(beats):
            if beat:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


# ── Main feature computation ──────────────────────────────────────────────────

def compute_features(
    bars: pd.DataFrame,
    vix_series: Optional[pd.Series] = None,
    vix_lookback: int = 252,
    eps_beats: Optional[int] = None,
) -> dict | None:
    """
    Compute all 10 breakout features for the MOST RECENT bar in `bars`.

    Parameters
    ----------
    bars : pd.DataFrame
        Daily OHLCV with columns: date (datetime.date), open, high, low, close, volume
        Sorted ascending by date. Needs at least 60 rows.
    vix_series : pd.Series, optional
        VIX close prices indexed by date. Used for vix_pct_rank.
    vix_lookback : int
        Rolling window for VIX percentile (default 252 = 1 year).
    eps_beats : int, optional
        Pre-computed consecutive_beats (avoids redundant yfinance calls).
        If None, consecutive_beats is returned as 0.

    Returns
    -------
    dict of feature_name -> float, or None if not enough data.
    """
    bars = bars.sort_values("date").reset_index(drop=True)

    if len(bars) < 45:
        return None   # need at least ~40 days for ATR(40)

    today = bars.iloc[-1]
    prev  = bars.iloc[-2] if len(bars) >= 2 else None
    prev3 = bars.iloc[-4] if len(bars) >= 4 else None

    # ── ret_1: today's 1-day return ──────────────────────────────────────────
    ret_1 = float(today["close"] / prev["close"] - 1) if prev is not None else 0.0

    # ── ret_3: 3-day return ──────────────────────────────────────────────────
    ret_3 = float(today["close"] / prev3["close"] - 1) if prev3 is not None else ret_1

    # ── surprise_reaction_gap: open vs prior close ───────────────────────────
    surprise_reaction_gap = (
        float(today["open"] / prev["close"] - 1) if prev is not None else 0.0
    )

    # ── earnings_reaction_pct: intraday (close / open - 1) ───────────────────
    earnings_reaction_pct = (
        float(today["close"] / today["open"] - 1)
        if today["open"] > 0 else 0.0
    )

    # ── range_expansion: today range / 20-day avg range ─────────────────────
    today_range   = float(today["high"] - today["low"])
    avg_range_20d = float((bars["high"] - bars["low"]).iloc[-21:-1].mean())
    range_expansion = today_range / avg_range_20d if avg_range_20d > 0 else 1.0

    # ── compression_ratio: ATR(10) / ATR(40) ─────────────────────────────────
    atr10 = _atr(bars, 10).iloc[-1]
    atr40 = _atr(bars, 40).iloc[-1]
    compression_ratio = float(atr10 / atr40) if atr40 > 0 else 1.0

    # ── is_50d_bo: close > 50-day SMA ────────────────────────────────────────
    ma50 = bars["close"].iloc[-51:-1].mean() if len(bars) >= 51 else bars["close"].mean()
    is_50d_bo = float(today["close"] > ma50)

    # ── vol_accel: 5-day avg vol / 20-day avg vol ────────────────────────────
    vol_ma5  = bars["volume"].iloc[-6:-1].mean()
    vol_ma20 = bars["volume"].iloc[-21:-1].mean()
    vol_accel = float(vol_ma5 / vol_ma20) if vol_ma20 > 0 else 1.0

    # ── vix_pct_rank ─────────────────────────────────────────────────────────
    vix_pct_rank = 0.5   # neutral fallback
    if vix_series is not None and not vix_series.empty:
        today_date = today["date"]
        if today_date in vix_series.index:
            vix_today = float(vix_series.loc[today_date])
            past_vix  = vix_series.loc[:today_date].iloc[-(vix_lookback + 1):-1]
            if len(past_vix) >= 30:
                vix_pct_rank = float(np.mean(past_vix.values < vix_today))

    # ── consecutive_beats ────────────────────────────────────────────────────
    consecutive_beats = eps_beats if eps_beats is not None else 0

    return {
        "ret_1"                 : ret_1,
        "ret_3"                 : ret_3,
        "range_expansion"       : range_expansion,
        "compression_ratio"     : compression_ratio,
        "is_50d_bo"             : is_50d_bo,
        "vol_accel"             : vol_accel,
        "surprise_reaction_gap" : surprise_reaction_gap,
        "earnings_reaction_pct" : earnings_reaction_pct,
        "vix_pct_rank"          : vix_pct_rank,
        "consecutive_beats"     : float(consecutive_beats),
    }


# ── Breakout filter ───────────────────────────────────────────────────────────

def passes_breakout_filter(
    bars: pd.DataFrame,
    min_price: float = 5.0,
    min_price_change_pct: float = 2.0,
    min_vol_ratio: float = 1.3,
) -> bool:
    """
    Quick pre-filter: is today's bar consistent with a post-earnings breakout?
    Requires: price move >= min_price_change_pct AND volume surge >= min_vol_ratio.
    """
    if len(bars) < 21:
        return False

    today = bars.iloc[-1]
    prev  = bars.iloc[-2]

    if float(today["close"]) < min_price:
        return False

    pct_change = (today["close"] / prev["close"] - 1) * 100
    if pct_change < min_price_change_pct:
        return False

    vol_ma20 = bars["volume"].iloc[-21:-1].mean()
    vol_ratio = today["volume"] / vol_ma20 if vol_ma20 > 0 else 0
    if vol_ratio < min_vol_ratio:
        return False

    return True
