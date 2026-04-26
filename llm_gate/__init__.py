"""LLM continuation gate for the Breakout Bot.

Filters post-earnings breakout candidates through Claude to:
  BOOST — strong catalyst, raise position size
  TAKE  — no major concern, proceed with quant signal
  SKIP  — fundamental concern overrides the quant signal

Public API:
    from llm_gate import run_llm_gate

    approved_df, boosted, skipped, analyses = run_llm_gate(
        candidates_df,
        signal_date="2024-08-01",
    )
"""

from __future__ import annotations

import time
import pandas as pd

from llm_gate.polygon_news import fetch_news, PolygonNewsError
from llm_gate.edgar import fetch_8k_filings, EdgarError
from llm_gate.analyzer import analyze_breakout, BreakoutAnalysis

NEWS_DAYS_BACK    = 5    # tight window — we want earnings-day news only
FILINGS_DAYS_BACK = 7
SLEEP_BETWEEN_API = 2    # seconds — respect Polygon free tier


def run_llm_gate(
    candidates: pd.DataFrame,
    signal_date: str,
    score_col: str = "model_score",
    ret_col: str = "ret_1d",
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict], list[dict], dict[str, BreakoutAnalysis]]:
    """Run the LLM continuation gate on a DataFrame of breakout candidates.

    Parameters
    ----------
    candidates : pd.DataFrame
        Each row is a scored breakout candidate.
        Must have: symbol, model_score, ret_1d, gap_pct, vol_ratio.
    signal_date : str
        The as-of date (YYYY-MM-DD). Used to cap news to today.
    score_col : str
        Column name for model probability scores.
    ret_col : str
        Column name for today's % price change.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    approved : pd.DataFrame
        Candidates that passed (TAKE or BOOST decisions).
    boosted : list[dict]
        List of {symbol, ...} for BOOST decisions (strong catalyst).
    skipped : list[dict]
        List of {symbol, ...} for SKIP decisions.
    analyses : dict[str, BreakoutAnalysis]
        Full analysis objects keyed by symbol.
    """
    approved_rows = []
    boosted       = []
    skipped       = []
    analyses      = {}

    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows(), 1):
        symbol      = str(row["symbol"])
        score       = float(row.get(score_col, 0.5))
        ret_1d      = float(row.get(ret_col, 0.0)) * 100   # convert to %
        gap_pct     = float(row.get("gap_pct", 0.0)) * 100
        vol_ratio   = float(row.get("vol_ratio", 1.0))

        if verbose:
            print(f"  LLM gate [{i}/{total}] {symbol}  score={score:.3f}  ret={ret_1d:+.1f}%  gap={gap_pct:+.1f}% …",
                  flush=True)

        # Fetch news
        try:
            articles = fetch_news(
                symbol, days_back=NEWS_DAYS_BACK, limit=20, as_of_date=signal_date
            )
        except PolygonNewsError as e:
            if verbose:
                print(f"    [news error] {e}")
            articles = []

        time.sleep(SLEEP_BETWEEN_API)

        # Fetch 8-K filings
        try:
            filings = fetch_8k_filings(symbol, days_back=FILINGS_DAYS_BACK)
        except EdgarError as e:
            if verbose:
                print(f"    [edgar error] {e}")
            filings = []

        # Ask Claude
        result = analyze_breakout(
            symbol           = symbol,
            price_change_pct = ret_1d,
            gap_pct          = gap_pct,
            vol_ratio        = vol_ratio,
            model_score      = score,
            news_articles    = articles,
            filings          = filings,
            days_back        = NEWS_DAYS_BACK,
        )

        analyses[symbol]        = result
        result._n_articles      = len(articles)
        result._n_filings       = len(filings)

        entry = {
            "symbol"     : symbol,
            "event_type" : result.event_type,
            "reason"     : result.reason,
            "score"      : result.sentiment_score,
            "confidence" : result.confidence,
            "headline"   : result.key_headline,
        }

        if result.skip_trade:
            skipped.append(entry)
            if verbose:
                print(f"    → SKIP  score={result.sentiment_score:+.2f}  "
                      f"conf={result.confidence:.0%}  [{result.event_type}]")
                print(f"       {result.reason[:100]}")
        else:
            approved_rows.append(row)
            if result.action == "BOOST":
                boosted.append(entry)
            if verbose:
                print(f"    → {result.action}  score={result.sentiment_score:+.2f}  "
                      f"conf={result.confidence:.0%}  [{result.event_type}]")

    approved = (
        pd.DataFrame(approved_rows).reset_index(drop=True)
        if approved_rows
        else candidates.iloc[0:0].copy()
    )
    return approved, boosted, skipped, analyses
