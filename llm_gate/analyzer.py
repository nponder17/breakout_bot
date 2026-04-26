"""Send breakout context to Claude and parse structured continuation gate response."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from llm_gate.polygon_news import format_for_prompt as fmt_news
from llm_gate.edgar import format_for_prompt as fmt_filings

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "breakout_gate.txt"
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.4


@dataclass
class BreakoutAnalysis:
    symbol          : str
    sentiment_score : float
    skip_trade      : bool
    confidence      : float
    event_type      : str
    reason          : str
    key_headline    : str
    raw_response    : str = field(repr=False, default="")
    error           : Optional[str] = None

    @property
    def is_uncertain(self) -> bool:
        return self.confidence < UNCERTAIN_CONFIDENCE_THRESHOLD

    @property
    def action(self) -> str:
        if self.skip_trade:
            return "SKIP"
        if self.sentiment_score >= 0.5:
            return "BOOST"
        return "TAKE"

    def summary(self) -> str:
        return (
            f"{self.symbol}: {self.action}  "
            f"score={self.sentiment_score:+.2f}  "
            f"conf={self.confidence:.0%}  "
            f"type={self.event_type}\n"
            f"  {self.reason}"
        )


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _parse_json_response(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def analyze_breakout(
    symbol          : str,
    price_change_pct: float,
    gap_pct         : float,
    vol_ratio       : float,
    model_score     : float,
    news_articles   : list[dict],
    filings         : list[dict],
    days_back       : int = 5,
    max_retries     : int = 2,
) -> BreakoutAnalysis:
    """
    Ask Claude whether this post-earnings breakout has continuation potential.

    Parameters
    ----------
    symbol           : ticker
    price_change_pct : today's total % change (pct, e.g. 8.5 for +8.5%)
    gap_pct          : gap at open vs prior close (pct)
    vol_ratio        : today's volume / 20-day avg volume
    model_score      : probability from the quant model (0–1)
    news_articles    : from polygon_news.fetch_news
    filings          : from edgar.fetch_8k_filings
    days_back        : days of news fetched (for prompt context)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    news_text    = fmt_news(news_articles, max_articles=10)
    filings_text = fmt_filings(filings)

    prompt = _load_prompt().format(
        symbol           = symbol,
        price_change_pct = price_change_pct,
        gap_pct          = gap_pct,
        vol_ratio        = vol_ratio,
        model_score      = model_score,
        days_back        = days_back,
        news_text        = news_text,
        filings_text     = filings_text,
    )

    client = anthropic.Anthropic(api_key=api_key)
    raw = ""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            message = client.messages.create(
                model      = model,
                max_tokens = 512,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text
            break
        except Exception as e:
            last_error = str(e)
            if attempt == 0:
                print(f"    [LLM error attempt {attempt+1}]: {type(e).__name__}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return BreakoutAnalysis(
                    symbol=symbol, sentiment_score=0.0, skip_trade=False,
                    confidence=0.0, event_type="uncertain",
                    reason="API call failed — defaulting to TAKE",
                    key_headline="none", raw_response="", error=last_error,
                )

    try:
        parsed = _parse_json_response(raw)
    except ValueError as e:
        return BreakoutAnalysis(
            symbol=symbol, sentiment_score=0.0, skip_trade=False,
            confidence=0.0, event_type="uncertain",
            reason="Could not parse Claude response — defaulting to TAKE",
            key_headline="none", raw_response=raw, error=str(e),
        )

    score      = max(-1.0, min(1.0, float(parsed.get("sentiment_score", 0.0))))
    confidence = max(0.0,  min(1.0, float(parsed.get("confidence", 0.5))))
    skip       = bool(parsed.get("skip_trade", score <= -0.5))

    return BreakoutAnalysis(
        symbol          = symbol,
        sentiment_score = score,
        skip_trade      = skip,
        confidence      = confidence,
        event_type      = parsed.get("event_type", "uncertain"),
        reason          = parsed.get("reason", ""),
        key_headline    = parsed.get("key_headline", "none"),
        raw_response    = raw,
    )
