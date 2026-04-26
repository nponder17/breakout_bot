"""
at_open.py — Breakout Bot execution at market open.

Run ~9:31 ET each trading day. Executes the plan from after_close.py:
  - SELL: market orders for positions flagged for exit (target/stop/time)
  - BUY:  market orders for new breakout positions

Sell side is executed first so capital is freed before buying.
"""

from datetime import datetime, timedelta
import pytz

from bot_config import require_env, BOT_NAME, NOTIONAL_BASE
from alpaca_utils import (
    get_trading_calendar,
    submit_market_order,
    get_position,
    wait_for_order_terminal,
    get_order,
    get_order_by_client_order_id,
)
from state_db import (
    init_db, get_plan, plan_already_executed, mark_plan_executed,
    log_event,
    add_lot_pending_entry, mark_lot_open_filled, mark_lot_failed,
    mark_lots_pending_exit, close_lots_for_symbol_filled,
    symbol_already_open, get_open_lots_for_symbol,
    get_pending_entries, get_pending_exits,
)
from telegram_utils import tg_send

ET = pytz.timezone("America/New_York")

DRY_RUN          = False
FORCE_EXEC_DATE  = None
MIN_SELL_QTY     = 1e-6
FILL_TIMEOUT_SEC = 75
FILL_POLL_SEC    = 1.5


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _order_summary(o: dict) -> str:
    return (f"status={o.get('status')} id={o.get('id')} "
            f"filled_qty={o.get('filled_qty')} avg={o.get('filled_avg_price')}")


def _reconcile_pending():
    msgs = []

    pe = get_pending_entries()
    if pe is not None and not pe.empty:
        for _, r in pe.iterrows():
            sym  = r["symbol"]
            coid = r.get("entry_client_order_id")
            oid  = r.get("entry_order_id")
            try:
                if oid:
                    o = get_order(oid)
                elif coid:
                    o = get_order_by_client_order_id(coid)
                    if o is None:
                        msgs.append(f"⚠️ PENDING_ENTRY {sym}: order not found")
                        continue
                else:
                    msgs.append(f"⚠️ PENDING_ENTRY {sym}: missing order ids")
                    continue

                st = (o.get("status") or "").lower()
                if st == "filled":
                    mark_lot_open_filled(
                        coid,
                        entry_order_id=o.get("id") or oid or "UNKNOWN",
                        qty=_safe_float(o.get("filled_qty", 0.0)),
                        avg_entry_price=_safe_float(o.get("filled_avg_price", 0.0)),
                        filled_notional=_safe_float(o.get("filled_qty", 0.0)) * _safe_float(o.get("filled_avg_price", 0.0)),
                        filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                    )
                    msgs.append(f"✅ Reconciled ENTRY: {sym}")
                elif st in ("canceled", "rejected", "expired"):
                    mark_lot_failed(coid, f"reconciled:{st}")
                    msgs.append(f"🛑 ENTRY terminal {sym}: {st}")
            except Exception as e:
                msgs.append(f"❌ Reconcile PENDING_ENTRY {sym}: {e}")

    px = get_pending_exits()
    if px is not None and not px.empty:
        seen = set()
        for _, r in px.iterrows():
            sym  = r["symbol"]
            if sym in seen:
                continue
            seen.add(sym)
            coid = r.get("exit_client_order_id")
            oid  = r.get("exit_order_id")
            try:
                if oid:
                    o = get_order(oid)
                elif coid:
                    o = get_order_by_client_order_id(coid)
                    if o is None:
                        msgs.append(f"⚠️ PENDING_EXIT {sym}: order not found")
                        continue
                else:
                    msgs.append(f"⚠️ PENDING_EXIT {sym}: missing ids")
                    continue

                st = (o.get("status") or "").lower()
                if st == "filled":
                    filled_qty = _safe_float(o.get("filled_qty", 0.0))
                    avg_exit   = _safe_float(o.get("filled_avg_price", 0.0))
                    close_lots_for_symbol_filled(
                        sym,
                        avg_exit_price=avg_exit,
                        filled_notional_exit=filled_qty * avg_exit,
                        filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                        sold_qty_total=filled_qty,
                        exit_date=str(datetime.now(ET).date()),
                    )
                    msgs.append(f"✅ Reconciled EXIT: {sym}")
            except Exception as e:
                msgs.append(f"❌ Reconcile PENDING_EXIT {sym}: {e}")

    return msgs


def main():
    require_env()
    init_db()

    now_et       = datetime.now(ET)
    run_date     = now_et.date()
    run_date_str = str(run_date)

    cal = get_trading_calendar(start=str(run_date - timedelta(days=10)),
                               end=str(run_date + timedelta(days=60)))
    if cal.empty:
        raise RuntimeError("Trading calendar empty.")

    cal_dates = set(cal["date"].tolist())
    if (not FORCE_EXEC_DATE) and (run_date not in cal_dates):
        print(f"Not a trading day ({run_date}); exiting.")
        return

    exec_date     = datetime.strptime(FORCE_EXEC_DATE, "%Y-%m-%d").date() if FORCE_EXEC_DATE else run_date
    exec_date_str = str(exec_date)

    # Reconcile any unresolved orders from prior runs
    rec_msgs = []
    if not DRY_RUN:
        rec_msgs = _reconcile_pending()

    plan = get_plan(exec_date_str)
    if plan is None:
        plan = {"buy_symbols": [], "sell_symbols": [], "executed": False}
        log_event("AT_OPEN", f"No plan for {exec_date_str}; nothing to do.")

    skip_buys = plan_already_executed(exec_date_str)

    # ── Exits first ───────────────────────────────────────────────────────────
    sell_msgs = []
    for sym in sorted(plan.get("sell_symbols", [])):
        try:
            lots = get_open_lots_for_symbol(sym)
            if lots is None or lots.empty:
                sell_msgs.append(f"⏭️ {sym}: no open lots")
                continue

            qty_to_sell = lots["qty"].apply(_safe_float).sum()
            if qty_to_sell <= MIN_SELL_QTY:
                sell_msgs.append(f"⏭️ {sym}: qty≈0")
                continue

            pos = get_position(sym)
            if not pos:
                sell_msgs.append(f"⚠️ No Alpaca position for {sym}")
                continue

            broker_qty  = abs(_safe_float(pos.get("qty", 0.0)))
            qty_to_sell = min(qty_to_sell, broker_qty)
            exit_coid   = f"bobot-{exec_date_str}-{sym}-sell"

            # Determine exit trigger from DB
            exit_trigger = "MANUAL"
            if not lots.empty and "exit_trigger" in lots.columns:
                triggers = lots["exit_trigger"].dropna().tolist()
                if triggers:
                    exit_trigger = triggers[0]

            if DRY_RUN:
                sell_msgs.append(f"🧪 DRY_RUN SELL {sym} qty={qty_to_sell:.4f} [{exit_trigger}]")
                continue

            resp = submit_market_order(symbol=sym, side="sell", qty=qty_to_sell,
                                       time_in_force="day", client_order_id=exit_coid)
            exit_oid = resp.get("id")

            mark_lots_pending_exit(sym, exit_coid, exit_oid, exit_trigger)

            o  = wait_for_order_terminal(order_id=exit_oid,
                                         timeout_sec=FILL_TIMEOUT_SEC,
                                         poll_sec=FILL_POLL_SEC)
            st = (o.get("status") or "").lower()
            if st != "filled":
                sell_msgs.append(f"⚠️ SELL not filled {sym}: {_order_summary(o)}")
                continue

            filled_qty = _safe_float(o.get("filled_qty", 0.0))
            avg_exit   = _safe_float(o.get("filled_avg_price", 0.0))
            close_lots_for_symbol_filled(
                sym,
                avg_exit_price=avg_exit,
                filled_notional_exit=filled_qty * avg_exit,
                filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                sold_qty_total=filled_qty,
                exit_date=exec_date_str,
            )
            sell_msgs.append(f"✅ SELL {sym} qty={filled_qty:.4f} avg=${avg_exit:.2f} [{exit_trigger}]")

        except Exception as e:
            sell_msgs.append(f"❌ SELL {sym} failed: {e}")

    if not sell_msgs:
        sell_msgs.append("No sells.")

    # ── Entries ───────────────────────────────────────────────────────────────
    buy_msgs    = []
    buy_success = 0

    if skip_buys:
        buy_msgs.append("No buys (plan already executed).")
    else:
        plan_notionals = plan.get("buy_notionals") or {}

        for sym in plan.get("buy_symbols", []):
            if symbol_already_open(sym):
                buy_msgs.append(f"⏭️ SKIP {sym}: already have open lot")
                continue

            notional_info = plan_notionals.get(sym, {})
            if isinstance(notional_info, dict):
                notional     = float(notional_info.get("notional", NOTIONAL_BASE))
                model_score  = notional_info.get("model_score")
                llm_action   = notional_info.get("llm_action", "TAKE")
                max_hold     = notional_info.get("max_hold")
            else:
                notional     = float(notional_info) if notional_info else NOTIONAL_BASE
                model_score  = None
                llm_action   = "TAKE"
                max_hold     = None

            entry_coid = f"bobot-{exec_date_str}-{sym}-buy"

            if DRY_RUN:
                buy_msgs.append(f"🧪 DRY_RUN BUY {sym} ${notional:.0f} [{llm_action}]")
                continue

            try:
                add_lot_pending_entry(
                    symbol                = sym,
                    entry_date            = exec_date_str,
                    notional              = notional,
                    entry_client_order_id = entry_coid,
                    model_score           = model_score,
                    llm_action            = llm_action,
                    max_hold_date         = max_hold,
                )

                resp = submit_market_order(symbol=sym, side="buy", notional=notional,
                                           time_in_force="day", client_order_id=entry_coid)
                entry_oid = resp.get("id")

                o  = wait_for_order_terminal(order_id=entry_oid,
                                             timeout_sec=FILL_TIMEOUT_SEC,
                                             poll_sec=FILL_POLL_SEC)
                st = (o.get("status") or "").lower()
                if st != "filled":
                    buy_msgs.append(f"⚠️ BUY not filled {sym}: {_order_summary(o)}")
                    mark_lot_failed(entry_coid, f"entry_not_filled:{st}")
                    continue

                filled_qty = _safe_float(o.get("filled_qty", 0.0))
                avg_entry  = _safe_float(o.get("filled_avg_price", 0.0))
                mark_lot_open_filled(
                    entry_coid,
                    entry_order_id=o.get("id") or entry_oid,
                    qty=filled_qty,
                    avg_entry_price=avg_entry,
                    filled_notional=filled_qty * avg_entry,
                    filled_at=o.get("filled_at") or datetime.now(ET).isoformat(),
                )
                buy_msgs.append(f"✅ BUY {sym} ${notional:.0f} qty={filled_qty:.4f} avg=${avg_entry:.2f} [{llm_action}]")
                buy_success += 1

            except Exception as e:
                buy_msgs.append(f"❌ BUY {sym} failed: {e}")
                try:
                    mark_lot_failed(entry_coid, f"exception:{e}")
                except Exception:
                    pass

    if (not DRY_RUN) and (not skip_buys):
        mark_plan_executed(exec_date_str)

    msg = [
        f"🚀 {BOT_NAME} At-Open Execution",
        f"Run date: {run_date_str}  |  Plan: {exec_date_str}",
        f"Mode: {'DRY_RUN' if DRY_RUN else 'LIVE-PAPER'}",
        "",
    ]
    if rec_msgs:
        msg += ["Reconcile:"] + rec_msgs + [""]
    msg += ["Exits:"] + sell_msgs + [""]
    msg += ["Entries:"] + buy_msgs

    tg_send("\n".join(msg))
    log_event("AT_OPEN", " | ".join(msg))
    print("\n".join(msg))


if __name__ == "__main__":
    main()
