"""Market price-history derivations (DERIVED from Kalshi candlesticks -- never invented).

A candle whose ``end_period_ts`` is ``E`` summarises trading in ``(E - interval, E]``.
Given a game's puck-drop timestamp ``start_ts`` we take a small set of *named price
points* from the candle series so the ledger stays compact while still supporting:

* ``open``   first candle with a quote          -> opening line
* ``t24h``   last candle ending <= start - 24h  -> day-before line
* ``t6h``    last candle ending <= start - 6h   -> morning-skate line (starters usually known)
* ``t1h``    last candle ending <= start - 1h
* ``close``  last candle ending <= start        -> closing line (the BACKTEST entry price)
* ``ig60``   last candle ending <= start + 60m  -> roughly end of the 1st period
* ``ig120``  last candle ending <= start + 120m -> roughly mid 3rd period
* ``final``  last candle in the series          -> price just before settlement

Each point records the executable offer (``ask``), the bid, the last trade, the period
mean, the period volume and the open interest, together with the candle timestamp so the
timing of every historical price is auditable.  Nothing is interpolated: when no candle
exists for a point, the point is simply absent.
"""

from __future__ import annotations

import math

from typing import Any, Iterable, Sequence

PRICE_POINTS = ("open", "t24h", "t6h", "t1h", "close", "ig60", "ig120", "final")

_OFFSETS = {"t24h": -24 * 3600, "t6h": -6 * 3600, "t1h": -3600, "close": 0,
            "ig60": 60 * 60, "ig120": 120 * 60}


def _has_quote(c: dict) -> bool:
    return any(c.get(k) is not None for k in ("ask_close", "bid_close", "price_close"))


def _last_at_or_before(candles: Sequence[dict], ts: int, *, after: int | None = None) -> dict | None:
    best = None
    for c in candles:
        e = c["end_period_ts"]
        if e > ts:
            break
        if after is not None and e <= after:
            continue
        if _has_quote(c):
            best = c
    return best


def derive_price_points(candles: Iterable[dict], *, start_ts: int) -> dict[str, dict[str, Any]]:
    """Pick the named points out of a (possibly unsorted) candle list."""
    cs = sorted((c for c in candles if c.get("end_period_ts") is not None),
                key=lambda c: c["end_period_ts"])
    out: dict[str, dict[str, Any]] = {}
    if not cs:
        return out
    first = next((c for c in cs if _has_quote(c)), None)
    if first is not None:
        out["open"] = _point(first)
    for name, off in _OFFSETS.items():
        after = start_ts if name.startswith("ig") else None
        c = _last_at_or_before(cs, start_ts + off, after=after)
        if c is not None:
            out[name] = _point(c)
    last = next((c for c in reversed(cs) if _has_quote(c)), None)
    if last is not None:
        out["final"] = _point(last)
    return out


def _point(c: dict) -> dict[str, Any]:
    return {
        "end_period_ts": int(c["end_period_ts"]),
        "bid": c.get("bid_close"),
        "ask": c.get("ask_close"),
        "last": c.get("price_close"),
        "mean": c.get("price_mean"),
        "volume": c.get("volume"),
        "open_interest": c.get("open_interest"),
    }


def implied_prob(ask: float | None, bid: float | None) -> float | None:
    """Market-implied probability: the mid when both sides exist, else whichever exists.

    The *executable* price for a buyer is the ask; the mid is only a fair-value proxy and
    is labelled as such wherever it is displayed.
    """
    if ask is not None and bid is not None:
        return round((ask + bid) / 2.0, 4)
    if ask is not None:
        return round(ask, 4)
    if bid is not None:
        return round(bid, 4)
    return None


def two_way_vig(ask_a: float | None, ask_b: float | None) -> float | None:
    """Overround when buying either side at the offer (0.04 = 4 cents of vig)."""
    if ask_a is None or ask_b is None:
        return None
    return round(ask_a + ask_b - 1.0, 4)


def american_to_prob(american: float | None) -> float | None:
    """Break-even probability implied by an American price (includes the book's margin)."""
    if american is None:
        return None
    a = float(american)
    if a == 0:
        return None
    if a > 0:
        return round(100.0 / (a + 100.0), 4)
    return round(-a / (-a + 100.0), 4)


def devig_pair(p_a: float | None, p_b: float | None) -> tuple[float | None, float | None]:
    """Proportional (multiplicative) de-vig of a two-way market."""
    if p_a is None or p_b is None or (p_a + p_b) <= 0:
        return None, None
    s = p_a + p_b
    return round(p_a / s, 4), round(p_b / s, 4)


# --------------------------------------------------------------------------- fees
#: Kalshi general trading (taker) fee, from the official fee schedule effective
#: 2026-07-07 (https://kalshi.com/docs/kalshi-fee-schedule.pdf, read 2026-09-20):
#: ``fees = round up(M x 0.07 x C x P x (1-P))`` with M = 1 unless a series is listed
#: with another multiplier.  KXNHLGAME is not on the non-standard list, so M = 1 applies.
#: Maker fees use ``0.0175`` with a default multiplier of 0, i.e. resting orders in
#: KXNHLGAME pay nothing -- but this engine always takes the offer, so it pays the taker fee.
KALSHI_TAKER_RATE = 0.07
KALSHI_TAKER_MULTIPLIER_KXNHLGAME = 1.0


def kalshi_taker_fee(price: float, contracts: float, *, multiplier: float = KALSHI_TAKER_MULTIPLIER_KXNHLGAME
                     ) -> float:
    """Taker fee in dollars for buying ``contracts`` at ``price``.

    Rounded UP to the next cent, which reproduces the schedule's own table (100 contracts
    at $0.05 -> $0.34 from a raw 0.3325; 1 contract at $0.01 -> $0.01).
    """
    if contracts <= 0 or not (0 < price < 1):
        return 0.0
    raw = multiplier * KALSHI_TAKER_RATE * contracts * price * (1.0 - price)
    return math.ceil(raw * 100 - 1e-9) / 100


def kalshi_fee_per_unit_staked(price: float, *, multiplier: float = KALSHI_TAKER_MULTIPLIER_KXNHLGAME) -> float:
    """Fee per $1 staked (contracts = 1/price): 0.07 x (1 - P), unrounded -- used by the
    flat-stake backtests where the contract count is fractional by construction."""
    if not (0 < price < 1):
        return 0.0
    return round(multiplier * KALSHI_TAKER_RATE * (1.0 - price), 6)
