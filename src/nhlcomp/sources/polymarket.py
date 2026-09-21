"""Adapter for Polymarket's Gamma API (``gamma-api.polymarket.com``).

A second, independent prediction market.  This project **executes on Kalshi**; Polymarket is
ingested as a cross-check, so nothing in this module places, sizes or simulates an order and
no Polymarket price is ever used to settle a Kalshi wager.

Verified from the Arena fetch tool on 2026-09-21 (the development sandbox has no outbound
network; CI does):

* ``GET /events?tag_slug=nhl&closed=false&limit=3`` -> HTTP 200, a JSON **array** of event
  objects, each with a nested ``markets`` array.  No key, no auth header, no rate-limit
  challenge on a handful of reads.
* The three open NHL events at that moment were all season-level ("2026-27 NHL Stanley Cup
  Champion"), not per-game.  One market retained in
  ``data/captured/polymarket_gamma_nhl_event_excerpt.json`` shows the fields this project
  parses: ``bestBid`` 0.021 / ``bestAsk`` 0.022, ``spread`` 0.001, ``lastTradePrice`` 0.021,
  ``liquidity`` "56107.48476", ``volume`` "24237.809822", ``volume24hr`` 14.38,
  ``outcomes`` '["Yes", "No"]' and ``outcomePrices`` '["0.0215", "0.9785"]' (both JSON
  **strings**, not arrays), ``clobTokenIds`` (two token ids as a JSON string),
  ``orderPriceMinTickSize`` 0.001, ``orderMinSize`` 5, ``enableOrderBook`` true,
  ``acceptingOrders`` true, ``feeType`` "sports_fees_v2" and
  ``feeSchedule`` {exponent 1, rate 0.03, takerOnly true, rebateRate 0.25}.
* ``tag_slug`` (or ``tag_id``) is required to scope the query to a sport: without it the same
  endpoint returns other sports' events, so an unscoped read is not an NHL read.

What that means for how the data is stored:

* ``outcomes``/``outcomePrices``/``clobTokenIds`` are kept as the published strings.  Parsing
  them into a list is a convenience for a reader; rewriting them is not, so the raw text is
  what lands in the ledger.
* Numeric fields arrive as strings (``liquidity``, ``volume``) *and* as numbers
  (``liquidityNum``, ``volumeNum``).  Both are read; when they disagree the disagreement is
  reported rather than resolved by preferring one.
* A market's ``game_id`` is only set when the market can be matched to one NHL game without
  ambiguity, and ``match_basis`` always says how the match was made or why it was not.  On
  2026-09-21 no NHL game market existed to match, which is recorded as a verified negative
  rather than treated as missing data.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..http import HttpClient, NetworkUnavailable

GAMMA = "https://gamma-api.polymarket.com"

#: Question wording that identifies a season-long market.  Used only to *label* what kind of
#: market a row is; it never creates a price or a payoff.
_FUTURES_RE = re.compile(
    r"(stanley cup|champion|conference|division|playoff|president|trophy|season|win the)",
    re.IGNORECASE)
#: A per-game market carries a game start time.  Field name observed in Polymarket's own
#: sports payload; when it is absent the market is not treated as a game market.
_GAME_TIME_KEYS = ("gameStartTime", "game_start_time", "eventStartTime")


def _f(v: Any) -> float | None:
    """Float or None.  Polymarket publishes numbers as strings, so both are accepted and a
    value that is not a number stays None instead of becoming 0."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _loads(v: Any) -> Any:
    """``outcomes``/``outcomePrices``/``clobTokenIds`` arrive as JSON text; return the parsed
    value, or the original text when it does not parse (never a fabricated list)."""
    if isinstance(v, (list, dict)) or v is None:
        return v
    try:
        return json.loads(str(v))
    except (TypeError, ValueError):
        return v


class PolymarketGamma:
    source_id = "polymarket.gamma"

    def __init__(self, http: HttpClient):
        self.http = http
        self.calls = 0

    def _get(self, url: str) -> Any:
        self.calls += 1
        return self.http.get(url).json

    # ------------------------------------------------------------------ reads
    def events(self, *, tag_slug: str = "nhl", closed: bool = False, limit: int = 100,
               offset: int = 0) -> list[dict]:
        """``GET /events`` scoped to a sport tag.

        Returns the array as published.  ``tag_slug`` is not optional in practice: an
        unscoped query returns other sports, which would put non-NHL rows into an NHL ledger.
        """
        url = (f"{GAMMA}/events?tag_slug={tag_slug}"
               f"&closed={'true' if closed else 'false'}&limit={int(limit)}&offset={int(offset)}")
        data = self._get(url)
        if isinstance(data, dict):        # an error body is not an empty result
            raise NetworkUnavailable(f"polymarket /events returned an object, not a list: "
                                     f"{json.dumps(data)[:200]}")
        return [e for e in (data or []) if isinstance(e, dict)]

    def markets(self, *, tag_slug: str = "nhl", closed: bool = False, limit: int = 100,
                offset: int = 0) -> list[dict]:
        """``GET /markets`` -- the flat market list, for when an event grouping is not wanted."""
        url = (f"{GAMMA}/markets?tag_slug={tag_slug}"
               f"&closed={'true' if closed else 'false'}&limit={int(limit)}&offset={int(offset)}")
        data = self._get(url)
        if isinstance(data, dict):
            raise NetworkUnavailable(f"polymarket /markets returned an object, not a list: "
                                     f"{json.dumps(data)[:200]}")
        return [m for m in (data or []) if isinstance(m, dict)]


def market_family(m: dict, event: dict | None = None) -> tuple[str, str]:
    """``(family, basis)`` where family is ``game`` | ``futures`` | ``other``.

    A *label*, derived from the market's own published fields and nothing else:

    * ``game`` only when the market carries a game start time, which is the field that makes
      it a per-game instrument;
    * ``futures`` when the question names a season-long outcome;
    * ``other`` otherwise -- an unclassified market is not quietly folded into either.
    """
    for k in _GAME_TIME_KEYS:
        if m.get(k):
            return "game", f"market publishes {k}={m.get(k)!r}"
    q = f"{m.get('question') or ''} {m.get('slug') or ''} {(event or {}).get('title') or ''}"
    if _FUTURES_RE.search(q):
        return "futures", "question/slug names a season-long outcome; no game start time"
    return "other", "no game start time and no season-long wording: left unclassified"


def normalize_event(event: dict, *, retrieved_at: str, source_url: str) -> list[dict[str, Any]]:
    """One Gamma event -> one row per nested market, flattened for storage.

    Event-level context (id, slug, title, liquidity, volume) is denormalized onto each market
    row so a reader never has to re-join to know which event a price came from.  Values are
    copied, not converted: the published strings stay strings in their own columns.
    """
    out: list[dict[str, Any]] = []
    markets = event.get("markets")
    if not isinstance(markets, Sequence) or isinstance(markets, (str, bytes)):
        markets = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        family, basis = market_family(m, event)
        liq = _f(m.get("liquidityNum"))
        if liq is None:
            liq = _f(m.get("liquidity"))
        vol = _f(m.get("volumeNum"))
        if vol is None:
            vol = _f(m.get("volume"))
        fee = m.get("feeSchedule") or {}
        if not isinstance(fee, dict):
            fee = {}
        outcomes = _loads(m.get("outcomes"))
        prices = _loads(m.get("outcomePrices"))
        out.append({
            "id": str(m.get("id") or ""),
            "event_id": (str(event.get("id")) if event.get("id") is not None else None),
            "event_slug": event.get("slug"),
            "event_title": event.get("title"),
            "slug": m.get("slug"),
            "question": m.get("question"),
            "condition_id": m.get("conditionId"),
            "group_item_title": m.get("groupItemTitle"),
            "outcomes": (m.get("outcomes") if isinstance(m.get("outcomes"), str)
                         else json.dumps(outcomes)),
            "outcome_prices": (m.get("outcomePrices") if isinstance(m.get("outcomePrices"), str)
                               else json.dumps(prices)),
            "best_bid": _f(m.get("bestBid")),
            "best_ask": _f(m.get("bestAsk")),
            "spread": _f(m.get("spread")),
            "last_trade_price": _f(m.get("lastTradePrice")),
            "liquidity": liq,
            "volume": vol,
            "volume_24hr": _f(m.get("volume24hr")),
            "tick_size": _f(m.get("orderPriceMinTickSize")),
            "min_size": _f(m.get("orderMinSize")),
            "fee_type": m.get("feeType"),
            "fee_rate": _f(fee.get("rate")),
            "fee_exponent": _f(fee.get("exponent")),
            "fee_taker_only": (1 if fee.get("takerOnly") else
                               (0 if fee.get("takerOnly") is not None else None)),
            "fee_rebate_rate": _f(fee.get("rebateRate")),
            "start_date": m.get("startDate"),
            "end_date": m.get("endDate"),
            "game_start_time": next((m.get(k) for k in _GAME_TIME_KEYS if m.get(k)), None),
            "active": (1 if m.get("active") else 0) if m.get("active") is not None else None,
            "closed": (1 if m.get("closed") else 0) if m.get("closed") is not None else None,
            "accepting_orders": ((1 if m.get("acceptingOrders") else 0)
                                 if m.get("acceptingOrders") is not None else None),
            "enable_order_book": ((1 if m.get("enableOrderBook") else 0)
                                  if m.get("enableOrderBook") is not None else None),
            "market_family": family,
            "match_basis": basis,
            "retrieved_at": retrieved_at,
            "source_url": source_url,
        })
    return out


def numeric_disagreements(rows: Sequence[dict[str, Any]],
                          raw_markets: Sequence[dict[str, Any]]) -> list[str]:
    """Where Polymarket's own string and numeric forms of the same figure disagree.

    ``liquidity``/"56107.48476" vs ``liquidityNum``/56107.48476 are the same number in two
    spellings; when they are *not* the same, that is a discrepancy inside one source and it
    is reported instead of being smoothed over by preferring one field.
    """
    out: list[str] = []
    by_id = {str(r.get("id")): r for r in rows}
    for m in raw_markets:
        r = by_id.get(str(m.get("id") or ""))
        if r is None:
            continue
        for str_key, num_key, col in (("liquidity", "liquidityNum", "liquidity"),
                                      ("volume", "volumeNum", "volume")):
            a, b = _f(m.get(str_key)), _f(m.get(num_key))
            if a is not None and b is not None and abs(a - b) > 0.01:
                out.append(f"market {r['id']}: {str_key}={a} disagrees with {num_key}={b} "
                           f"(stored {col}={r.get(col)})")
    return out
