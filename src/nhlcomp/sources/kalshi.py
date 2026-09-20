"""Kalshi prediction-market adapter (public market-data endpoints only).

Verified from this repository's build environment on 2026-09-20:
``GET https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNHLGAME``
returned HTTP 200 with live NHL game contracts including ``yes_bid_dollars``,
``yes_ask_dollars``, ``yes_bid_size_fp``, ``yes_ask_size_fp``, ``volume_fp`` and
``liquidity_dollars`` -- i.e. real quote depth, which is what the execution model needs.

No authentication is required for the market-data reads used here.  Order placement
would require credentials and is deliberately NOT implemented: this project is
paper-trading only.
"""

from __future__ import annotations

from typing import Any

from ..http import HttpClient

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Series tickers that have been observed or are plausible; each is verified at runtime
# by ``verify_series`` and flagged rather than assumed.
CANDIDATE_SERIES = [
    "KXNHLGAME",        # observed: per-game NHL winner markets
    "KXNHL",            # candidate: season-level NHL markets
    "KXNHLSTANLEYCUP",  # candidate: Stanley Cup futures
    "KXNHLCONF",        # candidate: conference winners
    "KXNHLAWARD",       # candidate: individual awards
]


def _f(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class KalshiApi:
    source_id = "kalshi.trade_api"

    def __init__(self, http: HttpClient):
        self.http = http

    # ------------------------------------------------------------ discovery
    def series(self, ticker: str) -> dict | None:
        try:
            return self.http.get(f"{BASE}/series/{ticker}").json
        except Exception:
            return None

    def events(self, series_ticker: str, *, limit: int = 200, status: str = "active") -> list[dict]:
        url = f"{BASE}/events?series_ticker={series_ticker}&limit={limit}&status={status}"
        try:
            return self.http.get(url).json.get("events", [])
        except Exception:
            return []

    def markets(self, series_ticker: str, *, limit: int = 200, status: str = "active",
                cursor: str | None = None, max_pages: int = 5) -> list[dict]:
        out: list[dict] = []
        for _ in range(max_pages):
            url = f"{BASE}/markets?series_ticker={series_ticker}&limit={limit}&status={status}"
            if cursor:
                url += f"&cursor={cursor}"
            try:
                payload = self.http.get(url).json
            except Exception:
                break
            out.extend(payload.get("markets", []))
            cursor = payload.get("cursor")
            if not cursor:
                break
        return out

    def settled_markets(self, series_ticker: str, *, limit: int = 200,
                        max_pages: int = 10) -> list[dict]:
        """Historical, settled contracts.

        Verified 2026-09-20: each row carries ``result`` (yes/no), ``previous_yes_bid_dollars``,
        ``previous_yes_ask_dollars``, ``volume_fp`` and ``settlement_ts``.  Kalshi does not
        document the exact timestamp behind the ``previous_*`` fields, so callers must treat
        them as "some quote that predates settlement" and label any derived bet accordingly.
        """
        return self.markets(series_ticker, limit=limit, status="settled", max_pages=max_pages)

    def orderbook(self, ticker: str) -> dict | None:
        try:
            return self.http.get(f"{BASE}/markets/{ticker}/orderbook").json
        except Exception:
            return None

    def trades(self, ticker: str, *, limit: int = 100) -> list[dict]:
        try:
            return self.http.get(f"{BASE}/markets/trades?ticker={ticker}&limit={limit}").json.get("trades", [])
        except Exception:
            return []

    def candles(self, ticker: str, *, start_ts: int, end_ts: int, period_interval: int = 60) -> list[dict]:
        """Historical OHLC candles.  If this returns rows it is genuine timestamped
        price history and may be used for BACKTEST labels; if it 404s the caller must
        fall back to FORWARD TEST."""
        url = (f"{BASE}/series/KXNHLGAME/markets/{ticker}/candles"
               f"?start_ts={start_ts}&end_ts={end_ts}&period_interval={period_interval}")
        try:
            return self.http.get(url).json.get("candles", [])
        except Exception:
            return []


def normalize_market(m: dict, *, ts_utc: str, retrieved_at: str,
                     source_url: str) -> dict[str, Any]:
    """Convert one Kalshi market into the normalized quote rows we store.

    Emits one row for the YES side and one for the NO side so that both sides of the
    book are visible to the execution model.
    """
    ticker = m.get("ticker") or ""
    game_date = (m.get("occurrence_datetime") or "")[:10] or None
    rows = []
    for side, bid_k, ask_k, bsz, asz in (
        ("YES", "yes_bid_dollars", "yes_ask_dollars", "yes_bid_size_fp", "yes_ask_size_fp"),
        ("NO", "no_bid_dollars", "no_ask_dollars", "no_bid_size_fp", "no_ask_size_fp"),
    ):
        bid, ask = _f(m, bid_k), _f(m, ask_k)
        spread = round(ask - bid, 6) if (bid is not None and ask is not None) else None
        rows.append({
            "provider": "kalshi",
            "market_key": m.get("event_ticker") or "",
            "contract": ticker,
            "game_date": game_date,
            "market_type": "moneyline" if (m.get("market_type") == "binary") else (m.get("market_type") or "binary"),
            "selection": m.get("title") or ticker,
            "side": side,
            "price": _f(m, "last_price_dollars"),
            "bid": bid, "ask": ask, "spread": spread,
            "bid_size": _f(m, bsz), "ask_size": _f(m, asz),
            "volume": _f(m, "volume_fp"),
            "liquidity": _f(m, "liquidity_dollars"),
            "last_price": _f(m, "last_price_dollars"),
            "ts_utc": ts_utc,
            "retrieved_at": retrieved_at,
            "source_url": source_url,
            "status": m.get("status"),
            "settlement": m.get("result") or None,
            "rules": m.get("rules_primary"),
        })
    return rows


def binary_price_to_decimal(price: float) -> float:
    """A YES contract bought at $p pays $1, so decimal odds = 1/p."""
    if price is None or price <= 0:
        return float("nan")
    return 1.0 / float(price)
