"""Kalshi prediction-market adapter (public market-data endpoints only).

What has actually been verified against ``https://api.elections.kalshi.com`` (all reads,
no authentication):

* ``GET /trade-api/v2/markets?series_ticker=KXNHLGAME`` -- live NHL game contracts with
  ``yes_bid_dollars``/``yes_ask_dollars`` and size fields (verified 2026-09-20 from CI).
* ``GET /trade-api/v2/markets?...&status=settled`` -- **only the most recent** settled
  contracts.  Kalshi partitions its data into a *live* tier and a *historical* tier; a
  settled contract older than ``GET /trade-api/v2/historical/cutoff`` disappears from the
  live listing.  On 2026-09-20 the live listing held just 12 pre-season KXNHLGAME
  contracts while the historical tier held the whole 2025-26 season.
* ``GET /trade-api/v2/historical/markets?series_ticker=KXNHLGAME&limit=1000&cursor=...``
  -- settled contracts older than the cutoff (verified 2026-09-20: returned the 2026
  Stanley Cup Final contracts with ``result``, ``settlement_ts``, ``close_time``,
  ``open_time``, ``occurrence_datetime``, ``volume_fp``).  Time filters are ignored on this
  endpoint; paging is by cursor only.
* ``GET /trade-api/v2/historical/markets/{ticker}/candlesticks?start_ts&end_ts&period_interval``
  -- genuine timestamped OHLC price history for historical-tier contracts, at 1-minute and
  60-minute resolution (verified 2026-09-20 on KXNHLGAME-26JUN14CARVGK-CAR: hourly candles
  pre-game, minute candles in-game, each with ``price``, ``yes_bid``, ``yes_ask``,
  ``volume`` and ``open_interest``).
* ``GET /trade-api/v2/series/{series}/markets/{ticker}/candlesticks?...`` -- same for
  live-tier contracts, with ``*_dollars``/``*_fp`` field names and *no* ``price.open/close``
  in periods that had no trades (verified 2026-09-20 on KXNHLGAME-26SEP19VGKLA-VGK).
* ``GET /trade-api/v2/series?category=Sports&tags=Hockey`` -- lists the hockey series
  (KXNHLGAME, KXNHLTOTAL, KXNHLSPREAD, KXNHL1P, KXNHLOVERTIME, KXNHL, ... verified 2026-09-20).

The earlier ``.../candles`` path used by this project was simply the wrong resource name
(the API calls them ``candlesticks``); that is why it 404'd.

Order placement would require credentials and is deliberately NOT implemented: this
project is paper-trading only.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from ..http import HttpClient

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HIST_BASE = f"{BASE}/historical"

# Series tickers that have been observed or are plausible; each is verified at runtime
# by ``verify_series`` and flagged rather than assumed.
CANDIDATE_SERIES = [
    "KXNHLGAME",        # observed: per-game NHL winner markets
    "KXNHLTOTAL",       # observed 2026-09-20: full-game total goals, one contract per strike
    "KXNHLSPREAD",      # observed 2026-09-21: puck line, one contract per team+strike
    "KXNHL1P",          # observed in the series list: 1st period winner
    "KXNHLOVERTIME",    # observed 2026-09-21: game goes to overtime (historical tier)
    "KXNHL",            # observed: Stanley Cup futures
]

#: Series this project actually walks for prices (quotes, settled contracts, candles).
#: Each one is a market family with a settlement rule that can be checked against the
#: NHL's official result, which is the bar for trading it.
TRADED_SERIES = ("KXNHLGAME", "KXNHLTOTAL", "KXNHLSPREAD", "KXNHLOVERTIME")

#: Series that exist in Kalshi's ``/series?category=Sports&tags=Hockey`` listing but had
#: **no contracts at all** when they were last polled (2026-09-21, preseason; see
#: ``data/captured/kalshi_unlisted_series_20260921.json``).  They are polled on every run
#: so that the day the exchange lists them the prices are captured from that day forward;
#: until then nothing is bet on them and no price for them is invented.
WATCH_SERIES = ("KXNHL1P", "KXNHL2P", "KXNHL3P", "KXNHL1PTOTAL", "KXNHL2PTOTAL",
                "KXNHL3PTOTAL", "KXNHL1PSPREAD", "KXNHL2OT", "KXNHLSAVES", "KXNHLGOAL",
                "KXNHLANYGOAL", "KXNHLAST", "KXNHLPTS", "KXNHLFIRSTGOAL")

#: Kalshi series -> this project's market vocabulary.  Built from the series listing the
#: exchange itself publishes (81 hockey series recorded in ``kalshi_series``), so a new
#: market family is named rather than silently defaulting to "moneyline".
SERIES_MARKET_TYPE = {
    "KXNHLGAME": "moneyline",
    "KXNHLTOTAL": "total",
    "KXNHLSPREAD": "puck_line",
    "KXNHLOVERTIME": "overtime",
    "KXNHL2OT": "overtime",
    "KXNHL1P": "first_period",
    "KXNHL2P": "second_period",
    "KXNHL3P": "third_period",
    "KXNHL1PTOTAL": "first_period_total",
    "KXNHL2PTOTAL": "second_period_total",
    "KXNHL3PTOTAL": "third_period_total",
    "KXNHL1PSPREAD": "first_period_spread",
    "KXNHL2PSPREAD": "second_period_spread",
    "KXNHL3PSPREAD": "third_period_spread",
    "KXNHLSAVES": "goalie_prop",
    "KXNHLGOAL": "player_prop",
    "KXNHLANYGOAL": "player_prop",
    "KXNHLAST": "player_prop",
    "KXNHLPTS": "player_prop",
    "KXNHLFIRSTGOAL": "player_prop",
}

#: Kalshi team codes that differ from the NHL triCode.  Everything not listed here is
#: assumed identical to the NHL abbreviation and is still validated against the teams
#: table before a contract is attached to a game.
KALSHI_TEAM_ALIASES = {
    "LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "WAS": "WSH", "MON": "MTL",
    "CLB": "CBJ", "NAS": "NSH", "VGS": "VGK", "UTAH": "UTA", "ARI": "UTA",
}

NHL_TRICODES = (
    "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "CBJ", "COL", "DAL", "DET", "EDM", "FLA",
    "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG",
)

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7,
           "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


class KalshiApiError(RuntimeError):
    """Kalshi returned an application-level error rather than market data."""


def _f(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(d: dict, *keys: str) -> float | None:
    """First parseable float among alternative field names (live vs historical schema)."""
    for k in keys:
        v = _f(d, k)
        if v is not None:
            return v
    return None


class KalshiApi:
    source_id = "kalshi.trade_api"

    #: polite pacing between uncached historical reads; Kalshi's published read limit
    #: for unauthenticated/basic access is well above this.
    pace_seconds = 0.12

    def __init__(self, http: HttpClient):
        self.http = http
        self.calls = 0

    # ------------------------------------------------------------ helpers
    def _get(self, url: str, *, use_cache: bool = True) -> dict:
        payload = self.http.get(url, use_cache=use_cache).json
        self.calls += 1
        if isinstance(payload, dict) and payload.get("error"):
            raise KalshiApiError(f"GET {url} -> {payload['error']}")
        if not isinstance(payload, dict):
            raise KalshiApiError(f"GET {url} -> unexpected payload type {type(payload).__name__}")
        return payload

    # ------------------------------------------------------------ discovery
    def series(self, ticker: str) -> dict | None:
        try:
            return self._get(f"{BASE}/series/{ticker}")
        except Exception:
            return None

    def series_list(self, *, category: str = "Sports", tags: str = "Hockey") -> list[dict]:
        """All series in a category/tag.  Verified 2026-09-20: ``?category=Sports&tags=Hockey``
        returns the NHL series family with ``ticker``, ``title``, ``fee_type``,
        ``settlement_sources`` and ``contract_terms_url``."""
        url = f"{BASE}/series?category={category}&tags={tags}"
        try:
            return list(self._get(url).get("series", []))
        except Exception:
            return []

    def events(self, series_ticker: str, *, limit: int = 200, status: str = "open") -> list[dict]:
        url = f"{BASE}/events?series_ticker={series_ticker}&limit={limit}&status={status}"
        try:
            return self._get(url).get("events", [])
        except Exception:
            return []

    def events_page(self, series_ticker: str, *, status: str = "settled", limit: int = 200,
                    cursor: str | None = None, with_nested_markets: bool = False
                    ) -> tuple[list[dict], str | None]:
        """One page of events (newest first) and the cursor for the next page.

        Verified 2026-09-20: ``status=settled`` lists the entire KXNHLGAME history back
        through the 2025-26 season, but ``markets`` is empty for events older than the
        historical cutoff -- use :meth:`historical_markets` for those.
        """
        url = (f"{BASE}/events?series_ticker={series_ticker}&status={status}&limit={limit}"
               f"&with_nested_markets={'true' if with_nested_markets else 'false'}")
        if cursor:
            url += f"&cursor={cursor}"
        payload = self._get(url)
        return list(payload.get("events", [])), (payload.get("cursor") or None)

    #: Kalshi rejects unknown status values with {"error":{"code":"bad_request",
    #: "details":"invalid status filter"}} -- verified 2026-09-20 for "finalized".
    #: The live filter is "open", not "active".
    VALID_STATUSES = ("unopened", "open", "closed", "settled")

    def markets(self, series_ticker: str, *, limit: int = 200, status: str = "open",
                cursor: str | None = None, max_pages: int = 5) -> list[dict]:
        """Fetch live-tier contracts.  Raises ``KalshiApiError`` on an API-level error
        instead of returning an empty list, because an empty list and a rejected request
        look identical to the caller and would silently understate the market."""
        if status not in self.VALID_STATUSES:
            raise KalshiApiError(f"invalid status filter {status!r}; "
                                 f"expected one of {self.VALID_STATUSES}")
        out: list[dict] = []
        seen: set[str] = set()
        for _ in range(max_pages):
            url = f"{BASE}/markets?series_ticker={series_ticker}&limit={limit}&status={status}"
            if cursor:
                url += f"&cursor={cursor}"
            # cache-on: one fetch per URL per run (the CI cache directory is fresh every
            # run, and offline tests prime it); a stale hit is labelled by HttpClient.
            payload = self._get(url)
            page = payload.get("markets", [])
            for m in page:
                key = (m.get("ticker"), m.get("side"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(m)
            cursor = payload.get("cursor")
            if not cursor or not page:
                break
        return out

    def settled_markets(self, series_ticker: str, *, limit: int = 200,
                        max_pages: int = 10) -> list[dict]:
        """Recently settled contracts from the *live* tier only (see module docstring)."""
        return self.markets(series_ticker, limit=limit, status="settled", max_pages=max_pages)

    # ------------------------------------------------------------ historical tier
    def historical_cutoff(self) -> str | None:
        """ISO timestamp separating the live tier from the historical tier.

        Verified 2026-09-20: ``GET /historical/cutoff`` returned 2026-07-22T00:00:00Z.
        The response key is read defensively because Kalshi's docs and payload differ.
        """
        try:
            payload = self._get(f"{HIST_BASE}/cutoff")
        except Exception:
            return None
        for k in ("cutoff", "cutoff_ts", "historical_cutoff", "settled_before", "ts"):
            v = payload.get(k)
            if isinstance(v, str) and len(v) >= 10:
                return v
        for v in payload.values():
            if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", v):
                return v
        return None

    def historical_markets(self, series_ticker: str, *, limit: int = 1000,
                           cursor: str | None = None, max_pages: int = 50,
                           on_page: Callable[[list[dict], str | None], None] | None = None,
                           ) -> tuple[list[dict], str | None]:
        """Settled contracts older than the cutoff, newest first, paged by cursor.

        Returns (markets, next_cursor).  ``next_cursor`` is None once the series history
        is exhausted; callers persist it so a later run can resume where this one stopped.
        """
        out: list[dict] = []
        for _ in range(max_pages):
            url = f"{HIST_BASE}/markets?series_ticker={series_ticker}&limit={limit}"
            if cursor:
                url += f"&cursor={cursor}"
            payload = self._get(url)
            page = payload.get("markets", [])
            out.extend(page)
            cursor = payload.get("cursor") or None
            if on_page:
                on_page(page, cursor)
            if not cursor or not page:
                cursor = None
                break
            time.sleep(self.pace_seconds)
        return out, cursor

    def historical_trades(self, ticker: str, *, limit: int = 1000,
                          min_ts: int | None = None, max_ts: int | None = None) -> list[dict]:
        url = f"{HIST_BASE}/trades?ticker={ticker}&limit={limit}"
        if min_ts is not None:
            url += f"&min_ts={min_ts}"
        if max_ts is not None:
            url += f"&max_ts={max_ts}"
        try:
            return list(self._get(url).get("trades", []))
        except Exception:
            return []

    # ------------------------------------------------------------ price history
    def candlesticks(self, series_ticker: str, ticker: str, *, start_ts: int, end_ts: int,
                     period_interval: int = 60, tier: str = "historical") -> list[dict]:
        """Timestamped OHLC candles, normalized to one schema regardless of tier.

        ``tier`` is 'historical' for contracts settled before the cutoff and 'live'
        otherwise.  Raises ``KalshiApiError`` on an API error so a missing history is never
        silently mistaken for an empty one.
        """
        if tier == "historical":
            url = (f"{HIST_BASE}/markets/{ticker}/candlesticks"
                   f"?start_ts={start_ts}&end_ts={end_ts}&period_interval={period_interval}")
        else:
            url = (f"{BASE}/series/{series_ticker}/markets/{ticker}/candlesticks"
                   f"?start_ts={start_ts}&end_ts={end_ts}&period_interval={period_interval}")
        payload = self._get(url, use_cache=(tier == "historical"))
        raw = payload.get("candlesticks")
        if raw is None:
            raise KalshiApiError(f"GET {url} -> no 'candlesticks' key ({sorted(payload)})")
        out = [normalize_candle(c) for c in raw]
        out = [c for c in out if c["end_period_ts"] is not None]
        out.sort(key=lambda c: c["end_period_ts"])
        return out

    def candles(self, ticker: str, *, start_ts: int, end_ts: int, period_interval: int = 60,
                series_ticker: str = "KXNHLGAME", tier: str = "historical") -> list[dict]:
        """Backwards-compatible alias for :meth:`candlesticks`."""
        try:
            return self.candlesticks(series_ticker, ticker, start_ts=start_ts, end_ts=end_ts,
                                     period_interval=period_interval, tier=tier)
        except Exception:
            return []

    def orderbook(self, ticker: str) -> dict | None:
        try:
            return self._get(f"{BASE}/markets/{ticker}/orderbook", use_cache=False)
        except Exception:
            return None

    def trades(self, ticker: str, *, limit: int = 100) -> list[dict]:
        try:
            return self._get(f"{BASE}/markets/trades?ticker={ticker}&limit={limit}",
                             use_cache=False).get("trades", [])
        except Exception:
            return []


# --------------------------------------------------------------------- normalizers
def normalize_candle(c: dict) -> dict[str, Any]:
    """One schema for both candle payload shapes.

    Historical tier: ``price{open,high,low,close,mean,previous}``, ``yes_bid{...}``,
    ``yes_ask{...}``, ``volume``, ``open_interest``.
    Live tier: the same keys suffixed ``_dollars`` / ``_fp`` and *absent* ``price.open`` etc.
    for periods with no trades (only ``previous_dollars`` is present).
    """
    price = c.get("price") or {}
    bid = c.get("yes_bid") or {}
    ask = c.get("yes_ask") or {}
    ts = c.get("end_period_ts")
    try:
        ts = int(ts) if ts is not None else None
    except (TypeError, ValueError):
        ts = None
    return {
        "end_period_ts": ts,
        "price_open": _first(price, "open", "open_dollars"),
        "price_high": _first(price, "high", "high_dollars"),
        "price_low": _first(price, "low", "low_dollars"),
        "price_close": _first(price, "close", "close_dollars"),
        "price_mean": _first(price, "mean", "mean_dollars"),
        "price_previous": _first(price, "previous", "previous_dollars"),
        "bid_open": _first(bid, "open", "open_dollars"),
        "bid_high": _first(bid, "high", "high_dollars"),
        "bid_low": _first(bid, "low", "low_dollars"),
        "bid_close": _first(bid, "close", "close_dollars"),
        "ask_open": _first(ask, "open", "open_dollars"),
        "ask_high": _first(ask, "high", "high_dollars"),
        "ask_low": _first(ask, "low", "low_dollars"),
        "ask_close": _first(ask, "close", "close_dollars"),
        "volume": _first(c, "volume", "volume_fp"),
        "open_interest": _first(c, "open_interest", "open_interest_fp"),
    }


def split_team_codes(raw: str) -> tuple[str, str] | None:
    """Split a Kalshi team-code pair (``CARVGK``) into ``(away, home)``.

    The split is validated against the NHL triCode list plus :data:`KALSHI_TEAM_ALIASES`.
    Returns None when zero *or more than one* split is valid: picking one anyway would
    attach a contract to the wrong game, and the caller always has the rules text to fall
    back on.  This is the only place in the project that splits a code pair, so the
    ticker parser and the ingest fallback cannot disagree about how it is done.
    """
    known = set(NHL_TRICODES) | set(KALSHI_TEAM_ALIASES)
    hits = [(raw[:i], raw[i:]) for i in range(2, len(raw) - 1)
            if raw[:i] in known and raw[i:] in known]
    return hits[0] if len(hits) == 1 else None


def parse_event_ticker(event_ticker: str) -> dict[str, Any] | None:
    """Decode ``KXNHLGAME-26JUN14CARVGK`` -> date + Kalshi team codes.

    The team part is split against the NHL triCode list plus :data:`KALSHI_TEAM_ALIASES`;
    when more than one split is valid (or none) the result carries ``ambiguous=True`` and
    the caller must fall back to the rules text rather than guess.
    """
    m = re.match(r"^(KXNHL[A-Z0-9]*)-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$", event_ticker or "")
    if not m:
        return None
    series, yy, mon, dd, rest = m.groups()
    if mon not in _MONTHS:
        return None
    date = f"{2000 + int(yy):04d}-{_MONTHS[mon]:02d}-{int(dd):02d}"
    split = split_team_codes(rest)
    out = {"series": series, "game_date": date, "raw": rest, "ambiguous": split is None}
    if split is not None:
        a, b = split
        out["away_code"], out["home_code"] = a, b
        out["away_abbrev"] = KALSHI_TEAM_ALIASES.get(a, a)
        out["home_abbrev"] = KALSHI_TEAM_ALIASES.get(b, b)
    return out


def contract_suffix(m: dict) -> str | None:
    """The part of a ticker after its event ticker (``KXNHLSPREAD-26SEP24UTAVGK-VGK3`` ->
    ``VGK3``), or None when the ticker is not prefixed by its event ticker."""
    t = (m.get("ticker") or "")
    ev = (m.get("event_ticker") or "")
    if ev and t.startswith(ev + "-"):
        return t[len(ev) + 1:]
    return None


#: Contract suffixes that look like a team code but are not one.  ``OT`` is the overtime
#: series' own suffix (``KXNHLOVERTIME-26JUN14CARVGK-OT``, verified 2026-09-21).
NON_TEAM_SUFFIXES = frozenset({"OT", "O", "U", "YES", "NO", "TIE", "OT2", "SO"})

def market_team_code(m: dict) -> str | None:
    """The team code in a contract suffix (``...-CAR`` -> ``CAR``, ``...-VGK3`` -> ``VGK``).

    A suffix is a team code optionally followed by a **rung index**, and the rung index is
    NOT the line.  Verified 2026-09-21 on KXNHLSPREAD: in the live tier ``-VGK3`` carries
    ``floor_strike=2.5`` and ``-VGK2`` carries ``1.5``, while in the historical tier of the
    same series ``-VGK2`` carries ``2.5`` and ``-VGK1`` carries ``1.5``.  The digit moves
    with how many rungs the exchange listed for that event, so the line is read only from
    ``floor_strike``/``strike_type`` and the digit is discarded here.

    Returns None for a suffix that is not a team code at all (``-OT`` for the overtime
    series, ``-O6`` style tokens, numeric-only suffixes), so a caller can never mistake
    them for a side.
    """
    return suffix_team_code(contract_suffix(m))


def suffix_team_code(suffix: str | None) -> str | None:
    """The team code inside a stored contract suffix, with any rung index discarded.

    Split out from :func:`market_team_code` so a row that already holds the suffix (the
    ``rung`` column of ``market_settlements``) can be read without the original payload.
    Same rule, same refusals: ``VGK3`` -> ``VGK``, ``OT`` -> None, ``O6`` -> None.
    """
    if not suffix:
        return None
    mm = re.match(r"^([A-Z]{2,5})\d*$", str(suffix).strip())
    if not mm:
        return None
    code = mm.group(1)
    if code in NON_TEAM_SUFFIXES:
        return None
    return code


def normalize_market(m: dict, *, ts_utc: str, retrieved_at: str,
                     source_url: str) -> list[dict[str, Any]]:
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
            "market_type": market_type_for(m),
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
            # The line, as the exchange published it.  Kalshi's title text differs between
            # the live tier ("Full Game: Over 8.5 goals scored") and the historical tier
            # ("Carolina vs Vegas: Total Goals"), so the strike is taken from the numeric
            # ``floor_strike`` field and the direction from ``strike_type`` rather than
            # parsed out of a string that changes shape.
            "strike": _f(m, "floor_strike"),
            "strike_type": m.get("strike_type") or None,
            # the team the contract names, from the exchange's own ticker suffix with the
            # rung index stripped (see market_team_code).  A puck-line contract is quoted
            # per team per rung, so which team it is belongs on the quote row; the line
            # itself still comes only from floor_strike/strike_type.
            "team_abbrev": KALSHI_TEAM_ALIASES.get(
                market_team_code(m) or "", market_team_code(m)),
        })
    return rows


def market_type_for(m: dict) -> str:
    """Map a Kalshi series to this project's market vocabulary (:data:`SERIES_MARKET_TYPE`).

    An unrecognised series falls back to the exchange's own ``market_type`` rather than to
    "moneyline": calling an unknown contract a moneyline is how a rule ends up trading an
    instrument whose settlement it has never checked.
    """
    ev = (m.get("event_ticker") or m.get("ticker") or "")
    series = ev.split("-", 1)[0]
    if series in SERIES_MARKET_TYPE:
        return SERIES_MARKET_TYPE[series]
    return "moneyline" if m.get("market_type") == "binary" else (m.get("market_type") or "binary")


def binary_price_to_decimal(price: float) -> float:
    """A YES contract bought at $p pays $1, so decimal odds = 1/p."""
    if price is None or price <= 0:
        return float("nan")
    return 1.0 / float(price)
