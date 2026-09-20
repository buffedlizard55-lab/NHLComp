"""The permanent NHL DATA SOURCE REGISTRY.

Every entry records the fields required by the project brief.  The ``status`` field is
only ever set to ``verified`` by :func:`verify_all`, which actually performs an HTTP
probe and stores the observation in ``source_verification``.  Entries below start as
``candidate`` with a human-readable note about what was observed during development;
the machine check is what promotes them.

Cost/licensing wording deliberately avoids treating "free tier" as "free": sources that
require a key, a trial, or paid credits are marked ``freemium`` or ``paid`` even when a
$0 entry point is advertised.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..store import Store, utcnow


@dataclass
class SourceSpec:
    source_id: str
    name: str
    url: str
    data_type: str
    nhl_relevance: str
    historical_depth: str = "unknown"
    live_available: bool = False
    update_frequency: str = "unknown"
    api_available: bool = False
    auth_required: str = "none"
    cost: str = "free"
    genuinely_free: str = "unknown"     # yes | no | freemium | unknown
    usage_limits: str = ""
    licensing: str = "not published"
    provenance: str = ""
    reliability: str = "unknown"
    accuracy: str = "unknown"
    granularity: str = ""
    automated_access: str = "unknown"
    known_limits: str = ""
    notes: str = ""
    probe_urls: tuple[str, ...] = ()
    status: str = "candidate"


SOURCES: tuple[SourceSpec, ...] = (
    # ---------------------------------------------------------- NHL first party
    SourceSpec(
        source_id="nhl.api_web",
        name="NHL Web API (api-web.nhle.com)",
        url="https://api-web.nhle.com/v1",
        data_type="games, schedules, standings, boxscores, play-by-play, rosters, EDGE featured stats",
        nhl_relevance="Primary source for league games, results, schedule, standings and per-game detail",
        historical_depth="Multi-season; per-game endpoints accept historical game ids",
        live_available=True,
        update_frequency="Live during games; standings/schedule update continuously",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="No published rate limit or ToS for programmatic use; observed 200s without a key",
        licensing="No machine-readable licence published; NHL.com terms of use apply",
        provenance="NHL (first party)",
        reliability="high",
        accuracy="authoritative for results and schedules",
        granularity="per event / per game / per season",
        automated_access="yes (plain HTTPS GET, JSON, no key)",
        known_limits="Unofficial endpoint: shape can change without notice; no SLA",
        notes="Observed HTTP 200 with live JSON from the build environment on 2026-09-20 "
              "(scoreboard/now returned 9 focused dates including 2026-09-19 preseason results).",
        probe_urls=("https://api-web.nhle.com/v1/scoreboard/now",
                    "https://api-web.nhle.com/v1/standings/now"),
    ),
    SourceSpec(
        source_id="nhl.stats_rest",
        name="NHL Records / Stats REST API (api.nhle.com)",
        url="https://api.nhle.com/stats/rest/en",
        data_type="historical records, franchise and player aggregates, paginated tables",
        nhl_relevance="Historical depth for team/player records across many seasons",
        historical_depth="Full league history (back to 1917 franchises)",
        live_available=False,
        update_frequency="Daily",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Paginated (start/limit); cayenneExp filter syntax",
        licensing="No machine-readable licence published",
        provenance="NHL (first party)",
        reliability="high",
        accuracy="authoritative for official records",
        granularity="season / game aggregates",
        automated_access="yes (plain HTTPS GET, JSON, no key)",
        known_limits="Filter syntax errors return 200 with an error message in the body, "
                     "so HTTP status alone is not proof of a valid query",
        notes="Observed HTTP 200 on /team (62 franchise rows) and a structured error body for a "
              "malformed cayenneExp on 2026-09-20.",
        probe_urls=("https://api.nhle.com/stats/rest/en/team",),
    ),
    SourceSpec(
        source_id="nhl.api_v1",
        name="NHL API v1 (api.nhl.com)",
        url="https://api.nhl.com/api/v1",
        data_type="teams, players, games (newer generation)",
        nhl_relevance="Potential replacement for the legacy statsapi host",
        historical_depth="unknown",
        live_available=False,
        update_frequency="unknown",
        api_available=True,
        auth_required="unknown",
        cost="free",
        genuinely_free="unknown",
        usage_limits="unknown",
        licensing="not published",
        provenance="NHL (first party)",
        reliability="unknown",
        accuracy="unknown",
        granularity="unknown",
        automated_access="unknown",
        known_limits="Two independent fetch attempts from the build sandbox failed; this may be a "
                     "sandbox egress restriction rather than a dead endpoint. MUST be re-probed "
                     "from a normal network before use.",
        notes="FLAGGED: not reachable from the development sandbox on 2026-09-20. Not rejected - "
              "the CI probe decides.",
        probe_urls=("https://api.nhl.com/api/v1/teams",),
        status="flagged",
    ),
    SourceSpec(
        source_id="nhl.legacy_statsapi",
        name="NHL legacy Stats API (statsapi.web.nhl.com)",
        url="https://statsapi.web.nhl.com/api/v1",
        data_type="teams, venues (with lat/lon), schedules, boxscores",
        nhl_relevance="Was the canonical NHL JSON API; only known public source of arena coordinates",
        historical_depth="Multi-season",
        live_available=False,
        update_frequency="unknown",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="unknown",
        usage_limits="unknown",
        licensing="not published",
        provenance="NHL (first party)",
        reliability="unknown",
        accuracy="was authoritative",
        granularity="per game",
        automated_access="unknown",
        known_limits="Fetch failed from the build sandbox on 2026-09-20. Widely reported as "
                     "deprecated in favour of api-web.nhle.com.",
        notes="FLAGGED: if this host stays unreachable, arena latitude/longitude is NOT available "
              "from a first-party source and travel distance must be derived from venue UTC "
              "offsets instead of invented coordinates.",
        probe_urls=("https://statsapi.web.nhl.com/api/v1/teams",
                    "https://statsapi.web.nhl.com/api/v1/venues"),
        status="flagged",
    ),
    SourceSpec(
        source_id="nhl.edge",
        name="NHL EDGE (puck & player tracking)",
        url="https://www.nhl.com/stats/edge",
        data_type="skating speed, skating distance, shot speed, shot location, zone time",
        nhl_relevance="Tracking metrics for player-prop and speed-based hypotheses",
        historical_depth="2021-22 onward (tracking rollout)",
        live_available=True,
        update_frequency="Live and daily",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Exposed through api-web player/team landing payloads rather than a "
                     "documented standalone API",
        licensing="not published",
        provenance="NHL (first party)",
        reliability="medium",
        accuracy="official tracking feed",
        granularity="per player / per game",
        automated_access="partial",
        known_limits="Not every metric is available for every game; coverage varies by arena "
                     "and season. No historical bulk export.",
        notes="Accessed via api-web player landing featuredStats; each metric must be checked "
              "for coverage before use.",
        probe_urls=("https://www.nhl.com/stats/edge",),
    ),
    # ---------------------------------------------------------- market data
    SourceSpec(
        source_id="kalshi.trade_api",
        name="Kalshi Trade API v2 (public market data)",
        url="https://api.elections.kalshi.com/trade-api/v2",
        data_type="binary contract quotes: bid, ask, sizes, volume, liquidity, settlement",
        nhl_relevance="Real, timestamped, tradable NHL prices and depth - the only verified "
                      "no-key source of live NHL market prices in this project",
        historical_depth="Candles endpoint may provide history; verified separately",
        live_available=True,
        update_frequency="Continuous",
        api_available=True,
        auth_required="none for market data reads (credentials required to trade)",
        cost="free",
        genuinely_free="yes",
        usage_limits="Public read endpoints; trading requires a funded US account and KYC",
        licensing="Kalshi API terms apply",
        provenance="Kalshi (first party exchange)",
        reliability="high",
        accuracy="exchange-grade quote data",
        granularity="per contract, per tick",
        automated_access="yes",
        known_limits="NHL liquidity is thin (observed bid/ask sizes of 5-555 contracts and "
                     "$0 liquidity on many preseason games), so the execution model must model "
                     "partial fills. Preseason markets may be sparse.",
        notes="Observed HTTP 200 with live KXNHLGAME contracts on 2026-09-20, e.g. "
              "KXNHLGAME-26SEP23MINDAL-MIN yes_ask 0.49 size 10, volume 7.71 on the DAL leg.",
        probe_urls=("https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNHLGAME&limit=5",),
    ),
    SourceSpec(
        source_id="kalshi.candles",
        name="Kalshi historical candles",
        url="https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candles",
        data_type="timestamped OHLC price history per contract",
        nhl_relevance="Would make fully timestamped price-based BACKTESTS possible",
        historical_depth="n/a",
        live_available=False,
        update_frequency="n/a",
        api_available=False,
        auth_required="none",
        cost="free",
        genuinely_free="unknown",
        usage_limits="n/a",
        licensing="Kalshi API terms apply",
        provenance="Kalshi (first party exchange)",
        reliability="n/a",
        accuracy="n/a",
        granularity="n/a",
        automated_access="no",
        known_limits="Both documented path shapes returned HTTP 404 'page not found' from the "
                     "build environment on 2026-09-20 for a known settled contract "
                     "(KXNHLGAME-26SEP19VGKLA-VGK). Treated as UNAVAILABLE.",
        notes="REJECTED: verified unavailable. Intraday price history cannot be reconstructed, so "
              "no intraday line-movement backtest is attempted anywhere in this project.",
        probe_urls=("https://api.elections.kalshi.com/trade-api/v2/series/KXNHLGAME/markets/"
                    "KXNHLGAME-26SEP19VGKLA-VGK/candles?start_ts=1789603200&end_ts=1789905600"
                    "&period_interval=60",
                    "https://api.elections.kalshi.com/trade-api/v2/markets/"
                    "KXNHLGAME-26SEP19VGKLA-VGK/candles?start_ts=1789603200&end_ts=1789905600"
                    "&period_interval=60"),
        status="rejected",
    ),
    SourceSpec(
        source_id="kalshi.settled_markets",
        name="Kalshi settled NHL game markets",
        url="https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNHLGAME&status=settled",
        data_type="settled binary contracts with pre-settlement bid/ask, volume, open interest "
                  "and the official result",
        nhl_relevance="The only verified source of real historical NHL prices available to this "
                      "project, and therefore the only legitimate basis for price-based BACKTESTS",
        historical_depth="As far back as Kalshi has listed NHL game markets",
        live_available=False,
        update_frequency="On settlement",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Paginated with a cursor; be conservative with page size",
        licensing="Kalshi API terms apply",
        provenance="Kalshi (first party exchange)",
        reliability="high",
        accuracy="exchange-grade",
        granularity="per contract",
        automated_access="yes",
        known_limits="The exact timestamp of previous_yes_bid_dollars / previous_yes_ask_dollars is "
                     "NOT documented by Kalshi. It is known only to precede settlement. Every bet "
                     "built from it is therefore stamped verification_status="
                     "'single_source_timing_unverified' and carries a caveat, rather than being "
                     "presented as a precisely timestamped entry.",
        notes="Observed HTTP 200 on 2026-09-20: KXNHLGAME-26SEP19VGKLA-VGK result=yes, "
              "previous_yes_ask 0.47, previous_yes_bid 0.44, volume 30711 contracts, "
              "settlement_ts 2026-09-20T04:32:34Z.",
        probe_urls=("https://api.elections.kalshi.com/trade-api/v2/markets"
                    "?series_ticker=KXNHLGAME&limit=3&status=settled",),
    ),
    SourceSpec(
        source_id="odds.the_odds_api",
        name="The Odds API",
        url="https://the-odds-api.com/",
        data_type="sportsbook moneyline/spread/total odds, live and some history",
        nhl_relevance="Multi-book closing prices needed for CLV measurement",
        historical_depth="Paid history endpoint",
        live_available=True,
        update_frequency="Continuous",
        api_available=True,
        auth_required="api_key",
        cost="freemium",
        genuinely_free="freemium",
        usage_limits="Advertised free tier is a limited monthly credit allowance; history is paid",
        licensing="Commercial licence",
        provenance="Aggregator (second party)",
        reliability="medium",
        accuracy="aggregated book prices",
        granularity="per book, per market, timestamped",
        automated_access="yes with key",
        known_limits="NOT USABLE IN THIS BUILD: no API key is available in this environment and "
                     "none may be invented. All sportsbook price work is therefore flagged, and "
                     "closing-line value is computed against Kalshi instead.",
        notes="Explicitly not treated as free. Rejected for automated use until a key exists.",
        status="rejected",
    ),
    # ---------------------------------------------------------- secondary / cross-check
    SourceSpec(
        source_id="espn.nhl_api",
        name="ESPN NHL site API",
        url="https://site.api.espn.com/apis/site/v2/sports/hockey/nhl",
        data_type="injuries with timestamps, scores, schedules, team/player news",
        nhl_relevance="Independent cross-check for injuries and results; primary injury feed",
        historical_depth="Current injury list only; no injury archive",
        live_available=True,
        update_frequency="Continuous",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Undocumented; be conservative with request volume",
        licensing="not published",
        provenance="ESPN (second party)",
        reliability="medium",
        accuracy="editorial; used only for cross-validation, never as the sole authority",
        granularity="per player",
        automated_access="yes",
        known_limits="Payload is extremely verbose (74 response chunks for one call in testing); "
                     "parse only the injury fields. ESPN team ids differ from NHL team ids.",
        notes="Observed HTTP 200 on /injuries on 2026-09-20 with a dated entry "
              "(Ryan Poehling, ANA, day-to-day, 2026-09-20T04:22Z).",
        probe_urls=("https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",),
    ),
    SourceSpec(
        source_id="openmeteo.archive",
        name="Open-Meteo Historical Weather Archive",
        url="https://archive-api.open-meteo.com/v1/archive",
        data_type="hourly historical temperature, wind, precipitation, snow",
        nhl_relevance="Outdoor / Winter Classic and Stadium Series games only",
        historical_depth="1940 onward (ERA5)",
        live_available=False,
        update_frequency="Daily",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Non-commercial free tier; 10k calls/day",
        licensing="CC BY 4.0 for the underlying reanalysis",
        provenance="Copernicus ERA5 / NOAA (third party redistribution)",
        reliability="high",
        accuracy="reanalysis, not station observation",
        granularity="hourly",
        automated_access="yes",
        known_limits="Only relevant to the handful of outdoor NHL games per season; indoor arenas "
                     "make this irrelevant for ~99% of games. Do not build a general weather strategy.",
        notes="Observed HTTP 200 on 2026-09-20 returning hourly values for a 2026-01-01 window.",
        probe_urls=("https://api.open-meteo.com/v1/forecast?latitude=40.75&longitude=-73.99&current=temperature_2m",),
    ),
    # ---------------------------------------------------------- analytics communities
    SourceSpec(
        source_id="nst.money_puck_evolved",
        name="Natural Stat Trick / MoneyPuck / Evolving-Hockey",
        url="https://www.naturalstattrick.com/",
        data_type="xG, high-danger chances, Corsi, Fenwick, zone entries",
        nhl_relevance="Advanced metrics that NHL does not publish in bulk",
        historical_depth="Multi-season",
        live_available=False,
        update_frequency="Daily",
        api_available=False,
        auth_required="none",
        cost="free",
        genuinely_free="freemium",
        usage_limits="HTML scraping; several sites paywall or rate-limit; terms generally prohibit "
                     "systematic scraping",
        licensing="not published",
        provenance="Third party analytics",
        reliability="medium",
        accuracy="model-derived (xG models differ between providers)",
        granularity="per game / per player",
        automated_access="not permitted without permission",
        known_limits="REJECTED for automated ingest: no API, scraping conflicts with site terms, "
                     "and xG is a model output that would be mislabelled as source data.",
        notes="Recorded so the absence is a decision, not an oversight. Revisit only with an "
              "explicit data agreement.",
        status="rejected",
    ),
    SourceSpec(
        source_id="research.public",
        name="Public hockey analytics research (papers, GitHub, forums)",
        url="https://github.com/topics/nhl",
        data_type="hypotheses, prior literature, methodology",
        nhl_relevance="Source of testable public strategy claims (brief section 27)",
        historical_depth="n/a",
        live_available=False,
        update_frequency="n/a",
        api_available=False,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Per-repository licences",
        licensing="varies",
        provenance="Community",
        reliability="low",
        accuracy="claims are unverified until tested here",
        granularity="n/a",
        automated_access="n/a",
        known_limits="A claim that a strategy 'wins' is treated as a hypothesis with zero prior "
                     "weight until it is reproduced on our own data.",
        notes="Used for hypothesis generation only, never as a data source.",
    ),
)


def seed_registry(store: Store, *, verified_probe: dict[str, tuple[int | None, str, str]] | None = None
                  ) -> int:
    """Insert/refresh every registry row.

    ``verified_probe`` maps source_id -> (http_status, verdict, evidence).  When supplied the
    row is promoted to ``verified``; otherwise it keeps its declared status and the probe is
    recorded as the evidence.
    """
    verified_probe = verified_probe or {}
    now = utcnow()
    n = 0
    for spec in SOURCES:
        d = asdict(spec)
        d["live_available"] = int(spec.live_available)
        d["api_available"] = int(spec.api_available)
        d["first_seen_at"] = now
        status = spec.status
        if spec.source_id in verified_probe:
            http_status, verdict, evidence = verified_probe[spec.source_id]
            store.execute(
                """INSERT INTO source_verification
                   (source_id, checked_at, method, url_probed, http_status, ok, evidence, verdict, notes)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (spec.source_id, now, "http_get", spec.probe_urls[0] if spec.probe_urls else spec.url,
                 http_status, 1 if verdict == "reachable" else 0, evidence, verdict,
                 "automated probe"),
            )
            if verdict == "reachable" and spec.status != "rejected":
                status = "verified"
        d["status"] = status
        d["last_verified_at"] = now if spec.source_id in verified_probe else None
        cols = [c for c in d if c != "probe_urls"]
        store.execute(
            f"""INSERT INTO source_registry({','.join(cols)})
                VALUES({','.join('?' * len(cols))})
                ON CONFLICT(source_id) DO UPDATE SET
                  {','.join(f'{c}=excluded.{c}' for c in cols if c != 'source_id')}""",
            tuple(d[c] for c in cols),
        )
        n += 1
    store.commit()
    return n


def probe_urls_for(spec: SourceSpec) -> list[str]:
    return list(spec.probe_urls)
