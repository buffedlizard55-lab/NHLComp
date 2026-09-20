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
        name="NHL EDGE tracking API (api-web.nhle.com/v1/edge)",
        url="https://api-web.nhle.com/v1/edge",
        data_type="team + skater + goalie tracking aggregates: skating distance, speed bursts, "
                  "shot speed, zone time, top-10 leaderboards",
        nhl_relevance="Tracking metrics for speed/shot-quality hypotheses (brief 25). Not assumed "
                      "predictive; snapshotted so a test becomes possible later.",
        historical_depth="Season-to-date aggregates from 2021-22 (seasonsWithEdgeStats); NO "
                         "per-game history endpoint found, so only dated snapshots accumulate here",
        live_available=False,
        update_frequency="Daily during the season",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Undocumented public JSON; treated as rate-sensitive (one call per team per run)",
        licensing="not published",
        provenance="NHL (first party)",
        reliability="medium",
        accuracy="official tracking feed",
        granularity="per team / per player, season-to-date",
        automated_access="yes",
        known_limits="The HTML page www.nhl.com/stats/edge is NOT reachable as a plain GET (the "
                     "earlier 'unreachable' verdict probed that page). The JSON API "
                     "/v1/edge/team-comparison/{teamId}/{season}/{gameType} and "
                     "/v1/edge/skater-speed-top-10/... returned HTTP 200 on 2026-09-20. Values "
                     "are season aggregates: a snapshot taken today cannot be used for a game "
                     "played last month without look-ahead, so EDGE is FORWARD-only until a "
                     "dated history exists.",
        notes="Snapshotted into edge_team_snapshots each run (SOURCE DATA, forward-only).",
        probe_urls=("https://api-web.nhle.com/v1/edge/skater-speed-top-10/F/max/20252026/2",),
    ),
    SourceSpec(
        source_id="nhl.stats_rest_game",
        name="NHL Stats REST per-game reports (team/summary, goalie/summary with isGame=true)",
        url="https://api.nhle.com/stats/rest/en",
        data_type="one row per team-game (PP%, PK%, shots for/against, faceoff%, GF/GA, "
                  "home/road, opponent) and one row per goalie-game (gamesStarted, saves, "
                  "shotsAgainst, savePct, TOI, decision)",
        nhl_relevance="Verified official per-game special-teams, shot-share and goalie logs -> the "
                      "point-in-time features for the goaltending / special-teams / PDO strategies",
        historical_depth="Multi-season (seasonId + gameTypeId filters); 2025-26 regular season "
                         "returned 2,624 team-game rows and 2,768 goalie-game rows",
        live_available=False,
        update_frequency="After each game",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="limit=100 pages with start offset; ~27 pages per season per report",
        licensing="not published",
        provenance="NHL (first party)",
        reliability="high",
        accuracy="official game statistics",
        granularity="per team-game / per goalie-game",
        automated_access="yes",
        known_limits="Rows appear only after a game is final, so for UPCOMING games the starter is "
                     "unknown from this feed (goalie-gated strategies stay WAITING FOR GOALIE). "
                     "For past games the starter identity is the post-game log: using it as a "
                     "pre-game feature is an explicit ASSUMPTION (starters are announced at the "
                     "morning skate) recorded on every such strategy.",
        notes="Verified 2026-09-20: team/summary?isAggregate=false&isGame=true&cayenneExp="
              "seasonId=20252026 and gameTypeId=2 -> total 2624; goalie/summary same filter -> "
              "total 2768, fields gamesStarted/saves/shotsAgainst/savePct/timeOnIce/teamAbbrev.",
        probe_urls=("https://api.nhle.com/stats/rest/en/goalie/summary?isAggregate=false&isGame=true"
                    "&limit=1&cayenneExp=seasonId=20252026%20and%20gameTypeId=2",),
    ),
    SourceSpec(
        source_id="nhl.partner_odds",
        name="NHL partner-game odds feed (DraftKings via api-web)",
        url="https://api-web.nhle.com/v1/partner-game/US/now",
        data_type="American-odds moneyline (MONEY_LINE_2_WAY), puck line and total for the current "
                  "slate, per partner",
        nhl_relevance="The only verified no-key sportsbook line in this project; used as a "
                      "REFERENCE price (de-vigged) next to each Kalshi forward signal, never as an "
                      "execution price",
        historical_depth="NONE: current slate only; snapshots accumulate from 2026-09-20",
        live_available=True,
        update_frequency="Continuous while the slate is open",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Undocumented public JSON; one call per run",
        licensing="not published (DraftKings odds redistributed by the NHL)",
        provenance="NHL (first party feed) carrying DraftKings (partnerId 9) prices",
        reliability="medium",
        accuracy="as published by the partner at retrieval time",
        granularity="per game / per market",
        automated_access="yes",
        known_limits="No history -> FORWARD TEST only. A sportsbook line is a quote with margin, not "
                     "a probability; it is de-vigged proportionally before comparison and the raw "
                     "American odds are stored unchanged.",
        notes="Verified 2026-09-20: partner-game/US/now returned games with MONEY_LINE_2_WAY, "
              "PUCK_LINE and OVER_UNDER entries (value = American odds, qualifier = line).",
        probe_urls=("https://api-web.nhle.com/v1/partner-game/US/now",),
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
        name="Kalshi candlesticks (live tier)",
        url="https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candlesticks",
        data_type="timestamped 1/60/1440-minute candles per contract: yes_bid/yes_ask OHLC, trade "
                  "price OHLC/mean, volume, open interest",
        nhl_relevance="Makes timestamped price-based BACKTESTS and closing-line value possible: "
                      "the last pre-game candle is the closing price; earlier candles give the "
                      "opening and T-24h/T-6h/T-1h lines.",
        historical_depth="Contracts settled after the historical cutoff (2026-07-22T00:00:00Z); "
                         "older contracts are served by kalshi.historical",
        live_available=True,
        update_frequency="Continuous",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Public read endpoint; this project paces uncached reads at ~8/s and caps "
                     "calls per run",
        licensing="Kalshi API terms apply",
        provenance="Kalshi (first party exchange)",
        reliability="high",
        accuracy="exchange-grade",
        granularity="per contract, per period",
        automated_access="yes",
        known_limits="The endpoint is named 'candlesticks'. The 2026-09-20 probes that returned "
                     "404 used the path segment 'candles', which does not exist -- that verdict "
                     "was a wrong-URL error, not an unavailable feed, and is corrected here. "
                     "Periods with no trades carry only price.previous plus bid/ask OHLC.",
        notes="Verified 2026-09-20: /series/KXNHLGAME/markets/KXNHLGAME-26SEP19VGKLA-VGK/"
              "candlesticks?period_interval=60 returned hourly candles (e.g. end_period_ts "
              "1789653600 yes_ask close 0.59, yes_bid close 0.51, volume 22.85).",
        probe_urls=("https://api.elections.kalshi.com/trade-api/v2/series/KXNHLGAME/markets/"
                    "KXNHLGAME-26SEP19VGKLA-VGK/candlesticks?start_ts=1789603200"
                    "&end_ts=1789905600&period_interval=60",),
    ),
    SourceSpec(
        source_id="kalshi.historical",
        name="Kalshi historical tier (markets + candlesticks before the cutoff)",
        url="https://api.elections.kalshi.com/trade-api/v2/historical",
        data_type="finalized contracts (result, settlement_ts, close_time, volume) and their "
                  "candlestick history for everything settled before /historical/cutoff",
        nhl_relevance="The full KXNHLGAME record back through the 2025-26 season (regular season "
                      "and playoffs) -- the basis for every priced BACKTEST in this project",
        historical_depth="All KXNHLGAME contracts settled before 2026-07-22T00:00:00Z",
        live_available=False,
        update_frequency="Static (contracts migrate here after the cutoff moves)",
        api_available=True,
        auth_required="none",
        cost="free",
        genuinely_free="yes",
        usage_limits="Cursor pagination up to 1000 rows/page; no time filter on the market list, "
                     "so the full series is walked once and resumed via a stored cursor",
        licensing="Kalshi API terms apply",
        provenance="Kalshi (first party exchange)",
        reliability="high",
        accuracy="exchange-grade",
        granularity="per contract; candles per period",
        automated_access="yes",
        known_limits="Only settled/finalized contracts. Candle timestamps are period ends; the "
                     "'close' used here is the last 60-minute candle ending at or before the NHL "
                     "scheduled start, so it can include trades up to puck drop.",
        notes="Verified 2026-09-20: /historical/cutoff -> 2026-07-22T00:00:00Z; "
              "/historical/markets?series_ticker=KXNHLGAME&limit=2 -> KXNHLGAME-26JUN14CARVGK-CAR "
              "result yes, volume_fp 5825067.90; /historical/markets/{ticker}/candlesticks -> "
              "hourly candles with yes_ask close 0.53 at end_period_ts 1781481600.",
        probe_urls=("https://api.elections.kalshi.com/trade-api/v2/historical/cutoff",),
    ),
    SourceSpec(
        source_id="kalshi.settled_markets",
        name="Kalshi settled NHL game markets",
        url="https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNHLGAME&status=settled",
        data_type="settled binary contracts with pre-settlement bid/ask, volume, open interest "
                  "and the official result",
        nhl_relevance="The only verified source of real historical NHL prices available to this "
                      "project, and therefore the only legitimate basis for price-based BACKTESTS",
        historical_depth="LIVE TIER ONLY: contracts settled after the historical cutoff "
                         "(12 KXNHLGAME contracts on 2026-09-20; cursor exhausted). Everything "
                         "older is served by kalshi.historical.",
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
        name="Natural Stat Trick / Evolving-Hockey (HTML)",
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
        usage_limits="HTML only; Evolving-Hockey paywalls most tables; systematic scraping is not "
                     "permitted by the sites",
        licensing="not published",
        provenance="Third party analytics",
        reliability="medium",
        accuracy="model-derived (xG models differ between providers)",
        granularity="per game / per player",
        automated_access="not permitted without permission",
        known_limits="REJECTED for automated ingest: no API and scraping conflicts with site "
                     "terms. xG would also be a MODEL OUTPUT, not source data.",
        notes="Recorded so the absence is a decision, not an oversight. Revisit only with an "
              "explicit data agreement.",
        status="rejected",
    ),
    SourceSpec(
        source_id="moneypuck.downloads",
        name="MoneyPuck published data downloads",
        url="https://moneypuck.com/data.htm",
        data_type="season/game/player summary CSVs and per-shot files (shots_{year}.zip) with "
                  "MoneyPuck's xGoal model output, 2007-08 onward",
        nhl_relevance="Shot-level xG history that could add a shot-quality feature family; a "
                      "third-party MODEL OUTPUT, so it must be labelled as such",
        historical_depth="2007-08 through the current season (per data.htm)",
        live_available=False,
        update_frequency="Daily during the season",
        api_available=False,
        auth_required="none",
        cost="free",
        genuinely_free="yes (non-commercial, credit required)",
        usage_limits="Only the downloads listed on data.htm; MoneyPuck asks that other pages "
                     "(predictions, betting pages) not be scraped",
        licensing="Free for non-commercial use with credit to MoneyPuck.com (stated on data.htm)",
        provenance="Third party (MoneyPuck)",
        reliability="medium",
        accuracy="model-derived xG; raw shot events mirror NHL play-by-play",
        granularity="per shot / per game / per player",
        automated_access="yes, for the listed files only",
        known_limits="Not ingested yet: files are large (per-season zips) and every derived feature "
                     "would be MODEL OUTPUT from a model this project cannot audit. Kept as a "
                     "candidate rather than rejected, because the published terms permit it.",
        notes="Terms read 2026-09-20 from data.htm; data dictionaries are linked from that page.",
        probe_urls=("https://moneypuck.com/data.htm",),
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
        probed = spec.source_id in verified_probe
        if probed:
            _http_status, verdict, _evidence = verified_probe[spec.source_id]
            if verdict == "reachable" and spec.status != "rejected":
                status = "verified"
        d["status"] = status
        d["last_verified_at"] = now if probed else None
        cols = [c for c in d if c != "probe_urls"]
        # the registry row must exist BEFORE the verification row: source_verification has a
        # foreign key onto it, and seeding the probe first raises IntegrityError.
        store.execute(
            f"""INSERT INTO source_registry({','.join(cols)})
                VALUES({','.join('?' * len(cols))})
                ON CONFLICT(source_id) DO UPDATE SET
                  {','.join(f'{c}=excluded.{c}' for c in cols if c != 'source_id')}""",
            tuple(d[c] for c in cols),
        )
        if probed:
            http_status, verdict, evidence = verified_probe[spec.source_id]
            store.execute(
                """INSERT INTO source_verification
                   (source_id, checked_at, method, url_probed, http_status, ok, evidence, verdict, notes)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (spec.source_id, now, "http_get",
                 spec.probe_urls[0] if spec.probe_urls else spec.url,
                 http_status, 1 if verdict == "reachable" else 0, evidence, verdict,
                 "automated probe"),
            )
        n += 1
    store.commit()
    return n


def probe_urls_for(spec: SourceSpec) -> list[str]:
    return list(spec.probe_urls)
