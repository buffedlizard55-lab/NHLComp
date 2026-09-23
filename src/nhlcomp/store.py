"""Persistent, normalized, auditable storage layer (SQLite, stdlib only).

Design rules enforced here:

* Every externally retrieved payload is recorded in ``raw_response`` with the URL,
  the retrieval timestamp and a SHA-256 digest, so any derived value can be traced
  back to the exact bytes it came from.
* Historical bets are append-only.  ``record_bet`` refuses to overwrite an existing
  ``bet_id``; corrections go through ``amend_bet`` which writes an audit row.
* Every row that is not SOURCE DATA is labelled with a ``provenance`` value so the
  UI can distinguish SOURCE / DERIVED / MODEL / ASSUMPTION / UNVERIFIED.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 8

#: Published feeds disagree on club abbreviation style.  This maps the *short* forms other
#: sources use onto the three-letter form the NHL API returns, so a row from ESPN can be
#: attached to the same franchise the rest of the ledger uses.  It is looked up only when the
#: abbreviation is not already a team in the ``teams`` table, so it can never shadow a real
#: club code (``SJ``, ``TB``, ``NJ`` and ``LA`` are not NHL codes).
#: Only pairs that were read off the two feeds for the *same* club are listed.  Nothing here
#: is inferred from a naming convention: a plausible-looking guess such as ``PHO`` -> Utah
#: would silently attach a historical Phoenix Coyotes row to the wrong modern franchise, and
#: an unmapped row that is reported is worth more than a mapped row that is wrong.
ABBREV_ALIASES: dict[str, str] = {
    "SJ": "SJS",   # ESPN injury feed / NHL api-web
    "TB": "TBL",
    "NJ": "NJD",
    "LA": "LAK",
}

# Columns added after their table already shipped.  Applied by Store._migrate so a
# committed ledger.db from an earlier version gains them without a destructive rebuild.
#: Indexes that reference a column added by :data:`ADDITIVE_COLUMNS`.  Applied by
#: ``_migrate`` after the ALTERs, never inside SCHEMA (see the comment there).
POST_MIGRATION_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_abbrev, position)",
    "CREATE INDEX IF NOT EXISTS idx_goalie_starts_game ON goalie_starts(game_id, snapshot_ts)",
    "CREATE INDEX IF NOT EXISTS idx_goalie_starts_snapshot ON goalie_starts(snapshot_ts)",
)

ADDITIVE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("injuries", "position", "TEXT"),
    ("injuries", "long_comment", "TEXT"),
    # schema v5: historical-tier Kalshi contracts carry more identifying detail
    ("market_settlements", "series_ticker", "TEXT"),
    ("market_settlements", "tier", "TEXT"),
    ("market_settlements", "floor_strike", "REAL"),
    ("market_settlements", "close_time", "TEXT"),
    ("market_settlements", "title", "TEXT"),
    ("market_settlements", "occurrence_datetime", "TEXT"),
    ("market_settlements", "team_abbrev", "TEXT"),
    ("market_settlements", "market_type", "TEXT"),
    ("market_settlements", "candles_state", "TEXT"),
    ("market_settlements", "strike_type", "TEXT"),   # kalshi: greater | less (Over | Under)
    # totals / puck-line quotes need the line the contract was written at
    ("market_quotes", "strike", "REAL"),
    ("market_quotes", "strike_type", "TEXT"),
    ("bets", "close_price_ts", "TEXT"),
    ("bets", "price_point", "TEXT"),
    ("bets", "fee", "REAL"),                     # exchange taker fee at the fill (dollars)
    ("bets", "strike", "REAL"),                  # totals line the wager was placed at
    ("bets", "price_basis", "TEXT"),             # exchange | derived (how ask was obtained)
    ("backtests", "avg_edge", "REAL"),
    ("backtests", "hit_rate", "REAL"),
    ("backtests", "base_rate", "REAL"),
    ("backtests", "n_games", "INTEGER"),
    ("backtests", "price_basis", "TEXT"),
    # schema v6: puck-line / overtime markets and the second prediction-market source
    ("market_quotes", "team_abbrev", "TEXT"),      # team named by the contract (kalshi suffix)
    ("bets", "exchange_side", "TEXT"),             # YES | NO -- which side of the book was bought
    ("bets", "depth_basis", "TEXT"),               # published_offer_size | traded_volume | declared_cap
    ("market_settlements", "rung", "TEXT"),        # the ticker suffix, kept verbatim, never parsed as a line
    ("kalshi_series", "open_contracts", "INTEGER"),   # from the last listing poll
    # schema v7: which phase of a season a rule is allowed to trade, and which phase a
    # wager was actually placed in (1 = preseason, 2 = regular season, 3 = playoffs)
    ("strategies", "game_types", "TEXT NOT NULL DEFAULT '[2, 3]'"),
    ("bets", "game_type", "INTEGER"),
    ("kalshi_series", "last_listed_check", "TEXT"),
    ("kalshi_series", "listing_note", "TEXT"),        # incl. verified-negative observations
    # schema v8: a verified PRE-GAME starter feed (ESPN `probables`) and the roster that
    # gives its names an NHL player_id.  Every column is additive; no historical row is
    # rewritten, and a row written before the column existed keeps NULL -- which reads as
    # "the source did not publish this", never as "no".
    ("goalie_starts", "status_type", "TEXT"),         # the feed's own word: expected|confirmed
    ("goalie_starts", "status_name", "TEXT"),         # the feed's own label: "Expected"
    ("goalie_starts", "source_player_id", "INTEGER"), # ESPN's id, which is not an NHL id
    ("goalie_starts", "snapshot_ts", "TEXT"),         # when this project read it
    ("goalie_starts", "venue", "TEXT"),               # part of the join key
    ("goalie_starts", "source_event_id", "TEXT"),
    ("goalie_starts", "match_basis", "TEXT"),
    ("goalie_starts", "side", "TEXT"),
    ("goalie_starts", "actual_goalie_name", "TEXT"),  # post-game cross-check, DERIVED
    ("goalie_starts", "conversion_checked_at", "TEXT"),
    ("players", "team_abbrev", "TEXT"),
    ("players", "season", "INTEGER"),
    ("players", "sweater", "INTEGER"),
    ("players", "roster_group", "TEXT"),
    ("players", "source_id", "TEXT"),
    ("players", "retrieved_at", "TEXT"),
    ("injuries", "player_id", "INTEGER"),
    ("injuries", "player_match_basis", "TEXT"),
    ("games", "split_squad_home", "INTEGER"),
    ("games", "split_squad_away", "INTEGER"),
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------- sources
CREATE TABLE IF NOT EXISTS source_registry (
    source_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    url             TEXT NOT NULL,
    data_type       TEXT NOT NULL,
    nhl_relevance   TEXT NOT NULL,
    historical_depth TEXT,
    live_available  INTEGER NOT NULL DEFAULT 0,
    update_frequency TEXT,
    api_available   INTEGER NOT NULL DEFAULT 0,
    auth_required   TEXT NOT NULL DEFAULT 'none',
    cost            TEXT NOT NULL DEFAULT 'free',
    genuinely_free  TEXT NOT NULL DEFAULT 'unknown',   -- yes | no | freemium | unknown
    usage_limits    TEXT,
    licensing       TEXT,
    provenance      TEXT,
    reliability     TEXT,
    accuracy        TEXT,
    granularity     TEXT,
    automated_access TEXT,
    known_limits    TEXT,
    first_seen_at   TEXT NOT NULL,
    last_verified_at TEXT,
    status          TEXT NOT NULL DEFAULT 'candidate',  -- candidate|verified|flagged|rejected
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS source_verification (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL REFERENCES source_registry(source_id),
    checked_at    TEXT NOT NULL,
    method        TEXT NOT NULL,          -- http_get | http_head | manual | docs
    url_probed    TEXT NOT NULL,
    http_status   INTEGER,
    ok            INTEGER NOT NULL,
    evidence      TEXT,                   -- what was actually observed
    verdict       TEXT NOT NULL,          -- reachable | unreachable | blocked | partial
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS raw_response (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT NOT NULL,
    source_id     TEXT,
    retrieved_at  TEXT NOT NULL,
    status_code   INTEGER,
    byte_size     INTEGER,
    sha256        TEXT NOT NULL,
    available_at  TEXT,                   -- when the underlying data became public
    notes         TEXT,
    UNIQUE (url, sha256)
);
CREATE INDEX IF NOT EXISTS idx_raw_url ON raw_response(url);

-- ---------------------------------------------------------------- reference
CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY,
    -- NOT unique: the records API returns one row per franchise, and a triCode repeats
    -- across renames (Utah Hockey Club and Utah Mammoth are both UTA; the 1917 and modern
    -- Ottawa Senators are both SEN). Disambiguation is by team_id + full_name + active.
    abbrev      TEXT NOT NULL,
    full_name   TEXT NOT NULL,
    franchise_id INTEGER,
    conference  TEXT,
    division    TEXT,
    timezone    TEXT,
    venue       TEXT,
    city        TEXT,
    lat         REAL,
    lon         REAL,
    active      INTEGER NOT NULL DEFAULT 1,
    provenance  TEXT NOT NULL DEFAULT 'SOURCE'
);

CREATE INDEX IF NOT EXISTS idx_teams_abbrev ON teams(abbrev);

-- Roster snapshot.  One row per player, carrying the season of the most recent roster
-- that named him: this table resolves identities (an injury name -> a player_id, an ESPN
-- probable starter -> an NHL goalie), it is not a roster history.  `roster_group` keeps the
-- feed's own grouping (forwards / defensemen / goalies) instead of inferring it from the
-- position letter, so a mismatch between the two is visible rather than hidden.
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    position    TEXT,
    shoots      TEXT,
    birth_date  TEXT,
    team_id     INTEGER,
    team_abbrev TEXT,
    season      INTEGER,
    sweater     INTEGER,
    roster_group TEXT,
    source_id   TEXT,
    retrieved_at TEXT,
    provenance  TEXT NOT NULL DEFAULT 'SOURCE'
);

-- ---------------------------------------------------------------- games
CREATE TABLE IF NOT EXISTS games (
    game_id       INTEGER PRIMARY KEY,
    season        INTEGER NOT NULL,
    game_type     INTEGER NOT NULL,        -- 1 pre, 2 reg, 3 post
    game_date     TEXT NOT NULL,           -- league-scheduled local date
    start_time_utc TEXT NOT NULL,
    home_id       INTEGER NOT NULL REFERENCES teams(team_id),
    away_id       INTEGER NOT NULL REFERENCES teams(team_id),
    venue         TEXT,
    venue_tz      TEXT,
    utc_offset    TEXT,
    eastern_offset TEXT,
    neutral_site  INTEGER NOT NULL DEFAULT 0,
    state         TEXT NOT NULL,           -- FUTURE|LIVE|FINAL|OFF|CANCELLED
    home_score    INTEGER,
    away_score    INTEGER,
    last_period_type TEXT,                 -- REG|OT|SO
    home_shots    INTEGER,
    away_shots    INTEGER,
    winner_id     INTEGER,
    source_id     TEXT,
    provenance    TEXT NOT NULL DEFAULT 'SOURCE',
    -- the schedule payload publishes these itself (verified 2026-09-22 on
    -- /v1/schedule/2026-09-23: the two OTT/TOR preseason games are both split-squad, the
    -- MIN at DAL and LAK at ANA games are not).  A split-squad game is not played by the
    -- club's NHL roster, which is a first-party reason to keep it out of a competition
    -- scored on NHL-roster performance.  NULL means the feed did not publish the field.
    split_squad_home INTEGER,
    split_squad_away INTEGER,
    UNIQUE (game_id)
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season, game_type);

-- team-level view of a game; DERIVED from games
CREATE TABLE IF NOT EXISTS team_game (
    game_id     INTEGER NOT NULL REFERENCES games(game_id),
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    opp_id      INTEGER NOT NULL,
    is_home     INTEGER NOT NULL,
    gf          INTEGER,
    ga          INTEGER,
    result      TEXT,                      -- W|L|OTL|T|NULL
    decided_in  TEXT,                      -- REG|OT|SO
    PRIMARY KEY (game_id, team_id)
);

CREATE TABLE IF NOT EXISTS standings_snapshot (
    as_of       TEXT NOT NULL,
    season      INTEGER NOT NULL,
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    gp          INTEGER, wins INTEGER, losses INTEGER, otl INTEGER,
    points      INTEGER, gf INTEGER, ga INTEGER,
    home_wins INTEGER, home_losses INTEGER, home_otl INTEGER, home_gf INTEGER, home_ga INTEGER,
    road_wins INTEGER, road_losses INTEGER, road_otl INTEGER, road_gf INTEGER, road_ga INTEGER,
    l10_wins INTEGER, l10_losses INTEGER, l10_otl INTEGER, l10_gf INTEGER, l10_ga INTEGER,
    streak_code TEXT, streak_count INTEGER,
    shootout_wins INTEGER, shootout_losses INTEGER,
    clinch      TEXT,
    source_id   TEXT,
    provenance  TEXT NOT NULL DEFAULT 'SOURCE',
    PRIMARY KEY (as_of, season, team_id)
);

CREATE TABLE IF NOT EXISTS venues (
    venue       TEXT PRIMARY KEY,
    city        TEXT,
    timezone    TEXT,
    utc_offset  TEXT,
    lat         REAL,
    lon         REAL,
    outdoor     INTEGER NOT NULL DEFAULT 0,
    provenance  TEXT NOT NULL DEFAULT 'DERIVED',
    notes       TEXT
);

-- injuries: SOURCE data from an editorial feed, used for cross-validation only
CREATE TABLE IF NOT EXISTS injuries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL,
    player_name   TEXT NOT NULL,
    team_abbrev   TEXT,
    status        TEXT,
    detail        TEXT,
    position      TEXT,                 -- 'G' marks a goalie; drives WAITING FOR GOALIE
    long_comment  TEXT,
    reported_at   TEXT,                 -- when the source says it was reported
    retrieved_at  TEXT NOT NULL,
    provenance    TEXT NOT NULL DEFAULT 'SOURCE',
    verified      INTEGER NOT NULL DEFAULT 0,
    -- resolved against the club's published roster by name.  DERIVED, never asserted:
    -- `player_match_basis` records how the row was matched and NULL means no roster named
    -- that player, which is a fact about the roster, not a licence to guess an id.
    player_id     INTEGER,
    player_match_basis TEXT,
    UNIQUE (source_id, player_name, reported_at)
);

-- Starting-goalie announcements.  NULL means "not published by any verified source", never
-- a guess.  A row records what the source SAID (`status_type`, `status_name` are the feed's
-- own words: ESPN publishes type='expected' for a probable starter) and when this project
-- read it (`snapshot_ts`).  Nothing downstream may use a row whose snapshot postdates the
-- decision it is being used for -- that is the point of storing the timestamp at all.
CREATE TABLE IF NOT EXISTS goalie_starts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       INTEGER,
    game_date     TEXT,
    team_id       INTEGER,
    team_abbrev   TEXT,
    goalie_name   TEXT,
    goalie_id     INTEGER,               -- NHL player_id when resolved by name, else NULL
    announced_at  TEXT,                  -- only when the source publishes a time
    source_id     TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    is_confirmed  INTEGER NOT NULL DEFAULT 0,
    provenance    TEXT NOT NULL DEFAULT 'SOURCE',
    status_type   TEXT,                  -- feed's own word: expected | confirmed | ...
    status_name   TEXT,                  -- feed's own label: "Expected"
    source_player_id INTEGER,            -- the feed's own player id (ESPN != NHL)
    snapshot_ts   TEXT,                  -- when this project read it (point-in-time guard)
    venue         TEXT,                  -- part of the join key: two clubs can meet twice a day
    source_event_id TEXT,                -- ESPN event id, kept verbatim
    match_basis   TEXT,                  -- how the NHL game_id was resolved
    side          TEXT,                  -- home | away
    -- the post-game cross-check, filled only after the NHL's own goalie log reports who
    -- actually started.  DERIVED: it is this project's comparison of two sources, and a
    -- NULL means "the game has not been played / the log has not been read yet".
    actual_goalie_name TEXT,
    conversion_checked_at TEXT,
    UNIQUE (game_id, team_id, goalie_name, source_id)
);

-- ---------------------------------------------------------------- markets
CREATE TABLE IF NOT EXISTS market_quotes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    provider     TEXT NOT NULL,            -- kalshi | sportsbook | n/a
    market_key   TEXT NOT NULL,            -- e.g. KXNHLGAME-26SEP23MINDAL
    contract     TEXT,                     -- e.g. ...-MIN
    game_id      INTEGER,
    game_date    TEXT,
    market_type  TEXT NOT NULL,            -- moneyline|total|puck_line|prop|futures
    selection    TEXT NOT NULL,            -- team abbrev or line
    side         TEXT NOT NULL,            -- YES|NO|OVER|UNDER|HOME|AWAY
    price        REAL,                     -- decimal odds for books, dollars for binary
    bid          REAL,
    ask          REAL,
    spread       REAL,
    bid_size     REAL,
    ask_size     REAL,
    volume       REAL,
    liquidity    REAL,
    last_price   REAL,
    ts_utc       TEXT NOT NULL,            -- quote timestamp
    retrieved_at TEXT NOT NULL,
    is_opening   INTEGER NOT NULL DEFAULT 0,
    is_closing   INTEGER NOT NULL DEFAULT 0,
    source_url   TEXT,
    raw_id       INTEGER,
    provenance   TEXT NOT NULL DEFAULT 'SOURCE',
    UNIQUE (provider, contract, side, ts_utc)
);
CREATE INDEX IF NOT EXISTS idx_mq_game ON market_quotes(game_id);
CREATE INDEX IF NOT EXISTS idx_mq_ts ON market_quotes(ts_utc);

-- Official per-goal rows from api-web.nhle.com/v1/score/{date} (SOURCE).  One row per goal
-- with the period it was scored in, which is what a period market settles on and what lets
-- the final score and lastPeriodType be cross-checked against a second NHL feed.
CREATE TABLE IF NOT EXISTS game_period_goals (
    game_id      INTEGER NOT NULL,
    period       INTEGER NOT NULL,
    period_type  TEXT,                    -- REG | OT | SO, as published per goal
    team_abbrev  TEXT NOT NULL,
    player_id    INTEGER NOT NULL DEFAULT 0,   -- 0 when the feed published no player
    player_name  TEXT,
    time_in_period TEXT,
    strength     TEXT,                    -- ev | pp | sh | en | ps
    goal_modifier TEXT,
    away_score   INTEGER,                 -- running score exactly as published
    home_score   INTEGER,
    source_id    TEXT NOT NULL,
    source_url   TEXT,
    retrieved_at TEXT NOT NULL,
    provenance   TEXT NOT NULL DEFAULT 'SOURCE',
    UNIQUE (game_id, period, time_in_period, team_abbrev, player_id)
);
CREATE INDEX IF NOT EXISTS idx_gpg_game ON game_period_goals(game_id);

-- Period scores DERIVED from game_period_goals, with the reconciliation against the official
-- final score recorded rather than assumed: a game decided in a shootout publishes no goal
-- row for the deciding shot, so the derived sum can legitimately differ from the final score
-- by the shootout goal.  When it does, that is stated here and never silently patched.
CREATE TABLE IF NOT EXISTS game_period_scores (
    game_id       INTEGER NOT NULL,
    period        INTEGER NOT NULL,
    period_type   TEXT,
    home_goals    INTEGER NOT NULL DEFAULT 0,
    away_goals    INTEGER NOT NULL DEFAULT 0,
    derived_from  TEXT NOT NULL DEFAULT 'nhl.score.goals',
    reconciled    INTEGER NOT NULL DEFAULT 0,   -- 1 when the period rows sum to the official final
    reconciliation_note TEXT,
    source_id     TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    provenance    TEXT NOT NULL DEFAULT 'DERIVED',
    PRIMARY KEY (game_id, period)
);

-- Polymarket (gamma-api.polymarket.com) NHL markets: a second, independent prediction-market
-- price.  Keyless and free (verified 2026-09-21).  Reference only: this project executes on
-- Kalshi, so nothing here funds a wager and Polymarket's own fee schedule is stored as
-- published rather than modelled into a fill.
CREATE TABLE IF NOT EXISTS polymarket_markets (
    id             TEXT PRIMARY KEY,
    event_id       TEXT,
    event_slug     TEXT,
    event_title    TEXT,
    slug           TEXT,
    question       TEXT,
    condition_id   TEXT,
    group_item_title TEXT,
    outcomes       TEXT,                  -- JSON array as published (a string, not a list)
    outcome_prices TEXT,                  -- JSON array as published
    best_bid       REAL,
    best_ask       REAL,
    spread         REAL,
    last_trade_price REAL,
    liquidity      REAL,
    volume         REAL,
    volume_24hr    REAL,
    tick_size      REAL,
    min_size       REAL,
    fee_type       TEXT,
    fee_rate       REAL,
    fee_exponent   REAL,
    fee_taker_only INTEGER,
    fee_rebate_rate REAL,
    start_date     TEXT,
    end_date       TEXT,
    game_start_time TEXT,
    active         INTEGER,
    closed         INTEGER,
    accepting_orders INTEGER,
    enable_order_book INTEGER,
    game_id        INTEGER,               -- matched NHL game, NULL when it is not a game market
    market_family  TEXT,                  -- DERIVED: futures | game | other
    match_basis    TEXT,                  -- how game_id was matched, or why it was not
    retrieved_at   TEXT NOT NULL,
    source_url     TEXT,
    provenance     TEXT NOT NULL DEFAULT 'SOURCE'
);
CREATE INDEX IF NOT EXISTS idx_pm_game ON polymarket_markets(game_id);
CREATE INDEX IF NOT EXISTS idx_pm_family ON polymarket_markets(market_family);

-- Settled exchange contracts: real historical price + real outcome, used for BACKTESTS.
CREATE TABLE IF NOT EXISTS market_settlements (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    provider      TEXT NOT NULL,
    event_ticker  TEXT NOT NULL,
    contract      TEXT NOT NULL,
    game_id       INTEGER,
    game_date     TEXT,
    selection     TEXT,
    side          TEXT NOT NULL,
    result        TEXT,                    -- yes | no | '' (unsettled)
    settle_price  REAL,
    price_before  REAL,
    bid_before    REAL,
    ask_before    REAL,
    volume        REAL,
    open_interest REAL,
    open_time     TEXT,
    settlement_ts TEXT,
    retrieved_at  TEXT NOT NULL,
    source_url    TEXT,
    provenance    TEXT NOT NULL DEFAULT 'SOURCE',
    UNIQUE (provider, contract, side)
);
CREATE INDEX IF NOT EXISTS idx_ms_game ON market_settlements(game_id);

-- ---------------------------------------------------------------- strategies
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id    TEXT NOT NULL,          -- NHL_GOALIE_EDGE
    version        INTEGER NOT NULL,
    username       TEXT NOT NULL,
    name           TEXT NOT NULL,
    category       TEXT NOT NULL,
    hypothesis     TEXT NOT NULL,
    data_used      TEXT NOT NULL,
    entry_rule     TEXT NOT NULL,
    price_rule     TEXT NOT NULL,
    settlement_rule TEXT NOT NULL,
    markets        TEXT NOT NULL,
    params_json    TEXT NOT NULL DEFAULT '{}',
    origin         TEXT NOT NULL DEFAULT 'self_generated',  -- self_generated|public_source|user
    origin_ref     TEXT,
    parent_version INTEGER,
    change_note    TEXT,
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'candidate',  -- candidate|active|paused|retired|rejected
    starting_bankroll REAL NOT NULL,
    bankroll       REAL NOT NULL,
    test_mode      TEXT NOT NULL DEFAULT 'FORWARD TEST',
    -- the NHL season phases this rule may trade: 2 = regular season, 3 = playoffs.
    -- Preseason (1) is deliberately excluded by default: a rule fitted on regular-season
    -- form has no verified claim on a September roster, and WC-1 below records why.
    game_types     TEXT NOT NULL DEFAULT '[2, 3]',
    leakage_checked INTEGER NOT NULL DEFAULT 0,
    provenance     TEXT NOT NULL DEFAULT 'MODEL',
    PRIMARY KEY (strategy_id, version)
);

CREATE TABLE IF NOT EXISTS strategy_lifecycle (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    ts_utc       TEXT NOT NULL,
    from_status  TEXT,
    to_status    TEXT NOT NULL,
    reason       TEXT,
    evidence     TEXT
);

-- ---------------------------------------------------------------- bets
CREATE TABLE IF NOT EXISTS bets (
    bet_id        TEXT PRIMARY KEY,
    strategy_id   TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    username      TEXT NOT NULL,
    test_mode     TEXT NOT NULL,           -- BACKTEST | FORWARD TEST
    season        INTEGER,
    game_id       INTEGER,
    game_date     TEXT,
    matchup       TEXT,
    market        TEXT NOT NULL,
    selection     TEXT NOT NULL,
    bet_type      TEXT NOT NULL,
    provider      TEXT NOT NULL,
    odds_format   TEXT NOT NULL,           -- decimal | american | binary
    price         REAL NOT NULL,           -- decimal odds (books) or dollars (binary)
    implied_prob  REAL NOT NULL,
    model_prob    REAL,
    edge          REAL,
    fair_price    REAL,
    decision_ts   TEXT NOT NULL,
    bet_ts        TEXT NOT NULL,
    stake         REAL NOT NULL,
    liquidity     REAL,
    filled_size   REAL,
    slippage      REAL,
    entry_price   REAL,
    close_price   REAL,
    clv           REAL,
    result        TEXT,                    -- WIN|LOSS|PUSH|OPEN|VOID
    settle_ts     TEXT,
    pnl           REAL,
    roi           REAL,
    source_url    TEXT,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    notes         TEXT,
    features_json TEXT,
    created_at    TEXT NOT NULL,
    amended       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bets_strategy ON bets(strategy_id, strategy_version);
CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(bet_ts);
CREATE INDEX IF NOT EXISTS idx_bets_mode ON bets(test_mode);

CREATE TABLE IF NOT EXISTS bet_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    bet_id      TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,
    action      TEXT NOT NULL,             -- INSERT | AMEND | SETTLE
    before_json TEXT,
    after_json  TEXT,
    reason      TEXT
);

-- ---------------------------------------------------------------- upcoming
CREATE TABLE IF NOT EXISTS upcoming_bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id   TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    username      TEXT NOT NULL,
    game_id       INTEGER,
    game_date     TEXT,
    matchup       TEXT,
    market        TEXT NOT NULL,
    selection     TEXT NOT NULL,
    provider      TEXT NOT NULL,
    current_price REAL,
    required_price REAL,
    model_prob    REAL,
    fair_price    REAL,
    edge          REAL,
    stake         REAL,
    decision_ts   TEXT NOT NULL,
    status        TEXT NOT NULL,           -- see STATUS vocabulary below
    blocking_reason TEXT,
    supporting_data TEXT,
    source_url    TEXT,
    UNIQUE (strategy_id, strategy_version, game_id, market, selection)
);
CREATE INDEX IF NOT EXISTS idx_up_status ON upcoming_bets(status);

-- ---------------------------------------------------------------- tests
CREATE TABLE IF NOT EXISTS backtests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    label        TEXT NOT NULL,
    train_from   TEXT, train_to TEXT,
    valid_from   TEXT, valid_to TEXT,
    test_from    TEXT, test_to TEXT,
    n_bets       INTEGER NOT NULL,
    n_wins       INTEGER NOT NULL,
    n_losses     INTEGER NOT NULL,
    n_push       INTEGER NOT NULL,
    staked       REAL, pnl REAL, roi REAL,
    max_drawdown REAL, sharpe REAL,
    avg_price    REAL, clv REAL,
    data_sufficient INTEGER NOT NULL DEFAULT 1,
    caveat       TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE (strategy_id, version, label)
);

CREATE TABLE IF NOT EXISTS forward_tests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    label        TEXT NOT NULL,
    window_from  TEXT, window_to TEXT,
    n_bets       INTEGER NOT NULL,
    staked       REAL, pnl REAL, roi REAL,
    note         TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE (strategy_id, version, label)
);

-- ---------------------------------------------------------------- research
CREATE TABLE IF NOT EXISTS hypotheses (
    hyp_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    question    TEXT NOT NULL,
    rationale   TEXT NOT NULL,
    required_data TEXT NOT NULL,
    data_available INTEGER NOT NULL,
    testable    INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'proposed',
    outcome     TEXT,
    origin      TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    exp_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    hypothesis  TEXT,
    strategy_id TEXT,
    version     INTEGER,
    kind        TEXT NOT NULL,             -- feature_scan | ablation | ab_test | replication
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    conclusion  TEXT,
    verdict     TEXT                       -- edge | no_edge | inconclusive | rejected
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    evidence    TEXT,
    confidence  TEXT NOT NULL DEFAULT 'low',   -- low|medium|high
    kind        TEXT NOT NULL DEFAULT 'observation'
);

-- ---------------------------------------------------------------- integrity
CREATE TABLE IF NOT EXISTS irregularities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'info',   -- info|warn|error|critical
    entity_type TEXT NOT NULL DEFAULT '',
    entity_id   TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL,
    sources     TEXT,                            -- the disagreeing sources
    auto_corrected INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'open',    -- open|resolved|wont_fix
    resolution  TEXT,
    resolved_at TEXT,
    UNIQUE (kind, entity_type, entity_id, detail)
);
CREATE INDEX IF NOT EXISTS idx_irr_status ON irregularities(status);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc    TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    entity    TEXT,
    detail    TEXT
);

-- ---------------------------------------------------------------- schema v5
-- Named price points derived from Kalshi candlesticks (see nhlcomp.market).  One row per
-- (contract, point); the candle timestamp makes the timing of every price auditable.
CREATE TABLE IF NOT EXISTS market_price_points (
    contract        TEXT NOT NULL,
    point           TEXT NOT NULL,            -- open|t24h|t6h|t1h|close|ig60|ig120|final|latest
    end_period_ts   INTEGER NOT NULL,         -- unix seconds, end of the candle period
    game_id         INTEGER,
    team_abbrev     TEXT,                     -- NHL triCode the YES side refers to (moneyline)
    series_ticker   TEXT,
    market_type     TEXT,
    bid             REAL,
    ask             REAL,
    last            REAL,
    mean            REAL,
    volume          REAL,
    open_interest   REAL,
    period_interval INTEGER NOT NULL DEFAULT 60,
    tier            TEXT NOT NULL,            -- historical|live
    retrieved_at    TEXT NOT NULL,
    source_url      TEXT,
    provenance      TEXT NOT NULL DEFAULT 'DERIVED',
    PRIMARY KEY (contract, point)
);
CREATE INDEX IF NOT EXISTS idx_mpp_game ON market_price_points(game_id);

-- Kalshi series discovered through /series?category=Sports&tags=Hockey.
CREATE TABLE IF NOT EXISTS kalshi_series (
    ticker             TEXT PRIMARY KEY,
    title              TEXT,
    category           TEXT,
    tags               TEXT,
    fee_type           TEXT,
    frequency          TEXT,
    settlement_sources TEXT,
    contract_terms_url TEXT,
    first_seen         TEXT NOT NULL,
    last_seen          TEXT NOT NULL,
    n_settled_events   INTEGER,
    earliest_event     TEXT,
    latest_event       TEXT,
    history_cursor     TEXT,                  -- resume point for /historical/markets paging
    history_complete   INTEGER NOT NULL DEFAULT 0
);

-- Per-team per-game lines from api.nhle.com/stats/rest team/summary?isGame=true.
CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id        INTEGER NOT NULL,
    team_id        INTEGER NOT NULL,
    game_date      TEXT,
    home_road      TEXT,
    opponent_abbrev TEXT,
    team_name      TEXT,
    gf REAL, ga REAL, sf REAL, sa REAL,
    pp_pct REAL, pk_pct REAL, pp_net_pct REAL, pk_net_pct REAL, fo_pct REAL,
    wins REAL, losses REAL, ot_losses REAL, points REAL, reg_wins REAL, so_wins REAL,
    season         INTEGER,
    game_type      INTEGER,
    source_id      TEXT NOT NULL,
    retrieved_at   TEXT NOT NULL,
    provenance     TEXT NOT NULL DEFAULT 'SOURCE',
    PRIMARY KEY (game_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_tgs_team_date ON team_game_stats(team_id, game_date);

-- Per-goalie per-game lines from api.nhle.com/stats/rest goalie/summary?isGame=true.
CREATE TABLE IF NOT EXISTS goalie_game_stats (
    game_id        INTEGER NOT NULL,
    player_id      INTEGER NOT NULL,
    goalie_name    TEXT,
    team_abbrev    TEXT,
    opponent_abbrev TEXT,
    home_road      TEXT,
    game_date      TEXT,
    started        INTEGER NOT NULL DEFAULT 0,
    saves REAL, shots_against REAL, goals_against REAL, save_pct REAL, toi_seconds REAL,
    decision       TEXT,
    season         INTEGER,
    game_type      INTEGER,
    source_id      TEXT NOT NULL,
    retrieved_at   TEXT NOT NULL,
    provenance     TEXT NOT NULL DEFAULT 'SOURCE',
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_ggs_team_date ON goalie_game_stats(team_abbrev, game_date);
CREATE INDEX IF NOT EXISTS idx_ggs_player_date ON goalie_game_stats(player_id, game_date);

-- Sportsbook prices published through the NHL's partner odds feed.  Snapshots only:
-- the feed has no history, so this table *is* the history from the day ingestion began.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL,
    partner         TEXT NOT NULL,
    market_desc     TEXT NOT NULL,            -- MONEY_LINE_2_WAY | PUCK_LINE | OVER_UNDER
    home_price      REAL,                     -- American odds exactly as published
    away_price      REAL,
    home_qualifier  TEXT,
    away_qualifier  TEXT,
    home_abbrev     TEXT,
    away_abbrev     TEXT,
    start_time_utc  TEXT,
    source_updated_utc TEXT,
    retrieved_at    TEXT NOT NULL,
    source_url      TEXT,
    provenance      TEXT NOT NULL DEFAULT 'SOURCE',
    UNIQUE (game_id, partner, market_desc, source_updated_utc)
);
CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_id);

-- NHL EDGE team aggregates.  Season-to-date snapshots (no per-game history), hence
-- usable as FORWARD-TEST inputs only.
CREATE TABLE IF NOT EXISTS edge_team_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id       INTEGER NOT NULL,
    season        INTEGER NOT NULL,
    game_type     INTEGER NOT NULL,
    retrieved_at  TEXT NOT NULL,
    games_played  INTEGER,
    avg_shot_speed REAL,
    shot_attempts_90_plus REAL,
    bursts_over_22 REAL,
    bursts_20_22  REAL,
    max_skating_speed REAL,
    distance_last10_avg REAL,
    payload_json  TEXT NOT NULL,
    source_url    TEXT,
    provenance    TEXT NOT NULL DEFAULT 'SOURCE',
    UNIQUE (team_id, season, game_type, retrieved_at)
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_name(name: str) -> str:
    """Case-, diacritic- and punctuation-insensitive form of a published player name.

    Used only as a *second* attempt at identity resolution, after an exact match on the
    roster's own ``full_name`` failed, and every match made this way is recorded as
    DERIVED with its basis so a reader can see the name was not published identically.
    """
    import unicodedata
    stripped = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    kept = "".join(ch for ch in ascii_only.lower() if ch.isalnum() or ch == " ")
    return " ".join(kept.split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Store:
    """Thin SQLite wrapper.  Callers never write SQL that mutates bets directly."""

    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        #: ``(kind, entity_id)`` pairs raised by ``flag`` *during this process*.  A check that
        #: runs on every pass re-raises every condition it still finds, so anything absent
        #: from this set no longer reproduces (see ``flag`` and
        #: ``Verifier.reconcile_stale_flags``).
        self.flag_seen: set[tuple[str, str]] = set()
        self.conn.commit()
        cur = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            self.conn.commit()
        elif row["value"] != str(SCHEMA_VERSION):
            self.conn.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))
            self.conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for a ledger.db that predates the current schema.

        ``CREATE TABLE IF NOT EXISTS`` never alters a table that already exists, and
        data/ledger.db is committed to the repo and reused by CI.  Without this, adding a
        column to the schema would leave the live database without it and the next INSERT
        would fail on a column that plainly exists in the source.  Only ADD COLUMN is
        performed -- never a drop or a rewrite, so historical rows are never touched.
        """
        for table, column, decl in ADDITIVE_COLUMNS:
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if table not in {r[0] for r in self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}:
                continue
            if column not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        # Indexes over additive columns cannot live in SCHEMA: `executescript(SCHEMA)` runs
        # before this migration, so on a ledger that predates the column the CREATE INDEX
        # would fail on a column that is about to exist.  They are created here instead,
        # after every ALTER has been applied.
        for sql in POST_MIGRATION_INDEXES:
            self.conn.execute(sql)

    # ------------------------------------------------------------- helpers
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)).fetchall())

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, tuple(params)).fetchone()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator["Store"]:
        try:
            yield self
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def team_id_for(self, abbrev: str, full_name: str | None = None) -> int | None:
        """Resolve an abbreviation to a team_id, preferring the active franchise and an
        exact full-name match.  Returns None when the source cannot disambiguate.

        Published feeds do not agree on abbreviation style.  The NHL uses three letters
        (``LAK``, ``NJD``, ``SJS``, ``TBL``); ESPN's injury feed uses two (``LA``, ``NJ``,
        ``SJ``, ``TB``).  Before this alias table existed, every ESPN injury row for those
        four clubs failed to resolve, so four franchises silently had **no** injury context
        feeding the injury-gated strategies while the queue recorded 13 unmappable entries.
        The alias is a mapping between two published vocabularies, not a guess: each pair was
        read off the two feeds for the same club, and an alias is only applied when the exact
        abbreviation is not already a team in its own right.
        """
        rows = self.query("SELECT team_id, full_name, active FROM teams WHERE abbrev=?", (abbrev,))
        if not rows:
            canon = ABBREV_ALIASES.get((abbrev or "").upper())
            if canon:
                rows = self.query("SELECT team_id, full_name, active FROM teams WHERE abbrev=?",
                                  (canon,))
        if not rows:
            return None
        if len(rows) == 1:
            return int(rows[0]["team_id"])
        if full_name:
            exact = [r for r in rows if (r["full_name"] or "") == full_name]
            if len(exact) == 1:
                return int(exact[0]["team_id"])
        act = [r for r in rows if r["active"]]
        if len(act) == 1:
            return int(act[0]["team_id"])
        return None

    def backfill_bet_game_type(self) -> int:
        """Fill ``bets.game_type`` from the game each wager was placed on.

        DERIVED DATA, not a correction: the season phase of a bet is a property of its game,
        which is SOURCE DATA already in the ``games`` table, and this only copies it across
        for rows written before the column existed.  No price, stake, result or P&L is
        touched, every affected row is left auditable through ``audit_log``, and a row whose
        game is not in the ledger stays NULL rather than being assigned a phase it never had.
        """
        rows = self.query(
            """SELECT g.game_type t, COUNT(*) n
                 FROM bets b JOIN games g ON g.game_id=b.game_id
                WHERE b.game_type IS NULL GROUP BY 1""")
        if not rows:
            return 0
        cur = self.execute(
            """UPDATE bets SET game_type = (SELECT g.game_type FROM games g
                                           WHERE g.game_id = bets.game_id)
                WHERE game_type IS NULL
                  AND game_id IN (SELECT game_id FROM games)""")
        self.commit()
        self.audit("store", "BACKFILL_BET_GAME_TYPE", "bets",
                   json.dumps({"rows": cur.rowcount,
                               "by_phase": {str(r["t"]): int(r["n"]) for r in rows},
                               "derived_from": "games.game_type (SOURCE DATA)",
                               "columns_touched": ["game_type"],
                               "note": "price, stake, result and pnl untouched"}))
        return cur.rowcount

    def derive_winners(self) -> int:
        """Fill games.winner_id from the published final scores.

        winner_id is DERIVED DATA: it is not a field the NHL API returns, it is computed
        from home_score/away_score, which are.  A game is only marked when the score is
        unambiguous -- a tie after regulation means the source data we hold is incomplete,
        and we leave winner_id NULL rather than guess which side took the extra point.
        """
        cur = self.execute(
            """UPDATE games SET winner_id = CASE
                   WHEN home_score > away_score THEN home_id
                   WHEN away_score > home_score THEN away_id
                   ELSE NULL END
               WHERE winner_id IS NULL
                 AND home_score IS NOT NULL AND away_score IS NOT NULL
                 AND state IN ('FINAL','OFF')""")
        n = cur.rowcount
        ties = self.one(
            """SELECT COUNT(*) AS c FROM games
               WHERE winner_id IS NULL AND home_score IS NOT NULL
                 AND away_score IS NOT NULL AND home_score = away_score
                 AND state IN ('FINAL','OFF')""")
        self.commit()
        n_ties = int(ties["c"]) if ties else 0
        if n_ties:
            # A regulation tie with no winner is not possible in the NHL, so a tie here
            # means our score snapshot is stale or partial.  Recorded, never guessed.
            self.flag(
                "ambiguous_result",
                f"{n_ties} completed game(s) carry equal home/away scores; winner_id left "
                f"NULL rather than inferred -- the score snapshot needs re-ingest",
                severity="error", entity_type="dataset", entity_id="games")
        return int(n)

    def mark_current_teams(self, season: int) -> int:
        """Flag the franchises that actually played in ``season`` as active, using the
        standings as the authority rather than a hard-coded club list."""
        ids = {int(r["team_id"]) for r in self.query(
            "SELECT DISTINCT team_id FROM standings_snapshot WHERE season=?", (season,))}
        if not ids:
            return 0
        self.execute("UPDATE teams SET active=0")
        qmarks = ",".join("?" * len(ids))
        self.execute(f"UPDATE teams SET active=1 WHERE team_id IN ({qmarks})", tuple(ids))
        self.commit()
        return len(ids)

    def audit(self, actor: str, action: str, entity: str = "", detail: str = "") -> None:
        self.execute(
            "INSERT INTO audit_log(ts_utc, actor, action, entity, detail) VALUES(?,?,?,?,?)",
            (utcnow(), actor, action, entity, detail),
        )

    # ------------------------------------------------------------- raw cache
    def record_raw(self, url: str, body: str, *, source_id: str | None = None,
                   status_code: int | None = 200, available_at: str | None = None,
                   notes: str | None = None, retrieved_at: str | None = None) -> int:
        digest = sha256_text(body)
        row = self.one("SELECT id FROM raw_response WHERE url=? AND sha256=?", (url, digest))
        if row:
            return int(row["id"])
        cur = self.execute(
            """INSERT INTO raw_response(url, source_id, retrieved_at, status_code, byte_size,
                                        sha256, available_at, notes)
               VALUES(?,?,?,?,?,?,?,?)""",
            (url, source_id, retrieved_at or utcnow(), status_code, len(body.encode()), digest,
             available_at, notes),
        )
        self.commit()
        return int(cur.lastrowid)

    # ------------------------------------------------------------- bets
    def record_bet(self, bet: dict[str, Any]) -> bool:
        """Append-only.  Returns False if the bet_id already exists."""
        if self.one("SELECT 1 FROM bets WHERE bet_id=?", (bet["bet_id"],)):
            return False
        cols = list(bet.keys())
        self.execute(
            f"INSERT INTO bets({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
            tuple(bet[c] for c in cols),
        )
        self.execute(
            "INSERT INTO bet_audit(bet_id, ts_utc, action, after_json, reason) VALUES(?,?,?,?,?)",
            (bet["bet_id"], utcnow(), "INSERT", json.dumps(bet, default=str), "initial placement"),
        )
        return True

    def settle_bet(self, bet_id: str, *, result: str, pnl: float, close_price: float | None = None,
                   clv: float | None = None, settle_ts: str | None = None,
                   reason: str = "settlement") -> None:
        before = self.one("SELECT * FROM bets WHERE bet_id=?", (bet_id,))
        if before is None:
            raise KeyError(f"unknown bet_id {bet_id}")
        if before["result"] in ("WIN", "LOSS", "PUSH", "VOID"):
            # re-settlement is only allowed as an explicit correction, keep audit trail
            reason = f"re-settlement over {before['result']}: {reason}"
        stake = float(before["stake"] or 0)
        roi = (pnl / stake) if stake else None
        self.execute(
            """UPDATE bets SET result=?, pnl=?, roi=?, close_price=?, clv=?, settle_ts=?, amended=amended
               WHERE bet_id=?""",
            (result, pnl, roi, close_price, clv, settle_ts or utcnow(), bet_id),
        )
        after = self.one("SELECT * FROM bets WHERE bet_id=?", (bet_id,))
        self.execute(
            "INSERT INTO bet_audit(bet_id, ts_utc, action, before_json, after_json, reason) VALUES(?,?,?,?,?,?)",
            (bet_id, utcnow(), "SETTLE", json.dumps(dict(before), default=str),
             json.dumps(dict(after), default=str), reason),
        )
        self.commit()

    def amend_bet(self, bet_id: str, fields: dict[str, Any], reason: str) -> None:
        before = self.one("SELECT * FROM bets WHERE bet_id=?", (bet_id,))
        if before is None:
            raise KeyError(f"unknown bet_id {bet_id}")
        if not fields:
            return
        allowed = {"price", "stake", "result", "pnl", "close_price", "clv", "close_price_ts",
                   "price_point", "verification_status", "notes"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"fields not amendable without explicit schema change: {sorted(bad)}")
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE bets SET {sets}, amended=1 WHERE bet_id=?", (*fields.values(), bet_id))
        after = self.one("SELECT * FROM bets WHERE bet_id=?", (bet_id,))
        self.execute(
            "INSERT INTO bet_audit(bet_id, ts_utc, action, before_json, after_json, reason) VALUES(?,?,?,?,?,?)",
            (bet_id, utcnow(), "AMEND", json.dumps(dict(before), default=str),
             json.dumps(dict(after), default=str), reason),
        )
        self.commit()

    # ------------------------------------------------------------- irregularities
    def flag(self, kind: str, detail: str, *, severity: str = "warn", entity_type: str | None = None,
             entity_id: str | None = None, sources: str | None = None,
             auto_corrected: int = 0, actor: str = "flag") -> bool:
        """Record an irregularity.  Never auto-correct silently: auto_corrected must be
        accompanied by a resolution note in the caller's own audit entry.

        Every call also marks ``(kind, entity_id)`` as *still reproducing* in
        :attr:`flag_seen`.  A condition-based check re-raises its flags on every run, so a
        flag that is open but was not re-raised this run describes something that is no
        longer true -- otherwise ``INSERT OR IGNORE`` would keep a fixed problem in the queue
        forever, which is exactly what happened to the two ``duplicate_game`` rows left over
        from the split-squad matching bug.  See ``Verifier.reconcile_stale_flags``, which acts
        on this and closes such rows with a resolution note rather than deleting them.

        If that row was *closed* and the identical condition is seen again, the row is
        re-opened (audited as ``REOPEN_IRREGULARITY`` under ``actor``) instead of being
        swallowed: a recurring problem is not a resolved one.
        """
        et, eid = entity_type or "", entity_id or ""
        cur = self.execute(
            """INSERT OR IGNORE INTO irregularities
               (ts_utc, kind, severity, entity_type, entity_id, detail, sources, auto_corrected)
               VALUES(?,?,?,?,?,?,?,?)""",
            # empty string rather than NULL: SQLite treats NULLs as distinct in UNIQUE
            # constraints, which would let the same irregularity be recorded repeatedly.
            (utcnow(), kind, severity, et, eid, detail, sources, auto_corrected),
        )
        reopened = 0
        if cur.rowcount == 0:
            # The identical row is already there.  If it was closed -- reconciled as stale
            # when the condition went away, or retired as a false positive -- it has to come
            # back when the condition does, or the queue would quietly forget a problem it
            # had already reported once.  Only the status is touched: the original detail and
            # first-seen timestamp stay as the record of when it was first observed.
            prev = self.one(
                """SELECT id, status FROM irregularities
                    WHERE kind=? AND entity_type=? AND entity_id=? AND detail=?""",
                (kind, et, eid, detail))
            if prev and prev["status"] != "open":
                self.execute(
                    "UPDATE irregularities SET status='open', resolution=NULL, resolved_at=NULL"
                    " WHERE id=?", (prev["id"],))
                self.audit(actor, "REOPEN_IRREGULARITY", kind,
                           json.dumps({"id": prev["id"], "entity_id": eid, "was": prev["status"],
                                       "detail": detail[:300]}))
                reopened = 1
        self.commit()
        self.flag_seen.add((kind, eid))
        return cur.rowcount > 0 or bool(reopened)

    def resolve_irregularity(self, irr_id: int, resolution: str, status: str = "resolved") -> None:
        self.execute(
            "UPDATE irregularities SET status=?, resolution=?, resolved_at=? WHERE id=?",
            (status, resolution, utcnow(), irr_id),
        )
        self.commit()

    def resolve_irregularities_where(self, kind: str, detail_prefix: str, resolution: str, *,
                                     entity_id: str | None = None, actor: str = "resolve") -> int:
        """Close every OPEN irregularity of ``kind`` whose detail starts with ``detail_prefix``.

        The complement of :meth:`flag` for conditions that *recovered*: a fetch that timed
        out yesterday and succeeded today, a coverage gap the two ingest walks have since
        closed, a market that was unmatchable until the schedule caught up.  Nothing is
        deleted -- each row keeps its original detail and first-seen timestamp and gains a
        resolution note -- and one ``audit_log`` entry records how many rows were closed and
        why, so a reader of the queue history sees the recovery, not just its absence.

        Returns the number of rows closed (0 is normal and means nothing was open).
        """
        rows = self.query(
            """SELECT id FROM irregularities
                WHERE kind=? AND status='open' AND detail LIKE ?
                  AND (? = '' OR entity_id = ?)""",
            (kind, detail_prefix + "%", entity_id or "", entity_id or ""))
        for r in rows:
            self.resolve_irregularity(int(r["id"]), resolution)
        if rows:
            self.audit(actor, "RESOLVE_RECOVERED", kind,
                       json.dumps({"closed": len(rows), "detail_prefix": detail_prefix,
                                   "entity_id": entity_id, "resolution": resolution[:300]}))
            self.commit()
        return len(rows)


    # ------------------------------------------------------------- strategies
    def upsert_strategy(self, s: dict[str, Any]) -> None:
        s = dict(s)
        # a newly registered version starts with its full starting bankroll unless told otherwise
        if "bankroll" not in s or s["bankroll"] is None:
            s["bankroll"] = s.get("starting_bankroll", 1000.0)
        cols = list(s.keys())
        self.execute(
            f"""INSERT INTO strategies({','.join(cols)}) VALUES({','.join('?' * len(cols))})
                ON CONFLICT(strategy_id, version) DO UPDATE SET
                  {','.join(f"{c}=excluded.{c}" for c in cols if c not in ('strategy_id', 'version'))}""",
            tuple(s[c] for c in cols),
        )
        self.commit()

    def set_strategy_status(self, strategy_id: str, version: int, status: str,
                            reason: str = "", evidence: str = "") -> None:
        prev = self.one("SELECT status FROM strategies WHERE strategy_id=? AND version=?",
                        (strategy_id, version))
        self.execute("UPDATE strategies SET status=? WHERE strategy_id=? AND version=?",
                     (status, strategy_id, version))
        self.execute(
            """INSERT INTO strategy_lifecycle(strategy_id, version, ts_utc, from_status, to_status,
                                              reason, evidence) VALUES(?,?,?,?,?,?,?)""",
            (strategy_id, version, utcnow(), prev["status"] if prev else None, status, reason, evidence),
        )
        self.commit()

    def latest_versions(self) -> list[sqlite3.Row]:
        return self.query(
            """SELECT * FROM strategies s
               WHERE version = (SELECT MAX(version) FROM strategies s2
                                WHERE s2.strategy_id = s.strategy_id)
               ORDER BY strategy_id"""
        )

    def sync_bankroll(self, strategy_id: str, version: int) -> None:
        # BACKTEST rows never touch the live bankroll: the brief forbids merging backtest
        # and forward-test results, and letting recovered historical PnL fund forward
        # stakes would do exactly that.
        row = self.one(
            """SELECT COALESCE(SUM(pnl),0) AS realized
               FROM bets WHERE strategy_id=? AND strategy_version=? AND result IN
                 ('WIN','LOSS','PUSH','VOID') AND test_mode='FORWARD TEST'""",
            (strategy_id, version),
        )
        s = self.one("SELECT starting_bankroll FROM strategies WHERE strategy_id=? AND version=?",
                     (strategy_id, version))
        if s is None:
            return
        start = float(s["starting_bankroll"])
        realized = float(row["realized"] or 0)
        # open exposure is reserved from the bankroll so strategies cannot over-commit
        open_exposure = self.one(
            """SELECT COALESCE(SUM(stake),0) AS e FROM bets
               WHERE strategy_id=? AND strategy_version=? AND result='OPEN'
                 AND test_mode='FORWARD TEST'""",
            (strategy_id, version),
        )["e"] or 0.0
        self.execute("UPDATE strategies SET bankroll=? WHERE strategy_id=? AND version=?",
                     (start + realized - float(open_exposure), strategy_id, version))
        self.commit()

    # ------------------------------------------------------- rosters and starters
    def upsert_player(self, p: dict[str, Any]) -> None:
        """Store one roster row.  Keyed on the NHL ``player_id`` the feed publishes.

        This table exists to resolve identities (an injury name to a player_id, an ESPN
        probable starter to an NHL goalie), so the most recent roster that names a player
        wins and the season of that roster is kept with the row.  It is a snapshot, not a
        roster history, and is labelled as one wherever it is displayed.
        """
        self.execute(
            """INSERT INTO players(player_id, full_name, position, shoots, birth_date, team_id,
                                   team_abbrev, season, sweater, roster_group, source_id,
                                   retrieved_at, provenance)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(player_id) DO UPDATE SET
                 full_name=excluded.full_name, position=excluded.position,
                 shoots=excluded.shoots, birth_date=excluded.birth_date,
                 team_id=excluded.team_id, team_abbrev=excluded.team_abbrev,
                 season=excluded.season, sweater=excluded.sweater,
                 roster_group=excluded.roster_group, source_id=excluded.source_id,
                 retrieved_at=excluded.retrieved_at, provenance=excluded.provenance""",
            (p["player_id"], p["full_name"], p.get("position"), p.get("shoots"),
             p.get("birth_date"), p.get("team_id"), p.get("team_abbrev"), p.get("season"),
             p.get("sweater"), p.get("roster_group"), p.get("source_id") or "nhl.roster",
             p.get("retrieved_at") or utcnow(), p.get("provenance") or "SOURCE"))

    def resolve_player(self, name: str, team_abbrev: str | None = None,
                       position: str | None = None) -> tuple[int | None, str]:
        """``(player_id, match_basis)`` for a name the roster publishes, or ``(None, why)``.

        Matching is exact on the published ``full_name`` first.  Only if that fails is a
        case/punctuation-insensitive form tried, and the basis string says which happened,
        because a fuzzy identity match is DERIVED data and has to be readable as such.
        Two different players on the same roster sharing a normalised name is a conflict:
        it returns ``(None, ...)`` rather than picking one.
        """
        want = (name or "").strip()
        if not want:
            return None, "no name published"
        clauses = ["full_name=?"]
        params: list[Any] = [want]
        if team_abbrev:
            clauses.append("(team_abbrev=? OR team_abbrev IS NULL)")
            params.append(team_abbrev)
        if position:
            clauses.append("(position=? OR position IS NULL)")
            params.append(position)
        rows = self.query(f"SELECT player_id FROM players WHERE {' AND '.join(clauses)}", params)
        if len(rows) == 1:
            return int(rows[0]["player_id"]), f"exact full_name match on the published roster"
        if len(rows) > 1:
            return None, (f"{len(rows)} roster rows publish that exact name; not resolved "
                          f"rather than picked")
        norm = _normalize_name(want)
        sql = "SELECT player_id, full_name FROM players WHERE 1=1"
        params2: list[Any] = []
        if team_abbrev:
            sql += " AND team_abbrev=?"
            params2.append(team_abbrev)
        if position:
            # scope applies to the fallback pass too: a team's G must never be borrowed
            # for a D, even when the name matches after normalisation
            sql += " AND (position=? OR position IS NULL)"
            params2.append(position)
        rows = self.query(sql, params2)
        hits = [r for r in rows if _normalize_name(r["full_name"] or "") == norm]
        if len(hits) == 1:
            return int(hits[0]["player_id"]), (
                f"normalised match: roster publishes '{hits[0]['full_name']}' for '{want}' "
                f"(case/diacritic-insensitive) -- DERIVED, not an exact source match")
        if len(hits) > 1:
            return None, (f"{len(hits)} roster rows normalise to '{want}'; not resolved "
                          f"rather than picked")
        return None, "no roster row published that name"

    def record_goalie_start(self, row: dict[str, Any]) -> bool:
        """Append a starter/probable-starter observation.  True when a new row was written.

        The UNIQUE key is (game_id, team_id, goalie_name, source_id): a source that names a
        *different* goalie for the same game and team on a later run writes a second row, so
        the change of announcement is preserved as two observations instead of overwriting
        the first one.  Rows are never deleted here.
        """
        before = self.one("SELECT COUNT(*) c FROM goalie_starts")["c"]
        self.execute(
            """INSERT OR IGNORE INTO goalie_starts(
                 game_id, game_date, team_id, team_abbrev, goalie_name, goalie_id,
                 announced_at, source_id, retrieved_at, is_confirmed, provenance,
                 status_type, status_name, source_player_id, snapshot_ts, venue,
                 source_event_id, match_basis, side)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row.get("game_id"), row.get("game_date"), row.get("team_id"),
             row.get("team_abbrev"), row.get("goalie_name"), row.get("goalie_id"),
             row.get("announced_at"), row["source_id"], row["retrieved_at"],
             1 if row.get("is_confirmed") else 0, row.get("provenance") or "SOURCE",
             row.get("status_type"), row.get("status_name"), row.get("source_player_id"),
             row.get("snapshot_ts") or row["retrieved_at"], row.get("venue"),
             row.get("source_event_id"), row.get("match_basis"), row.get("side")))
        after = self.one("SELECT COUNT(*) c FROM goalie_starts")["c"]
        return after > before

    def starters_asof(self, decision_ts: str | None = None,
                      *, confirmed_only: bool = False) -> dict[int, dict[int, dict[str, Any]]]:
        """``game_id -> team_id -> starter`` from rows this project had read by ``decision_ts``.

        The timestamp filter is the point of the whole table: a probable starter read at
        21:00Z may not be used to justify a decision stamped 18:00Z.  When two rows exist
        for one (game, team) -- the announcement changed -- the most recent row that still
        precedes the decision wins, and both stay in the ledger.
        """
        sql = "SELECT * FROM goalie_starts WHERE goalie_name IS NOT NULL"
        params: list[Any] = []
        if decision_ts:
            sql += " AND COALESCE(snapshot_ts, retrieved_at) <= ?"
            params.append(decision_ts)
        if confirmed_only:
            sql += " AND is_confirmed=1"
        sql += " ORDER BY COALESCE(snapshot_ts, retrieved_at), id"
        out: dict[int, dict[int, dict[str, Any]]] = {}
        for r in self.query(sql, params):
            if r["game_id"] is None or r["team_id"] is None:
                continue
            out.setdefault(int(r["game_id"]), {})[int(r["team_id"])] = dict(r)
        return out
