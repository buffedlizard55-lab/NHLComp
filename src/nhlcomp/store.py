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

SCHEMA_VERSION = 5

# Columns added after their table already shipped.  Applied by Store._migrate so a
# committed ledger.db from an earlier version gains them without a destructive rebuild.
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
    ("bets", "close_price_ts", "TEXT"),
    ("bets", "price_point", "TEXT"),
    ("bets", "fee", "REAL"),                     # exchange taker fee at the fill (dollars)
    ("backtests", "avg_edge", "REAL"),
    ("backtests", "hit_rate", "REAL"),
    ("backtests", "base_rate", "REAL"),
    ("backtests", "n_games", "INTEGER"),
    ("backtests", "price_basis", "TEXT"),
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

CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    position    TEXT,
    shoots      TEXT,
    birth_date  TEXT,
    team_id     INTEGER,
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
    UNIQUE (source_id, player_name, reported_at)
);

-- starting-goalie announcements; NULL means "not yet announced", never a guess
CREATE TABLE IF NOT EXISTS goalie_starts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       INTEGER,
    game_date     TEXT,
    team_id       INTEGER,
    team_abbrev   TEXT,
    goalie_name   TEXT,
    goalie_id     INTEGER,
    announced_at  TEXT,
    source_id     TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    is_confirmed  INTEGER NOT NULL DEFAULT 0,
    provenance    TEXT NOT NULL DEFAULT 'SOURCE',
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
        exact full-name match.  Returns None when the source cannot disambiguate."""
        rows = self.query("SELECT team_id, full_name, active FROM teams WHERE abbrev=?", (abbrev,))
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
             auto_corrected: int = 0) -> bool:
        """Record an irregularity.  Never auto-correct silently: auto_corrected must be
        accompanied by a resolution note in the caller's own audit entry."""
        cur = self.execute(
            """INSERT OR IGNORE INTO irregularities
               (ts_utc, kind, severity, entity_type, entity_id, detail, sources, auto_corrected)
               VALUES(?,?,?,?,?,?,?,?)""",
            # empty string rather than NULL: SQLite treats NULLs as distinct in UNIQUE
            # constraints, which would let the same irregularity be recorded repeatedly.
            (utcnow(), kind, severity, entity_type or "", entity_id or "", detail, sources,
             auto_corrected),
        )
        self.commit()
        return cur.rowcount > 0

    def resolve_irregularity(self, irr_id: int, resolution: str, status: str = "resolved") -> None:
        self.execute(
            "UPDATE irregularities SET status=?, resolution=?, resolved_at=? WHERE id=?",
            (status, resolution, utcnow(), irr_id),
        )
        self.commit()

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
