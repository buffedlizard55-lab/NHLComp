"""The autonomous research log, as ledger rows.

Requirement 2 asks the system to research any legitimate source that can improve NHL
prediction, and requirement 16 asks the final report to separate what was discovered,
verified and rejected.  This module is where a research session's verified observations are
written into the ``findings`` table -- each with its evidence (URLs, HTTP results, dates) --
so the site's Research Lab page and the final report are generated from the ledger rather
than from prose that could drift from it.

Every entry states its verification date and method.  Nothing here is assumed from memory:
each item was fetched during the session that logged it.
"""

from __future__ import annotations

import json
from typing import Any

#: The session that produced these entries (kept out of the finding bodies so re-recording
#: on a later run stays accurate).
SESSION_DATE = "2026-09-22"

#: (finding_id, title, body, evidence_dict, confidence, kind)
RESEARCH_FINDINGS: tuple[dict[str, Any], ...] = (
    {
        "finding_id": "FIND_ODDS_SNAPSHOTS_NO_NHL",
        "title": "sports-odds-datasets (ParlayAPI) verified to contain no NHL data",
        "body": (
            "The 'open snapshots of sportsbook odds data' repository recommended by the "
            "awesome-sports-betting-data index was evaluated as a possible second (sportsbook) "
            "closing-line source to cross-check Kalshi closes. Its file list is Super Bowl LX "
            "closing lines (13 rows), one MLB day (2026-08-23, 1,151 rows) and a 50,000-row "
            "player-prop sample spanning 2026-02-08..2026-08-25 -- no hockey or NHL file of "
            "any kind. The underlying archive is a commercial freemium API (1,000 credits/"
            "month trial tier, which this project does not treat as permanently free). "
            "Registered as REJECTED; the registry entry records the evidence so the absence "
            "is a decision, not an oversight."),
        "evidence": {
            "url": "https://github.com/JacobiusMakes/sports-odds-datasets",
            "fetched": SESSION_DATE,
            "files_seen": ["superbowl_lx_closing_lines.csv", "mlb_2026-08-23_closing_lines.csv",
                           "prop_closing_lines_sample_50k.csv"],
            "license": "CC BY 4.0 (samples)",
        },
        "confidence": "high",
        "kind": "source_review",
    },
    {
        "finding_id": "FIND_KAGGLE_NHL_ODDS_REJECTED",
        "title": "Kaggle 'NHL Full Game & Betting Statistics' rejected on access and provenance",
        "body": (
            "A third-party Kaggle dataset claiming NHL game data 2004-2025 including betting "
            "odds was evaluated as a historical sportsbook price source. Rejected for two "
            "independent reasons. (1) Access: downloading requires a Kaggle account/API key; "
            "none is available to this project and none may be invented. (2) Provenance: the "
            "odds columns are said to come from ESPN JSON responses with no per-price "
            "timestamps, and this project's own verified probe (2026-09-21, 2025-11-15 slate) "
            "found ESPN's public scoreboard carries NO odds block on completed games -- so "
            "the dataset's odds cannot be reconciled with anything this project can verify. "
            "Under the never-invent-odds rule, an unverifiable third party cannot be promoted "
            "into a backtest price. Revisit only with an authenticated download path and a "
            "spot-check tying several of its prices to an independent source at a stated time."),
        "evidence": {
            "url": "https://www.kaggle.com/datasets/jonathanncoletti/nhl-historical-game-data",
            "fetched": SESSION_DATE,
            "cross_reference": "espn.nhl_scoreboard registry entry (no odds block on completed games)",
        },
        "confidence": "high",
        "kind": "source_review",
    },
    {
        "finding_id": "FIND_MONEYPUCK_OFFSEASON_STALE",
        "title": "MoneyPuck downloads verified live but dormant for the 2026 off-season",
        "body": (
            "The MoneyPuck data page was re-verified live: the published terms permit "
            "non-commercial use with credit, and only the listed downloads are offered "
            "(scraping other pages requires approval). The page stated 'last updated "
            "2026-06-15 Eastern Time' and the newest season file is 2025-2026 -- i.e. during "
            "the 2026 off-season there is no 2026-27 file and no update stream, so MoneyPuck "
            "cannot supply current-season point-in-time features until updates resume. It "
            "remains a candidate (not ingested): shot-level xG is a third-party MODEL OUTPUT "
            "and would be labelled as such, never as source data."),
        "evidence": {
            "url": "https://moneypuck.com/data.htm",
            "fetched": SESSION_DATE,
            "page_last_updated": "2026-06-15 (stated on page)",
            "terms": "free for non-commercial use with credit; scraping other pages requires approval",
        },
        "confidence": "high",
        "kind": "source_review",
    },
    {
        "finding_id": "FIND_NO_NHL_PROBABLE_GOALIE_ENDPOINT",
        "title": "NHL endpoints publish no pre-game starter; the block was lifted via ESPN instead",
        "body": (
            "PROBE (2026-09-22): api-web.nhle.com exposes no probable/announced-goalies "
            "endpoint -- gamecenter 2026010051 probable-goalies and the gamecenter landing "
            "page for a future game both returned HTTP 404, consistent with the documented "
            "API surface: the stats REST goalie log identifies the starter only AFTER a game "
            "is final. RESOLUTION (2026-09-22, same day): the block lifted through a "
            "DIFFERENT source -- ESPN's public scoreboard publishes a "
            "probableStartingGoalie per competitor of a scheduled game; see "
            "FIND_ESPN_PROBABLE_STARTERS. The NHL-side negative stands: within the NHL's "
            "own endpoints the starter remains post-game only, so the post-game log stays "
            "an ASSUMPTION-based identity source for backtests and the ESPN feed is the "
            "only pre-game name this project has."),
        "evidence": {
            "url": "https://api-web.nhle.com/v1/gamecenter/2026010051/probable-goalies",
            "fetched": SESSION_DATE,
            "result": "HTTP 404 Not Found",
            "also_probed": "https://api-web.nhle.com/v1/gamecentre/2026010034/landing -> 404 (future game)",
            "resolution": "FIND_ESPN_PROBABLE_STARTERS",
        },
        "confidence": "high",
        "kind": "source_review",
    },
    {
        "finding_id": "FIND_ESPN_PROBABLE_STARTERS",
        "title": "VERIFIED: ESPN scoreboard publishes probable starting goalies before puck drop",
        "body": (
            "The largest blocked input in this project -- a pre-game starting-goalie name -- "
            "was unblocked on 2026-09-22. ESPN's public NHL scoreboard "
            "(site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=YYYYMMDD, "
            "keyless) publishes, per competitor of a SCHEDULED game, a probables[] entry "
            "named probableStartingGoalie with the athlete's name, ESPN player id and a "
            "status object. Verified live on the 2026-09-23 slate: TOR at OTT (Canadian "
            "Tire Centre) carried Linus Ullmark (OTT) and Anthony Stolarz (TOR), both with "
            "status type 'expected'. Consequences, in order of caution: (1) the status "
            "observed is EXPECTED, never confirmed -- strict goalie-gated rules keep "
            "waiting; (2) no announcement timestamp is published, so the feed supports "
            "FORWARD tests only and can never price a historical decision; (3) ESPN event "
            "ids are not NHL game ids and the same two clubs can meet twice in an evening, "
            "so the join is date + home/away + venue and an ambiguous event is stored "
            "unmatched and flagged, never guessed; (4) names are resolved to NHL player_ids "
            "through the club rosters, and that resolution is recorded as DERIVED with its "
            "basis. Registered as espn.nhl_probables in the source registry; capture at "
            "data/captured/espn_scoreboard_probable_goalies_20260923.json. New probable-"
            "starter versions of the goalie strategies were seeded (GOALIE_EDGE v3, "
            "GOALIE_FATIGUE v2, GOALIE_NEWS v3); all are FORWARD TEST only."),
        "evidence": {
            "url": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=20260923",
            "fetched": SESSION_DATE,
            "capture": "data/captured/espn_scoreboard_probable_goalies_20260923.json",
            "status_observed": "type='expected' (both competitors)",
            "join_key": "UTC date + home/away abbreviations, venue as tie-breaker",
            "registry_id": "espn.nhl_probables",
        },
        "confidence": "high",
        "kind": "source_review",
    },
    {
        "finding_id": "FIND_SPLIT_SQUAD_FLAGS",
        "title": "DISCOVERED: the NHL schedule payload publishes split-squad flags",
        "body": (
            "Reading the /v1/schedule/{date} payload (not the scoreboard feed) for the "
            "2026-09-23 slate surfaced two fields this project had never stored: "
            "awaySplitSquad and homeSplitSquad. Both OTT/TOR preseason games that evening "
            "are split-squad (true/true) -- each club fields two squads that night -- while "
            "MIN at DAL and LAK at ANA are false. A split-squad game is not played by the "
            "club's NHL roster, which is a first-party scope fact rather than a heuristic: "
            "such games are now stored on the games row and the strategy scope gate (which "
            "already keeps preseason games out of the scored competition) has a per-game "
            "flag it can consult. Capture at "
            "data/captured/nhl_schedule_20260923_excerpt.json."),
        "evidence": {
            "url": "https://api-web.nhle.com/v1/schedule/2026-09-23",
            "fetched": SESSION_DATE,
            "capture": "data/captured/nhl_schedule_20260923_excerpt.json",
            "fields": ["homeSplitSquad", "awaySplitSquad"],
            "note": "the /v1/scoreboard feed does NOT publish these fields; only /v1/schedule does",
        },
        "confidence": "high",
        "kind": "source_review",
    },
    {
        "finding_id": "FIND_PLAYERS_TABLE_WAS_EMPTY",
        "title": "VERIFIED GAP CLOSED: club rosters now ingested into the players table",
        "body": (
            "The players table existed since schema v1 and was EMPTY in every committed "
            "ledger -- NhlApi.roster() was implemented and never called, so requirement 10's "
            "normalized player/roster storage existed only as a schema. The feed was "
            "re-verified live (keyless, free, per club per season; /v1/roster/DAL/20262027 "
            "and the /current alias), captured, and the ingest now stores one snapshot row "
            "per player with the roster's own grouping, sweater number and season. Two "
            "downstream joins this enables, both recorded with their match basis: injury "
            "rows (name + team abbrev only) resolve to player_ids, and ESPN probable-starter "
            "names resolve to NHL goalie ids. The table is a SNAPSHOT, not a roster "
            "history, and is labelled as one. Capture at "
            "data/captured/nhl_roster_DAL_20262027_excerpt.json."),
        "evidence": {
            "url": "https://api-web.nhle.com/v1/roster/DAL/20262027",
            "alias": "https://api-web.nhle.com/v1/roster/TOR/current serves the same shape",
            "fetched": SESSION_DATE,
            "capture": "data/captured/nhl_roster_DAL_20262027_excerpt.json",
            "caveats": ["sweaterNumber and shootsCatches are optional in the real payload",
                        "a September roster lists camp invitees; it is not an opening-night roster"],
        },
        "confidence": "high",
        "kind": "status",
    },
    {
        "finding_id": "FIND_DISCOVERY_FULL_SCAN_RECORDED",
        "title": "AUDIT: every discovery trigger now lands in the experiments ledger",
        "body": (
            "The discovery scan previously wrote ledger rows for two outcomes -- promoted "
            "candidates and loud rejections -- and let every other trigger (inconclusive "
            "verdicts, thin priced samples) leave no trace, so the search's footprint could "
            "not be reconstructed from the ledger. This pass records one experiments row "
            "per scanned trigger with its train/validation hit rates, priced result, "
            "one-sided p-value, Holm-adjusted threshold and verdict, keyed by a hash of the "
            "trigger so re-runs are idempotent and nothing is duplicated. The scan is now "
            "fully auditable end to end: hypotheses -> experiments -> strategies -> bets."),
        "evidence": {
            "table": "experiments",
            "kind": "feature_scan",
            "id_scheme": "EXP_SCAN_<sha1(trigger)>",
            "fetched": SESSION_DATE,
        },
        "confidence": "high",
        "kind": "audit",
    },
    {
        "finding_id": "FIND_QUEUE_RECOVERY_PASS",
        "title": "Fourth-session audit: the verification queue now closes recovered conditions",
        "body": (
            "A queue that only grows trains a reader to ignore it. This pass added "
            "evidence-carrying recovery for every raiser whose condition can later be "
            "observed to have gone away: transient stats-REST timeouts resolve when the "
            "identical fetch succeeds; no_markets resolves when the same query returns "
            "contracts; non-NHL exhibition opponents (e.g. EHC Red Bull Muenchen, team 7509, "
            "2024 Global Series) are identified from the schedule payload's own abbrev and "
            "supersede the unknown_team error; totals/strike coverage gaps resolve when the "
            "candle and feature walks meet; the overtime settlement rule resolves when "
            "settled shootout evidence exists; unmatched_market flags resolve when a later "
            "ingest matches the event; and evidence-carrying manual research resolutions are "
            "applied idempotently with an audit row. Nothing is deleted: resolution notes, "
            "original details and audit_log entries are kept. First application on the "
            "2026-09-22 ledger closed the KXNHLSPREAD-26JAN26LACBJ listing anomaly (the "
            "exchange now returns not_found for it and no NHL game exists for the ticker) "
            "and left 24 open irregularities, all standing conditions rather than stale "
            "history."),
        "evidence": {
            "resolved_example": "irregularity 319 (unmatched_market KXNHLSPREAD-26JAN26LACBJ)",
            "kalshi_probe": "GET /trade-api/v2/markets/KXNHLSPREAD-26JAN26LACBJ -> not_found, 2026-09-22",
            "fetched": SESSION_DATE,
            "nhl_schedule_check": "club-schedule CBJ 2026-01 has no game on the 26th; LAK@DET was game 2025020833 on the 27th",
        },
        "confidence": "high",
        "kind": "audit",
    },
    {
        "finding_id": "FIND_TOTALS_PRICED_BACKTESTS_LIVE",
        "title": "Totals and puck-line priced backtests are now populated (walks met)",
        "body": (
            "The Kalshi historical candle walk and the NHL per-game stats walk have reached "
            "the same games: the 2026-09-22 run reports 7,741 settled totals contracts with a "
            "pre-puck-drop offer across 1,261 games, of which 1,159 have point-in-time "
            "features (KXNHLSPREAD: 3,169 contracts / 793 games / 691 backtestable). Totals "
            "backtests are therefore priced, not accuracy-only; the former "
            "totals_price_coverage gap is closed by the recovery pass. KXNHLOVERTIME is the "
            "remaining gap (48 contracts with an offer, 0 backtestable), and its settlement "
            "rule is still shootout-unverified -- both stay flagged."),
        "evidence": {
            "report_keys": ["totals_price_coverage", "puck_line_price_coverage",
                            "overtime_price_coverage"],
            "fetched": SESSION_DATE,
            "run_date": SESSION_DATE,
        },
        "confidence": "high",
        "kind": "status",
    },
)


def record(store) -> int:
    """Write the research log into ``findings`` (idempotent; evidence replaces in place)."""
    n = 0
    for f in RESEARCH_FINDINGS:
        store.execute(
            """INSERT OR REPLACE INTO findings(finding_id, created_at, title, body, evidence,
                                               confidence, kind) VALUES(?,?,?,?,?,?,?)""",
            (f["finding_id"], utcnow_iso(), f["title"], f["body"],
             json.dumps(f["evidence"], indent=1), f["confidence"], f["kind"]))
        n += 1
    store.commit()
    return n


def utcnow_iso() -> str:
    from .store import utcnow
    return utcnow()
