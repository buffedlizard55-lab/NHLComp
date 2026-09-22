"""Regression tests for the fourth-session audit (2026-09-22): queue recovery.

A verification queue that only ever grows trains a reader to ignore it, which is how real
defects hide.  This pass adds the complement of ``Store.flag`` -- evidence-carrying
*recovery* -- to every raiser whose condition can later be observed to have gone away, and
pins each behavior here:

1. a transient ``broken_api`` (stats REST read timeout) resolves when the identical fetch
   succeeds in a later run;
2. ``no_markets`` resolves when the same query returns contracts again;
3. non-NHL exhibition opponents (2024 Global Series: EHC Red Bull Muenchen, team id 7509,
   abbrev ``MUN``) are identified from the schedule payload's own abbrev, recorded as
   excluded-by-design, and supersede the old ``unknown_team`` error;
4. a regular-season or abbreviation-less unknown team still raises the ``unknown_team``
   error -- the recovery must not open a guessing loophole;
5. coverage gaps (totals and strike markets) resolve when the candle walk and the feature
   walk finally reach the same games;
6. the overtime settlement rule resolves when settled shootout evidence exists;
7. an ``unmatched_market`` flag resolves when a later ingest matches the event;
8. evidence-carrying manual research resolutions apply idempotently and are audited.

Nothing anywhere deletes a row: recovery sets ``status='resolved'`` with a resolution note
and writes one ``audit_log`` entry per closed group.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp import research
from nhlcomp.http import HttpClient
from nhlcomp.ingest import Ingestor
from nhlcomp.pipeline import Pipeline
from nhlcomp.sources.nhl import normalize_game
from nhlcomp.store import Store, utcnow
from nhlcomp.verify import Verifier

STATS_REST = "https://api.nhle.com/stats/rest/en"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.addCleanup(self.store.close)
        # a few NHL franchises, as the teams endpoint would publish them
        for tid, ab in ((7, "BUF"), (29, "CBJ"), (26, "LAK"), (19, "DET")):
            self.store.execute(
                "INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                (tid, ab, f"Team {ab}"))
        self.store.commit()


class TestResolveWhere(Base):
    def test_closes_only_matching_open_rows_and_writes_audit(self):
        self.store.flag("broken_api", "stats REST team/summary isGame 20242025/2: timed out",
                        entity_type="source", entity_id="nhl.stats_rest")
        self.store.flag("broken_api", "stats REST team/summary isGame 20252026/2: timed out",
                        entity_type="source", entity_id="nhl.stats_rest")
        self.store.flag("broken_api", "scoreboard 2026-09-21: timed out",
                        entity_type="source", entity_id="nhl.api_web")
        n = self.store.resolve_irregularities_where(
            "broken_api", "stats REST team/summary isGame 20242025/2:",
            "recovered: the same fetch succeeded", actor="ingest_ext")
        self.assertEqual(n, 1)
        rows = self.store.query(
            "SELECT detail, status FROM irregularities WHERE kind='broken_api' "
            "ORDER BY detail")
        by = {r["detail"]: r["status"] for r in rows}
        self.assertEqual(by["stats REST team/summary isGame 20242025/2: timed out"], "resolved")
        self.assertEqual(by["stats REST team/summary isGame 20252026/2: timed out"], "open")
        self.assertEqual(by["scoreboard 2026-09-21: timed out"], "open")
        audit = self.store.query(
            "SELECT * FROM audit_log WHERE action='RESOLVE_RECOVERED'")
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["actor"], "ingest_ext")

    def test_entity_id_scope(self):
        self.store.flag("totals_price_coverage", "Totals price history and feature history gap",
                        entity_type="market", entity_id="KXNHLTOTAL")
        self.store.flag("totals_price_coverage", "Totals price history and feature history gap",
                        entity_type="market", entity_id="OTHER")
        n = self.store.resolve_irregularities_where(
            "totals_price_coverage", "Totals price history", "walks met",
            entity_id="KXNHLTOTAL", actor="pipeline")
        self.assertEqual(n, 1)
        left = self.store.one(
            "SELECT status FROM irregularities WHERE entity_id='OTHER' "
            "AND kind='totals_price_coverage'")
        self.assertEqual(left["status"], "open")

    def test_no_open_rows_is_a_normal_zero(self):
        self.assertEqual(self.store.resolve_irregularities_where(
            "no_markets", "no KXNHLGAME markets", "x", actor="ingest"), 0)


class TestStatsRestRecovery(Base):
    TEAM_SUMMARY_URL = (f"{STATS_REST}/team/summary?isAggregate=false&isGame=true"
                        f"&start=0&limit=100&cayenneExp=seasonId=20242025%20and%20gameTypeId=2")

    def _summary_payload(self):
        row = {"gameId": 2024020001, "teamId": 7, "gameDate": "2024-10-09",
               "homeRoad": "H", "opponentTeamAbbrev": "CBJ", "teamFullName": "Team BUF",
               "goalsFor": 4, "goalsAgainst": 2, "shotsForPerGame": 31.0,
               "shotsAgainstPerGame": 28.0, "powerPlayPct": 20.0, "penaltyKillPct": 80.0,
               "powerPlayNetPct": 15.0, "penaltyKillNetPct": 75.0, "faceoffWinPct": 50.0,
               "wins": 1, "losses": 0, "otLosses": 0, "points": 2,
               "winsInRegulation": 1, "winsInShootout": 0}
        return {"data": [row], "total": 1}

    def test_timeout_flag_resolves_when_the_same_fetch_succeeds(self):
        http = HttpClient(os.path.join(self.tmp.name, "cache"))
        http._store(self.TEAM_SUMMARY_URL, 200, json.dumps(self._summary_payload()))
        ing = Ingestor(self.store, http, verbose=False)
        # yesterday's run timed out mid-walk and flagged it
        self.store.flag("broken_api",
                        "stats REST team/summary isGame 20242025/2: The read operation timed out",
                        severity="warn", entity_type="source", entity_id="nhl.stats_rest")
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
            " home_id, away_id, state) VALUES(2024020001, 20242025, 2, '2024-10-09',"
            " '2024-10-09T23:00:00Z', 7, 29, 'OFF')")
        self.store.commit()
        # a past season with current_season set walks from cache (as CI does)
        n = ing.nhl_team_game_stats([20242025], current_season=20262027)
        self.assertEqual(n, 1)
        row = self.store.one(
            "SELECT status, resolution FROM irregularities WHERE kind='broken_api'")
        self.assertEqual(row["status"], "resolved")
        self.assertIn("transient", row["resolution"])
        # and the recovery is auditable
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM audit_log WHERE action='RESOLVE_RECOVERED'")["c"], 1)


class TestNonNhlOpponent(Base):
    @staticmethod
    def _schedule_game(gid, game_date, home_id, away_id, home_ab, away_ab, gtype=1):
        return normalize_game({
            "id": gid, "season": 20242025, "gameType": gtype, "gameDate": game_date,
            "startTimeUTC": f"{game_date}T23:00:00Z", "venue": {"default": "SAP Garden"},
            "venueUTCOffset": "+02:00", "venueTimezone": "US/Eastern",
            "gameState": "FINAL", "gameScheduleState": "OK",
            "homeTeam": {"id": home_id, "abbrev": home_ab, "score": 0},
            "awayTeam": {"id": away_id, "abbrev": away_ab, "score": 5},
            "periodDescriptor": {"number": 3, "periodType": "REG"},
        })

    def test_global_series_opponent_is_identified_and_unknown_team_superseded(self):
        ing = Ingestor(self.store, HttpClient(os.path.join(self.tmp.name, "cache")),
                       verbose=False)
        # game 2024010106: Buffalo vs team id 7509 (EHC Red Bull Muenchen, "MUN"),
        # exactly the row the live ledger flagged as unknown_team on 2026-09-20
        g = self._schedule_game(2024010106, "2024-09-27", 7509, 7, "MUN", "BUF")
        self.assertIsNotNone(g)
        ing._insert_games([g], "nhl.api_web")
        flag = self.store.one(
            "SELECT * FROM irregularities WHERE entity_id='2024010106'")
        self.assertIsNotNone(flag)
        self.assertEqual(flag["kind"], "non_nhl_opponent_excluded")
        self.assertEqual(flag["severity"], "info")
        self.assertIn("MUN", flag["detail"])
        # the game is NOT in the ledger: excluded by design, not attached to a guess
        self.assertIsNone(self.store.one(
            "SELECT 1 FROM games WHERE game_id=2024010106"))
        # an unknown_team error from an earlier run is superseded, with the evidence
        self.store.flag("unknown_team",
                        "game 2024010106 on 2024-09-27 references team id(s) [7509] which are "
                        "not in the teams table; game skipped rather than attached to a guessed "
                        "franchise", severity="error", entity_type="game",
                        entity_id="2024010106")
        ing._insert_games([g], "nhl.api_web")
        row = self.store.one(
            "SELECT status, resolution FROM irregularities WHERE kind='unknown_team' "
            "AND entity_id='2024010106'")
        self.assertEqual(row["status"], "resolved")
        self.assertIn("non-NHL exhibition opponent", row["resolution"])

    def test_regular_season_unknown_team_still_raises_the_error(self):
        ing = Ingestor(self.store, HttpClient(os.path.join(self.tmp.name, "cache")),
                       verbose=False)
        g = self._schedule_game(2024020099, "2024-11-01", 7509, 7, "MUN", "BUF", gtype=2)
        ing._insert_games([g], "nhl.api_web")
        flag = self.store.one(
            "SELECT kind, severity FROM irregularities WHERE entity_id='2024020099'")
        self.assertEqual(flag["kind"], "unknown_team")
        self.assertEqual(flag["severity"], "error")

    def test_unknown_nhl_abbrev_still_raises_the_error_even_in_preseason(self):
        ing = Ingestor(self.store, HttpClient(os.path.join(self.tmp.name, "cache")),
                       verbose=False)
        # abbrev claims to be an NHL club (BUF) while the id is not in the teams table:
        # that is a data problem, not an exhibition opponent, and must stay an error
        g = self._schedule_game(2024010110, "2024-09-28", 9999, 7, "BUF", "CBJ")
        ing._insert_games([g], "nhl.api_web")
        flag = self.store.one(
            "SELECT kind, severity FROM irregularities WHERE entity_id='2024010110'")
        self.assertEqual(flag["kind"], "unknown_team")
        self.assertEqual(flag["severity"], "error")


class _CoverageBase(Base):
    def setUp(self):
        super().setUp()
        self.pipe = Pipeline(self.store, HttpClient(os.path.join(self.tmp.name, "cache")),
                             verbose=False)

    def _game(self, gid, start, decided=True):
        from nhlcomp.features import GameRef
        return GameRef(
            game_id=gid, start=start, game_date=start.strftime("%Y-%m-%d"),
            home_id=29, away_id=7, venue="Nationwide Arena", venue_tz=None,
            utc_offset="-05:00", season=20252026, game_type=2,
            home_score=4 if decided else None, away_score=2 if decided else None,
            last_period_type="REG" if decided else None)


class TestCoverageRecovery(_CoverageBase):
    def test_strike_market_gap_resolves_when_the_walks_meet(self):
        from nhlcomp.features import GameRef  # noqa: F401  (import guard)
        past = NOW - timedelta(days=30)
        g = self._game(2025020500, past)
        row = {"contract": "KXNHLSPREAD-26AUG22BUFCBJ-CBJ1", "game_id": g.game_id,
               "points": {"close": {"ask": 0.4}}}
        # first pass: no feature row -> the gap is flagged
        self.pipe._strike_market_coverage([row], {g.game_id: g}, {}, series="KXNHLSPREAD",
                                          report_key="puck_line_price_coverage",
                                          flag_kind="puck_line_price_coverage")
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE kind='puck_line_price_coverage'")["status"],
            "open")
        # later pass: the feature walk caught up -> the flag is resolved, not deleted
        cov = self.pipe._strike_market_coverage(
            [row], {g.game_id: g}, {g.game_id: {"lam_home": 3.1, "lam_away": 2.7}},
            series="KXNHLSPREAD", report_key="puck_line_price_coverage",
            flag_kind="puck_line_price_coverage")
        self.assertEqual(cov["backtestable_games"], 1)
        row2 = self.store.one(
            "SELECT status, resolution FROM irregularities WHERE kind='puck_line_price_coverage'")
        self.assertEqual(row2["status"], "resolved")
        self.assertIn("overlap", row2["resolution"])

    def test_totals_gap_resolves_when_the_walks_meet(self):
        past = NOW - timedelta(days=30)
        g = self._game(2025020501, past)
        totals = [{"contract": "KXNHLTOTAL-26AUG22BUFCBJ-5", "game_id": g.game_id,
                   "points": {"close": {"ask": 0.5}}}]
        self.pipe._totals_coverage(totals, {g.game_id: g}, {})
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE kind='totals_price_coverage'")["status"],
            "open")
        self.pipe._totals_coverage(totals, {g.game_id: g},
                                   {g.game_id: {"lam_home": 3.0, "lam_away": 2.8}})
        row = self.store.one(
            "SELECT status, resolution FROM irregularities WHERE kind='totals_price_coverage'")
        self.assertEqual(row["status"], "resolved")
        self.assertIn("1 totals game", row["resolution"])


class TestOvertimeRuleRecovery(Base):
    def _insert_so_evidence(self):
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
            " home_id, away_id, state, home_score, away_score, last_period_type)"
            " VALUES(2025020601, 20252026, 2, '2026-01-05', '2026-01-06T00:00:00Z',"
            " 29, 7, 'OFF', 3, 2, 'SO')")
        self.store.execute(
            """INSERT INTO market_settlements(provider, event_ticker, contract, game_id,
                     game_date, side, result, settlement_ts, retrieved_at, series_ticker)
               VALUES('kalshi', 'KXNHLOVERTIME-26JAN05CBJBUF',
                      'KXNHLOVERTIME-26JAN05CBJBUF-OT', 2025020601, '2026-01-05',
                      'YES', 'yes', '2026-01-06T03:00:00Z', ?, 'KXNHLOVERTIME')""",
            (_iso(NOW),))
        self.store.commit()

    def test_unverified_rule_resolves_when_shootout_evidence_exists(self):
        v = Verifier(self.store)
        # while no settled shootout contract exists, the rule is flagged unverified
        out = v.cross_validate_overtime_settlements()
        self.assertEqual(out["shootout_evidence"], [])
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE kind='settlement_rule_unverified'")["status"],
            "open")
        # the walk reaches a shootout game and the exchange settled it -> evidence
        self._insert_so_evidence()
        out = v.cross_validate_overtime_settlements()
        self.assertEqual(len(out["shootout_evidence"]), 1)
        row = self.store.one(
            "SELECT status, resolution FROM irregularities "
            "WHERE kind='settlement_rule_unverified'")
        self.assertEqual(row["status"], "resolved")
        self.assertIn("shootout", row["resolution"])


class TestUnmatchedMarketRecovery(Base):
    def test_resolves_once_the_event_has_quotes(self):
        self.store.flag("unmatched_market",
                        "kalshi KXNHLSPREAD-26JAN26LACBJ could not be matched to an NHL game "
                        "(2026-01-26 ('LAK', 'CBJ'))", severity="warn", entity_type="quote",
                        entity_id="KXNHLSPREAD-26JAN26LACBJ")
        v = Verifier(self.store)
        self.assertEqual(v.reconcile_unmatched_markets(), 0)  # nothing matched yet
        self.store.execute(
            """INSERT INTO market_quotes(market_key, contract, provider, market_type, selection, side,
                          retrieved_at, ts_utc)
               VALUES('KXNHLSPREAD-26JAN26LACBJ', 'KXNHLSPREAD-26JAN26LACBJ-CBJ1',
                      'kalshi', 'puck_line', 'home -1.5', 'YES', ?, ?)""",
            (_iso(NOW), _iso(NOW)))
        self.store.commit()
        self.assertEqual(v.reconcile_unmatched_markets(), 1)
        row = self.store.one(
            "SELECT status FROM irregularities WHERE kind='unmatched_market'")
        self.assertEqual(row["status"], "resolved")

    def test_non_kalshi_entity_ids_are_left_alone(self):
        self.store.flag("unmatched_market", "some other provider mismatch",
                        entity_type="quote", entity_id="OTHER-1")
        v = Verifier(self.store)
        self.assertEqual(v.reconcile_unmatched_markets(), 0)
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE entity_id='OTHER-1'")["status"], "open")


class TestManualResolutions(Base):
    def test_lacbj_resolution_applies_with_evidence_and_is_idempotent(self):
        self.store.flag("unmatched_market",
                        "kalshi KXNHLSPREAD-26JAN26LACBJ could not be matched to an NHL game "
                        "(2026-01-26 ('LAK', 'CBJ'))", severity="warn", entity_type="quote",
                        entity_id="KXNHLSPREAD-26JAN26LACBJ")
        v = Verifier(self.store)
        self.assertEqual(v.apply_manual_resolutions(), 1)
        row = self.store.one(
            "SELECT status, resolution FROM irregularities "
            "WHERE entity_id='KXNHLSPREAD-26JAN26LACBJ' AND kind='unmatched_market'")
        self.assertEqual(row["status"], "resolved")
        self.assertIn("not_found", row["resolution"])
        self.assertIn("manual research resolution", row["resolution"])
        audit = self.store.one(
            "SELECT * FROM audit_log WHERE action='MANUAL_RESOLUTION'")
        self.assertEqual(audit["entity"], "unmatched_market")
        # idempotent: a resolved row is not resolved again
        self.assertEqual(v.apply_manual_resolutions(), 0)

    def test_an_unknown_entity_is_untouched(self):
        self.store.flag("unmatched_market", "some other market",
                        entity_type="quote", entity_id="KXNHLGAME-26JAN26XXYYYY")
        v = Verifier(self.store)
        self.assertEqual(v.apply_manual_resolutions(), 0)
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE entity_id='KXNHLGAME-26JAN26XXYYYY'",
        )["status"], "open")


class TestResearchLog(Base):
    def test_research_findings_are_recorded_idempotently_with_evidence(self):
        from nhlcomp import research
        n = research.record(self.store)
        self.assertEqual(n, len(research.RESEARCH_FINDINGS))
        row = self.store.one(
            "SELECT * FROM findings WHERE finding_id='FIND_ODDS_SNAPSHOTS_NO_NHL'")
        self.assertIsNotNone(row)
        self.assertEqual(row["kind"], "source_review")
        ev = json.loads(row["evidence"])
        self.assertIn("fetched", ev)
        self.assertIn("url", ev)
        # every entry carries a verification method and date
        for f in self.store.query("SELECT finding_id, evidence FROM findings"):
            self.assertIn("fetched", json.loads(f["evidence"]), f["finding_id"])
        # idempotent: re-recording replaces, never duplicates
        self.assertEqual(research.record(self.store), n)
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM findings WHERE finding_id='FIND_ODDS_SNAPSHOTS_NO_NHL'")["c"],
            1)

    def test_pipeline_stage_writes_the_log(self):
        from nhlcomp.http import HttpClient
        from nhlcomp.pipeline import Pipeline
        pipe = Pipeline(self.store, HttpClient(os.path.join(self.tmp.name, "cache")),
                        verbose=False)
        n = pipe.stage_research_log()
        self.assertEqual(n, len(research.RESEARCH_FINDINGS))
        self.assertEqual(pipe.report["research_log_entries"], n)
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM findings WHERE kind IN ('source_review','audit','status') "
            "AND finding_id LIKE 'FIND_%'")["c"], len(research.RESEARCH_FINDINGS))


if __name__ == "__main__":
    unittest.main()
