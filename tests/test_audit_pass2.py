"""Regression tests for the second-pass audit (2026-09-22).

Each test here pins one defect that was found by reading the *ledger* rather than the code,
and each one is written so that reverting the fix fails the test:

1. **Shared book depth.**  A published offer size describes the book, not the strategy
   looking at it.  On the 2026-09-21 ledger, ``KXNHLGAME-26SEP21NYRNJ-NYR`` published six
   contracts and seven strategies booked 1,443.7 contracts against it.
2. **Strike-aware duplicate detection.**  One totals rule may legitimately hold an Over 5.5
   *and* an Over 7.5 on one game; grouping without the strike produced 64 false positives.
3. **Stale flags.**  A condition-based check re-raises what it still finds, so a flag that is
   open but not re-raised no longer reproduces.  Two ``duplicate_game`` rows describing real
   preseason split-squad doubleheaders stayed in the queue for a day after the bug was fixed.
4. **Feed abbreviation aliases.**  ESPN publishes ``SJ``/``TB``/``NJ``/``LA`` where the NHL
   publishes ``SJS``/``TBL``/``NJD``/``LAK``, so four clubs had no injury context at all.
5. **Season scope.**  All 102 forward-test wagers on the 2026-09-21 ledger were preseason
   games, traded by rules fitted on regular-season form.
6. **Season-phase reporting.**  Those wagers must be visible *and* kept out of the scored
   competition totals, never silently dropped and never merged in.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp.analysis import ALL_PHASES, COMPETITION_PHASES, Performance
from nhlcomp.paper import (DEPTH_BASIS_SHARED, UNKNOWN_DEPTH_CAP_CONTRACTS,
                           WIDE_BOOK_MAX_SPREAD, PaperEngine, evidence_depth, simulate_fill)
from nhlcomp.store import ABBREV_ALIASES, Store, utcnow
from nhlcomp.strategies import (PLAYOFFS, PRESEASON, REGULAR_SEASON, Signal, ThresholdStrategy)
from nhlcomp.strategies import Quote
from nhlcomp.verify import Verifier

TEAMS = ((1, "NJD"), (10, "TOR"), (14, "TBL"), (24, "ANA"), (26, "LAK"), (28, "SJS"),
         (29, "CBJ"))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.addCleanup(self.store.close)
        for tid, ab in TEAMS:
            self.store.execute(
                "INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                (tid, ab, f"Team {ab}"))
        self.store.commit()

    def make_bet(self, bet_id, *, gtype=None, mode="BACKTEST", result="WIN", pnl=5.0,
                 market="moneyline", selection="home", strike=None, stake=10.0, game_id=1):
        """Write a bet through the supported append path, with every NOT NULL column set."""
        row = {
            "bet_id": bet_id, "strategy_id": "S", "strategy_version": 1, "username": "S_001",
            "test_mode": mode, "game_id": game_id, "game_date": "2026-01-01",
            "matchup": "away@home", "market": market, "selection": selection,
            "bet_type": "binary_contract", "provider": "kalshi", "odds_format": "binary",
            "price": 0.5, "implied_prob": 0.5, "model_prob": 0.6, "edge": 0.1,
            "fair_price": 0.5, "decision_ts": "2026-01-01T00:00:00Z",
            "bet_ts": "2026-01-01T00:00:00Z", "stake": stake, "entry_price": 0.5,
            "result": result, "pnl": pnl, "verification_status": "test",
            "created_at": utcnow(), "amended": 0, "strike": strike, "game_type": gtype,
        }
        self.assertTrue(self.store.record_bet(row), bet_id)
        self.store.commit()

    def add_game(self, game_id, *, game_type, start=None, state="FUT", hs=None, as_=None,
                 home=10, away=24, season=20262027):
        start = start or (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=2))
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
            " away_id, venue, utc_offset, state, home_score, away_score, source_id, provenance)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'test','SOURCE')",
            (game_id, season, game_type, start.date().isoformat(), _iso(start), home, away,
             "Arena", "-04:00", state, hs, as_))
        self.store.commit()
        return start


# ---------------------------------------------------------------------------- 1. depth
class TestSharedBookDepth(Base):
    """Published depth is a property of the market and is shared between strategies."""

    def test_evidence_depth_ranks_offer_over_volume_over_cap(self):
        self.assertEqual(evidence_depth(40.0, 900.0)[0], 40.0)
        self.assertEqual(evidence_depth(None, 900.0)[0], 900.0)
        self.assertEqual(evidence_depth(None, None)[0], UNKNOWN_DEPTH_CAP_CONTRACTS)

    def test_a_second_wager_cannot_take_depth_the_first_one_already_took(self):
        first = simulate_fill(50.0, 0.50, 10.0)
        self.assertEqual(first.contracts_filled, 10.0)
        # 10 contracts existed at that level and the first wager took all of them
        second = simulate_fill(50.0, 0.50, 10.0, remaining=0.0)
        self.assertEqual(second.contracts_filled, 0.0)
        self.assertEqual(second.depth_basis, DEPTH_BASIS_SHARED)

    def test_a_partly_used_level_fills_only_the_remainder(self):
        f = simulate_fill(500.0, 0.50, 100.0, remaining=30.0)
        self.assertEqual(f.contracts_filled, 30.0)
        self.assertAlmostEqual(f.stake, 15.0)
        self.assertEqual(f.depth_remaining_after, 0.0)

    def _sig(self, *, strategy_id, ask=0.50, size=10.0, stake=50.0, contract="K-A", side="YES"):
        q = Quote(provider="kalshi", market_key="K", contract=contract, game_id=1,
                  game_date="2026-01-01", market_type="moneyline", selection="home", side=side,
                  bid=round(ask - 0.02, 4), ask=ask, bid_size=size, ask_size=size, volume=1.0,
                  liquidity=None, ts_utc=utcnow())
        return Signal(strategy_id=strategy_id, version=1, username=f"{strategy_id}_001",
                      game_id=1, game_date="2026-01-01", matchup="away@home", market="moneyline",
                      selection="home", side=side, model_prob=0.80, fair_price=ask,
                      required_price=0.60, stake=stake, status="READY TO BET", quote=q)

    def test_two_strategies_on_one_quote_together_take_only_the_published_size(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        eng = PaperEngine(self.store)
        ts = "2026-01-01T00:00:00Z"
        a = eng.place(self._sig(strategy_id="AAA"), decision_ts=ts, test_mode="FORWARD TEST")
        b = eng.place(self._sig(strategy_id="BBB"), decision_ts=ts, test_mode="FORWARD TEST")
        self.assertIsNotNone(a)
        self.assertIsNone(b, "the book level was fully claimed by the first wager")
        rows = self.store.query("SELECT filled_size FROM bets")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["filled_size"]), 10.0)
        self.assertIn("K-A", eng.depth_blocked)
        self.assertEqual(eng.depth_blocked["K-A"]["strategies"], ["BBB"])

    def test_a_later_quote_at_a_new_timestamp_is_a_fresh_book_level(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        eng = PaperEngine(self.store)
        a = eng.place(self._sig(strategy_id="AAA"), decision_ts="2026-01-01T00:00:00Z",
                      test_mode="FORWARD TEST")
        b = eng.place(self._sig(strategy_id="BBB"), decision_ts="2026-01-01T00:10:00Z",
                      test_mode="FORWARD TEST")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b, "a new snapshot of the book is a new level")

    def test_depth_is_seeded_from_the_ledger_so_a_rerun_cannot_double_claim(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        ts = "2026-01-01T00:00:00Z"
        PaperEngine(self.store).place(self._sig(strategy_id="AAA"), decision_ts=ts,
                                      test_mode="FORWARD TEST")
        # a brand-new engine (a later process) must still see the level as spent
        fresh = PaperEngine(self.store)
        self.assertEqual(fresh.depth_remaining("K-A", "YES", ts, 10.0), 0.0)
        self.assertIsNone(fresh.place(self._sig(strategy_id="BBB"), decision_ts=ts,
                                      test_mode="FORWARD TEST"))

    def test_a_duplicate_bet_id_does_not_consume_depth(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        eng = PaperEngine(self.store)
        ts = "2026-01-01T00:00:00Z"
        sig = self._sig(strategy_id="AAA")
        self.assertIsNotNone(eng.place(sig, decision_ts=ts, test_mode="FORWARD TEST"))
        # same bet_id again -> refused by record_bet; the level must stay as it was
        self.assertIsNone(eng.place(sig, decision_ts=ts, test_mode="FORWARD TEST"))
        self.assertEqual(PaperEngine(self.store).depth_remaining("K-A", "YES", ts, 10.0), 0.0)
        self.assertEqual(len(self.store.query("SELECT 1 FROM bets")), 1)

    def test_the_claim_is_reported_on_the_row(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        eng = PaperEngine(self.store)
        bid = eng.place(self._sig(strategy_id="AAA", stake=2.0), decision_ts="2026-01-01T00:00:00Z",
                        test_mode="FORWARD TEST")
        notes = json.loads(self.store.one("SELECT notes FROM bets WHERE bet_id=?", (bid,))["notes"])
        self.assertEqual(notes["depth_remaining_after"], 6.0)   # 4 contracts taken of 10
        self.assertEqual(notes["entry_bid"], 0.48)
        self.assertAlmostEqual(notes["entry_spread"], 0.02)


# -------------------------------------------------------------------- 1b. wide books
class TestWideBookRefusal(Base):
    def _sig(self, bid, ask):
        q = Quote(provider="kalshi", market_key="K", contract="K-W", game_id=1,
                  game_date="2026-01-01", market_type="moneyline", selection="home", side="YES",
                  bid=bid, ask=ask, bid_size=50.0, ask_size=50.0, volume=0.0, liquidity=None,
                  ts_utc=utcnow())
        return Signal(strategy_id="AAA", version=1, username="AAA_001", game_id=1,
                      game_date="2026-01-01", matchup="away@home", market="moneyline",
                      selection="home", side="YES", model_prob=0.95, fair_price=ask,
                      required_price=0.99, stake=20.0, status="READY TO BET", quote=q)

    def test_an_implausibly_wide_book_is_refused_with_the_numbers_recorded(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        eng = PaperEngine(self.store)
        # the real 2026-09-21 row: bid 0.11 / ask 0.74 on KXNHLGAME-26SEP24NJNYR-NYR
        self.assertIsNone(eng.place(self._sig(0.11, 0.74), decision_ts="2026-01-01T00:00:00Z",
                                    test_mode="FORWARD TEST"))
        self.assertEqual(eng.wide_book_skipped["K-W"]["spread"], 0.63)
        self.assertEqual(eng.wide_book_skipped["K-W"]["max_spread"], WIDE_BOOK_MAX_SPREAD)
        self.assertEqual(self.store.query("SELECT 1 FROM bets"), [])

    def test_a_normal_book_is_not_touched_by_the_threshold(self):
        self.add_game(1, game_type=REGULAR_SEASON)
        eng = PaperEngine(self.store)
        bid = eng.place(self._sig(0.68, 0.72), decision_ts="2026-01-01T00:00:00Z",
                        test_mode="FORWARD TEST")
        self.assertIsNotNone(bid)
        self.assertEqual(eng.wide_book_skipped, {})


# ------------------------------------------------------- 2. duplicates and 3. stale flags
class TestDuplicateDetectionAndReconciliation(Base):
    def _bet(self, bet_id, **kw):
        self.make_bet(bet_id, market=kw.pop("market", "total"),
                      selection=kw.pop("selection", "over"), **kw)

    def test_two_rungs_of_one_totals_ladder_are_not_a_duplicate(self):
        self._bet("S-v1-1-total-over-5.5", strike=5.5)
        self._bet("S-v1-1-total-over-7.5", strike=7.5)
        counts = Verifier(self.store).check_bets()
        self.assertEqual(counts["duplicate_bet"], 0)
        self.assertEqual(self.store.query(
            "SELECT 1 FROM irregularities WHERE kind='duplicate_bet'"), [])

    def test_a_genuine_double_entry_at_the_same_line_is_still_caught(self):
        self._bet("S-v1-1-total-over-5.5", strike=5.5)
        self._bet("S-v1-1-total-over-5.5b", strike=5.5)
        counts = Verifier(self.store).check_bets()
        self.assertEqual(counts["duplicate_bet"], 1)
        self.assertTrue(self.store.query(
            "SELECT 1 FROM irregularities WHERE kind='duplicate_bet'"))

    def test_a_flag_that_no_longer_reproduces_is_closed_not_deleted(self):
        # the real stale row: an NHL preseason split-squad doubleheader, matched on
        # date/home/away by an older verifier that ignored the differing start times
        # inserted directly, exactly as the older verifier left it: a real preseason
        # split-squad doubleheader (18:00Z and 22:00Z, two game ids, two final scores)
        self.store.execute(
            """INSERT INTO irregularities(ts_utc, kind, severity, entity_type, entity_id,
                                          detail, auto_corrected)
               VALUES(?,?,?,?,?,?,0)""",
            ("2026-09-20T20:38:12Z", "duplicate_game", "warn", "game", "2024-09-22:13:18",
             "2024-09-22 18@13 appears 2 times"))
        self.store.commit()
        verifier = Verifier(self.store)
        closed = verifier.reconcile_stale_flags()
        self.assertEqual(closed.get("duplicate_game"), 1)
        row = self.store.one("SELECT * FROM irregularities WHERE kind='duplicate_game'")
        self.assertEqual(row["status"], "resolved")
        self.assertIn("did not reproduce", row["resolution"])
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM irregularities WHERE kind='duplicate_game'")["c"], 1,
            "nothing is deleted")
        self.assertTrue(self.store.query(
            "SELECT 1 FROM audit_log WHERE action='RECONCILE_STALE_FLAG'"))

    def test_a_flag_still_reproducing_is_left_open(self):
        self._bet("S-v1-1-total-over-5.5", strike=5.5)
        self._bet("S-v1-1-total-over-5.5b", strike=5.5)
        verifier = Verifier(self.store)
        verifier.check_bets()
        self.assertEqual(verifier.reconcile_stale_flags(), {})
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE kind='duplicate_bet'")["status"], "open")

    def test_a_kind_this_verifier_does_not_derive_is_never_touched(self):
        # an ingest-time flag: silence from the verifier proves nothing about it
        self.store.execute(
            """INSERT INTO irregularities(ts_utc, kind, severity, entity_type, entity_id,
                                          detail, auto_corrected)
               VALUES(?,?,?,?,?,?,0)""",
            ("2026-09-20T20:38:12Z", "broken_api", "error", "source", "kalshi.trade_api",
             "kalshi 503"))
        self.store.commit()
        self.assertEqual(Verifier(self.store).reconcile_stale_flags(), {})
        self.assertEqual(self.store.one(
            "SELECT status FROM irregularities WHERE kind='broken_api'")["status"], "open")


# ------------------------------------------------------------------ 4. abbreviation aliases
class TestFeedAbbreviationAliases(Base):
    def test_the_espn_short_forms_resolve_to_the_nhl_club(self):
        for short, canon in ABBREV_ALIASES.items():
            self.assertIsNotNone(self.store.team_id_for(short), short)
            self.assertEqual(self.store.team_id_for(short), self.store.team_id_for(canon))
        self.assertEqual(self.store.team_id_for("LA"), 26)
        self.assertEqual(self.store.team_id_for("SJ"), 28)
        self.assertEqual(self.store.team_id_for("TB"), 14)

    def test_an_alias_never_shadows_a_real_club_code(self):
        # CBJ is a real code; if a future feed used "CBJ" for something else the alias table
        # must not be consulted at all
        self.assertEqual(self.store.team_id_for("CBJ"), 29)

    def test_an_unknown_code_still_resolves_to_nothing_rather_than_a_guess(self):
        self.assertIsNone(self.store.team_id_for("ZZZ"))
        self.assertIsNone(self.store.team_id_for(""))


# ------------------------------------------------------------------------- 5. season scope
class TestSeasonScope(Base):
    def test_the_default_scope_is_regular_season_and_playoffs(self):
        s = ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f")
        self.assertEqual(tuple(s.game_types), (REGULAR_SEASON, PLAYOFFS))
        self.assertIn("game_types", json.loads(s.describe()["params_json"]))

    def test_the_scope_survives_a_database_round_trip(self):
        from nhlcomp.pipeline import _hydrate
        s = ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f",
                              game_types=(PRESEASON, REGULAR_SEASON, PLAYOFFS))
        row = dict(s.describe())
        row.update({"game_types": json.dumps(list(s.game_types)), "status": "active",
                    "created_at": utcnow(), "test_mode": "FORWARD TEST", "leakage_checked": 1})
        self.store.upsert_strategy(row)
        stored = self.store.one("SELECT * FROM strategies WHERE strategy_id='X'")
        revived = _hydrate(stored)
        self.assertEqual(tuple(revived.game_types), (1, 2, 3))

    def test_a_regular_season_rule_refuses_a_preseason_game_forward(self):
        from nhlcomp.pipeline import Pipeline
        from nhlcomp.http import HttpClient
        start = self.add_game(1, game_type=PRESEASON)
        pipe = Pipeline(self.store, HttpClient(cache_dir=os.path.join(self.tmp.name, "c"),
                                               offline=True), verbose=False)
        sig = Signal(strategy_id="X", version=1, username="X_001", game_id=1,
                     game_date=start.date().isoformat(), matchup="a@h", market="moneyline",
                     selection="home", side="YES", model_prob=0.9, fair_price=0.5,
                     required_price=0.6, stake=10.0, status="READY TO BET", quote=None)
        self.assertFalse(pipe._in_scope(
            ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f"),
            type("G", (), {"game_type": PRESEASON})()))
        self.assertTrue(pipe._in_scope(
            ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f",
                              game_types=(PRESEASON,)), type("G", (), {"game_type": 1})()))
        self.assertTrue(pipe._in_scope(
            ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f"),
            type("G", (), {"game_type": PLAYOFFS})()))
        self.assertIsNotNone(sig)

    def test_a_bet_records_the_phase_of_its_game(self):
        self.add_game(1, game_type=PRESEASON)
        q = Quote(provider="kalshi", market_key="K", contract="K-A", game_id=1,
                  game_date="2026-01-01", market_type="moneyline", selection="home", side="YES",
                  bid=0.48, ask=0.50, bid_size=50.0, ask_size=50.0, volume=1.0, liquidity=None,
                  ts_utc=utcnow())
        sig = Signal(strategy_id="X", version=1, username="X_001", game_id=1,
                     game_date="2026-01-01", matchup="a@h", market="moneyline",
                     selection="home", side="YES", model_prob=0.8, fair_price=0.5,
                     required_price=0.6, stake=10.0, status="READY TO BET", quote=q)
        bid = PaperEngine(self.store).place(sig, decision_ts="2026-01-01T00:00:00Z",
                                            test_mode="FORWARD TEST")
        self.assertEqual(self.store.one("SELECT game_type FROM bets WHERE bet_id=?",
                                        (bid,))["game_type"], PRESEASON)

    def test_a_legacy_row_without_a_phase_is_backfilled_from_its_game(self):
        self.add_game(1, game_type=PRESEASON)
        self.make_bet("B1", mode="FORWARD TEST", result="OPEN", pnl=None)
        self.assertIsNone(self.store.one("SELECT game_type FROM bets WHERE bet_id='B1'")["game_type"])
        self.assertEqual(self.store.backfill_bet_game_type(), 1)
        self.assertEqual(self.store.one(
            "SELECT game_type FROM bets WHERE bet_id='B1'")["game_type"], PRESEASON)
        self.assertTrue(self.store.query(
            "SELECT 1 FROM audit_log WHERE action='BACKFILL_BET_GAME_TYPE'"))
        # idempotent, and it never invents a phase for a bet whose game is unknown
        self.assertEqual(self.store.backfill_bet_game_type(), 0)


# ------------------------------------------------------------------- 6. phase reporting
class TestSeasonPhaseReporting(Base):
    def _bet(self, bet_id, *, gtype, mode, result, pnl, stake=10.0):
        self.make_bet(bet_id, gtype=gtype, mode=mode, result=result, pnl=pnl, stake=stake)

    def test_the_scored_total_excludes_preseason_by_default(self):
        self._bet("A", gtype=REGULAR_SEASON, mode="FORWARD TEST", result="WIN", pnl=5.0)
        self._bet("B", gtype=PRESEASON, mode="FORWARD TEST", result="LOSS", pnl=-10.0)
        perf = Performance(self.store)
        scored = perf.competition_totals(test_mode="FORWARD TEST")
        self.assertEqual(scored["bets"], 1)
        self.assertEqual(scored["pnl"], 5.0)
        everything = perf.competition_totals(test_mode="FORWARD TEST", phases=ALL_PHASES)
        self.assertEqual(everything["bets"], 2)
        self.assertEqual(everything["pnl"], -5.0)

    def test_the_preseason_wagers_are_still_reported_never_dropped(self):
        self._bet("A", gtype=REGULAR_SEASON, mode="FORWARD TEST", result="WIN", pnl=5.0)
        self._bet("B", gtype=PRESEASON, mode="FORWARD TEST", result="LOSS", pnl=-10.0)
        rows = Performance(self.store).phase_breakdown(test_mode="FORWARD TEST")
        by_phase = {r["phase"]: r for r in rows}
        self.assertIn("preseason", by_phase)
        self.assertFalse(by_phase["preseason"]["scored_in_competition"])
        self.assertEqual(by_phase["preseason"]["pnl"], -10.0)
        self.assertEqual(by_phase["regular season"]["pnl"], 5.0)
        self.assertTrue(by_phase["regular season"]["scored_in_competition"])

    def test_a_strategy_summary_carries_its_phase_split(self):
        self.store.upsert_strategy({
            "strategy_id": "S", "version": 1, "username": "S_001", "name": "s",
            "category": "c", "hypothesis": "h", "data_used": "d", "entry_rule": "e",
            "price_rule": "p", "settlement_rule": "r", "markets": "moneyline",
            "params_json": "{}", "origin": "seed", "origin_ref": None,
            "starting_bankroll": 1000.0, "bankroll": 1000.0, "status": "active",
            "created_at": utcnow(), "test_mode": "FORWARD TEST", "leakage_checked": 1})
        self._bet("A", gtype=REGULAR_SEASON, mode="FORWARD TEST", result="WIN", pnl=5.0)
        self._bet("B", gtype=PRESEASON, mode="FORWARD TEST", result="LOSS", pnl=-10.0)
        summ = Performance(self.store).summarize("S", 1)
        self.assertEqual(summ["preseason_bets"], 1)
        self.assertEqual(summ["phases"]["preseason"]["pnl"], -10.0)
        self.assertEqual(summ["phases"]["regular season"]["pnl"], 5.0)
        # the strategy's own ledger still includes both -- its bankroll must be consistent
        self.assertEqual(summ["FORWARD TEST"]["pnl"], -5.0)

    def test_the_competition_phase_tuple_is_regular_season_and_playoffs(self):
        self.assertEqual(tuple(COMPETITION_PHASES), (2, 3))
        self.assertEqual(tuple(ALL_PHASES), (1, 2, 3))


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
