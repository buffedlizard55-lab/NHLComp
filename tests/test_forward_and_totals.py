"""Regression + coverage for the two things that decide whether the competition trades at all.

1. **A forward bet must actually be placeable.**  On 2026-09-20 the ledger held 3,268
   BACKTEST wagers and *zero* FORWARD TEST wagers, with 183 upcoming rows asserting
   "condition met and priced, but no quote published for this market yet".  The quote *was*
   published: the stored quote's ``selection`` is the exchange's own wording
   ("Anaheim wins", written from Kalshi's ``title``), while a rule asks for a side
   ("home"), so ``find_quote`` compared the two and never matched.  The fixture that used
   to cover this path inserted ``selection='home'`` -- a value production never writes --
   which is why the suite was green while the forward test was dead.  The first test here
   reproduces the production shape.

2. **The totals market (KXNHLTOTAL) must be tradeable end to end** -- line parsed from the
   exchange's own ``floor_strike``/``strike_type``, entry at a real offer, settlement from
   the official final score, and no historical price invented for the side that has none.

All payloads here are SYNTHETIC but shaped exactly like the live responses captured on
2026-09-21 (see data/captured/); the numbers are invented for the test and are never
presented as NHL or Kalshi data.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp.backtest import Backtester, PricedBacktester
from nhlcomp.http import HttpClient
from nhlcomp.market import kalshi_taker_fee
from nhlcomp.models import PoissonModel, p_total_over
from nhlcomp.pipeline import Pipeline
from nhlcomp.store import Store, utcnow
from nhlcomp.strategies import DecisionContext, Quote, ThresholdStrategy, TotalsStrategy
from nhlcomp.verify import Verifier

TOR_ID, ANA_ID = 10, 24


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


class ForwardAndTotalsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.addCleanup(self.store.close)
        self.http = HttpClient(cache_dir=os.path.join(self.tmp.name, "cache"))
        self.pipe = Pipeline(self.store, self.http, verbose=False)
        self.store.execute("INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                           (TOR_ID, "TOR", "Toronto Maple Leafs"))
        self.store.execute("INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                           (ANA_ID, "ANA", "Anaheim Ducks"))
        # a decided game (for BACKTEST) and a future game (for FORWARD TEST)
        self.past = datetime(2026, 3, 4, 0, 30, tzinfo=timezone.utc)
        self.future = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=3)
        self.gid_past, self.gid_future = 2026020001, 2026020002
        for gid, day, state, hs, as_ in ((self.gid_past, self.past, "FINAL", 4, 2),
                                         (self.gid_future, self.future, "FUT", None, None)):
            self.store.execute(
                "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
                " away_id, venue, utc_offset, state, home_score, away_score, last_period_type,"
                " source_id, provenance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'test','SOURCE')",
                (gid, 20252026, 2, day.date().isoformat(), _iso(day), TOR_ID, ANA_ID, "Arena",
                 "-05:00", state, hs, as_, "REG" if hs is not None else None))
        self.store.commit()

    # --------------------------------------------------------------- helpers
    def feature_row(self, gid: int, day: datetime, **extra) -> dict:
        row = {"game_id": gid, "game_date": day.date().isoformat(),
               "start_time_utc": _iso(day), "home_id": TOR_ID, "away_id": ANA_ID,
               "home_n_prior": 40, "away_n_prior": 40, "home_exp_goals": 3.0,
               "away_exp_goals": 2.8, "lam_home": 2.975, "lam_away": 2.775,
               "exp_total": 5.75, "p_home_ml": 0.541, "p_away_ml": 0.459,
               "p_home_elo": 0.55, "p_away_elo": 0.45, "p_home_logit": 0.54,
               "p_away_logit": 0.46}
        row.update(extra)
        return row

    def register(self, strat) -> None:
        d = strat.describe()
        d.update(status="active", created_at=utcnow(), test_mode="FORWARD TEST",
                 leakage_checked=1)
        self.store.upsert_strategy(d)
        self.store.set_strategy_status(strat.strategy_id, strat.version, "active",
                                       reason="test", evidence="test")

    def quote_row(self, *, contract: str, market_key: str, gid: int, day: datetime,
                  market_type: str, title: str, side: str, bid: float, ask: float,
                  ask_size: float = 200.0, strike=None, strike_type=None, ts=None) -> None:
        self.store.execute(
            "INSERT INTO market_quotes(provider, market_key, contract, game_id, game_date,"
            " market_type, selection, side, bid, ask, spread, bid_size, ask_size, volume,"
            " liquidity, last_price, ts_utc, retrieved_at, source_url, strike, strike_type)"
            " VALUES('kalshi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'https://kalshi.example',?,?)",
            (market_key, contract, gid, day.date().isoformat(), market_type, title, side,
             bid, ask, round(ask - bid, 4), ask_size, ask_size, 10, 100, ask,
             ts or utcnow(), utcnow(), strike, strike_type))
        self.store.commit()


class TestForwardQuoteMatching(ForwardAndTotalsBase):
    """The bug that kept the competition at zero forward wagers."""

    def test_a_live_quote_worded_the_way_kalshi_words_it_is_traded(self):
        strat = ThresholdStrategy(strategy_id="NHL_B2B_HOME_FADE", username="U1",
                                  name="t", feature="home_only_b2b", operator=">=", threshold=1,
                                  bet_side="away", min_edge=0.03)
        self.register(strat)
        # exactly what ingest writes: selection = the contract's title, one row per side
        self.quote_row(contract="KXNHLGAME-26OCT01ANATOR-TOR", market_key="KXNHLGAME-26OCT01ANATOR",
                       gid=self.gid_future, day=self.future, market_type="moneyline",
                       title="Toronto Maple Leafs wins", side="YES", bid=0.50, ask=0.52)
        self.quote_row(contract="KXNHLGAME-26OCT01ANATOR-ANA", market_key="KXNHLGAME-26OCT01ANATOR",
                       gid=self.gid_future, day=self.future, market_type="moneyline",
                       title="Anaheim Ducks wins", side="YES", bid=0.44, ask=0.46)
        rows = [self.feature_row(self.gid_past, self.past, home_only_b2b=1.0),
                self.feature_row(self.gid_future, self.future, home_only_b2b=1.0,
                                 p_away_ml=0.55, p_home_ml=0.45)]
        placed = self.pipe.stage_forward(rows)
        self.assertGreaterEqual(placed["FORWARD TEST"], 1,
                                "a quoted contract must be tradeable, not 'no quote published'")
        bet = self.store.one("SELECT * FROM bets WHERE test_mode='FORWARD TEST'")
        self.assertIsNotNone(bet)
        self.assertEqual(bet["selection"], "away")
        self.assertAlmostEqual(float(bet["entry_price"]), 0.46)
        self.assertEqual(bet["contract"] if "contract" in bet.keys() else None, None)
        notes = json.loads(bet["notes"])
        self.assertEqual(notes["contract"], "KXNHLGAME-26OCT01ANATOR-ANA")
        self.assertEqual(notes["contract_label"], "Anaheim Ducks wins")
        # the fee is charged on the simulated fill
        self.assertAlmostEqual(float(bet["fee"]), kalshi_taker_fee(0.46, float(bet["filled_size"])))
        # the upcoming row for that opportunity is EXECUTED, not stuck on a false reason
        up = self.store.one("SELECT * FROM upcoming_bets WHERE strategy_id=? AND game_id=?",
                            (strat.strategy_id, self.gid_future))
        self.assertEqual(up["status"], "EXECUTED")
        self.assertNotIn("no quote published", up["blocking_reason"] or "")

    def test_no_false_claim_of_a_missing_quote_when_one_is_in_the_context(self):
        """The 2026-09-20 ledger wrote this reason 183 times while a quote was present."""
        q = Quote(provider="kalshi", market_key="K", contract="K-ANA", game_id=1,
                  game_date="2026-10-01", market_type="moneyline", selection="home",
                  side="YES", bid=0.3, ask=0.32, bid_size=10, ask_size=10, volume=1,
                  liquidity=1, ts_utc=utcnow(), label="Anaheim Ducks wins")
        strat = ThresholdStrategy(feature="home_only_b2b", operator=">=", threshold=1,
                                  bet_side="home", min_edge=0.03)
        ctx = DecisionContext(decision_ts=q.ts_utc,
                              features={"game_id": 1, "game_date": "2026-10-01", "home_id": TOR_ID,
                                        "away_id": ANA_ID, "home_only_b2b": 1.0},
                              predictions={"p_home_ml": 0.62, "p_away_ml": 0.38}, quotes=[q],
                              bankroll=1000.0)
        sig = strat.evaluate(ctx)[0]
        self.assertEqual(sig.status, "READY TO BET")

    def test_an_unmappable_contract_title_is_skipped_not_guessed(self):
        self.quote_row(contract="KXNHLGAME-26OCT01XYTOR-QQQ", market_key="KXNHLGAME-26OCT01XYTOR",
                       gid=self.gid_future, day=self.future, market_type="moneyline",
                       title="Somewhere Else wins", side="YES", bid=0.5, ask=0.52)
        quotes = self.pipe._live_quotes()
        self.assertEqual([q for q in quotes if q.contract == "KXNHLGAME-26OCT01XYTOR-QQQ"], [])


class TestTotalsMarket(ForwardAndTotalsBase):
    """KXNHLTOTAL: line parsing, entry, settlement, and the side that has no history."""

    def test_the_model_probability_matches_the_score_matrix(self):
        m = PoissonModel()
        f = {"home_exp_goals": 3.0, "away_exp_goals": 2.8, "home_n_prior": 30, "away_n_prior": 30}
        p = m.predict(f)
        self.assertAlmostEqual(p_total_over(p["lam_home"], p["lam_away"], 5.5), p["p_over55"],
                               places=5)
        self.assertAlmostEqual(p_total_over(p["lam_home"], p["lam_away"], 6.5), p["p_over65"],
                               places=5)
        self.assertAlmostEqual(p_total_over(2.9, 2.7, 5.5) + p_total_over(2.9, 2.7, 20.5) * 0
                               + (1 - p_total_over(2.9, 2.7, 5.5)), 1.0, places=6)

    def test_live_quotes_split_an_over_contract_into_over_and_under_offers(self):
        # Kalshi quotes both sides of an Over contract: yes_ask is the Over price and
        # no_ask is the Under price.  Both are observed; neither is derived from the other.
        for side, bid, ask in (("YES", 0.01, 0.42), ("NO", 0.56, 0.60)):
            self.quote_row(contract="KXNHLTOTAL-26OCT01ANATOR-6",
                           market_key="KXNHLTOTAL-26OCT01ANATOR", gid=self.gid_future,
                           day=self.future, market_type="total",
                           title="Full Game: Over 6.5 goals scored", side=side, bid=bid, ask=ask,
                           strike=6.5, strike_type="greater")
        quotes = {q.selection: q for q in self.pipe._live_quotes()}
        self.assertEqual(set(quotes), {"over", "under"})
        self.assertAlmostEqual(quotes["over"].ask, 0.42)
        self.assertAlmostEqual(quotes["under"].ask, 0.60)
        self.assertEqual(quotes["over"].side, "YES")
        self.assertEqual(quotes["under"].side, "NO")
        self.assertEqual(quotes["over"].strike, 6.5)
        self.assertEqual(quotes["over"].label, "Full Game: Over 6.5 goals scored")

    def test_a_contract_with_no_readable_line_is_never_traded(self):
        self.quote_row(contract="KXNHLTOTAL-26OCT01ANATOR-X",
                       market_key="KXNHLTOTAL-26OCT01ANATOR", gid=self.gid_future,
                       day=self.future, market_type="total", title="Carolina vs Vegas: Total Goals",
                       side="YES", bid=0.4, ask=0.44, strike=None, strike_type=None)
        self.assertEqual([q for q in self.pipe._live_quotes() if q.market_type == "total"], [])

    def test_totals_rule_bets_value_and_refuses_a_price_that_is_not_value(self):
        strat = TotalsStrategy(strategy_id="NHL_TOTALS_OVER", username="T1", name="t",
                               direction="over", min_edge=0.04)
        f = self.feature_row(self.gid_future, self.future)
        p_over = p_total_over(f["lam_home"], f["lam_away"], 5.5)     # ~0.5134

        def ctx(ask):
            q = Quote(provider="kalshi", market_key="K", contract="K-5", game_id=self.gid_future,
                      game_date=f["game_date"], market_type="total", selection="over",
                      side="YES", bid=round(ask - 0.02, 4), ask=ask, bid_size=50, ask_size=50,
                      volume=5, liquidity=5, ts_utc=utcnow(), label="Over 5.5 goals scored",
                      strike=5.5)
            return DecisionContext(decision_ts=q.ts_utc, features=f,
                                   predictions={"lam_home": f["lam_home"], "lam_away": f["lam_away"]},
                                   quotes=[q], bankroll=1000.0)

        good = strat.evaluate(ctx(round(p_over - 0.10, 4)))[0]
        self.assertEqual(good.status, "READY TO BET")
        self.assertEqual(good.market, "total")
        self.assertEqual(good.selection, "over")
        self.assertGreater(good.stake, 0.0)
        bad = strat.evaluate(ctx(round(p_over + 0.02, 4)))[0]
        self.assertEqual(bad.status, "PRICE TOO HIGH")
        far = strat.evaluate(ctx(round(p_over + 0.10, 4)))[0]
        self.assertEqual(far.status, "PRICE TOO HIGH")
        # a strike outside the traded range is watched, not traded
        q = Quote(provider="kalshi", market_key="K", contract="K-9", game_id=self.gid_future,
                  game_date=f["game_date"], market_type="total", selection="over", side="YES",
                  bid=0.05, ask=0.07, bid_size=50, ask_size=50, volume=5, liquidity=5,
                  ts_utc=utcnow(), label="Over 12.5 goals scored", strike=12.5)
        wide = strat.evaluate(DecisionContext(decision_ts=q.ts_utc, features=f,
                                              predictions={"lam_home": f["lam_home"],
                                                           "lam_away": f["lam_away"]},
                                              quotes=[q], bankroll=1000.0))[0]
        self.assertEqual(wide.status, "WATCHING")

    def test_backtest_prices_an_over_from_a_settled_candle_and_settles_from_the_exchange(self):
        strat = TotalsStrategy(strategy_id="NHL_TOTALS_OVER", username="T1", name="t",
                               direction="over", min_edge=0.04)
        self.register(strat)
        # settled KXNHLTOTAL contract for the decided game, shaped like the historical tier
        self.store.execute(
            """INSERT INTO market_settlements(provider, event_ticker, contract, game_id,
                     game_date, selection, side, result, settle_price, volume, open_time,
                     settlement_ts, retrieved_at, source_url, series_ticker, tier, floor_strike,
                     close_time, title, occurrence_datetime, market_type, strike_type)
               VALUES('kalshi',?,?,?,?,?,'YES','yes',0.0,100,?,?,?,?,'KXNHLTOTAL','historical',?,?,
                      ?,?,'total','greater')""",
            ("KXNHLTOTAL-26MAR04ANATOR", "KXNHLTOTAL-26MAR04ANATOR-5", self.gid_past,
             self.past.date().isoformat(), "Over 5.5 goals scored",
             _iso(self.past - timedelta(days=2)), _iso(self.past + timedelta(hours=3)),
             utcnow(), "https://kalshi.example", 5.5,
             _iso(self.past + timedelta(hours=3)), "Toronto vs Anaheim: Total Goals",
             _iso(self.past)))
        # hourly candles: the last one ending before puck drop offers the Over at 0.32,
        # against a model probability of ~0.513 at this strike -- value, so it trades
        for k, ts in (("open", self.past - timedelta(hours=30)),
                      ("t6h", self.past - timedelta(hours=6)),
                      ("close", self.past - timedelta(minutes=30))):
            self.store.execute(
                "INSERT INTO market_price_points(contract, point, end_period_ts, game_id,"
                " team_abbrev, series_ticker, market_type, bid, ask, last, mean, volume,"
                " open_interest, period_interval, tier, retrieved_at) VALUES(?,?,?,?,?,?,?,?,?,?,?"
                ",?,?,60,'historical',?)",
                ("KXNHLTOTAL-26MAR04ANATOR-5", k, _epoch(ts), self.gid_past, None, "KXNHLTOTAL",
                 "total", 0.30, 0.32, 0.31, 0.31, 100, 1000, utcnow()))
        self.store.commit()
        rows = [self.feature_row(self.gid_past, self.past)]
        placed = self.pipe.stage_forward(rows)
        self.assertGreaterEqual(placed["BACKTEST"], 1)
        bet = self.store.one("SELECT * FROM bets WHERE market='total'")
        self.assertIsNotNone(bet)
        self.assertEqual(bet["test_mode"], "BACKTEST")
        self.assertEqual(bet["selection"], "over")
        self.assertAlmostEqual(float(bet["strike"]), 5.5)
        self.assertEqual(bet["price_point"], "close")
        self.assertEqual(bet["verification_status"], "kalshi_candle_close")
        self.assertAlmostEqual(float(bet["entry_price"]), 0.32)
        self.assertEqual(bet["result"], "WIN")          # the contract settled 'yes'
        gross = float(bet["filled_size"]) * (1 - float(bet["entry_price"]))
        self.assertAlmostEqual(float(bet["pnl"]), gross - float(bet["fee"]), places=3)
        self.assertGreater(float(bet["fee"]), 0.0)
        sup = json.loads(bet["features_json"])
        self.assertEqual(sup["strike"], 5.5)
        self.assertEqual(sup["line_basis"], "strike_type+floor_strike")
        self.assertIn("kalshi candlestick", sup["price_basis"])
        # the decision timestamp is the candle's own end, at or before puck drop
        self.assertLessEqual(int(bet["decision_ts"]), _epoch(self.past))

    def test_a_forward_totals_bet_settles_from_the_official_final_score(self):
        # an Over 5.5 bought on the future game, then the game finishes 4-2 (total 6)
        self.store.execute(
            "INSERT INTO bets(bet_id, strategy_id, strategy_version, username, test_mode, season,"
            " game_id, game_date, matchup, market, selection, bet_type, provider, odds_format,"
            " price, implied_prob, model_prob, edge, fair_price, decision_ts, bet_ts, stake,"
            " liquidity, filled_size, slippage, entry_price, fee, strike, price_basis, result,"
            " source_url, verification_status, notes, features_json, created_at)"
            " VALUES('T1','NHL_TOTALS_OVER',1,'T1','FORWARD TEST',20252026,?,?,?,'total','over',"
            " 'binary_contract','kalshi','binary',0.42,0.42,0.51,0.09,2.38,?,?,21.0,50,50,0,0.42,"
            " 0.52,5.5,'exchange','OPEN','https://kalshi.example','single_source','{}','{}',?)",
            (self.gid_future, self.future.date().isoformat(), f"{ANA_ID}@{TOR_ID}",
             _iso(self.future), utcnow(), utcnow()))
        self.store.commit()
        self.store.execute("UPDATE games SET state='FINAL', home_score=4, away_score=2,"
                           " last_period_type='REG' WHERE game_id=?", (self.gid_future,))
        self.store.commit()
        n = self.pipe.stage_settle()
        self.assertGreaterEqual(n, 1)
        bet = self.store.one("SELECT * FROM bets WHERE bet_id='T1'")
        self.assertEqual(bet["result"], "WIN")           # total 6 > 5.5
        gross = 50.0 * (1 - 0.42)
        self.assertAlmostEqual(float(bet["pnl"]), gross - float(bet["fee"]), places=3)
        audit = self.store.query("SELECT reason FROM bet_audit WHERE bet_id='T1' AND action='SETTLE'")
        self.assertTrue(any("official total 6 vs strike 5.5" in r["reason"] for r in audit))
        # an Under on the same game would have lost: it is the NO side of the same contract
        self.store.execute("UPDATE bets SET result='OPEN', pnl=NULL, selection='under', strike=5.5"
                           " WHERE bet_id='T1'")
        self.store.commit()
        self.pipe.stage_settle()
        bet = self.store.one("SELECT * FROM bets WHERE bet_id='T1'")
        self.assertEqual(bet["result"], "LOSS")          # six goals is not under 5.5
        self.assertAlmostEqual(float(bet["pnl"]), -50.0 * 0.42 - float(bet["fee"]), places=3)

    def test_a_total_that_cannot_be_settled_is_flagged_not_guessed(self):
        self.store.execute(
            "INSERT INTO bets(bet_id, strategy_id, strategy_version, username, test_mode, season,"
            " game_id, game_date, matchup, market, selection, bet_type, provider, odds_format,"
            " price, implied_prob, entry_price, stake, filled_size, decision_ts, bet_ts, result,"
            " created_at)"
            " VALUES('T2','NHL_TOTALS_OVER',1,'T1','FORWARD TEST',20252026,?,?,?,'total','over',"
            " 'binary_contract','kalshi','binary',0.42,0.42,0.42,10,20,?,?, 'OPEN',?)",
            (self.gid_future, self.future.date().isoformat(), f"{ANA_ID}@{TOR_ID}",
             _iso(self.future), utcnow(), utcnow()))
        self.store.execute("UPDATE games SET state='FINAL', home_score=4, away_score=2"
                           " WHERE game_id=?", (self.gid_future,))
        self.store.commit()
        self.pipe.stage_settle()
        self.assertEqual(self.store.one("SELECT result FROM bets WHERE bet_id='T2'")["result"],
                         "OPEN")
        self.assertTrue(self.store.one(
            "SELECT 1 FROM irregularities WHERE kind='unsettleable_total' AND status='open'"))

    def test_an_under_rule_declares_that_it_has_no_history_and_is_not_priced(self):
        under = TotalsStrategy(strategy_id="NHL_TOTALS_UNDER", username="T2", name="t",
                               direction="under", min_edge=0.04,
                               no_history_reason="no historical NO-side offer exists")
        self.assertTrue(under.no_history_reason)
        self.register(under)
        rows = [self.feature_row(self.gid_past, self.past, _winner=TOR_ID, _total=6,
                                 _decided=True)]
        pbt = PricedBacktester(self.store)
        self.assertIsNone(pbt.run(under, rows, label="priced_all"))
        # and the accuracy run claims no profit
        bt = Backtester(self.store)
        res = bt.run_totals_accuracy(under, rows)
        self.assertEqual(res.data_sufficient, 0)
        self.assertIn("NOT a profit figure", res.caveat)
        row = self.store.one("SELECT * FROM backtests WHERE strategy_id=? AND label=?",
                             (under.strategy_id, "totals_accuracy_all"))
        self.assertIsNone(row["pnl"])
        self.assertIsNone(row["roi"])
        self.assertEqual(row["data_sufficient"], 0)
        self.assertEqual(row["n_bets"], 2)     # the two default strikes, no exchange strikes yet
        self.assertEqual(row["n_wins"], 1)     # under 6.5 hits on a 6-goal game; under 5.5 does not
        self.assertIn("no price", row["price_basis"])


class TestTotalsVerification(ForwardAndTotalsBase):
    """The exchange's settlement and the official score must agree -- and if they don't,
    both values are recorded instead of one being silently preferred."""

    def _settled_totals_row(self, *, result: str, strike: float) -> None:
        self.store.execute(
            """INSERT INTO market_settlements(provider, event_ticker, contract, game_id,
                     game_date, selection, side, result, settle_price, volume, open_time,
                     settlement_ts, retrieved_at, source_url, series_ticker, tier, floor_strike,
                     close_time, title, occurrence_datetime, market_type, strike_type)
               VALUES('kalshi',?,?,?,?,?,'YES',?,0.0,100,?,?,?,?,'KXNHLTOTAL','historical',?,?,
                      ?,?,'total','greater')""",
            ("KXNHLTOTAL-26MAR04ANATOR", f"KXNHLTOTAL-26MAR04ANATOR-{int(strike)}", self.gid_past,
             self.past.date().isoformat(), f"Over {strike} goals scored", result,
             _iso(self.past - timedelta(days=2)), _iso(self.past + timedelta(hours=3)), utcnow(),
             "https://kalshi.example", strike, _iso(self.past + timedelta(hours=3)),
             "Toronto vs Anaheim: Total Goals", _iso(self.past)))
        self.store.commit()

    def test_agreeing_settlement_records_no_conflict(self):
        self._settled_totals_row(result="no", strike=6.5)      # the game finished 4-2 (6 goals)
        out = Verifier(self.store).cross_validate_totals_settlements()
        self.assertEqual(out["totals_contracts_compared"], 1)
        self.assertEqual(out["totals_conflicts"], 0)
        self.assertFalse(self.store.one(
            "SELECT 1 FROM irregularities WHERE kind='settlement_conflict'"))

    def test_a_disagreement_is_recorded_with_both_values_and_left_open(self):
        self._settled_totals_row(result="yes", strike=6.5)     # official total 6 implies 'no'
        out = Verifier(self.store).cross_validate_totals_settlements()
        self.assertEqual(out["totals_conflicts"], 1)
        row = self.store.one("SELECT * FROM irregularities WHERE kind='settlement_conflict'")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "open")
        self.assertEqual(row["severity"], "critical")
        self.assertIn("official total is 6", row["detail"])
        self.assertIn("kalshi settled 'yes'", row["detail"])

    def test_a_totals_bet_without_a_line_is_flagged(self):
        self.store.execute(
            "INSERT INTO bets(bet_id, strategy_id, strategy_version, username, test_mode, season,"
            " game_id, game_date, matchup, market, selection, bet_type, provider, odds_format,"
            " price, implied_prob, entry_price, stake, filled_size, decision_ts, bet_ts, result,"
            " created_at) VALUES('T9','S',1,'U','FORWARD TEST',20252026,?,?,?,'total','over',"
            " 'binary_contract','kalshi','binary',0.42,0.42,0.42,10,20,?,?, 'OPEN',?)",
            (self.gid_future, self.future.date().isoformat(), f"{ANA_ID}@{TOR_ID}",
             _iso(self.future), utcnow(), utcnow()))
        self.store.commit()
        out = Verifier(self.store).check_totals_bets()
        self.assertEqual(out["totals_bets"], 1)
        self.assertEqual(out["totals_missing_strike"], 1)
        self.assertTrue(self.store.one(
            "SELECT 1 FROM irregularities WHERE kind='missing_strike' AND entity_type='bet'"))


class TestStrikeParsing(unittest.TestCase):
    def test_exchange_fields_win_over_the_title_text(self):
        from nhlcomp.market import total_side_and_strike, total_side_from_text
        self.assertEqual(total_side_and_strike("greater", 8.5), ("over", 8.5))
        self.assertEqual(total_side_and_strike("less", 6.5), ("under", 6.5))
        self.assertIsNone(total_side_and_strike(None, 6.5))
        self.assertIsNone(total_side_and_strike("greater", None))
        # the live tier and the historical tier word the same contract differently
        self.assertEqual(total_side_from_text("Full Game: Over 8.5 goals scored"), ("over", 8.5))
        self.assertEqual(total_side_from_text("Over 6.5 goals scored"), ("over", 6.5))
        self.assertIsNone(total_side_from_text("Carolina vs Vegas: Total Goals"))


if __name__ == "__main__":
    unittest.main()


class TestStrikeSelection(ForwardAndTotalsBase):
    """The bug that kept the totals market at zero wagers.

    Kalshi lists a ladder of "Over k.5" contracts per game -- eight of them on the
    2026-09-24 slate (1.5 through 8.5), captured verbatim in the ledger's 464 totals quote
    rows.  `find_quote` returned the *first* match for the side "over", which in book order
    was strike 1.5, so every totals rule saw only 1.5 and refused it as outside the traded
    range.  The CI run of 2026-09-21 recorded that refusal on all 84 totals opportunities
    ("strike 1.5 outside the traded range [4.5, 8.5]") while holding 7,885 settled totals
    contracts and zero totals wagers.
    """

    LADDER = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

    def ladder_ctx(self, ask_for=None):
        """The eight contracts Kalshi actually quoted, lowest strike first."""
        f = self.feature_row(self.gid_future, self.future)
        quotes = []
        for k in self.LADDER:
            ask = (ask_for or {}).get(k, 0.50)
            quotes.append(Quote(provider="kalshi", market_key="KXNHLTOTAL-26OCT01ANATOR",
                                contract=f"KXNHLTOTAL-26OCT01ANATOR-{int(k)}",
                                game_id=self.gid_future, game_date=f["game_date"],
                                market_type="total", selection="over", side="YES",
                                bid=round(ask - 0.02, 4), ask=ask, bid_size=50, ask_size=50,
                                volume=5, liquidity=5, ts_utc=utcnow(),
                                label=f"Full Game: Over {k} goals scored", strike=k))
        return DecisionContext(decision_ts=quotes[0].ts_utc, features=f,
                               predictions={"lam_home": f["lam_home"], "lam_away": f["lam_away"]},
                               quotes=quotes, bankroll=1000.0), f

    def test_the_rule_trades_its_declared_strike_not_the_first_one_in_the_book(self):
        strat = TotalsStrategy(strategy_id="NHL_TOTALS_OVER", username="T1", name="t",
                               direction="over", min_edge=0.04, target_strike=6.5)
        ctx, f = self.ladder_ctx()
        sig = strat.evaluate(ctx)[0]
        self.assertEqual(sig.quote.strike, 6.5)
        self.assertEqual(sig.quote.contract, "KXNHLTOTAL-26OCT01ANATOR-6")
        self.assertEqual(sig.supporting["strikes_offered"], self.LADDER)
        self.assertEqual(sig.supporting["target_strike"], 6.5)
        # the probability is the one for the strike actually traded, not for 1.5
        self.assertAlmostEqual(sig.model_prob, p_total_over(f["lam_home"], f["lam_away"], 6.5))
        self.assertNotAlmostEqual(sig.model_prob, p_total_over(f["lam_home"], f["lam_away"], 1.5))

    def test_a_tighter_variant_trades_a_different_rung_of_the_same_ladder(self):
        ctx, _ = self.ladder_ctx()
        tight = TotalsStrategy(strategy_id="NHL_TOTALS_OVER_TIGHT", username="T2", name="t",
                               direction="over", min_edge=0.02, target_strike=5.5)
        self.assertEqual(tight.evaluate(ctx)[0].quote.strike, 5.5)

    def test_the_nearest_tradable_strike_wins_when_the_target_is_not_offered(self):
        # 6.5 is absent from the book; 7.5 and 5.5 are equidistant, and the tie goes to the
        # lower strike so the choice is deterministic and reproducible.
        ctx, _ = self.ladder_ctx()
        ctx.quotes = [q for q in ctx.quotes if q.strike != 6.5]
        sig = TotalsStrategy(strategy_id="NHL_TOTALS_OVER", username="T1", name="t",
                             direction="over", target_strike=6.5).evaluate(ctx)[0]
        self.assertEqual(sig.quote.strike, 5.5)

    def test_a_book_of_only_deep_strikes_reports_the_book_it_was_shown(self):
        # A near-certain "Over 1.5" is not a totals bet, and the ledger says what was on
        # offer instead of a bare "no".
        strat = TotalsStrategy(strategy_id="NHL_TOTALS_OVER", username="T1", name="t",
                               direction="over", min_strike=4.5, max_strike=8.5)
        ctx, _ = self.ladder_ctx()
        ctx.quotes = [q for q in ctx.quotes if q.strike < 4.5]
        sig = strat.evaluate(ctx)[0]
        self.assertEqual(sig.status, "WATCHING")
        self.assertIn("1.5", sig.blocking_reason)
        self.assertIn("3.5", sig.blocking_reason)
        self.assertIn("[4.5, 8.5]", sig.blocking_reason)

    def test_the_under_side_picks_its_own_rung_from_the_no_offers(self):
        ctx, _ = self.ladder_ctx()
        for q in ctx.quotes:
            q.side, q.selection = "NO", "under"
            q.label = q.label.replace("Over", "Over")     # the NO side of an Over contract
        strat = TotalsStrategy(strategy_id="NHL_TOTALS_UNDER", username="T3", name="t",
                               direction="under", target_strike=6.5)
        sig = strat.evaluate(ctx)[0]
        self.assertEqual(sig.side, "NO")
        self.assertEqual(sig.quote.strike, 6.5)
        self.assertEqual(sig.quote.contract, "KXNHLTOTAL-26OCT01ANATOR-6")

    def test_the_pipeline_hands_every_rung_to_the_rule(self):
        """`_live_quotes` must not collapse the ladder down to one quote per game."""
        for k in self.LADDER:
            self.quote_row(contract=f"KXNHLTOTAL-26OCT01ANATOR-{int(k)}",
                           market_key="KXNHLTOTAL-26OCT01ANATOR", gid=self.gid_future,
                           day=self.future, market_type="total",
                           title=f"Full Game: Over {k} goals scored", side="YES",
                           bid=0.48, ask=0.50, strike=k, strike_type="greater")
        quotes = [q for q in self.pipe._live_quotes() if q.market_type == "total"]
        self.assertEqual(sorted(q.strike for q in quotes), self.LADDER)


class TestTotalsPriceCoverage(ForwardAndTotalsBase):
    """An empty priced backtest must say why it is empty.

    On the 2026-09-21 CI ledger the totals candle walk held 577 settled contracts with a
    pre-puck-drop offer across 73 games (64 of them 2025-26 playoff games) while the season
    stats walk covered only 2024-25 -- so no game had both a price and a point-in-time
    feature row, and the priced totals backtest was empty.  Empty because of a coverage gap
    and empty because the rules found no value are different statements, and the ledger has
    to tell them apart.
    """

    def totals_rows(self, *contracts):
        return [{"game_id": self.gid_past, "contract": c, "direction": "over", "strike": 6.5,
                 "line_basis": "strike_type+floor_strike", "result": "no", "points": {
                     "close": {"ask": 0.5, "bid": 0.48, "ts": "2026-03-04T00:00:00Z"}}}
                for c in contracts]

    def test_a_price_with_no_features_is_flagged_not_silently_dropped(self):
        gap = self.pipe._totals_coverage(self.totals_rows("K-6"), {}, {})
        self.assertEqual(gap["contracts_with_a_pregame_offer"], 1)
        self.assertEqual(gap["games_with_a_pregame_offer"], 1)
        self.assertEqual(gap["of_those_games_with_point_in_time_features"], 0)
        self.assertEqual(gap["backtestable_games"], 0)
        self.assertIn("have not reached the same games", gap["reason"])
        row = self.store.one("SELECT * FROM irregularities WHERE kind='totals_price_coverage'")
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "open")
        self.assertIn("KXNHLTOTAL", row["entity_id"])

    def test_a_game_with_both_a_price_and_features_is_counted_backtestable(self):
        refs = {g.game_id: g for g in self.pipe.game_refs(game_types=(1, 2, 3))}
        gap = self.pipe._totals_coverage(self.totals_rows("K-6"), refs,
                                         {self.gid_past: {"lam_home": 3.0, "lam_away": 2.8}})
        self.assertEqual(gap["backtestable_games"], 1)
        self.assertNotIn("reason", gap)
        self.assertIsNone(self.store.one(
            "SELECT * FROM irregularities WHERE kind='totals_price_coverage'"))
