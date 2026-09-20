"""Tests for point-in-time features, models, execution, backtesting and verification."""

import os
import tempfile
import unittest
from datetime import datetime, timezone

from nhlcomp.backtest import Backtester, CAVEAT_NO_PRICE
from nhlcomp.features import FeatureBuilder, GameRef
from nhlcomp.models import (EloModel, HomeIceOnly, LogisticRest, PoissonModel, brier, log_loss,
                            wilson_interval)
from nhlcomp.paper import PaperEngine, binary_settlement, simulate_fill
from nhlcomp.store import Store, utcnow
from nhlcomp.strategies import DecisionContext, Quote, ThresholdStrategy
from nhlcomp.verify import Verifier


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def game(gid, start, home, away, hs=None, as_=None, off="-05:00", period="REG", season=20252026,
         gtype=2):
    return GameRef(game_id=gid, start=dt(start), game_date=start[:10], home_id=home,
                   away_id=away, venue=f"Arena{home}", venue_tz=None, utc_offset=off,
                   season=season, game_type=gtype, home_score=hs, away_score=as_,
                   last_period_type=period if hs is not None else None)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Team 1 plays every other day; team 2 has a back-to-back right before game 100.
        self.games = [
            game(1, "2026-01-01T00:00:00Z", 1, 3, 3, 1),
            game(2, "2026-01-03T00:00:00Z", 2, 4, 2, 2, period="OT"),
            game(3, "2026-01-05T00:00:00Z", 1, 4, 1, 4),
            game(4, "2026-01-07T00:00:00Z", 5, 2, 2, 5),
            game(5, "2026-01-08T00:00:00Z", 2, 6, 3, 0),          # team 2 back-to-back
            game(100, "2026-01-10T00:00:00Z", 1, 2, None, None),  # the game we predict
        ]
        self.fb = FeatureBuilder(self.games)

    def test_rest_and_back_to_back(self):
        f = self.fb.pair_features(self.games[-1])
        self.assertAlmostEqual(f["home_rest_days"], 5.0)
        self.assertAlmostEqual(f["away_rest_days"], 2.0)
        self.assertEqual(f["away_back_to_back"], 0)
        self.assertEqual(f["home_back_to_back"], 0)
        self.assertEqual(f["rest_diff"], 3.0)

    def test_back_to_back_is_detected(self):
        g = self.games[4]                      # team 2's second night
        f = self.fb.pair_features(g)
        self.assertEqual(f["home_back_to_back"], 1)
        self.assertEqual(f["home_calendar_rest_days"], 1)
        # team 2 has only two prior games (Jan 3 and Jan 7), so a 3-in-4 has not happened
        self.assertEqual(f["home_g3_in_4"], 0)

    def test_three_in_four_nights_is_detected(self):
        games = [game(1, "2026-01-01T00:00:00Z", 1, 3, 3, 1),
                 game(2, "2026-01-02T00:00:00Z", 4, 1, 1, 2),
                 game(3, "2026-01-04T00:00:00Z", 1, 5, 2, 1),
                 game(4, "2026-01-06T00:00:00Z", 6, 1, None, None)]
        f = FeatureBuilder(games).pair_features(games[3])
        self.assertEqual(f["away_g3_in_4"], 1)
        self.assertEqual(f["away_g4_in_6"], 1)

    def test_no_future_information_leaks(self):
        """The prediction for game 100 must not include game 100's own result."""
        g = self.games[-1]
        f = self.fb.pair_features(g)
        # team 1 has exactly 2 prior decided games (1 and 3), not 3
        self.assertEqual(f["home_n_prior"], 2)
        self.assertEqual(f["away_n_prior"], 3)
        # goals-for average must exclude game 100 entirely
        self.assertAlmostEqual(f["home_gf_avg"], (3 + 1) / 2, places=4)
        self.assertAlmostEqual(f["away_gf_avg"], (5 + 3 + 2) / 3, places=4)

    def test_travel_uses_published_offsets_not_invented_coordinates(self):
        games = [game(1, "2026-01-01T00:00:00Z", 1, 3, 3, 1, off="-05:00"),
                 game(2, "2026-01-05T00:00:00Z", 1, 4, 3, 1, off="-08:00")]
        fb = FeatureBuilder(games)
        f = fb.pair_features(games[1])
        self.assertEqual(f["home_tz_shift"], -3.0)
        self.assertEqual(f["home_travel_direction"], "west")

    def test_prev_overtime_flag(self):
        games = [game(1, "2026-01-01T00:00:00Z", 2, 3, 3, 2, period="OT"),
                 game(2, "2026-01-03T00:00:00Z", 4, 2, None, None)]
        f = FeatureBuilder(games).pair_features(games[1])
        self.assertEqual(f["away_prev_ot"], 1)

    def test_missing_offset_yields_none_not_zero(self):
        games = [game(1, "2026-01-01T00:00:00Z", 1, 3, 3, 1, off=None),
                 game(2, "2026-01-05T00:00:00Z", 1, 4, 3, 1, off="-05:00")]
        f = FeatureBuilder(games).pair_features(games[1])
        self.assertIsNone(f["home_tz_shift"])


class TestModels(unittest.TestCase):
    def test_poisson_probabilities_are_consistent(self):
        p = PoissonModel().predict({"home_exp_goals": 3.0, "away_exp_goals": 2.5,
                                    "home_n_prior": 40, "away_n_prior": 40})
        self.assertAlmostEqual(p["p_home_reg"] + p["p_away_reg"] + p["p_tie_reg"], 1.0, places=6)
        self.assertAlmostEqual(p["p_home_ml"] + p["p_away_ml"], 1.0, places=6)
        self.assertTrue(0 < p["p_home_ml"] < 1)
        self.assertAlmostEqual(p["p_over55"] + p["p_under55"], 1.0, places=6)

    def test_poisson_shrinks_to_league_average_on_thin_samples(self):
        m = PoissonModel(league_home_gf=2.9, league_away_gf=2.7)
        p = m.predict({"home_exp_goals": 6.0, "away_exp_goals": 0.5,
                       "home_n_prior": 0, "away_n_prior": 0})
        self.assertLess(p["lam_home"], 3.5)
        self.assertGreater(p["lam_away"], 2.0)

    def test_elo_moves_toward_winner_and_respects_home_advantage(self):
        e = EloModel()
        self.assertGreater(e.prob_home(1, 2), 0.5)          # home advantage alone
        for _ in range(5):
            e.observe(game(0, "2026-01-01T00:00:00Z", 1, 2, 4, 1))
        self.assertGreater(e.rating(1), e.rating(2))
        self.assertGreater(e.prob_home(1, 2), 0.5)

    def test_logistic_learns_a_separable_pattern(self):
        rows = [{"elo_diff": 1.0, "rest_diff": 2.0, "home_only_b2b": 0.0, "away_only_b2b": 0.0},
                {"elo_diff": -1.0, "rest_diff": -2.0, "home_only_b2b": 1.0, "away_only_b2b": 0.0}]
        labels = [1, 0]
        m = LogisticRest().fit(rows * 40, labels * 40)
        self.assertGreater(m.predict_p(rows[0]), 0.5)
        self.assertLess(m.predict_p(rows[1]), 0.5)

    def test_wilson_interval_is_wide_for_small_n(self):
        lo, hi = wilson_interval(2, 3)
        self.assertGreater(hi - lo, 0.4)
        lo, hi = wilson_interval(60, 100)
        self.assertLess(hi - lo, 0.2)

    def test_log_loss_and_brier_penalty(self):
        self.assertGreater(log_loss([0.9, 0.9], [0, 0]), log_loss([0.9, 0.9], [1, 1]))
        self.assertAlmostEqual(brier([0.5, 0.5], [1, 0]), 0.25)


class TestExecution(unittest.TestCase):
    def test_partial_fill_when_stake_exceeds_liquidity(self):
        f = simulate_fill(stake=500.0, ask=0.50, ask_size=200.0)
        self.assertEqual(f.contracts_filled, 200.0)
        self.assertAlmostEqual(f.stake, 100.0)
        self.assertAlmostEqual(f.unfilled_stake, 400.0)

    def test_full_fill_when_liquidity_is_sufficient(self):
        f = simulate_fill(stake=50.0, ask=0.50, ask_size=200.0)
        self.assertEqual(f.contracts_filled, 100.0)
        self.assertAlmostEqual(f.unfilled_stake, 0.0)

    def test_no_fill_on_zero_size_offer(self):
        f = simulate_fill(stake=50.0, ask=0.50, ask_size=0.0)
        self.assertEqual(f.contracts_filled, 0.0)
        self.assertAlmostEqual(f.stake, 0.0)

    def test_binary_settlement_math(self):
        self.assertAlmostEqual(binary_settlement(0.40, 100.0, True), 60.0)
        self.assertAlmostEqual(binary_settlement(0.40, 100.0, False), -40.0)
        self.assertAlmostEqual(binary_settlement(0.40, 100.0, True) +
                               binary_settlement(0.40, 100.0, False), 20.0)

    def test_kelly_is_capped(self):
        s = ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f",
                              stake_fraction=1.0, max_stake_pct=0.10)
        stake = s.size_stake(p=0.95, price=0.30, bankroll=1000.0, open_exposure=0.0)
        self.assertLessEqual(stake, 100.0)

    def test_kelly_refuses_a_negative_edge(self):
        s = ThresholdStrategy(strategy_id="X", version=1, username="X_001", feature="f")
        self.assertEqual(s.kelly(p=0.30, price=0.60), 0.0)


class _StoreTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.s = Store(self.path)
        for tid, ab in ((1, "AAA"), (2, "BBB"), (3, "CCC")):
            self.s.execute("INSERT INTO teams(team_id, abbrev, full_name) VALUES(?,?,?)",
                           (tid, ab, f"Team {ab}"))
        self.s.commit()

    def tearDown(self):
        self.s.close()


class TestPaperEngine(_StoreTestCase):
    def _sig(self, ask=0.50, size=200.0, stake=50.0):
        q = Quote(provider="kalshi", market_key="K", contract="K-A", game_id=1,
                  game_date="2026-01-01", market_type="moneyline", selection="home", side="YES",
                  bid=0.45, ask=ask, bid_size=100.0, ask_size=size, volume=10.0, liquidity=None,
                  ts_utc=utcnow())
        return ThresholdStrategy(strategy_id="S", version=1, username="S_001", feature="f").__class__(
            strategy_id="S", version=1, username="S_001", feature="f"), q

    def test_ready_to_bet_places_and_settles_a_win(self):
        from nhlcomp.strategies import Signal
        strat, q = self._sig()
        self.s.execute("INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
                       " home_id, away_id, state) VALUES(1,20252026,2,'2026-01-01',"
                       "'2026-01-01T00:00:00Z',1,2,'FUTURE')")
        self.s.commit()
        sig = Signal(strategy_id="S", version=1, username="S_001", game_id=1,
                     game_date="2026-01-01", matchup="2@1", market="moneyline", selection="home",
                     side="YES", model_prob=0.65, fair_price=0.50, required_price=0.60,
                     stake=50.0, status="READY TO BET", quote=q)
        eng = PaperEngine(self.s)
        bid = eng.place(sig, decision_ts=utcnow(), test_mode="FORWARD TEST")
        self.assertIsNotNone(bid)
        row = self.s.one("SELECT * FROM bets WHERE bet_id=?", (bid,))
        self.assertEqual(row["result"], "OPEN")
        self.assertAlmostEqual(float(row["stake"]), 50.0)
        n = eng.settle_game(1, home_id=1, away_id=2, home_score=3, away_score=1, state="FINAL",
                            last_period_type="REG")
        self.assertEqual(n, 1)
        row = self.s.one("SELECT result, pnl FROM bets WHERE bet_id=?", (bid,))
        self.assertEqual(row["result"], "WIN")
        self.assertAlmostEqual(float(row["pnl"]), 50.0)

    def test_tied_final_score_is_flagged_and_not_settled(self):
        from nhlcomp.strategies import Signal
        strat, q = self._sig()
        sig = Signal(strategy_id="S", version=1, username="S_001", game_id=2,
                     game_date="2026-01-02", matchup="2@1", market="moneyline", selection="home",
                     side="YES", model_prob=0.6, fair_price=0.5, required_price=0.6,
                     stake=10.0, status="READY TO BET", quote=q)
        PaperEngine(self.s).place(sig, decision_ts=utcnow(), test_mode="FORWARD TEST")
        n = PaperEngine(self.s).settle_game(2, home_id=1, away_id=2, home_score=3, away_score=3,
                                            state="FINAL", last_period_type="REG")
        self.assertEqual(n, 0)
        irr = self.s.one("SELECT kind FROM irregularities WHERE kind='impossible_result'")
        self.assertIsNotNone(irr)

    def test_non_ready_signals_never_become_bets(self):
        from nhlcomp.strategies import Signal
        strat, q = self._sig()
        sig = Signal(strategy_id="S", version=1, username="S_001", game_id=3,
                     game_date="2026-01-03", matchup="2@1", market="moneyline", selection="home",
                     side="YES", model_prob=0.6, fair_price=0.5, required_price=0.6,
                     stake=10.0, status="PRICE TOO HIGH", blocking_reason="ask too high", quote=q)
        self.assertIsNone(PaperEngine(self.s).place(sig, decision_ts=utcnow(),
                                                    test_mode="FORWARD TEST"))
        self.assertEqual(self.s.one("SELECT COUNT(*) c FROM bets")["c"], 0)


class TestVerifier(_StoreTestCase):
    def test_detects_tied_final_and_impossible_score(self):
        self.s.execute("INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
                       " home_id, away_id, state, home_score, away_score) VALUES"
                       "(1,20252026,2,'2026-01-01','2026-01-01T00:00:00Z',1,2,'FINAL',3,3)")
        self.s.execute("INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
                       " home_id, away_id, state, home_score, away_score) VALUES"
                       "(2,20252026,2,'2026-01-02','2026-01-02T00:00:00Z',1,2,'FINAL',40,1)")
        self.s.commit()
        c = Verifier(self.s).check_games()
        self.assertEqual(c["tied_final"], 1)
        self.assertEqual(c["impossible_score"], 1)

    def test_detects_duplicate_games(self):
        for gid in (1, 2):
            self.s.execute("INSERT INTO games(game_id, season, game_type, game_date,"
                           " start_time_utc, home_id, away_id, state) VALUES"
                           "(?,20252026,2,'2026-01-01','2026-01-01T00:00:00Z',1,2,'FINAL')",
                           (gid,))
        self.s.commit()
        self.assertEqual(Verifier(self.s).check_games()["duplicate"], 1)

    def test_detects_crossed_book_and_zero_liquidity(self):
        self.s.execute("INSERT INTO market_quotes(provider, market_key, contract, market_type,"
                       " selection, side, bid, ask, ts_utc, retrieved_at) VALUES"
                       "('kalshi','K','K-A','moneyline','home','YES',0.60,0.40,?,?)",
                       (utcnow(), utcnow()))
        self.s.execute("INSERT INTO market_quotes(provider, market_key, contract, market_type,"
                       " selection, side, bid, ask, bid_size, ask_size, ts_utc, retrieved_at)"
                       " VALUES"
                       "('kalshi','K','K-B','moneyline','away','YES',0.10,0.20,0,0,?,?)",
                       (utcnow(), utcnow()))
        self.s.commit()
        c = Verifier(self.s).check_quotes()
        self.assertEqual(c["crossed_book"], 1)
        self.assertEqual(c["zero_liquidity"], 1)

    def test_detects_incorrect_pnl(self):
        self.s.record_bet({"bet_id": "X1", "strategy_id": "S", "strategy_version": 1,
                           "username": "S_001", "test_mode": "FORWARD TEST", "market": "moneyline",
                           "selection": "home", "bet_type": "binary_contract",
                           "provider": "kalshi", "odds_format": "binary", "price": 0.50,
                           "implied_prob": 0.5, "decision_ts": utcnow(), "bet_ts": utcnow(),
                           "stake": 100.0, "filled_size": 200.0, "entry_price": 0.50,
                           "result": "WIN", "pnl": 999.0, "created_at": utcnow()})
        self.s.commit()
        self.assertEqual(Verifier(self.s).check_bets()["bad_pnl"], 1)

    def test_conflicting_sources_are_recorded_not_resolved(self):
        n = Verifier(self.s).cross_validate_games(
            [("games_table", {1: (3, 1)}), ("club_schedule", {1: (2, 1)})])
        self.assertEqual(n, 1)
        row = self.s.one("SELECT * FROM irregularities WHERE kind='conflicting_source'")
        self.assertIn("games_table says 3-1", row["detail"])
        self.assertIn("club_schedule says 2-1", row["detail"])


class TestBacktestLabels(unittest.TestCase):
    def test_backtest_without_price_data_reports_no_pnl(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = Store(path)
        strat = ThresholdStrategy(strategy_id="S", version=1, username="S_001",
                                  feature="home_back_to_back", operator="<=", threshold=0,
                                  bet_side="home")
        rows = []
        for i in range(60):
            rows.append({"game_id": i, "game_date": f"2026-01-{(i % 28) + 1:02d}",
                         "home_id": 1, "away_id": 2, "home_back_to_back": 0,
                         "home_n_prior": 20, "away_n_prior": 20,
                         "p_home_ml": 0.55, "p_away_ml": 0.45,
                         "_winner": 1 if i % 3 else 2})
        res = Backtester(s).run(strat, rows, label="all")
        self.assertEqual(res.data_sufficient, 0)
        self.assertEqual(res.caveat, CAVEAT_NO_PRICE)
        stored = s.one("SELECT pnl, roi, staked, data_sufficient, caveat FROM backtests")
        self.assertIsNone(stored["pnl"])
        self.assertIsNone(stored["roi"])
        self.assertIsNone(stored["staked"])
        self.assertEqual(stored["data_sufficient"], 0)
        self.assertGreater(res.n_bets, 0)
        s.close()

    def test_splits_are_chronological(self):
        rows = [{"i": i} for i in range(10)]
        parts = Backtester.split(rows)
        self.assertEqual([r["i"] for r in parts["train"]], [0, 1, 2, 3, 4, 5])
        self.assertEqual([r["i"] for r in parts["valid"]], [6, 7])
        self.assertEqual([r["i"] for r in parts["test"]], [8, 9])


if __name__ == "__main__":
    unittest.main()
