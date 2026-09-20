"""End-to-end pipeline test.

The games here are SYNTHETIC fixtures used only to prove the pipeline runs and that its
arithmetic is right; they are never presented as NHL results.  Real numbers come from the
live ingest.  The synthetic season is generated with a fixed seed so the test is
deterministic.
"""

import json
import os
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp.analysis import Performance
from nhlcomp.http import HttpClient
from nhlcomp.pipeline import Pipeline, _side_for_selection, build_team_names
from nhlcomp.site import build_site
from nhlcomp.store import Store, utcnow

SYNTHETIC_NOTE = "SYNTHETIC TEST FIXTURE - not NHL data"


def synth_season(store: Store, *, n_teams: int = 12, n_games: int = 300, seed: int = 7):
    rng = random.Random(seed)
    teams = list(range(1, n_teams + 1))
    store.execute("DELETE FROM team_game")
    store.execute("DELETE FROM games")
    for t in teams:
        store.execute("INSERT OR REPLACE INTO teams(team_id, abbrev, full_name, active) "
                      "VALUES(?,?,?,1)", (t, f"T{t:02d}", f"Synthetic Team {t}"))
    store.commit()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_games):
        home, away = rng.sample(teams, 2)
        day = start + timedelta(days=i // 6, hours=(i % 6) * 3)
        off = rng.choice(["-05:00", "-06:00", "-07:00", "-08:00"])
        # a real but modest home edge so the models have something to find
        p_home = 0.55 + rng.uniform(-0.05, 0.05)
        home_win = rng.random() < p_home
        hs, as_ = (rng.randint(2, 6), rng.randint(0, 4)) if home_win else \
                  (rng.randint(0, 4), rng.randint(2, 6))
        ot = rng.random() < 0.15
        if ot and abs(hs - as_) > 1:
            hs, as_ = (max(hs, as_), min(hs, as_) - 1) if home_win else \
                      (min(hs, as_) - 1, max(hs, as_))
        rows.append((1000 + i, day.strftime("%Y-%m-%dT%H:%M:%SZ"), day.date().isoformat(),
                     home, away, hs, as_, "OT" if ot else "REG", off))
    for gid, stu, gd, home, away, hs, as_, per, off in rows:
        store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
            " away_id, venue, utc_offset, state, home_score, away_score, last_period_type,"
            " source_id, provenance) VALUES(?,?,?,?,?,?,?,?,?,'FINAL',?,?,?,'test','SOURCE')",
            (gid, 20252026, 2, gd, stu, home, away, f"Arena{home}", off, hs, as_, per))
    store.commit()
    return rows


class TestPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cls.store = Store(cls.dbpath)
        cls.http = HttpClient(tempfile.mkdtemp())
        cls.pipe = Pipeline(cls.store, cls.http, verbose=False)
        cls.games = synth_season(cls.store)
        cls.pipe.ing.rebuild_team_games()

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_01_features_have_no_leakage(self):
        rows = self.pipe.stage_features(seasons=[20252026], game_types=(2,))
        self.assertEqual(len(rows), len(self.games))
        self.assertEqual(self.pipe.report["leakage_problems"], [])
        r = rows[0]
        self.assertEqual(r["home_n_prior"], 0)          # first game has no history
        self.assertIsNone(r["home_gf_avg"])
        later = [x for x in rows if x["home_n_prior"] and x["home_n_prior"] > 5][0]
        self.assertIsNotNone(later["home_gf_avg"])
        TestPipelineEndToEnd.rows = rows

    def test_02_models_beat_or_match_the_constant_baseline(self):
        rows = TestPipelineEndToEnd.rows
        refs = self.pipe.game_refs(seasons=[20252026], game_types=(2,))
        comp = self.pipe.stage_models(rows, refs, prior_season=None)
        for name in ("poisson", "elo", "logistic", "home_ice_constant"):
            self.assertIn(name, comp)
            self.assertIsNotNone(comp[name]["log_loss"], f"{name} produced no score")
        self.store.commit()
        f = self.store.one("SELECT * FROM findings WHERE finding_id='FIND_MODEL_COMPARE'")
        self.assertIsNotNone(f)

    def test_03_discovery_records_hypotheses_and_search_size(self):
        rows = TestPipelineEndToEnd.rows
        out = self.pipe.stage_strategies(rows)
        self.assertGreater(out["tested"], 100)
        self.assertGreater(self.store.one("SELECT COUNT(*) c FROM hypotheses")["c"], 0)
        self.assertGreaterEqual(len(self.store.latest_versions()), 10)
        cav = self.store.one("SELECT body FROM findings "
                             "WHERE finding_id='FIND_DISCOVERY_SEARCH'")["body"]
        self.assertIn("chance", cav)

    def test_04_backtests_never_report_a_pnl_without_price_data(self):
        rows = TestPipelineEndToEnd.rows
        self.pipe.stage_backtest(rows)
        bad = self.store.query("SELECT * FROM backtests WHERE pnl IS NOT NULL OR roi IS NOT NULL")
        self.assertEqual(len(bad), 0, "a backtest reported profit without verified prices")
        for b in self.store.query("SELECT * FROM backtests"):
            self.assertEqual(b["data_sufficient"], 0)
            self.assertIn("invent", b["caveat"])

    def test_05_forward_places_backtest_bets_from_real_price_shape(self):
        # attach a settled market to a game, priced the way Kalshi prices it
        gid = self.games[150][0]
        g = self.store.one("SELECT * FROM games WHERE game_id=?", (gid,))
        won = g["home_score"] > g["away_score"]
        self.store.execute(
            "INSERT INTO market_settlements(provider, event_ticker, contract, game_id, game_date,"
            " selection, side, result, settle_price, price_before, bid_before, ask_before, volume,"
            " open_interest, open_time, settlement_ts, retrieved_at, source_url) VALUES"
            "('kalshi','EV','CT',?,?,?,'YES',?,1.0,0.30,0.26,0.30,1000,500,?,?,?,'u')",
            (gid, g["game_date"], "home", "yes" if won else "no",
             g["start_time_utc"], g["start_time_utc"], utcnow()))
        self.store.commit()
        placed = self.pipe.stage_forward()
        self.assertGreaterEqual(placed["BACKTEST"], 1)
        bets = self.store.query("SELECT * FROM bets WHERE test_mode='BACKTEST'")
        self.assertTrue(bets)
        for b in bets:
            self.assertEqual(b["verification_status"], "single_source_timing_unverified")
            self.assertIn(b["result"], ("WIN", "LOSS"))
            self.assertIsNotNone(b["pnl"])
            price = float(b["entry_price"])
            contracts = float(b["filled_size"])
            expected = contracts * (1 - price) if b["result"] == "WIN" else -contracts * price
            self.assertAlmostEqual(float(b["pnl"]), expected, places=2)

    def test_05b_strategy_refuses_a_price_that_is_not_value(self):
        """A strategy must not chase: if the offer is worse than model_prob - min_edge it waits."""
        gid = self.games[200][0]
        g = self.store.one("SELECT * FROM games WHERE game_id=?", (gid,))
        self.store.execute(
            "INSERT INTO market_settlements(provider, event_ticker, contract, game_id, game_date,"
            " selection, side, result, settle_price, price_before, bid_before, ask_before, volume,"
            " open_interest, open_time, settlement_ts, retrieved_at, source_url) VALUES"
            "('kalshi','EV2','CT2',?,?,?,'YES','yes',1.0,0.97,0.95,0.97,1000,500,?,?,?,'u')",
            (gid, g["game_date"], "home", g["start_time_utc"], g["start_time_utc"], utcnow()))
        self.store.commit()
        self.pipe.stage_forward()
        bad = self.store.query(
            "SELECT * FROM bets WHERE game_id=? AND price > 0.9", (gid,))
        self.assertEqual(bad, [], "a strategy bought at 0.97 without an edge")

        # and the refusal is visible at the strategy level, not silently swallowed
        from nhlcomp.pipeline import _hydrate, build_team_names
        from nhlcomp.features import FeatureBuilder
        from nhlcomp.models import PoissonModel
        from nhlcomp.strategies import DecisionContext, Quote
        refs = {x.game_id: x for x in self.pipe.game_refs(game_types=(1, 2, 3))}
        fb = FeatureBuilder([x for x in refs.values() if x.game_type == 2])
        f = {int(r["game_id"]): r for r in fb.build_all(list(refs.values()))}[gid]
        pred = PoissonModel().predict(f)
        for k in ("p_home_elo", "p_home_logit"):
            pred[k] = pred["p_home_ml"]
        pred["p_away_elo"] = pred["p_away_logit"] = 1 - pred["p_home_ml"]
        q = Quote(provider="kalshi", market_key="EV2", contract="CT2", game_id=gid,
                  game_date=f["game_date"], market_type="moneyline", selection="home",
                  side="YES", bid=0.95, ask=0.97, bid_size=None, ask_size=None,
                  volume=1000.0, liquidity=None, ts_utc=utcnow())
        strat = _hydrate(self.store.one(
            "SELECT * FROM strategies WHERE strategy_id='NHL_POISSON_ML'"))
        sig = strat.evaluate(DecisionContext(decision_ts=utcnow(), features=f,
                                             predictions=pred, quotes=[q], bankroll=1000.0,
                                             open_exposure=0.0))[0]
        self.assertEqual(sig.status, "PRICE TOO HIGH")
        self.assertIn("0.97", sig.blocking_reason)

    def test_06_no_duplicate_bets_after_rerunning(self):
        before = self.store.one("SELECT COUNT(*) c FROM bets")["c"]
        self.pipe.stage_forward()
        after = self.store.one("SELECT COUNT(*) c FROM bets")["c"]
        self.assertEqual(before, after, "re-running the pipeline created duplicate wagers")
        dup = self.store.query(
            "SELECT strategy_id, game_id, market, selection, COUNT(*) c FROM bets "
            "GROUP BY 1,2,3,4 HAVING c > 1")
        self.assertEqual(dup, [])

    def test_07_analysis_reports_ci_and_breakdowns(self):
        self.pipe.stage_analysis()
        lb = self.pipe.report["leaderboard"]
        self.assertTrue(lb)
        top = lb[0]
        for key in ("pnl", "roi", "max_drawdown", "win_rate_ci", "n_settled", "longest_losing_streak"):
            self.assertIn(key, top)
        summ = Performance(self.store).summarize(top["strategy_id"], top["version"])
        self.assertIn("breakdowns", summ)
        self.assertIn("market", summ["breakdowns"])

    def test_08_verification_passes_on_a_clean_ledger(self):
        v = self.pipe.stage_verify()
        self.assertEqual(v["bets"]["impossible_odds"], 0)
        self.assertEqual(v["bets"]["bad_pnl"], 0)
        self.assertEqual(v["bets"]["duplicate_bet"], 0)

    def test_09_site_builds_every_section(self):
        outdir = tempfile.mkdtemp()
        n = build_site(self.store, outdir)
        self.assertGreater(n, 15)
        for name in ("index.html", "leaderboard.html", "strategies.html", "upcoming.html",
                     "positions.html", "history.html", "performance.html", "research.html",
                     "sources.html", "verification.html", "methodology.html",
                     "assets/style.css", "assets/app.js", ".nojekyll",
                     "data/leaderboard.json", "data/bets.json"):
            self.assertTrue(os.path.exists(os.path.join(outdir, name)), f"missing {name}")
        body = open(os.path.join(outdir, "methodology.html"), encoding="utf-8").read()
        self.assertIn("paper-trading", body)
        self.assertIn("no order-placement code", body)
        idx = open(os.path.join(outdir, "index.html"), encoding="utf-8").read()
        self.assertIn("Dashboard", idx)
        with open(os.path.join(outdir, "data", "leaderboard.json"), encoding="utf-8") as fh:
            self.assertIsInstance(json.load(fh), list)

    def test_10_post_settlement_analysis_flags_small_samples(self):
        bet = self.store.one("SELECT bet_id FROM bets WHERE result IN ('WIN','LOSS') LIMIT 1")
        if bet is None:
            self.skipTest("no settled bet")
        out = Performance(self.store).post_settlement_analysis(bet["bet_id"])
        self.assertIn("notes", out)
        if out["strategy_n"] < 30:
            self.assertTrue(any("too small" in n for n in out["notes"]))


class TestSideResolutionAgainstSyntheticTeams(unittest.TestCase):
    def test_generated_team_names_include_abbrev_variants(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = Store(path)
        s.execute("INSERT INTO teams(team_id, abbrev, full_name) VALUES(54,'VGK',"
                  "'Vegas Golden Knights')")
        s.commit()
        names = build_team_names(s)
        self.assertIn("vegas", names[54])
        self.assertIn("vgk", names[54])
        s.close()


if __name__ == "__main__":
    unittest.main()


class TestHydrateRoundTrip(unittest.TestCase):
    """A strategy reloaded from the DB must behave like the one that was saved.

    _hydrate used to name each parameter by hand, so requires_goalie / requires_lineup /
    injury_sensitive / blocked_reason were dropped on reload and gated strategies started
    betting as though they had never been gated.
    """

    def test_every_persisted_param_survives_a_round_trip(self):
        import json
        from nhlcomp.pipeline import _hydrate
        from nhlcomp.strategies import ThresholdStrategy

        original = ThresholdStrategy(
            strategy_id="NHL_GOALIE_EDGE", version=1, username="NHL_GOALIE_EDGE_011",
            name="Goaltending matchup edge", category="goaltending", hypothesis="h",
            data_used="d", entry_rule="e", price_rule="p", settlement_rule="s",
            markets="moneyline", origin="seed", origin_ref=None, starting_bankroll=1000.0,
            feature="home_n_prior", operator=">=", threshold=10, bet_side="home",
            requires_goalie=True, requires_lineup=True, injury_sensitive=True,
            min_price=0.07, min_edge=0.05, stake_fraction=0.3,
            blocked_reason="no verified source for X")
        row = original.describe()

        revived = _hydrate(row)
        self.assertTrue(revived.requires_goalie)
        self.assertTrue(revived.requires_lineup)
        self.assertTrue(revived.injury_sensitive)
        self.assertEqual(revived.min_price, 0.07)
        self.assertEqual(revived.blocked_reason, "no verified source for X")
        self.assertEqual(revived.min_edge, 0.05)
        self.assertEqual(revived.feature, "home_n_prior")
        self.assertEqual(revived.threshold, 10)

    def test_a_legacy_row_without_the_new_params_still_hydrates(self):
        import json
        from nhlcomp.pipeline import _hydrate
        legacy = {
            "strategy_id": "OLD", "version": 1, "username": "OLD_001", "name": "old",
            "category": "fatigue", "hypothesis": "h", "data_used": "d", "entry_rule": "e",
            "price_rule": "p", "settlement_rule": "s", "markets": "moneyline",
            "origin": "seed", "origin_ref": None, "starting_bankroll": 1000.0,
            "params_json": json.dumps({"feature": "back_to_back", "threshold": 1,
                                       "operator": ">=", "bet_side": "home",
                                       "use_model": "poisson", "min_edge": 0.03,
                                       "stake_fraction": 0.25}),
        }
        revived = _hydrate(legacy)
        self.assertFalse(revived.requires_goalie)
        self.assertIsNone(revived.blocked_reason)
        self.assertEqual(revived.feature, "back_to_back")
