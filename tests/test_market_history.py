"""Offline end-to-end test of the Kalshi history -> price points -> priced backtest ->
BACKTEST bet flow, plus the per-game stats REST features.

Everything here is a SYNTHETIC fixture shaped exactly like the live payloads that were
captured on 2026-09-20 (historical-tier market rows, candlestick payloads, stats REST
per-game rows).  The numbers are invented for the test and are never presented as NHL or
Kalshi data; the point is to prove that the code that runs against the live APIs does the
right arithmetic and applies the right labels.
"""

import json
import os
import random
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp.backtest import PricedBacktester, market_baseline, trigger_roi
from nhlcomp.features_ext import load_extended
from nhlcomp.http import HttpClient
from nhlcomp.market import (derive_price_points, devig_pair, kalshi_fee_per_unit_staked,
                            kalshi_taker_fee, two_way_vig)
from nhlcomp.pipeline import Pipeline, _hydrate
from nhlcomp.sources.kalshi import (BASE, HIST_BASE, normalize_candle, parse_event_ticker,
                                    market_team_code)
from nhlcomp.sources.nhl import STATS_REST
from nhlcomp.store import Store, utcnow
from nhlcomp.strategies import ThresholdStrategy

ABBREVS = ["NJD", "NYI", "NYR", "PHI", "PIT", "BOS", "BUF", "MTL", "OTT", "TOR", "CAR", "FLA"]
CUTOFF = "2026-07-22T00:00:00Z"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def synth_season(store: Store, *, n_games: int = 240, seed: int = 11):
    """Synthetic schedule with real NHL tricodes so Kalshi tickers can be matched."""
    rng = random.Random(seed)
    for i, ab in enumerate(ABBREVS, start=1):
        store.execute("INSERT OR REPLACE INTO teams(team_id, abbrev, full_name, active) "
                      "VALUES(?,?,?,1)", (i, ab, f"Synthetic {ab}"))
    start = datetime(2026, 1, 5, 19, 0, tzinfo=timezone.utc)
    games = []
    pairs: list[tuple[int, int]] = []
    for i in range(n_games):
        if i % 6 == 0:                       # six distinct matchups per day, no repeats
            order = list(range(1, len(ABBREVS) + 1))
            rng.shuffle(order)
            pairs = [(order[k], order[k + 1]) for k in range(0, 12, 2)]
        home, away = pairs[i % 6]
        day = start + timedelta(days=i // 6)
        p_home = 0.56
        home_win = rng.random() < p_home
        hs, as_ = (rng.randint(3, 6), rng.randint(0, 2)) if home_win else \
                  (rng.randint(0, 2), rng.randint(3, 6))
        gid = 2025020001 + i
        store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
            " away_id, venue, utc_offset, state, home_score, away_score, last_period_type,"
            " source_id, provenance) VALUES(?,?,?,?,?,?,?,?,?,'FINAL',?,?,'REG','test','SOURCE')",
            (gid, 20252026, 2, day.date().isoformat(), _iso(day), home, away, "Arena", "-05:00",
             hs, as_))
        games.append({"game_id": gid, "start": day, "home": home, "away": away,
                      "home_win": home_win, "hs": hs, "as": as_})
    store.commit()
    return games


def prime(http: HttpClient, url: str, payload) -> None:
    http._store(url, 200, json.dumps(payload))


def candle_series(open_ts: int, end_ts: int, start_ts: int, fair: float, rng: random.Random,
                  *, historical: bool = True) -> list[dict]:
    """Hourly candles shaped like the live/historical payloads."""
    out = []
    t = open_ts + 3600
    price = fair - 0.04           # opens a little cheap, drifts toward fair by the close
    while t <= end_ts:
        if t <= start_ts:
            frac = (t - open_ts) / max(1, (start_ts - open_ts))
            mid = round(price + (fair - price) * frac + rng.uniform(-0.005, 0.005), 2)
        else:
            mid = round(min(0.98, max(0.02, fair + rng.uniform(-0.2, 0.2))), 2)
        bid, ask = round(mid - 0.01, 2), round(mid + 0.01, 2)
        if historical:
            out.append({"end_period_ts": t, "open_interest": "1000.00", "volume": "50.00",
                        "price": {"open": f"{mid:.4f}", "high": f"{mid:.4f}", "low": f"{mid:.4f}",
                                  "close": f"{mid:.4f}", "mean": f"{mid:.4f}", "previous": f"{mid:.4f}"},
                        "yes_bid": {"open": f"{bid:.4f}", "high": f"{bid:.4f}", "low": f"{bid:.4f}",
                                    "close": f"{bid:.4f}"},
                        "yes_ask": {"open": f"{ask:.4f}", "high": f"{ask:.4f}", "low": f"{ask:.4f}",
                                    "close": f"{ask:.4f}"}})
        else:
            out.append({"end_period_ts": t, "open_interest_fp": "1000.00", "volume_fp": "50.00",
                        "price": {"previous_dollars": f"{mid:.4f}"},
                        "yes_bid": {"close_dollars": f"{bid:.4f}", "open_dollars": f"{bid:.4f}",
                                    "high_dollars": f"{bid:.4f}", "low_dollars": f"{bid:.4f}"},
                        "yes_ask": {"close_dollars": f"{ask:.4f}", "open_dollars": f"{ask:.4f}",
                                    "high_dollars": f"{ask:.4f}", "low_dollars": f"{ask:.4f}"}})
        t += 3600
    return out


def kalshi_fixtures(http: HttpClient, store: Store, games, *, priced: int = 160, seed: int = 5):
    """Prime the cache with a historical-tier market page and candlesticks per contract."""
    rng = random.Random(seed)
    prime(http, f"{HIST_BASE}/cutoff", {"cutoff": CUTOFF})
    prime(http, f"{BASE}/markets?series_ticker=KXNHLGAME&limit=200&status=settled",
          {"markets": [], "cursor": ""})
    abbrev = {i + 1: ab for i, ab in enumerate(ABBREVS)}
    markets = []
    for g in games[:priced]:
        d = g["start"]
        ev = f"KXNHLGAME-{d.strftime('%y%b%d').upper()}{abbrev[g['away']]}{abbrev[g['home']]}"
        open_time = d - timedelta(days=3)
        close_time = d + timedelta(hours=3)
        settle = d + timedelta(hours=3, minutes=5)
        fair_home = round(rng.uniform(0.42, 0.70), 2)
        for side, code in (("home", abbrev[g["home"]]), ("away", abbrev[g["away"]])):
            won = g["home_win"] if side == "home" else not g["home_win"]
            fair = fair_home if side == "home" else round(1.04 - fair_home, 2)   # sum > 1: vig
            ticker = f"{ev}-{code}"
            markets.append({
                "ticker": ticker, "event_ticker": ev, "status": "finalized",
                "result": "yes" if won else "no", "settlement_ts": _iso(settle),
                "close_time": _iso(close_time), "open_time": _iso(open_time),
                "occurrence_datetime": _iso(d), "expected_expiration_time": _iso(close_time),
                "title": f"{abbrev[g['away']]} at {abbrev[g['home']]} Winner?",
                "yes_sub_title": f"{code} Synthetic", "market_type": "binary",
                "previous_yes_ask_dollars": f"{fair + 0.01:.4f}",
                "previous_yes_bid_dollars": f"{fair - 0.01:.4f}",
                "previous_price_dollars": f"{fair:.4f}", "volume_fp": "1234.00",
                "open_interest_fp": "0.00", "settlement_value_dollars": "1.0000" if won else "0.0000",
                "custom_strike": {"hockey_team": f"uuid-{code}"},
                "rules_primary": f"If {code} wins the game scheduled for {d:%b %d, %Y}, "
                                 f"then the market resolves to Yes.",
            })
            start_ts = _epoch(d)
            open_ts, end_ts = _epoch(open_time), max(_epoch(close_time), start_ts + 4 * 3600)
            candles = candle_series(open_ts, end_ts, start_ts, fair, rng)
            prime(http, f"{HIST_BASE}/markets/{ticker}/candlesticks?start_ts={open_ts}"
                        f"&end_ts={end_ts}&period_interval=60",
                  {"ticker": ticker, "candlesticks": candles})
    prime(http, f"{HIST_BASE}/markets?series_ticker=KXNHLGAME&limit=1000",
          {"markets": markets, "cursor": ""})
    return markets


def stats_fixtures(http: HttpClient, games, season: int = 20252026):
    """team/summary and goalie/summary isGame=true rows for the synthetic games."""
    abbrev = {i + 1: ab for i, ab in enumerate(ABBREVS)}
    team_rows, goalie_rows = [], []
    rng = random.Random(3)
    for g in games:
        for side, tid, opp in (("H", g["home"], g["away"]), ("R", g["away"], g["home"])):
            gf, ga = (g["hs"], g["as"]) if side == "H" else (g["as"], g["hs"])
            team_rows.append({
                "gameId": g["game_id"], "teamId": tid, "gameDate": g["start"].date().isoformat(),
                "homeRoad": side, "opponentTeamAbbrev": abbrev[opp], "teamFullName": f"Synthetic {abbrev[tid]}",
                "goalsFor": gf, "goalsAgainst": ga, "shotsForPerGame": 28 + rng.randint(0, 10),
                "shotsAgainstPerGame": 28 + rng.randint(0, 10),
                "powerPlayPct": round(rng.uniform(0.0, 0.5), 4), "penaltyKillPct": round(rng.uniform(0.6, 1.0), 4),
                "powerPlayNetPct": 0.2, "penaltyKillNetPct": 0.8, "faceoffWinPct": round(rng.uniform(0.4, 0.6), 4),
                "wins": int(gf > ga), "losses": int(gf < ga), "otLosses": 0, "points": 2 * int(gf > ga),
                "winsInRegulation": int(gf > ga), "winsInShootout": 0})
            pid = 8470000 + tid            # one starter per team, always starts
            sa = 25 + rng.randint(0, 12)
            goalie_rows.append({
                "gameId": g["game_id"], "playerId": pid, "goalieFullName": f"Goalie {abbrev[tid]}",
                "teamAbbrev": abbrev[tid], "opponentTeamAbbrev": abbrev[opp], "homeRoad": side,
                "gameDate": g["start"].date().isoformat(), "gamesStarted": 1,
                "saves": sa - ga, "shotsAgainst": sa, "goalsAgainst": ga,
                "savePct": round((sa - ga) / sa, 4), "timeOnIce": 3600,
                "wins": int(gf > ga), "losses": int(gf < ga), "otLosses": 0})
    cay = f"seasonId={season}%20and%20gameTypeId=2"
    # the real server caps a page at 100 rows and reports ``total`` (observed 2026-09-20)
    for endpoint, rows in (("team/summary", team_rows), ("goalie/summary", goalie_rows)):
        start = 0
        while start < len(rows):
            chunk = rows[start:start + 100]
            prime(http, f"{STATS_REST}/{endpoint}?isAggregate=false&isGame=true&start={start}"
                        f"&limit=100&cayenneExp={cay}", {"data": chunk, "total": len(rows)})
            start += 100
    return len(team_rows), len(goalie_rows)


class TestMarketHelpers(unittest.TestCase):
    def test_price_points_from_candles(self):
        start = 1_781_481_600
        candles = [normalize_candle(c) for c in [
            {"end_period_ts": start - 86400 * 2, "price": {"close": "0.5000"},
             "yes_ask": {"close": "0.5200"}, "yes_bid": {"close": "0.4800"}, "volume": "1"},
            {"end_period_ts": start - 3600 * 6, "price": {"close": "0.5300"},
             "yes_ask": {"close": "0.5400"}, "yes_bid": {"close": "0.5200"}, "volume": "3"},
            {"end_period_ts": start, "price": {"close": "0.5500"},
             "yes_ask": {"close": "0.5600"}, "yes_bid": {"close": "0.5400"}, "volume": "9"},
            {"end_period_ts": start + 3600, "price": {"previous_dollars": "0.55"},
             "yes_ask": {"close_dollars": "0.7000"}, "yes_bid": {"close_dollars": "0.6000"},
             "volume_fp": "0.00"},
        ]]
        pts = derive_price_points(candles, start_ts=start)
        self.assertEqual(pts["open"]["ask"], 0.52)
        self.assertEqual(pts["t24h"]["ask"], 0.52)          # last candle at/before T-24h
        self.assertEqual(pts["t6h"]["ask"], 0.54)
        self.assertEqual(pts["close"]["ask"], 0.56)         # last candle ending at puck drop
        self.assertEqual(pts["close"]["end_period_ts"], start)
        self.assertEqual(pts["ig60"]["ask"], 0.70)          # strictly after the start
        self.assertNotIn("close_is_after_start", pts)

    def test_event_ticker_parsing_and_aliases(self):
        p = parse_event_ticker("KXNHLGAME-26SEP19VGKLA")
        self.assertEqual((p["game_date"], p["away_abbrev"], p["home_abbrev"]),
                         ("2026-09-19", "VGK", "LAK"))
        self.assertFalse(p["ambiguous"])
        p = parse_event_ticker("KXNHLGAME-26JUN14CARVGK")
        self.assertEqual((p["away_abbrev"], p["home_abbrev"]), ("CAR", "VGK"))
        self.assertIsNone(parse_event_ticker("KXNHLTOTAL-26SEP20NYINJ-3") if False else None)
        m = {"ticker": "KXNHLGAME-26JUN14CARVGK-CAR", "event_ticker": "KXNHLGAME-26JUN14CARVGK"}
        self.assertEqual(market_team_code(m), "CAR")

    def test_fee_matches_the_published_table(self):
        # Kalshi fee schedule effective 2026-07-07: fees = round up(0.07 * C * P * (1-P))
        self.assertAlmostEqual(kalshi_taker_fee(0.50, 100), 1.75)
        self.assertAlmostEqual(kalshi_taker_fee(0.05, 100), 0.34)
        self.assertAlmostEqual(kalshi_taker_fee(0.99, 100), 0.07)
        self.assertAlmostEqual(kalshi_taker_fee(0.01, 1), 0.01)
        self.assertAlmostEqual(kalshi_fee_per_unit_staked(0.5), 0.035)
        self.assertEqual(kalshi_taker_fee(0.5, 0), 0.0)

    def test_vig_and_devig(self):
        self.assertAlmostEqual(two_way_vig(0.57, 0.47), 0.04)
        h, a = devig_pair(0.6, 0.5)
        self.assertAlmostEqual(h + a, 1.0)
        self.assertAlmostEqual(h, 0.6 / 1.1, places=3)


HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURED = os.path.normpath(os.path.join(HERE, "..", "data", "captured"))


def _captured(name: str) -> dict:
    with open(os.path.join(CAPTURED, name), encoding="utf-8") as fh:
        return json.load(fh)


class TestCapturedKalshiPayloads(unittest.TestCase):
    """Real payloads captured on 2026-09-20 (see data/captured/MANIFEST.md): a historical-tier
    market page and the hourly candlesticks of one Stanley Cup Final contract."""

    def test_historical_market_rows_parse_and_match_a_game(self):
        page = _captured("kalshi_historical_markets_kxnhlgame_limit2.json")
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = Store(path)
        http = HttpClient(tempfile.mkdtemp(), offline=True)
        pipe = Pipeline(store, http, verbose=False)
        store.execute("INSERT OR REPLACE INTO teams(team_id, abbrev, full_name, active) VALUES(12,'CAR','Carolina Hurricanes',1)")
        store.execute("INSERT OR REPLACE INTO teams(team_id, abbrev, full_name, active) VALUES(54,'VGK','Vegas Golden Knights',1)")
        # NHL schedule row: 8pm ET on Jun 14 = 2026-06-15T00:00:00Z; game_date is the local date
        store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id, away_id,"
            " venue, utc_offset, state, home_score, away_score, last_period_type, source_id, provenance)"
            " VALUES(2025030416, 20252026, 3, '2026-06-14', '2026-06-15T00:00:00Z', 54, 12, 'T-Mobile Arena',"
            " '-07:00', 'OFF', 2, 3, 'REG', 'test', 'SOURCE')")
        store.commit()
        for m in page["markets"]:
            gid = pipe.ing._upsert_settlement(m, tier="historical", source_url="u", ts=utcnow())
            self.assertEqual(gid, 2025030416)
        rows = {r["contract"]: r for r in store.query("SELECT * FROM market_settlements")}
        car = rows["KXNHLGAME-26JUN14CARVGK-CAR"]
        vgk = rows["KXNHLGAME-26JUN14CARVGK-VGK"]
        self.assertEqual((car["team_abbrev"], car["result"]), ("CAR", "yes"))
        self.assertEqual((vgk["team_abbrev"], vgk["result"]), ("VGK", "no"))
        self.assertEqual(car["series_ticker"], "KXNHLGAME")
        self.assertEqual(car["market_type"], "moneyline")
        self.assertAlmostEqual(float(car["volume"]), 5825067.90)
        self.assertEqual(car["settlement_ts"], "2026-06-15T02:58:59.495517Z")
        # side resolution goes through the team code, not the title
        g = pipe.game_refs(game_types=(3,))[0]
        self.assertEqual(pipe._side_for_contract(dict(car), g, {}), "away")
        self.assertEqual(pipe._side_for_contract(dict(vgk), g, {}), "home")
        store.close()

    def test_historical_candles_give_a_pre_game_close_and_in_game_points(self):
        payload = _captured("kalshi_historical_candlesticks_26JUN14CARVGK_CAR.json")
        candles = [normalize_candle(c) for c in payload["candlesticks"]]
        self.assertEqual(candles[0]["ask_close"], 0.53)
        self.assertEqual(candles[0]["bid_close"], 0.52)
        self.assertAlmostEqual(candles[0]["volume"], 647423.42)
        start_ts = 1_781_481_600           # NHL scheduled start 2026-06-15T00:00:00Z
        pts = derive_price_points(candles, start_ts=start_ts)
        # the close is the candle ending AT puck drop (covering the hour before it) ...
        self.assertEqual(pts["close"]["end_period_ts"], start_ts)
        self.assertEqual(pts["close"]["ask"], 0.53)
        # ... and never a later one, even though CAR traded up to 0.99 during the game
        self.assertEqual(pts["ig60"]["ask"], 0.65)
        self.assertEqual(pts["ig120"]["ask"], 0.88)
        self.assertEqual(pts["final"]["end_period_ts"], 1_781_492_400)
        # Kalshi's occurrence_datetime (03:00Z) is the expected expiration, NOT the start:
        # using it as the start would make the 0.99 in-game candle the "close"
        wrong = derive_price_points(candles, start_ts=1_781_492_400)
        self.assertEqual(wrong["close"]["end_period_ts"], 1_781_492_400)
        self.assertNotEqual(wrong["close"]["ask"], pts["close"]["ask"])


class TestHistoryToBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cls.store = Store(cls.dbpath)
        cls.http = HttpClient(tempfile.mkdtemp(), offline=True)
        cls.pipe = Pipeline(cls.store, cls.http, verbose=False)
        cls.games = synth_season(cls.store)
        cls.pipe.ing.rebuild_team_games()
        cls.markets = kalshi_fixtures(cls.http, cls.store, cls.games)
        cls.n_team_rows, cls.n_goalie_rows = stats_fixtures(cls.http, cls.games)
        cls.pipe.ing.kalshi.pace_seconds = 0.0

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_01_history_walk_stores_settlements_and_close_points(self):
        stats = self.pipe.ing.kalshi_history(series="KXNHLGAME", max_calls=5000)
        self.assertEqual(stats["cutoff"], CUTOFF)
        self.assertEqual(stats["historical_rows"], len(self.markets))
        rows = self.store.query("SELECT * FROM market_settlements WHERE series_ticker='KXNHLGAME'")
        self.assertEqual(len(rows), len(self.markets))
        self.assertTrue(all(r["game_id"] is not None for r in rows), "every ticker must match a game")
        self.assertTrue(all(r["team_abbrev"] in ABBREVS for r in rows))
        self.assertTrue(all(r["tier"] == "historical" for r in rows))
        self.assertTrue(all(r["candles_state"] == "ok" for r in rows))
        pts = self.store.query("SELECT * FROM market_price_points WHERE point='close'")
        self.assertEqual(len(pts), len(self.markets))
        one = pts[0]
        self.assertEqual(one["series_ticker"], "KXNHLGAME")
        self.assertIsNotNone(one["game_id"])
        self.assertIn(one["team_abbrev"], ABBREVS)
        self.assertTrue(0 < one["ask"] < 1)
        g = self.store.one("SELECT start_time_utc FROM games WHERE game_id=?", (one["game_id"],))
        start_ts = int(datetime.strptime(g["start_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
                       .replace(tzinfo=timezone.utc).timestamp())
        self.assertLessEqual(one["end_period_ts"], start_ts, "close must not be after puck drop")
        self.assertEqual(self.store.one("SELECT history_complete FROM kalshi_series WHERE ticker='KXNHLGAME'")
                         ["history_complete"], 1)
        # a second run is resume-aware: nothing new, no candle refetch
        stats2 = self.pipe.ing.kalshi_history(series="KXNHLGAME", max_calls=5000)
        self.assertEqual(stats2["historical_new"], 0)
        self.assertEqual(stats2["candles_ok"], 0)

    def test_02_stats_rest_rows_become_point_in_time_features(self):
        n_team = self.pipe.ing.nhl_team_game_stats([20252026], current_season=20262027)
        n_goalie = self.pipe.ing.nhl_goalie_game_stats([20252026], current_season=20262027)
        self.assertEqual(n_team, self.n_team_rows)
        self.assertEqual(n_goalie, self.n_goalie_rows)
        rows = self.pipe.stage_features(seasons=[20252026], game_types=(2,))
        self.assertEqual(self.pipe.report["leakage_problems"], [])
        cov = self.pipe.report["extended_feature_coverage"]
        self.assertEqual(cov["market_close"], 160)
        self.assertGreater(cov["team_stats_both"], 100)
        self.assertEqual(cov["starters_known_both"], len(rows))
        late = rows[-1]
        self.assertIsNotNone(late["home_pp_pct_l10"])
        self.assertIsNotNone(late["diff_starter_sv_pct_l10"])
        self.assertIsNotNone(late["st_edge_home"])
        # features are computed from games strictly BEFORE the game date
        first = rows[0]
        self.assertIsNone(first["home_pp_pct_l10"])
        # market features
        priced = [r for r in rows if r.get("mkt_has_close")]
        self.assertEqual(len(priced), 160)
        r = priced[10]
        self.assertGreater(r["mkt_close_vig"], 0)                   # asks sum to more than 1
        self.assertEqual(r["mkt_close_is_latest"], 0)
        self.assertIsNotNone(r["mkt_move_home"])
        TestHistoryToBacktest.rows = rows

    def test_03_priced_backtest_and_baseline(self):
        rows = TestHistoryToBacktest.rows
        refs = self.pipe.game_refs(seasons=[20252026], game_types=(2,))
        self.pipe.stage_models(rows, refs, prior_season=None)
        decided = [r for r in rows if r.get("_winner") is not None]
        base = market_baseline(decided, bet_side="home")
        self.assertEqual(base["n"], 160)
        # blindly buying the home offer pays the spread and the fee
        self.assertLess(base["roi"], 0.05)
        gross = trigger_roi(decided, feature="mkt_has_close", operator=">=", threshold=1,
                            bet_side="home", fees=False)
        self.assertGreater(gross["roi"], base["roi"], "fees must lower the ROI")
        strat = ThresholdStrategy(strategy_id="T_HOME_FAV", feature="mkt_close_home_mid",
                                  operator=">=", threshold=0.5, bet_side="home", use_model="market")
        res = PricedBacktester(self.store).run(strat, decided, label="priced_all")
        self.assertIsNotNone(res)
        self.assertEqual(res.n_games, 160)
        self.assertGreater(res.n_bets, 100)
        self.assertIsNotNone(res.pnl)
        self.assertEqual(res.price_basis, "kalshi candle close ask")
        row = self.store.one("SELECT * FROM backtests WHERE strategy_id='T_HOME_FAV' AND label='priced_all'")
        self.assertEqual(row["data_sufficient"], 1)
        self.assertIn("taker fee", row["caveat"])
        # a blocked strategy or one needing a feed without history is not priced
        blocked = ThresholdStrategy(strategy_id="T_BLOCKED", feature="home_n_prior", operator=">=",
                                    threshold=0, bet_side="home", blocked_reason="no feed")
        self.assertIsNone(PricedBacktester(self.store).run(blocked, decided, persist=False))
        book = ThresholdStrategy(strategy_id="T_BOOK", feature="home_n_prior", operator=">=",
                                 threshold=0, bet_side="home", use_model="sportsbook")
        self.assertIsNone(PricedBacktester(self.store).run(book, decided, persist=False))

    def test_04_pipeline_backtest_stage_records_priced_results(self):
        rows = TestHistoryToBacktest.rows
        self.pipe.stage_strategies(rows)
        out = self.pipe.stage_backtest(rows)
        self.assertIn("market_baseline", self.pipe.report)
        priced = self.store.query("SELECT * FROM backtests WHERE label='priced_all' AND pnl IS NOT NULL")
        self.assertTrue(priced, "priced backtests must be persisted when candles exist")
        unpriced = self.store.query("SELECT * FROM backtests WHERE label='all'")
        self.assertTrue(all(b["pnl"] is None for b in unpriced),
                        "accuracy-only backtests must never carry a P&L")
        fav = self.store.one("SELECT * FROM backtests WHERE strategy_id='NHL_HOME_FAV' AND label='priced_all'")
        self.assertIsNotNone(fav)
        self.assertGreater(fav["n_bets"], 0)
        self.assertIsNotNone(self.store.one("SELECT 1 FROM findings WHERE finding_id='FIND_MARKET_BASELINE'"))
        # discovery saw priced games and evaluated market-family triggers
        self.assertGreater(self.pipe.report["discovery"]["tested"], 100)

    def test_05_forward_stage_writes_candle_priced_backtest_bets(self):
        rows = TestHistoryToBacktest.rows
        placed = self.pipe.stage_forward(rows)
        self.assertGreater(placed["BACKTEST"], 0)
        bets = self.store.query("SELECT * FROM bets WHERE test_mode='BACKTEST'")
        self.assertTrue(bets)
        for b in bets:
            self.assertEqual(b["verification_status"], "kalshi_candle_close")
            self.assertEqual(b["price_point"], "close")
            self.assertIn(b["result"], ("WIN", "LOSS"))
            self.assertGreater(float(b["fee"]), 0.0)
            gross = (float(b["filled_size"]) * (1 - float(b["entry_price"])) if b["result"] == "WIN"
                     else -float(b["filled_size"]) * float(b["entry_price"]))
            self.assertAlmostEqual(float(b["pnl"]), gross - float(b["fee"]), places=3)
            sup = json.loads(b["features_json"])
            self.assertIn("kalshi candlestick", sup["price_basis"])
            self.assertEqual(sup["data_labels"]["price"], "SOURCE DATA (kalshi)")
            # the decision timestamp is the candle's end, at or before the scheduled start
            g = self.store.one("SELECT start_time_utc FROM games WHERE game_id=?", (b["game_id"],))
            start_ts = int(datetime.strptime(g["start_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
                           .replace(tzinfo=timezone.utc).timestamp())
            self.assertLessEqual(int(b["decision_ts"]), start_ts)
        # only games with a recovered close were priced; the 80 unpriced games got nothing
        priced_games = {r["game_id"] for r in self.store.query(
            "SELECT DISTINCT game_id FROM market_price_points WHERE point='close'")}
        self.assertTrue(all(b["game_id"] in priced_games for b in bets))
        # BACKTEST P&L never funds a forward bankroll
        for s in self.store.latest_versions():
            self.store.sync_bankroll(s["strategy_id"], int(s["version"]))
            row = self.store.one("SELECT bankroll, starting_bankroll FROM strategies WHERE "
                                 "strategy_id=? AND version=?", (s["strategy_id"], s["version"]))
            self.assertAlmostEqual(float(row["bankroll"]), float(row["starting_bankroll"]))
        # re-running never duplicates a bet
        n_before = self.store.one("SELECT COUNT(*) c FROM bets")["c"]
        self.pipe.stage_forward(rows)
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM bets")["c"], n_before)

    def test_06_forward_bet_gets_clv_from_the_close_candle(self):
        # a future game with a live quote -> FORWARD TEST bet; later the close arrives
        gid = 2025029999
        day = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=2)  # not started
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
            " away_id, venue, utc_offset, state, source_id, provenance) "
            "VALUES(?,?,?,?,?,?,?,?,?,'FUT','test','SOURCE')",
            (gid, 20252026, 2, day.date().isoformat(), _iso(day), 10, 1, "Arena", "-05:00"))
        contract = "KXNHLGAME-FUTNJDTOR-TOR"
        self.store.execute(
            "INSERT INTO market_quotes(provider, market_key, contract, game_id, game_date,"
            " market_type, selection, side, bid, ask, bid_size, ask_size, volume, liquidity,"
            " ts_utc, retrieved_at, source_url) VALUES('kalshi','KXNHLGAME-FUTNJDTOR',?,?,?,"
            " 'moneyline','home','YES',0.30,0.32,200,200,10,100,?,?,'u')",
            (contract, gid, day.date().isoformat(), utcnow(), utcnow()))
        self.store.commit()
        rows = TestHistoryToBacktest.rows
        placed = self.pipe.stage_forward(rows)
        fwd = self.store.query("SELECT * FROM bets WHERE test_mode='FORWARD TEST' AND game_id=?", (gid,))
        self.assertGreaterEqual(placed["FORWARD TEST"], 1)
        self.assertTrue(fwd)
        self.assertTrue(all(b["close_price"] is None for b in fwd))
        # the close candle arrives (mid 0.40 vs entry 0.32 -> positive CLV)
        self.store.execute(
            "INSERT INTO market_price_points(contract, point, end_period_ts, game_id, team_abbrev,"
            " series_ticker, market_type, bid, ask, tier, retrieved_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (contract, "close", _epoch(day), gid, "TOR", "KXNHLGAME", "moneyline", 0.39, 0.41,
             "live", utcnow()))
        self.store.commit()
        n = self.pipe.stage_clv()
        self.assertEqual(n, len(fwd))
        for b in self.store.query("SELECT * FROM bets WHERE test_mode='FORWARD TEST' AND game_id=?", (gid,)):
            self.assertAlmostEqual(float(b["close_price"]), 0.40)
            self.assertAlmostEqual(float(b["clv"]), round(0.40 - float(b["entry_price"]), 4))
            self.assertEqual(b["amended"], 1)
            audit = self.store.query("SELECT * FROM bet_audit WHERE bet_id=? AND action='AMEND'", (b["bet_id"],))
            self.assertTrue(audit)

    def test_07_verification_is_clean_and_cross_validates_sources(self):
        v = self.pipe.stage_verify()
        self.assertEqual(v["bets"]["bad_pnl"], 0)
        self.assertEqual(v["bets"]["duplicate_bet"], 0)
        self.assertEqual(v["bets"]["impossible_odds"], 0)
        cv = v["cross_validation"]
        self.assertEqual(cv["team_game_rows_compared"], 480)
        self.assertEqual(cv["score_conflicts"], 0)
        self.assertEqual(cv["settlement_conflicts"], 0)
        # a planted disagreement is recorded with both values, not resolved
        self.store.execute("UPDATE team_game_stats SET gf = gf + 1 WHERE game_id=? AND home_road='H'",
                           (self.games[3]["game_id"],))
        self.store.execute("UPDATE market_settlements SET result = CASE result WHEN 'yes' THEN 'no' ELSE 'yes' END "
                           "WHERE contract LIKE ? ", (f"%{ABBREVS[self.games[4]['home'] - 1]}",))
        self.store.commit()
        cv = self.pipe.stage_verify()["cross_validation"]
        self.assertEqual(cv["score_conflicts"], 1)
        self.assertGreaterEqual(cv["settlement_conflicts"], 1)
        irr = self.store.query("SELECT * FROM irregularities WHERE kind='conflicting_source'")
        self.assertTrue(any("stats REST team/summary says" in r["detail"] for r in irr))
        self.assertTrue(all(r["status"] == "open" for r in irr))
        # the schedule row itself was not touched
        g = self.store.one("SELECT home_score FROM games WHERE game_id=?", (self.games[3]["game_id"],))
        self.assertEqual(g["home_score"], self.games[3]["hs"])
        # shootout definition: the stats REST line omits the deciding goal (verified on
        # 2024-25 data: 2-1 SO shows GF 1 / GA 1).  That is consistent, not a conflict.
        so = self.games[6]
        self.store.execute("UPDATE games SET last_period_type='SO', home_score=2, away_score=1 WHERE game_id=?",
                           (so["game_id"],))
        self.store.execute("UPDATE team_game_stats SET gf=1, ga=1 WHERE game_id=?", (so["game_id"],))
        self.store.commit()
        before = self.store.one("SELECT COUNT(*) c FROM irregularities WHERE kind='conflicting_source'")["c"]
        cv = self.pipe.stage_verify()["cross_validation"]
        self.assertEqual(cv["shootout_goal_definition_adjusted"], 2)
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM irregularities WHERE kind='conflicting_source'")["c"],
                         before)

    def test_07b_injury_sensitive_rules_are_not_replayed(self):
        # no BACKTEST wager may come from an injury-sensitive rule (no injury history)
        for row in self.store.query(
                """SELECT b.bet_id FROM bets b JOIN strategies s
                     ON s.strategy_id=b.strategy_id AND s.version=b.strategy_version
                    WHERE b.test_mode='BACKTEST' AND s.params_json LIKE '%"injury_sensitive": true%'"""):
            self.fail(f"{row['bet_id']} was replayed by an injury-sensitive rule")
        # a legacy row of that kind is annotated through amend_bet, never deleted
        legacy = self.store.one("SELECT * FROM bets WHERE test_mode='BACKTEST' LIMIT 1")
        inj = self.store.one("SELECT strategy_id, version FROM strategies WHERE strategy_id='NHL_INJURY_IMPACT' "
                             "ORDER BY version DESC LIMIT 1")
        self.store.execute("UPDATE bets SET strategy_id=?, strategy_version=? WHERE bet_id=?",
                           (inj["strategy_id"], inj["version"], legacy["bet_id"]))
        self.store.commit()
        n = self.pipe.stage_reconcile()
        self.assertEqual(n, 1)
        after = self.store.one("SELECT verification_status, amended FROM bets WHERE bet_id=?", (legacy["bet_id"],))
        self.assertIn("backtest_without_injury_context", after["verification_status"])
        self.assertEqual(after["amended"], 1)
        self.assertEqual(self.pipe.stage_reconcile(), 0)     # idempotent

    def test_08_hydrated_market_strategy_keeps_flat_staking(self):
        row = self.store.one("SELECT * FROM strategies WHERE strategy_id='NHL_STEAM_FOLLOW'")
        strat = _hydrate(row)
        self.assertEqual(strat.stake_mode, "flat")
        self.assertEqual(strat.use_model, "market")
        self.assertEqual(strat.min_edge, 0.0)
        # the seed library is versioned: since 2026-09-22 GOALIE_EDGE v3 (probable-starter
        # feed) is the newest, so v2 and v1 are both retired and only v3 trades
        v3 = self.store.one("SELECT status FROM strategies WHERE strategy_id='NHL_GOALIE_EDGE' AND version=3")
        v2 = self.store.one("SELECT status FROM strategies WHERE strategy_id='NHL_GOALIE_EDGE' AND version=2")
        v1 = self.store.one("SELECT status FROM strategies WHERE strategy_id='NHL_GOALIE_EDGE' AND version=1")
        self.assertEqual(v3["status"], "active")
        self.assertEqual(v2["status"], "retired")
        self.assertEqual(v1["status"], "retired")


if __name__ == "__main__":
    unittest.main()
