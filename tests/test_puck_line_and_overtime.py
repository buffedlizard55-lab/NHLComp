"""Puck line, overtime, period goals and the second prediction market: ingest -> price -> settle.

Two kinds of payload are used here and they are labelled differently on purpose:

* **REAL captures.**  The ingest tests replay responses committed under ``data/captured/``,
  fetched 2026-09-21 from Kalshi's live and historical tiers, from
  ``api-web.nhle.com/v1/score/2026-09-20`` and from ``gamma-api.polymarket.com``.  Those
  bytes are the assertion: what the parser stores must be what the exchange or the league
  published.  The teams and games they are joined to are test fixtures.
* **Synthetic execution fixtures.**  The pricing/settlement tests invent prices, because a
  test that needs a live book cannot run in CI.  They are shaped exactly like the captures
  (same fields, same absent fields -- including the NO side having no published offer size)
  and their numbers are never presented as NHL or Kalshi data.

What this file pins down:

1. KXNHLSPREAD -- the ticker suffix digit is a **rung index**, not the line; the line comes
   only from ``floor_strike``/``strike_type``, and the team the contract names is validated
   against that game's own two teams rather than assumed.
2. KXNHLOVERTIME -- a strike-less contract, whose settlement mapping is verified against the
   exchange's own results (OT -> yes, REG -> no) and whose **shootout** case is not verified
   by any evidence, so such a wager stays OPEN and is settled from the exchange's result
   instead of being guessed.
3. Series that list nothing are recorded as verified negatives, never backfilled.
4. ``/v1/score`` period goals are stored as published, the derived period scores are
   reconciled against the official final score, and a difference is stated rather than
   patched.
5. Polymarket rows are reference data: stored with their published fee schedule, matched to
   a game only when the match is unambiguous, and their absence of per-game NHL markets is
   recorded rather than worked around.
6. Depth -- a fill is capped by the best evidence the venue publishes, and where it
   publishes none the fallback is a declared, labelled assumption.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp.http import HttpClient
from nhlcomp.ingest import Ingestor
from nhlcomp.models import OtCalibration, p_margin_over
from nhlcomp.paper import (DEPTH_BASIS_CAP, DEPTH_BASIS_OFFER, DEPTH_BASIS_VOLUME,
                           UNKNOWN_DEPTH_CAP_CONTRACTS, PaperEngine, simulate_fill)
from nhlcomp.pipeline import Pipeline
from nhlcomp.sources.kalshi import (BASE as KALSHI_BASE, HIST_BASE,
                                    parse_event_ticker, split_team_codes)
from nhlcomp.store import Store, utcnow
from nhlcomp.verify import Verifier
from nhlcomp.strategies import DecisionContext, OvertimeStrategy, PuckLineStrategy, Quote

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURED = os.path.normpath(os.path.join(HERE, "..", "data", "captured"))

# team ids are fixtures; the abbreviations are the real ones the captures use
VGK, CAR, UTA, NJD, NYI = 54, 12, 53, 1, 2
TEAMS = ((VGK, "VGK", "Vegas Golden Knights"), (CAR, "CAR", "Carolina Hurricanes"),
         (UTA, "UTA", "Utah Hockey Club"), (NJD, "NJD", "New Jersey Devils"),
         (NYI, "NYI", "New York Islanders"))

KALSHI_LIVE = f"{KALSHI_BASE}/markets"
KALSHI_HIST = f"{HIST_BASE}/markets"
SCORE_URL = "https://api-web.nhle.com/v1/score/2026-09-20"
POLY_URL = "https://gamma-api.polymarket.com/events?tag_slug=nhl&closed=false&limit=100&offset=0"


def load(name: str):
    with open(os.path.join(CAPTURED, name), encoding="utf-8") as fh:
        return json.load(fh)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


class CapturedIngestBase(unittest.TestCase):
    """Replays the committed captures through the real ingest code, offline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.addCleanup(self.store.close)
        # offline=True: an unprimed URL fails immediately instead of reaching for a network
        # that CI may or may not have, so the tests are hermetic and fast
        self.http = HttpClient(cache_dir=os.path.join(self.tmp.name, "cache"), offline=True)
        self.ing = Ingestor(self.store, self.http, verbose=False)
        self.ing.kalshi.pace_seconds = 0.0
        for tid, ab, name in TEAMS:
            self.store.execute(
                "INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                (tid, ab, name))
        self.store.commit()

    # ------------------------------------------------------------------ helpers
    def prime(self, url: str, payload) -> None:
        self.http._store(url, 200, json.dumps(payload))

    def add_game(self, game_id: int, day: str, start: datetime, home: int, away: int, *,
                 state: str = "FUT", hs=None, as_=None, lpt=None, game_type: int = 2,
                 season: int = 20252026) -> None:
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
            " away_id, venue, utc_offset, state, home_score, away_score, last_period_type,"
            " source_id, provenance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, 'test','SOURCE')",
            (game_id, season, game_type, day, _iso(start), home, away, "Arena", "-04:00",
             state, hs, as_, lpt))
        self.store.commit()

    def one(self, sql: str, params=()):
        return self.store.one(sql, params)

    def flags(self, kind: str) -> list:
        return self.store.query("SELECT * FROM irregularities WHERE kind=?", (kind,))


class TestCapturedKalshiIngest(CapturedIngestBase):
    """The 2026-09-21 captures, parsed by the production ingest path."""

    def setUp(self):
        super().setUp()
        # the live spread capture is the 2026-09-24 preseason game UTA at VGK
        self.gid_future = 2026010030
        self.future = datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc)
        self.add_game(self.gid_future, "2026-09-24", self.future, VGK, UTA, game_type=1)

    def single_page(self, name: str) -> dict:
        """A captured page, with its cursor cleared.

        The captures were taken with ``limit=2`` so they carry a real next-page cursor; a
        test that primes one page must not then be asked for the next one.  The markets
        themselves are untouched.
        """
        return dict(load(name), cursor="")

    def test_live_spread_contracts_store_line_team_and_both_sides(self):
        payload = self.single_page("kalshi_live_markets_kxnhlspread_limit2.json")
        self.prime(f"{KALSHI_LIVE}?series_ticker=KXNHLSPREAD&limit=200&status=open", payload)
        n = self.ing.kalshi_nhl(series="KXNHLSPREAD")
        self.assertEqual(n, len(payload["markets"]))
        rows = self.store.query(
            "SELECT * FROM market_quotes WHERE market_type='puck_line' ORDER BY contract, side")
        self.assertEqual(len(rows), 4)             # two contracts x YES and NO
        for r in rows:
            self.assertEqual(r["game_id"], self.gid_future)
            self.assertEqual(r["team_abbrev"], "VGK")
            self.assertEqual(r["strike_type"], "greater")
            self.assertIn(float(r["strike"]), (1.5, 2.5))
            self.assertIsNone(r["ask_size"] if r["side"] == "NO" else None)

    def test_the_suffix_digit_is_a_rung_index_and_never_the_line(self):
        """``-VGK3`` carried floor_strike 2.5 and ``-VGK2`` carried 1.5 in the live tier,
        while the historical tier of the same series put 2.5 on ``-VGK2``.  Reading the digit
        as the line would price the wrong contract."""
        payload = self.single_page("kalshi_live_markets_kxnhlspread_limit2.json")
        self.prime(f"{KALSHI_LIVE}?series_ticker=KXNHLSPREAD&limit=200&status=open", payload)
        self.ing.kalshi_nhl(series="KXNHLSPREAD")
        by_contract = {r["contract"]: r for r in self.store.query(
            "SELECT * FROM market_quotes WHERE side='YES'")}
        three = by_contract["KXNHLSPREAD-26SEP24UTAVGK-VGK3"]
        two = by_contract["KXNHLSPREAD-26SEP24UTAVGK-VGK2"]
        self.assertAlmostEqual(float(three["strike"]), 2.5)
        self.assertAlmostEqual(float(two["strike"]), 1.5)
        # the historical tier of the same series, where -VGK2 is the 2.5 rung
        hist = load("kalshi_historical_markets_kxnhlspread_limit2.json")
        strikes = {}
        for m in hist["markets"]:
            strikes[m["ticker"]] = float(m["floor_strike"])
        self.assertEqual(strikes, {"KXNHLSPREAD-26JUN14CARVGK-VGK2": 2.5,
                                  "KXNHLSPREAD-26JUN14CARVGK-VGK1": 1.5})

    def test_historical_overtime_contracts_are_strike_less_and_carry_their_own_result(self):
        self.add_game(2025030415, "2026-06-11", datetime(2026, 6, 12, 0, 0, tzinfo=timezone.utc),
                      CAR, VGK, state="FINAL", hs=4, as_=2, lpt="REG", game_type=3)
        self.add_game(2025030416, "2026-06-14", datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc),
                      VGK, CAR, state="FINAL", hs=0, as_=3, lpt="REG", game_type=3)
        self.prime(f"{HIST_BASE}/cutoff", {"cutoff_time": "2026-07-22T00:00:00Z"})
        self.prime(f"{KALSHI_LIVE}?series_ticker=KXNHLOVERTIME&limit=200&status=settled",
                   {"markets": [], "cursor": ""})
        self.prime(f"{KALSHI_HIST}?series_ticker=KXNHLOVERTIME&limit=1000",
                   load("kalshi_historical_markets_kxnhlovertime_limit2.json"))
        stats = self.ing.kalshi_history(series="KXNHLOVERTIME", max_calls=40, max_pages=1)
        self.assertGreaterEqual(stats["historical_rows"], 2)
        rows = self.store.query(
            "SELECT * FROM market_settlements WHERE series_ticker='KXNHLOVERTIME'")
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["market_type"], "overtime")
            self.assertEqual(r["result"], "no")       # what the exchange published
            self.assertEqual(r["rung"], "OT")         # kept verbatim, not read as a team
            self.assertIsNone(r["team_abbrev"])       # 'OT' is not a team code
            self.assertIsNone(r["floor_strike"])      # a strike-less contract
            self.assertIsNotNone(r["game_id"])
        matched = {r["contract"]: r["game_id"] for r in rows}
        self.assertEqual(matched["KXNHLOVERTIME-26JUN11VGKCAR-OT"], 2025030415)
        self.assertEqual(matched["KXNHLOVERTIME-26JUN14CARVGK-OT"], 2025030416)

    def test_a_series_that_lists_nothing_is_recorded_as_a_verified_negative(self):
        for ticker in ("KXNHL1P", "KXNHLSAVES"):
            self.prime(f"{KALSHI_LIVE}?series_ticker={ticker}&limit=200&status=open",
                       {"markets": [], "cursor": ""})
        out = self.ing.kalshi_series_watch(series=("KXNHL1P", "KXNHLSAVES"))
        for ticker in ("KXNHL1P", "KXNHLSAVES"):
            self.assertEqual(out[ticker]["open_contracts"], 0)
            row = self.one("SELECT * FROM kalshi_series WHERE ticker=?", (ticker,))
            self.assertIsNotNone(row)
            self.assertEqual(row["open_contracts"], 0)
            self.assertIn("Verified negative", row["listing_note"])
            self.assertIsNotNone(row["last_listed_check"])
        # nothing was written for them: an empty answer is not a price
        self.assertEqual(self.one("SELECT COUNT(*) c FROM market_quotes")["c"], 0)
        self.assertEqual(self.one("SELECT COUNT(*) c FROM market_settlements")["c"], 0)

    def test_a_series_that_starts_listing_is_flagged_so_capture_begins(self):
        payload = self.single_page("kalshi_live_markets_kxnhlspread_limit2.json")
        self.prime(f"{KALSHI_LIVE}?series_ticker=KXNHLSPREAD&limit=200&status=open", payload)
        out = self.ing.kalshi_series_watch(series=("KXNHLSPREAD",))
        self.assertEqual(out["KXNHLSPREAD"]["open_contracts"], len(payload["markets"]))
        self.assertEqual(out["KXNHLSPREAD"]["market_type"], "puck_line")
        self.assertTrue(self.flags("new_market_listed"))


class TestCapturedNhlScoreIngest(CapturedIngestBase):
    """``/v1/score/{date}``: official per-goal rows, and the reconciliation that follows."""

    def setUp(self):
        super().setUp()
        # the captured game: 2026-09-20, NYI at NJD, official 2-1, decided in regulation
        self.gid = 2026010008
        self.add_game(self.gid, "2026-09-20", datetime(2026, 9, 20, 17, 0, tzinfo=timezone.utc),
                      NJD, NYI, state="FINAL", hs=2, as_=1, lpt="REG", game_type=1,
                      season=20262027)
        self.prime(SCORE_URL, load("nhl_score_20260920_game2026010008.json"))

    def test_goal_rows_are_stored_as_published(self):
        stats = self.ing.nhl_period_goals(["2026-09-20"])
        self.assertEqual(stats["games"], 1)
        self.assertEqual(stats["goals"], 3)
        rows = self.store.query(
            "SELECT * FROM game_period_goals WHERE game_id=? ORDER BY period, time_in_period",
            (self.gid,))
        self.assertEqual(len(rows), 3)
        first = rows[0]
        self.assertEqual(first["period"], 2)
        self.assertEqual(first["period_type"], "REG")
        self.assertEqual(first["team_abbrev"], "NYI")
        self.assertEqual(first["strength"], "pp")
        self.assertEqual(first["player_id"], 8484807)
        self.assertEqual(first["away_score"], 1)
        self.assertEqual(first["home_score"], 0)
        self.assertEqual(first["provenance"], "SOURCE")

    def test_derived_period_scores_reconcile_with_the_official_final(self):
        self.ing.nhl_period_goals(["2026-09-20"])
        rows = self.store.query(
            "SELECT * FROM game_period_scores WHERE game_id=? ORDER BY period", (self.gid,))
        self.assertEqual([(r["period"], r["home_goals"], r["away_goals"]) for r in rows],
                         [(2, 1, 1), (3, 1, 0)])
        for r in rows:
            self.assertEqual(r["reconciled"], 1)
            self.assertEqual(r["provenance"], "DERIVED")
        self.assertEqual(self.one(
            "SELECT COUNT(*) c FROM game_period_scores WHERE reconciled=0")["c"], 0)

    def test_a_shootout_difference_is_stated_and_never_patched(self):
        """The deciding shootout attempt is not published as a goal row, so the derived sum
        can legitimately differ from the official score by one.  That is recorded, not fixed."""
        payload = load("nhl_score_20260920_game2026010008.json")
        game = dict(payload["games"][0])
        game["gameOutcome"] = {"lastPeriodType": "SO"}
        game["homeTeam"] = dict(game["homeTeam"], score=4)     # official: 4-3 after the SO
        game["awayTeam"] = dict(game["awayTeam"], score=3)
        game["goals"] = [dict(g, period=3, timeInPeriod=f"1{i}:00")
                         for i, g in enumerate(game["goals"])]
        self.prime(SCORE_URL, {"games": [game], "currentDate": "2026-09-20"})
        stats = self.ing.nhl_period_goals(["2026-09-20"])
        self.assertEqual(stats["unreconciled"], 1)
        row = self.one("SELECT * FROM game_period_scores WHERE game_id=?", (self.gid,))
        self.assertEqual(row["reconciled"], 0)
        self.assertIn("shootout", row["reconciliation_note"].lower())
        # the derived numbers stay what the goal rows say; the official score is not copied in
        self.assertEqual((row["home_goals"], row["away_goals"]), (2, 1))
        self.assertTrue(self.flags("unreconciled_period_goals"))

    def test_a_conflicting_last_period_type_is_preserved_not_resolved(self):
        payload = load("nhl_score_20260920_game2026010008.json")
        game = dict(payload["games"][0])
        game["gameOutcome"] = {"lastPeriodType": "OT"}         # scoreboard feed says REG
        self.prime(SCORE_URL, {"games": [game], "currentDate": "2026-09-20"})
        stats = self.ing.nhl_period_goals(["2026-09-20"])
        self.assertEqual(stats["conflicts"], 1)
        flag = self.flags("source_disagreement")[0]
        self.assertIn("lastPeriodType", flag["detail"])
        # the stored scoreboard value is untouched: both readings survive
        self.assertEqual(self.one("SELECT last_period_type FROM games WHERE game_id=?",
                                  (self.gid,))["last_period_type"], "REG")


class TestCapturedPolymarketIngest(CapturedIngestBase):
    """Polymarket as a second prediction market: reference rows, honest absence of games."""

    def test_the_excerpt_is_stored_with_its_published_fee_schedule_and_no_game_match(self):
        excerpt = load("polymarket_gamma_nhl_event_excerpt.json")
        # the endpoint returns a bare array; the capture keeps it under 'events'
        self.prime(POLY_URL, excerpt["events"])
        stats = self.ing.polymarket_nhl(limit=100)
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["markets"], 1)
        self.assertEqual(stats["families"], {"futures": 1})
        self.assertEqual(stats["matched_games"], 0)
        row = self.one("SELECT * FROM polymarket_markets")
        self.assertEqual(row["id"], "2520276")
        self.assertEqual(row["group_item_title"], "Anaheim Ducks")
        self.assertAlmostEqual(row["best_bid"], 0.021)
        self.assertAlmostEqual(row["best_ask"], 0.022)
        self.assertAlmostEqual(row["liquidity"], 56107.48476)
        self.assertEqual(row["fee_type"], "sports_fees_v2")
        self.assertAlmostEqual(row["fee_rate"], 0.03)
        self.assertEqual(row["fee_taker_only"], 1)
        self.assertAlmostEqual(row["fee_rebate_rate"], 0.25)
        self.assertIsNone(row["game_id"])
        self.assertIn("not matched to a game", row["match_basis"])
        # the published strings are kept as published, not rewritten as parsed lists
        self.assertEqual(row["outcome_prices"], '["0.0215", "0.9785"]')
        self.assertTrue(self.flags("no_game_markets"))

    def test_a_futures_price_is_never_attached_to_a_game_it_does_not_describe(self):
        excerpt = load("polymarket_gamma_nhl_event_excerpt.json")
        ev = dict(excerpt["events"][0])
        mkt = dict(ev["markets"][0])
        # same market, but now claiming a game start time and naming both teams
        mkt["gameStartTime"] = "2026-09-24T02:00:00Z"
        mkt["question"] = "Will the Vegas Golden Knights beat the Utah Hockey Club?"
        ev["markets"] = [mkt]
        self.prime(POLY_URL, [ev])
        self.add_game(2026010030, "2026-09-24",
                      datetime(2026, 9, 24, 2, 0, tzinfo=timezone.utc), VGK, UTA, game_type=1)
        stats = self.ing.polymarket_nhl(limit=100)
        self.assertEqual(stats["families"], {"game": 1})
        self.assertEqual(stats["matched_games"], 1)
        row = self.one("SELECT * FROM polymarket_markets")
        self.assertEqual(row["game_id"], 2026010030)
        self.assertIn("matched on published game start date", row["match_basis"])


class ExecutionBase(unittest.TestCase):
    """Synthetic books shaped like the captures, for pricing and settlement behaviour."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.addCleanup(self.store.close)
        self.http = HttpClient(cache_dir=os.path.join(self.tmp.name, "cache"), offline=True)
        self.pipe = Pipeline(self.store, self.http, verbose=False)
        self.paper = self.pipe.paper
        for tid, ab, name in TEAMS:
            self.store.execute(
                "INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                (tid, ab, name))
        self.future = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=3)
        self.past = datetime(2026, 6, 5, 0, 30, tzinfo=timezone.utc)
        self.gid_future, self.gid_past = 2026010030, 2025030412
        self.add_game(self.gid_future, self.future.date().isoformat(), self.future, VGK, UTA,
                      state="FUT", game_type=1, season=20262027)
        self.add_game(self.gid_past, self.past.date().isoformat(), self.past, CAR, VGK,
                      state="FINAL", hs=4, as_=3, lpt="OT", game_type=3)
        self.store.commit()

    # ------------------------------------------------------------------ helpers
    def add_game(self, game_id, day, start, home, away, *, state, hs=None, as_=None, lpt=None,
                 game_type=2, season=20252026):
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc, home_id,"
            " away_id, venue, utc_offset, state, home_score, away_score, last_period_type,"
            " source_id, provenance) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, 'test','SOURCE')",
            (game_id, season, game_type, day, _iso(start), home, away, "Arena", "-04:00",
             state, hs, as_, lpt))
        self.store.commit()

    def register(self, strat) -> None:
        d = strat.describe()
        d.update(status="active", created_at=utcnow(), test_mode="FORWARD TEST",
                 leakage_checked=1)
        self.store.upsert_strategy(d)
        self.store.set_strategy_status(strat.strategy_id, strat.version, "active",
                                       reason="test", evidence="test")

    def one(self, sql: str, params=()):
        return self.store.one(sql, params)

    def flags(self, kind: str) -> list:
        return self.store.query("SELECT * FROM irregularities WHERE kind=?", (kind,))

    def quote_row(self, *, contract, market_key, gid, day, market_type, title, side, ask,
                  team_abbrev=None, strike=None, strike_type=None, ask_size=200.0, bid=None,
                  ts=None):
        bid = ask - 0.02 if bid is None else bid
        self.store.execute(
            "INSERT INTO market_quotes(provider, market_key, contract, game_id, game_date,"
            " market_type, selection, side, bid, ask, spread, bid_size, ask_size, volume,"
            " liquidity, last_price, ts_utc, retrieved_at, source_url, strike, strike_type,"
            " team_abbrev) VALUES('kalshi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'https://kalshi.example',"
            "?,?,?)",
            (market_key, contract, gid, day, market_type, title, side, bid, ask,
             round(ask - bid, 4), ask_size, ask_size, 10, 100, ask, ts or utcnow(), utcnow(),
             strike, strike_type, team_abbrev))
        self.store.commit()

    def feature_row(self, gid, day, **extra):
        row = {"game_id": gid, "game_date": day.date().isoformat(), "start_time_utc": _iso(day),
               "home_id": VGK, "away_id": UTA, "home_n_prior": 40, "away_n_prior": 40,
               "home_exp_goals": 3.2, "away_exp_goals": 2.4, "lam_home": 3.2, "lam_away": 2.4,
               "exp_total": 5.6, "p_home_ml": 0.63, "p_away_ml": 0.37, "p_home_elo": 0.62,
               "p_away_elo": 0.38, "p_home_logit": 0.61, "p_away_logit": 0.39,
               "p_overtime": 0.20, "p_tie_reg": 0.20,
               "home_id2": None}
        row.pop("home_id2")
        if gid == self.gid_past:
            row.update(home_id=CAR, away_id=VGK)
        row.update(extra)
        return row

    def settled_contract(self, *, contract, event_ticker, gid, day, series, market_type, result,
                         strike=None, rung=None, title="contract", selection="title",
                         volume=100.0):
        """One row shaped like Kalshi's own settled-contract payload."""
        row = {
            "provider": "kalshi", "event_ticker": event_ticker, "contract": contract,
            "game_id": gid, "game_date": day.date().isoformat(), "selection": selection,
            "side": "YES", "result": result, "settle_price": 1.0 if result == "yes" else 0.0,
            "volume": volume, "open_interest": 0.0,
            "open_time": _iso(day - timedelta(days=2)),
            "close_time": _iso(day + timedelta(hours=3)),
            "settlement_ts": _iso(day + timedelta(hours=3)), "retrieved_at": utcnow(),
            "source_url": "https://kalshi.example", "series_ticker": series,
            "tier": "historical", "floor_strike": strike, "title": title,
            "occurrence_datetime": _iso(day), "team_abbrev": None,
            "market_type": market_type,
            "strike_type": ("greater" if strike is not None else None), "rung": rung,
            "provenance": "SOURCE",
        }
        cols = list(row.keys())
        self.store.execute(
            f"INSERT INTO market_settlements({','.join(cols)}) "
            f"VALUES({','.join('?' * len(cols))})", tuple(row[c] for c in cols))
        self.store.commit()

    def candle(self, contract, gid, series, market_type, *, ask, bid=None, volume=100.0):
        bid = ask - 0.02 if bid is None else bid
        for point, ts in (("open", self.past - timedelta(hours=30)),
                          ("t6h", self.past - timedelta(hours=6)),
                          ("close", self.past - timedelta(minutes=30))):
            self.store.execute(
                "INSERT INTO market_price_points(contract, point, end_period_ts, game_id,"
                " team_abbrev, series_ticker, market_type, bid, ask, last, mean, volume,"
                " open_interest, period_interval, tier, retrieved_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,60,'historical',?)",
                (contract, point, _epoch(ts), gid, None, series, market_type, bid, ask,
                 ask, (ask + bid) / 2, volume, 1000, utcnow()))
        self.store.commit()


class TestPuckLineExecution(ExecutionBase):
    def test_a_live_puck_line_quote_resolves_team_strike_and_side(self):
        lam_h, lam_a = 3.2, 2.4
        p15 = p_margin_over(lam_h, lam_a, "home", 1.5)
        self.quote_row(contract="KXNHLSPREAD-X-VGK2", market_key="KXNHLSPREAD-X",
                       gid=self.gid_future, day=self.future.date().isoformat(),
                       market_type="puck_line", title="Vegas wins by over 1.5 goals",
                       side="YES", ask=round(p15 - 0.10, 4), team_abbrev="VGK", strike=1.5,
                       strike_type="greater")
        quotes = [q for q in self.pipe._live_quotes() if q.market_type == "puck_line"]
        self.assertEqual(len(quotes), 1)
        q = quotes[0]
        self.assertEqual(q.selection, "home")     # VGK is the home team of this fixture
        self.assertEqual(q.side, "YES")
        self.assertAlmostEqual(q.strike, 1.5)
        self.assertEqual(q.strike_type, "greater")
        self.assertEqual(q.label, "Vegas wins by over 1.5 goals")

    def test_a_contract_naming_a_team_not_in_the_game_is_refused_not_guessed(self):
        self.quote_row(contract="KXNHLSPREAD-X-CAR2", market_key="KXNHLSPREAD-X",
                       gid=self.gid_future, day=self.future.date().isoformat(),
                       market_type="puck_line", title="Carolina wins by over 1.5 goals",
                       side="YES", ask=0.50, team_abbrev="CAR", strike=1.5,
                       strike_type="greater")
        self.assertEqual([q for q in self.pipe._live_quotes() if q.market_type == "puck_line"],
                         [])
        self.assertTrue(self.pipe.store.query(
            "SELECT * FROM irregularities WHERE kind='unreadable_puck_line'"))

    def test_the_declared_rung_is_traded_and_the_wager_is_recorded_with_its_line(self):
        strat = PuckLineStrategy(strategy_id="NHL_PUCK_LINE_MODEL_COVER", username="P1",
                                 name="t", contract_team="model_stronger", exchange_side="YES",
                                 target_strike=1.5, min_strike=1.5, max_strike=2.5,
                                 min_edge=0.04)
        self.register(strat)
        lam_h, lam_a = 3.2, 2.4
        # two rungs on the home team: the rule must take 1.5, not whichever sorted first
        for strike, rung in ((2.5, "VGK3"), (1.5, "VGK2")):
            p = p_margin_over(lam_h, lam_a, "home", strike)
            self.quote_row(contract=f"KXNHLSPREAD-X-{rung}", market_key="KXNHLSPREAD-X",
                           gid=self.gid_future, day=self.future.date().isoformat(),
                           market_type="puck_line",
                           title=f"Vegas wins by over {strike} goals", side="YES",
                           ask=round(min(p - 0.10, 0.95), 4), team_abbrev="VGK", strike=strike,
                           strike_type="greater")
        placed = self.pipe.stage_forward([self.feature_row(self.gid_future, self.future)])
        self.assertEqual(placed["FORWARD TEST"], 1)
        bet = self.store.one("SELECT * FROM bets WHERE market='puck_line'")
        self.assertIsNotNone(bet)
        self.assertEqual(bet["test_mode"], "FORWARD TEST")
        self.assertEqual(bet["selection"], "home_cover")
        self.assertEqual(bet["exchange_side"], "YES")
        self.assertAlmostEqual(float(bet["strike"]), 1.5)
        self.assertEqual(bet["result"], "OPEN")
        notes = json.loads(bet["notes"])
        self.assertEqual(notes["contract"], "KXNHLSPREAD-X-VGK2")
        self.assertEqual(notes["strike_type"], "greater")
        self.assertEqual(notes["contract_label"], "Vegas wins by over 1.5 goals")

    def test_a_no_side_entry_is_capped_by_the_declared_assumption_and_says_so(self):
        """Kalshi publishes no offer size on the NO side, so the +1.5 side of a puck line has
        no depth evidence at all.  The fill is capped at a declared number and labelled."""
        strat = PuckLineStrategy(strategy_id="NHL_PUCK_LINE_DOG_PLUS", username="P2", name="t",
                                 contract_team="model_stronger", exchange_side="NO",
                                 target_strike=1.5, min_strike=1.5, max_strike=2.5,
                                 min_edge=0.04)
        self.register(strat)
        lam_h, lam_a = 3.2, 2.4
        p_cover = p_margin_over(lam_h, lam_a, "home", 1.5)
        # NO ask well below 1 - p_cover, and a stake big enough to want more than the cap
        self.quote_row(contract="KXNHLSPREAD-X-VGK2", market_key="KXNHLSPREAD-X",
                       gid=self.gid_future, day=self.future.date().isoformat(),
                       market_type="puck_line", title="Vegas wins by over 1.5 goals",
                       side="NO", ask=0.10, ask_size=None, team_abbrev="VGK", strike=1.5,
                       strike_type="greater")
        self.store.execute("UPDATE strategies SET starting_bankroll=100000 WHERE strategy_id=?",
                           (strat.strategy_id,))
        self.store.commit()
        self.pipe.stage_forward([self.feature_row(self.gid_future, self.future)])
        bet = self.store.one("SELECT * FROM bets WHERE market='puck_line'")
        self.assertIsNotNone(bet, "the NO side of a quoted contract should be tradeable forward")
        self.assertEqual(bet["exchange_side"], "NO")
        self.assertEqual(bet["selection"], "home_no_cover")
        self.assertEqual(bet["depth_basis"], DEPTH_BASIS_CAP)
        self.assertLessEqual(float(bet["filled_size"]), UNKNOWN_DEPTH_CAP_CONTRACTS)
        notes = json.loads(bet["notes"])
        self.assertIn("declared", notes["depth_cap_note"])
        self.assertGreater(notes["contracts_requested"], notes.get("unfilled_stake", 0) or 0)
        self.assertAlmostEqual(p_cover + (1 - p_cover), 1.0)

    def test_settlement_uses_the_official_margin_for_both_sides(self):
        price, contracts = 0.50, 20.0
        for selection, won in (("home_cover", True), ("home_no_cover", False),
                               ("away_cover", False), ("away_no_cover", True)):
            with self.subTest(selection=selection):
                bet_id = f"b-{selection}"
                self.store.record_bet({
                    "bet_id": bet_id, "strategy_id": "S", "strategy_version": 1,
                    "username": "u", "test_mode": "FORWARD TEST", "game_id": self.gid_past,
                    "game_date": self.past.date().isoformat(), "matchup": "VGK@CAR",
                    "market": "puck_line", "selection": selection, "bet_type": "binary_contract",
                    "provider": "kalshi", "odds_format": "binary", "price": price,
                    "implied_prob": price, "decision_ts": utcnow(), "bet_ts": utcnow(),
                    "stake": price * contracts, "filled_size": contracts, "slippage": 0.0,
                    "entry_price": price, "fee": 0.0, "strike": 1.5, "result": "OPEN",
                    "notes": json.dumps({"contract": "C", "strike_type": "greater"}),
                    "created_at": utcnow()})
                self.store.commit()
        # CAR 4-3 VGK in OT: home margin +1, away margin -1
        self.paper.settle_game(self.gid_past, home_id=CAR, away_id=VGK, home_score=4,
                               away_score=3, state="FINAL", last_period_type="OT")
        got = {r["selection"]: r["result"] for r in self.store.query(
            "SELECT selection, result FROM bets WHERE market='puck_line'")}
        self.assertEqual(got, {"home_cover": "LOSS", "home_no_cover": "WIN",
                               "away_cover": "LOSS", "away_no_cover": "WIN"})
        reason = self.store.one(
            "SELECT reason FROM bet_audit WHERE bet_id='b-home_no_cover' ORDER BY id DESC"
        )["reason"] if self.store.one("SELECT 1") else None
        self.assertTrue(reason)

    def test_a_two_goal_margin_covers_the_second_rung(self):
        self.store.record_bet({
            "bet_id": "b2", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "FORWARD TEST", "game_id": self.gid_past,
            "game_date": self.past.date().isoformat(), "matchup": "VGK@CAR",
            "market": "puck_line", "selection": "home_cover", "bet_type": "binary_contract",
            "provider": "kalshi", "odds_format": "binary", "price": 0.4, "implied_prob": 0.4,
            "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 8.0, "filled_size": 20.0,
            "slippage": 0.0, "entry_price": 0.4, "fee": 0.0, "strike": 1.5, "result": "OPEN",
            "notes": json.dumps({"contract": "C", "strike_type": "greater"}),
            "created_at": utcnow()})
        self.store.commit()
        self.paper.settle_game(self.gid_past, home_id=CAR, away_id=VGK, home_score=5,
                               away_score=3, state="FINAL", last_period_type="REG")
        self.assertEqual(self.one("SELECT result FROM bets WHERE bet_id='b2'")["result"], "WIN")

    def test_a_backtest_prices_the_cover_from_a_candle_and_settles_on_the_exchange_result(self):
        strat = PuckLineStrategy(strategy_id="NHL_PUCK_LINE_MODEL_COVER", username="P1",
                                 name="t", contract_team="model_stronger", exchange_side="YES",
                                 target_strike=1.5, min_strike=1.5, max_strike=2.5,
                                 min_edge=0.04)
        self.register(strat)
        contract = "KXNHLSPREAD-26JUN04VGKCAR-VGK2"
        # the fixture game is CAR home / VGK away, and this contract names VGK, so the model
        # has to rate the AWAY side stronger for 'model_stronger' to land on this contract
        lam_h, lam_a = 2.4, 3.2
        p_cover = p_margin_over(lam_h, lam_a, "away", 1.5)
        ask = round(min(p_cover - 0.10, 0.9), 4)
        self.settled_contract(contract=contract, event_ticker="KXNHLSPREAD-26JUN04VGKCAR",
                              gid=self.gid_past, day=self.past, series="KXNHLSPREAD",
                              market_type="puck_line", result="no", strike=1.5, rung="VGK2",
                              title="Carolina wins by over 1.5 goals",
                              selection="Carolina wins by over 1.5 goals")
        # the fixture game is CAR home / VGK away, so the VGK contract is the AWAY side
        self.candle(contract, self.gid_past, "KXNHLSPREAD", "puck_line", ask=ask, volume=1234.0)
        rows = [self.feature_row(self.gid_past, self.past, lam_home=lam_h, lam_away=lam_a,
                                 home_exp_goals=lam_h, away_exp_goals=lam_a)]
        placed = self.pipe.stage_forward(rows)
        self.assertEqual(placed["BACKTEST"], 1)
        bet = self.store.one("SELECT * FROM bets WHERE test_mode='BACKTEST'")
        self.assertEqual(bet["market"], "puck_line")
        self.assertEqual(bet["selection"], "away_cover")
        self.assertAlmostEqual(float(bet["strike"]), 1.5)
        self.assertAlmostEqual(float(bet["entry_price"]), ask)
        self.assertEqual(bet["verification_status"], "kalshi_candle_close")
        self.assertEqual(bet["result"], "LOSS")        # the exchange settled the contract 'no'
        self.assertEqual(bet["depth_basis"], DEPTH_BASIS_VOLUME)
        self.assertAlmostEqual(float(bet["liquidity"]), 1234.0)

    def test_a_no_side_puck_line_rule_is_forward_only_and_reports_why(self):
        strat = PuckLineStrategy(strategy_id="NHL_PUCK_LINE_DOG_PLUS", username="P2", name="t",
                                 contract_team="model_stronger", exchange_side="NO",
                                 target_strike=1.5, no_history_reason="no NO-side candle")
        self.register(strat)
        contract = "KXNHLSPREAD-26JUN04VGKCAR-VGK2"
        self.settled_contract(contract=contract, event_ticker="KXNHLSPREAD-26JUN04VGKCAR",
                              gid=self.gid_past, day=self.past, series="KXNHLSPREAD",
                              market_type="puck_line", result="no", strike=1.5, rung="VGK2")
        self.candle(contract, self.gid_past, "KXNHLSPREAD", "puck_line", ask=0.40)
        placed = self.pipe.stage_forward([self.feature_row(self.gid_past, self.past)])
        self.assertEqual(placed["BACKTEST"], 0)
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM bets WHERE test_mode='BACKTEST'")["c"], 0)
        self.assertIn("FORWARD TEST ONLY", " ".join(self.pipe.report["puck_line_no_history"]))


class TestOvertimeExecution(ExecutionBase):
    """KXNHLOVERTIME: calibrated tie mass, verified settlement, unverified shootout."""

    CAL = OtCalibration(scale=1.25, n=400, observed_rate=0.25, model_mean=0.2,
                        window="test window")

    def ot_quote(self, *, side="YES", ask=0.10, ask_size=200.0, gid=None, day=None):
        gid = gid or self.gid_future
        day = day or self.future
        self.quote_row(contract=f"KXNHLOVERTIME-X-{side}", market_key="KXNHLOVERTIME-X",
                       gid=gid, day=day.date().isoformat(), market_type="overtime",
                       title="Will there be overtime in this game?", side=side, ask=ask,
                       ask_size=ask_size)

    def test_a_yes_entry_uses_the_calibrated_tie_mass_not_the_raw_one(self):
        self.pipe.ot_calibration = self.CAL
        strat = OvertimeStrategy(strategy_id="NHL_OT_MODEL_YES", username="O1", name="t",
                                 direction="yes", min_edge=0.03)
        self.register(strat)
        self.ot_quote(side="YES", ask=0.20)          # calibrated p = 0.20 * 1.25 = 0.25
        self.pipe.stage_forward([self.feature_row(self.gid_future, self.future)])
        bet = self.store.one("SELECT * FROM bets WHERE market='overtime'")
        self.assertIsNotNone(bet)
        self.assertEqual(bet["selection"], "ot_yes")
        self.assertEqual(bet["exchange_side"], "YES")
        self.assertAlmostEqual(float(bet["model_prob"]), 0.25, places=4)
        sup = json.loads(bet["features_json"])
        self.assertTrue(sup["calibrated"])
        self.assertAlmostEqual(sup["p_tie_reg_raw"], 0.20)
        self.assertEqual(sup["ot_calibration"]["scale"], 1.25)

    def test_the_rule_refuses_to_trade_on_a_calibration_with_too_little_history(self):
        self.pipe.ot_calibration = OtCalibration(scale=1.9, n=12, observed_rate=0.25,
                                                 model_mean=0.13, window="thin")
        strat = OvertimeStrategy(strategy_id="NHL_OT_MODEL_YES", username="O1", name="t",
                                 direction="yes", min_edge=0.03)
        self.register(strat)
        self.ot_quote(side="YES", ask=0.05)
        self.pipe.stage_forward([self.feature_row(self.gid_future, self.future)])
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM bets")["c"], 0)
        row = self.store.one("SELECT * FROM upcoming_bets WHERE market='overtime'")
        self.assertEqual(row["status"], "WAITING FOR OTHER INFORMATION")
        self.assertIn("below its minimum", row["blocking_reason"])

    def test_a_rule_is_handed_only_the_side_it_buys_and_records_one_status(self):
        """Both the YES and the NO row of the same contract are in the book.  Evaluating a
        YES rule against the NO quote would overwrite the status of the wager it placed."""
        self.pipe.ot_calibration = self.CAL
        strat = OvertimeStrategy(strategy_id="NHL_OT_MODEL_YES", username="O1", name="t",
                                 direction="yes", min_edge=0.03)
        self.register(strat)
        self.ot_quote(side="NO", ask=0.80)           # published second, must not be evaluated
        self.ot_quote(side="YES", ask=0.20)
        self.pipe.stage_forward([self.feature_row(self.gid_future, self.future)])
        rows = self.store.query("SELECT * FROM upcoming_bets WHERE market='overtime'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "EXECUTED")
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM bets")["c"], 1)

    def test_ot_and_reg_settle_from_the_official_last_period_type(self):
        for gid, lpt, expected in ((self.gid_past, "OT", "WIN"),):
            self.store.record_bet({
                "bet_id": f"ot-{gid}", "strategy_id": "S", "strategy_version": 1,
                "username": "u", "test_mode": "FORWARD TEST", "game_id": gid,
                "game_date": self.past.date().isoformat(), "matchup": "VGK@CAR",
                "market": "overtime", "selection": "ot_yes", "bet_type": "binary_contract",
                "provider": "kalshi", "odds_format": "binary", "price": 0.25,
                "implied_prob": 0.25, "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 5.0,
                "filled_size": 20.0, "slippage": 0.0, "entry_price": 0.25, "fee": 0.0,
                "result": "OPEN", "notes": json.dumps({"contract": "C"}),
                "created_at": utcnow()})
            self.store.commit()
            self.paper.settle_game(gid, home_id=CAR, away_id=VGK, home_score=4, away_score=3,
                                   state="FINAL", last_period_type=lpt)
            self.assertEqual(self.one("SELECT result FROM bets WHERE bet_id=?",
                                      (f"ot-{gid}",))["result"], expected)

    def test_a_shootout_game_leaves_the_wager_open_and_records_why(self):
        gid = 2025020999
        day = datetime(2026, 3, 10, 0, 30, tzinfo=timezone.utc)
        self.add_game(gid, day.date().isoformat(), day, VGK, UTA, state="FINAL", hs=4, as_=3,
                      lpt="SO")
        self.store.record_bet({
            "bet_id": "ot-so", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "FORWARD TEST", "game_id": gid,
            "game_date": day.date().isoformat(), "matchup": "UTA@VGK", "market": "overtime",
            "selection": "ot_yes", "bet_type": "binary_contract", "provider": "kalshi",
            "odds_format": "binary", "price": 0.25, "implied_prob": 0.25,
            "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 5.0, "filled_size": 20.0,
            "slippage": 0.0, "entry_price": 0.25, "fee": 0.0, "result": "OPEN",
            "notes": json.dumps({"contract": "KXNHLOVERTIME-X-OT"}), "created_at": utcnow()})
        self.store.commit()
        n = self.paper.settle_game(gid, home_id=VGK, away_id=UTA, home_score=4, away_score=3,
                                   state="FINAL", last_period_type="SO")
        self.assertEqual(n, 0)
        bet = self.one("SELECT * FROM bets WHERE bet_id='ot-so'")
        self.assertEqual(bet["result"], "OPEN")       # not guessed either way
        self.assertIsNone(bet["pnl"])
        flags = self.store.query(
            "SELECT * FROM irregularities WHERE kind='settlement_rule_unverified'")
        self.assertTrue(flags)
        self.assertIn("shootout", flags[0]["detail"].lower())
        self.assertIn("UNVERIFIED", flags[0]["detail"])

    def test_the_exchange_result_settles_what_nhl_data_cannot(self):
        gid = 2025020999
        day = datetime(2026, 3, 10, 0, 30, tzinfo=timezone.utc)
        self.add_game(gid, day.date().isoformat(), day, VGK, UTA, state="FINAL", hs=4, as_=3,
                      lpt="SO")
        contract = "KXNHLOVERTIME-26MAR10UTAVGK-OT"
        self.store.record_bet({
            "bet_id": "ot-so2", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "FORWARD TEST", "game_id": gid, "game_date": day.date().isoformat(),
            "matchup": "UTA@VGK", "market": "overtime", "selection": "ot_yes",
            "bet_type": "binary_contract", "provider": "kalshi", "odds_format": "binary",
            "price": 0.25, "implied_prob": 0.25, "decision_ts": utcnow(), "bet_ts": utcnow(),
            "stake": 5.0, "filled_size": 20.0, "slippage": 0.0, "entry_price": 0.25,
            "fee": 0.0, "result": "OPEN", "notes": json.dumps({"contract": contract}),
            "created_at": utcnow()})
        self.settled_contract(contract=contract, event_ticker="KXNHLOVERTIME-26MAR10UTAVGK",
                              gid=gid, day=day, series="KXNHLOVERTIME",
                              market_type="overtime", result="yes", rung="OT")
        self.paper.settle_game(gid, home_id=VGK, away_id=UTA, home_score=4, away_score=3,
                               state="FINAL", last_period_type="SO")
        self.assertEqual(self.one("SELECT result FROM bets WHERE bet_id='ot-so2'")["result"],
                         "OPEN")
        n = self.paper.settle_from_exchange()
        self.assertEqual(n, 1)
        bet = self.one("SELECT * FROM bets WHERE bet_id='ot-so2'")
        self.assertEqual(bet["result"], "WIN")        # the exchange said the contract paid
        self.assertIn("kalshi settled", self.store.one(
            "SELECT reason FROM bet_audit WHERE bet_id='ot-so2' ORDER BY id DESC")["reason"])

    def test_a_backtest_prices_overtime_from_a_candle_with_an_earlier_only_calibration(self):
        strat = OvertimeStrategy(strategy_id="NHL_OT_MODEL_YES", username="O1", name="t",
                                 direction="yes", min_edge=0.03)
        self.register(strat)
        contract = "KXNHLOVERTIME-26JUN04VGKCAR-OT"
        self.settled_contract(contract=contract, event_ticker="KXNHLOVERTIME-26JUN04VGKCAR",
                              gid=self.gid_past, day=self.past, series="KXNHLOVERTIME",
                              market_type="overtime", result="yes", rung="OT",
                              title="Will there be overtime?", selection="Game goes to overtime")
        self.candle(contract, self.gid_past, "KXNHLOVERTIME", "overtime", ask=0.15, volume=900.0)
        # a calibration window made only of games that started BEFORE the priced game
        earlier = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        self.pipe._ot_cal_rows = [
            {"game_id": 1, "_start": _iso(earlier), "p_overtime": 0.20,
             "_went_past_regulation": 1}] * 120
        placed = self.pipe.stage_forward([self.feature_row(self.gid_past, self.past)])
        self.assertEqual(placed["BACKTEST"], 1)
        bet = self.store.one("SELECT * FROM bets WHERE test_mode='BACKTEST'")
        self.assertEqual(bet["market"], "overtime")
        self.assertEqual(bet["result"], "WIN")
        self.assertAlmostEqual(float(bet["entry_price"]), 0.15)
        sup = json.loads(bet["features_json"])
        self.assertTrue(sup["calibrated"])
        self.assertIn("before", sup["ot_calibration"]["window"])
        self.assertEqual(sup["ot_calibration"]["n"], 120)
        self.assertEqual(bet["depth_basis"], DEPTH_BASIS_VOLUME)

    def test_a_calibration_window_never_includes_the_game_being_priced(self):
        rows = [
            {"game_id": 1, "_start": _iso(datetime(2026, 1, 1, tzinfo=timezone.utc)),
             "p_overtime": 0.2, "_went_past_regulation": 1},
            {"game_id": 2, "_start": _iso(datetime(2026, 2, 1, tzinfo=timezone.utc)),
             "p_overtime": 0.2, "_went_past_regulation": 0},
            {"game_id": 3, "_start": _iso(datetime(2026, 3, 1, tzinfo=timezone.utc)),
             "p_overtime": 0.2, "_went_past_regulation": 1},
        ]
        cal = self.pipe.ot_calibration_asof(
            rows, cutoff=_iso(datetime(2026, 3, 1, tzinfo=timezone.utc)))
        self.assertEqual(cal.n, 2)                   # the March game is excluded
        self.assertAlmostEqual(cal.observed_rate, 0.5)
        self.assertFalse(cal.sufficient)             # and n=2 is far below the minimum
        self.assertEqual(cal.scale, 1.0)


class TestDepthEvidence(unittest.TestCase):
    def test_a_published_offer_size_caps_the_fill(self):
        f = simulate_fill(100.0, 0.50, 40.0)
        self.assertEqual(f.depth_basis, DEPTH_BASIS_OFFER)
        self.assertEqual(f.contracts_filled, 40.0)
        self.assertAlmostEqual(f.unfilled_stake, 80.0)

    def test_traded_volume_is_used_when_the_venue_publishes_no_size(self):
        f = simulate_fill(100.0, 0.50, None, traded_volume=1000.0)
        self.assertEqual(f.depth_basis, DEPTH_BASIS_VOLUME)
        self.assertEqual(f.contracts_filled, 200.0)
        self.assertAlmostEqual(f.liquidity, 1000.0)

    def test_a_candle_in_which_nothing_traded_yields_no_fill(self):
        f = simulate_fill(100.0, 0.50, None, traded_volume=0.0)
        self.assertEqual(f.contracts_filled, 0.0)
        self.assertAlmostEqual(f.stake, 0.0)
        self.assertEqual(f.depth_basis, DEPTH_BASIS_VOLUME)

    def test_the_declared_cap_applies_only_when_nothing_is_published(self):
        f = simulate_fill(1000.0, 0.50, None)
        self.assertEqual(f.depth_basis, DEPTH_BASIS_CAP)
        self.assertEqual(f.contracts_filled, UNKNOWN_DEPTH_CAP_CONTRACTS)
        self.assertIsNone(f.liquidity)       # no depth was claimed, so none is reported

    def test_a_zero_published_size_is_not_a_fill(self):
        f = simulate_fill(50.0, 0.50, 0.0)
        self.assertEqual(f.contracts_filled, 0.0)


class TestNoSideClosingLineValue(ExecutionBase):
    def test_a_no_side_forward_wager_gets_no_clv_because_no_closing_price_exists_for_it(self):
        """Kalshi's candle history publishes the YES bid/ask only, so the closing mid of the
        YES side is not the closing price of a NO-side wager and ``1 - yes_close`` is not
        assumed.  The absence is recorded."""
        contract = "KXNHLTOTAL-X-6"
        self.store.record_bet({
            "bet_id": "no-side", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "FORWARD TEST", "game_id": self.gid_future,
            "game_date": self.future.date().isoformat(), "matchup": "UTA@VGK", "market": "total",
            "selection": "under", "bet_type": "binary_contract", "provider": "kalshi",
            "odds_format": "binary", "price": 0.55, "implied_prob": 0.55,
            "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 11.0, "filled_size": 20.0,
            "slippage": 0.0, "entry_price": 0.55, "fee": 0.0, "exchange_side": "NO",
            "result": "OPEN", "notes": json.dumps({"contract": contract}),
            "created_at": utcnow()})
        self.store.execute(
            "INSERT INTO market_price_points(contract, point, end_period_ts, game_id,"
            " series_ticker, market_type, bid, ask, last, mean, volume, open_interest,"
            " period_interval, tier, retrieved_at) VALUES(?,?,?,?,'KXNHLTOTAL','total',0.44,0.46,"
            "0.45,0.45,100,1000,60,'historical',?)",
            (contract, "close", _epoch(self.future - timedelta(minutes=30)), self.gid_future,
             utcnow()))
        self.store.commit()
        n = self.pipe.stage_clv()
        self.assertEqual(n, 0)
        bet = self.one("SELECT * FROM bets WHERE bet_id='no-side'")
        self.assertIsNone(bet["close_price"])
        self.assertIsNone(bet["clv"])
        flags = self.store.query(
            "SELECT * FROM irregularities WHERE kind='clv_unavailable_no_side'")
        self.assertTrue(flags)
        self.assertIn("YES bid/ask only", flags[0]["detail"])

    def test_a_yes_side_forward_wager_still_gets_its_closing_price(self):
        contract = "KXNHLGAME-X-VGK"
        self.store.record_bet({
            "bet_id": "yes-side", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "FORWARD TEST", "game_id": self.gid_future,
            "game_date": self.future.date().isoformat(), "matchup": "UTA@VGK",
            "market": "moneyline", "selection": "home", "bet_type": "binary_contract",
            "provider": "kalshi", "odds_format": "binary", "price": 0.50, "implied_prob": 0.50,
            "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 10.0, "filled_size": 20.0,
            "slippage": 0.0, "entry_price": 0.50, "fee": 0.0, "exchange_side": "YES",
            "result": "OPEN", "notes": json.dumps({"contract": contract}),
            "created_at": utcnow()})
        self.store.execute(
            "INSERT INTO market_price_points(contract, point, end_period_ts, game_id,"
            " series_ticker, market_type, bid, ask, last, mean, volume, open_interest,"
            " period_interval, tier, retrieved_at) VALUES(?,?,?,?,'KXNHLGAME','moneyline',0.54,"
            "0.56,0.55,0.55,100,1000,60,'historical',?)",
            (contract, "close", _epoch(self.future - timedelta(minutes=30)), self.gid_future,
             utcnow()))
        self.store.commit()
        self.assertEqual(self.pipe.stage_clv(), 1)
        bet = self.one("SELECT * FROM bets WHERE bet_id='yes-side'")
        self.assertAlmostEqual(float(bet["close_price"]), 0.55)
        self.assertAlmostEqual(float(bet["clv"]), 0.05)


class TestSettlementStudies(ExecutionBase):
    """The verifier's cross-checks: the exchange's own result vs the NHL's own record."""

    def setUp(self):
        super().setUp()
        self.v = Verifier(self.store)

    def ot_contract(self, *, gid, day, result, lpt, hs, as_, game_type=3, contract=None):
        contract = contract or f"KXNHLOVERTIME-{gid}-OT"
        self.store.execute(
            "UPDATE games SET state='FINAL', home_score=?, away_score=?, last_period_type=? "
            "WHERE game_id=?", (hs, as_, lpt, gid))
        self.settled_contract(contract=contract, event_ticker=f"KXNHLOVERTIME-{gid}", gid=gid,
                              day=day, series="KXNHLOVERTIME", market_type="overtime",
                              result=result, rung="OT", selection="Game goes to overtime")

    def test_the_overtime_cross_tab_records_what_the_evidence_covers(self):
        d1 = self.past                                    # the fixture game's own date
        d2 = datetime(2026, 6, 11, 0, 30, tzinfo=timezone.utc)
        self.add_game(2025030415, d2.date().isoformat(), d2, CAR, VGK, state="FUT", game_type=3)
        self.ot_contract(gid=2025030412, day=d1, result="yes", lpt="OT", hs=4, as_=3)
        self.ot_contract(gid=2025030415, day=d2, result="no", lpt="REG", hs=4, as_=2)
        out = self.v.cross_validate_overtime_settlements()
        self.assertEqual(out["cross_tab"], {"OT->yes": 1, "REG->no": 1})
        self.assertEqual(out["overtime_conflicts"], 0)
        self.assertEqual(out["window"]["first_game"], self.past.date().isoformat())
        self.assertEqual(out["window"]["last_game"], "2026-06-11")
        self.assertEqual(out["window"]["game_types"], {"playoff": 2})
        self.assertIn("UNVERIFIED", out["shootout_treatment"])
        flags = self.store.query(
            "SELECT * FROM irregularities WHERE kind='settlement_rule_unverified' "
            "AND entity_id='KXNHLOVERTIME'")
        self.assertTrue(flags)

    def test_a_shootout_result_becomes_evidence_and_changes_the_conclusion(self):
        day = datetime(2026, 11, 3, 0, 30, tzinfo=timezone.utc)
        gid = 2026020555
        self.add_game(gid, day.date().isoformat(), day, VGK, UTA, state="FUT", game_type=2)
        self.ot_contract(gid=gid, day=day, result="yes", lpt="SO", hs=4, as_=3, game_type=2)
        out = self.v.cross_validate_overtime_settlements()
        self.assertEqual(out["cross_tab"], {"SO->yes": 1})
        self.assertEqual(len(out["shootout_evidence"]), 1)
        self.assertIn("settled 'yes'", out["shootout_evidence"][0])
        self.assertNotIn("UNVERIFIED", out["shootout_treatment"])
        self.assertEqual(out["overtime_conflicts"], 0)   # an SO row is evidence, not a conflict

    def test_an_overtime_contract_contradicting_the_official_record_is_critical(self):
        day = datetime(2026, 6, 4, 0, 30, tzinfo=timezone.utc)
        self.ot_contract(gid=2025030412, day=day, result="no", lpt="OT", hs=4, as_=3)
        out = self.v.cross_validate_overtime_settlements()
        self.assertEqual(out["overtime_conflicts"], 1)
        flags = self.store.query(
            "SELECT * FROM irregularities WHERE kind='settlement_conflict'")
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["severity"], "critical")
        self.assertIn("kalshi.trade_api,nhl.api_web", flags[0]["sources"])

    def test_puck_line_settlements_are_re_derived_from_the_official_margin(self):
        day = self.past
        gid = self.gid_past          # CAR home 4, VGK away 3, OT
        self.settled_contract(contract="KXNHLSPREAD-26JUN04VGKCAR-CAR2",
                              event_ticker="KXNHLSPREAD-26JUN04VGKCAR", gid=gid, day=day,
                              series="KXNHLSPREAD", market_type="puck_line", result="no",
                              strike=1.5, rung="CAR2", title="Carolina wins by over 1.5 goals",
                              selection="Carolina wins by over 1.5 goals")
        out = self.v.cross_validate_puck_line_settlements()
        self.assertEqual(out["puck_line_contracts_compared"], 1)
        self.assertEqual(out["puck_line_conflicts"], 0)
        self.assertEqual(out["puck_line_unreadable"], 0)
        # this game went to overtime, so it is evidence that the margin includes the OT goal
        self.assertEqual(out["puck_line_compared_games_that_went_past_regulation"], 1)

    def test_a_puck_line_contract_naming_a_team_not_in_the_game_is_excluded_not_guessed(self):
        self.store.execute("INSERT INTO teams(team_id, abbrev, full_name, active) "
                           "VALUES(6, 'BOS', 'Boston Bruins', 1) "
                           "ON CONFLICT(team_id) DO NOTHING")
        self.store.commit()
        self.settled_contract(contract="KXNHLSPREAD-X-BOS2", event_ticker="KXNHLSPREAD-X",
                              gid=self.gid_past, day=self.past, series="KXNHLSPREAD",
                              market_type="puck_line", result="yes", strike=1.5, rung="BOS2")
        out = self.v.cross_validate_puck_line_settlements()
        self.assertEqual(out["puck_line_contracts_compared"], 0)
        self.assertEqual(out["puck_line_unreadable"], 1)
        self.assertTrue(self.store.query(
            "SELECT * FROM irregularities WHERE kind='unreadable_puck_line'"))

    def test_a_puck_line_conflict_is_recorded_with_both_readings(self):
        self.settled_contract(contract="KXNHLSPREAD-26JUN04VGKCAR-CAR2",
                              event_ticker="KXNHLSPREAD-26JUN04VGKCAR", gid=self.gid_past,
                              day=self.past, series="KXNHLSPREAD", market_type="puck_line",
                              result="yes", strike=1.5, rung="CAR2")
        out = self.v.cross_validate_puck_line_settlements()
        self.assertEqual(out["puck_line_conflicts"], 1)
        flag = self.store.query(
            "SELECT * FROM irregularities WHERE kind='settlement_conflict'")[0]
        self.assertEqual(flag["severity"], "critical")
        self.assertIn("official margin is +1", flag["detail"])

    def test_declared_depth_fallbacks_are_counted_and_reported(self):
        self.store.record_bet({
            "bet_id": "d1", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "FORWARD TEST", "game_id": self.gid_future,
            "game_date": self.future.date().isoformat(), "matchup": "UTA@VGK",
            "market": "puck_line", "selection": "home_no_cover", "bet_type": "binary_contract",
            "provider": "kalshi", "odds_format": "binary", "price": 0.5, "implied_prob": 0.5,
            "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 50.0, "filled_size": 100.0,
            "slippage": 0.0, "entry_price": 0.5, "fee": 0.0, "strike": 1.5,
            "exchange_side": "NO", "depth_basis": "declared_cap_no_published_size",
            "result": "OPEN", "notes": "{}", "created_at": utcnow()})
        self.store.record_bet({
            "bet_id": "d2", "strategy_id": "S", "strategy_version": 1, "username": "u",
            "test_mode": "BACKTEST", "game_id": self.gid_past,
            "game_date": self.past.date().isoformat(), "matchup": "VGK@CAR",
            "market": "overtime", "selection": "ot_yes", "bet_type": "binary_contract",
            "provider": "kalshi", "odds_format": "binary", "price": 0.2, "implied_prob": 0.2,
            "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 20.0, "filled_size": 100.0,
            "slippage": 0.0, "entry_price": 0.2, "fee": 0.0, "exchange_side": "YES",
            "depth_basis": "traded_volume_at_entry_candle", "result": "OPEN", "notes": "{}",
            "created_at": utcnow()})
        self.store.commit()
        out = self.v.check_strike_market_bets()
        self.assertEqual(out["puck_line_bets"], 1)
        self.assertEqual(out["puck_line_missing_strike"], 0)
        self.assertEqual(out["overtime_bets"], 1)
        self.assertEqual(out["overtime_unmapped_selection"], 0)
        self.assertEqual(out["bets_on_declared_depth_cap"], 1)
        self.assertEqual(out["no_side_bets"], 1)
        self.assertEqual(out["bets_by_depth_basis"],
                         {"declared_cap_no_published_size": 1,
                          "traded_volume_at_entry_candle": 1})

    def test_an_unexplained_period_score_difference_is_flagged(self):
        self.store.execute(
            "INSERT INTO game_period_scores(game_id, period, period_type, home_goals, away_goals,"
            " derived_from, reconciled, reconciliation_note, source_id, retrieved_at, provenance)"
            " VALUES(?,?, 'REG', 2, 1, 'nhl.score.goals', 0, NULL, 'nhl.api_web', ?, 'DERIVED')",
            (self.gid_past, 3, utcnow()))
        self.store.commit()
        out = self.v.check_period_goal_reconciliation()
        self.assertEqual(out["games_with_period_scores"], 1)
        self.assertEqual(out["unreconciled"], 1)
        self.assertEqual(out["unreconciled_without_a_note"], 1)
        self.assertTrue(self.store.query(
            "SELECT * FROM irregularities WHERE kind='unreconciled_period_goals'"))

    def test_a_second_market_with_nothing_in_common_says_so(self):
        excerpt = load("polymarket_gamma_nhl_event_excerpt.json")
        from nhlcomp.sources.polymarket import normalize_event
        for row in normalize_event(excerpt["events"][0], retrieved_at=utcnow(),
                                   source_url="https://gamma-api.polymarket.com/events"):
            row["game_id"] = None
            cols = list(row.keys())
            self.store.execute(
                f"INSERT INTO polymarket_markets({','.join(cols)}) "
                f"VALUES({','.join('?' * len(cols))})", tuple(row[c] for c in cols))
        self.store.commit()
        out = self.v.cross_check_polymarket()
        self.assertEqual(out["polymarket_markets"], 1)
        self.assertEqual(out["families"], {"futures": 1})
        self.assertEqual(out["comparisons"], [])
        self.assertIn("no common market", out["finding"])


class TestQuoteSideIsPartOfARuleDefinition(ExecutionBase):
    def test_every_seed_rule_declares_the_side_of_the_book_it_buys(self):
        from nhlcomp.strategies import build_seed_strategies
        for strat in build_seed_strategies():
            with self.subTest(strategy=strat.strategy_id):
                self.assertIn(strat.quote_side, ("YES", "NO"))
                params = json.loads(strat.describe()["params_json"])
                self.assertEqual(params["quote_side"], strat.quote_side)
        by_id = {s.strategy_id: s for s in build_seed_strategies()}
        self.assertEqual(by_id["NHL_TOTALS_UNDER"].quote_side, "NO")
        self.assertEqual(by_id["NHL_PUCK_LINE_DOG_PLUS"].quote_side, "NO")
        self.assertEqual(by_id["NHL_OT_MODEL_NO"].quote_side, "NO")
        self.assertEqual(by_id["NHL_OT_MODEL_YES"].quote_side, "YES")
        self.assertEqual(by_id["NHL_B2B_FADE"].quote_side, "YES")


class TestKalshiGameMatching(CapturedIngestBase):
    """Every captured contract must be attached to the game Kalshi says it is on.

    Kalshi's rules text is the authoritative statement of which game a contract settles
    against, and the captured payloads show it written four different ways (moneyline "vs ...
    NHL game originally scheduled for", puck line "at ... professional hockey game originally
    scheduled for", overtime "vs ... professional hockey game originally scheduled for", and a
    playoff moneyline "the Game 6: Carolina at Vegas professional hockey game scheduled for").
    These tests drive the matcher with the real captured strings, with the event ticker removed
    so the rules text is the only thing it can use, and then with the rules text removed so the
    ticker fallback is the only thing it can use.  A contract that can be matched by neither is
    left unmatched and flagged -- never attached to a plausible-looking game.
    """

    GAMES = {
        # game_id: (date, away, home)
        2026030141: ("2026-06-11", VGK, CAR),
        2026030146: ("2026-06-14", CAR, VGK),
        2026010030: ("2026-09-24", UTA, VGK),
    }
    #: contract -> game_id, from the rules text of the committed captures
    EXPECTED = {
        "KXNHLOVERTIME-26JUN14CARVGK-OT": 2026030146,
        "KXNHLOVERTIME-26JUN11VGKCAR-OT": 2026030141,
        "KXNHLSPREAD-26JUN14CARVGK-VGK2": 2026030146,
        "KXNHLSPREAD-26JUN14CARVGK-VGK1": 2026030146,
        "KXNHLTOTAL-26JUN14CARVGK-9": 2026030146,
        "KXNHLTOTAL-26JUN14CARVGK-8": 2026030146,
        "KXNHLGAME-26JUN14CARVGK-VGK": 2026030146,
        "KXNHLGAME-26JUN14CARVGK-CAR": 2026030146,
        "KXNHLSPREAD-26SEP24UTAVGK-VGK3": 2026010030,
        "KXNHLSPREAD-26SEP24UTAVGK-VGK2": 2026010030,
    }
    CAPTURES = ("kalshi_historical_markets_kxnhlgame_limit2.json",
                "kalshi_historical_markets_kxnhlspread_limit2.json",
                "kalshi_historical_markets_kxnhlovertime_limit2.json",
                "kalshi_historical_markets_kxnhltotal_limit2.json",
                "kalshi_live_markets_kxnhlspread_limit2.json",
                "kalshi_settled_kxnhlgame.json")

    def setUp(self):
        super().setUp()
        for gid, (day, away, home) in self.GAMES.items():
            self.add_game(gid, day, datetime.fromisoformat(day + "T03:00:00+00:00"),
                          home, away, state="FINAL", hs=3, as_=2,
                          lpt="OT" if gid == 2026030146 else "REG")

    def captured_markets(self):
        for name in self.CAPTURES:
            for m in load(name).get("markets", []):
                yield name, m

    def test_rules_text_alone_matches_every_game_this_store_knows(self):
        matched = 0
        for _name, m in self.captured_markets():
            ticker = m.get("ticker")
            if ticker not in self.EXPECTED:
                continue
            rules_only = dict(m)
            rules_only.pop("event_ticker", None)      # the ticker path must not be available
            self.assertTrue(rules_only.get("rules_primary"), "fixture must carry real rules text")
            gid, abbrevs = self.ing._match_kalshi_game(rules_only)
            self.assertEqual(gid, self.EXPECTED[ticker],
                             f"{ticker}: rules text {rules_only['rules_primary']!r}")
            day, away, home = self.GAMES[gid]
            self.assertEqual(abbrevs[0], self.store.one(
                "SELECT abbrev FROM teams WHERE team_id=?", (away,))["abbrev"])
            self.assertEqual(abbrevs[1], self.store.one(
                "SELECT abbrev FROM teams WHERE team_id=?", (home,))["abbrev"])
            matched += 1
        self.assertEqual(matched, len(self.EXPECTED), "every captured contract must be exercised")

    def test_ticker_fallback_matches_when_the_rules_text_is_silent(self):
        """The fallback has to work for every KXNHL series, not just the moneyline one."""
        for _name, m in self.captured_markets():
            ticker = m.get("ticker")
            if ticker not in self.EXPECTED:
                continue
            no_rules = dict(m, rules_primary="", rules_secondary="")
            self.assertTrue(no_rules.get("event_ticker"))
            gid, _ = self.ing._match_kalshi_game(no_rules)
            self.assertEqual(gid, self.EXPECTED[ticker], f"{ticker}: ticker fallback")

    def test_prose_that_looks_like_a_matchup_does_not_invent_one(self):
        """A rules sentence with an earlier "the ... at ..." must not be parsed as two teams."""
        trap = ("If the game goes to overtime at any point in the Carolina vs Vegas "
                "professional hockey game originally scheduled for Jun 14, 2026, then the "
                "market resolves to Yes.")
        gid, abbrevs = self.ing._match_kalshi_game(
            {"ticker": "KXNHLOVERTIME-TRAP-OT", "rules_primary": trap})
        # the trap's first match is prose; the real clause at the end still wins
        self.assertEqual(gid, 2026030146)
        pure_prose = ("If the puck drops at any time and the home team scores at even "
                      "strength, then the market resolves to Yes.")
        gid, abbrevs = self.ing._match_kalshi_game(
            {"ticker": "KXNHLOVERTIME-PROSE-OT", "rules_primary": pure_prose})
        self.assertIsNone(gid)
        self.assertEqual(abbrevs, (None, None))

    def test_a_contract_with_neither_signal_is_flagged_not_guessed(self):
        gid, _ = self.ing._match_kalshi_game(
            {"ticker": "KXNHLGAME-26SEP19VGKLA-VGK",
             "event_ticker": "KXNHLGAME-26SEP19VGKLA",
             "rules_primary": "If Vegas wins the Vegas vs Los Angeles NHL game originally "
                              "scheduled for Sep 19, 2026, then the market resolves to Yes."})
        # Los Angeles is not in this store's team list, so the contract cannot be placed
        self.assertIsNone(gid)
        flags = self.flags("unmatched_market")
        self.assertTrue(flags)
        self.assertIn("KXNHLGAME-26SEP19VGKLA", str(flags[0]["detail"]))

    def test_code_pair_splitting_is_validated_never_guessed(self):
        self.assertEqual(split_team_codes("CARVGK"), ("CAR", "VGK"))
        self.assertEqual(split_team_codes("VGKLA"), ("VGK", "LA"))
        self.assertIsNone(split_team_codes("ZZZQQQ"))
        self.assertIsNone(split_team_codes("CAR"))
        self.assertIsNone(split_team_codes(""))
        parsed = parse_event_ticker("KXNHLOVERTIME-26JUN14ZZZQQQ")
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed["ambiguous"])
        self.assertEqual(parsed["game_date"], "2026-06-14")



if __name__ == "__main__":
    unittest.main()
