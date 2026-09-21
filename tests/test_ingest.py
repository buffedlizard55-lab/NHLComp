"""Tests for the ingest layer, driven by real captured API payloads.

The payloads in ``data/captured`` were retrieved from live endpoints on 2026-09-20 and are
replayed through the real ``HttpClient`` cache path, so the code under test is the same code
that runs against the live APIs.
"""

import json
import os
import tempfile
import unittest

from nhlcomp.http import HttpClient
from nhlcomp.ingest import Ingestor
from nhlcomp.pipeline import build_team_names, _side_for_selection
from nhlcomp.features import GameRef
from nhlcomp.sources.kalshi import (KalshiApi, KalshiApiError, binary_price_to_decimal,
                                    contract_suffix, normalize_market)
from nhlcomp.sources.nhl import derive_team_games, normalize_game, parse_scoreboard_games
from nhlcomp.sources.registry import SOURCES, seed_registry
from nhlcomp.store import Store
from nhlcomp.http import parse_iso

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURED = os.path.normpath(os.path.join(HERE, "..", "data", "captured"))


def load(name: str) -> dict:
    with open(os.path.join(CAPTURED, name), encoding="utf-8") as fh:
        return json.load(fh)


def prime_cache(http: HttpClient, url: str, payload: dict) -> None:
    http._store(url, 200, json.dumps(payload))


class TestRealPayloadParsing(unittest.TestCase):
    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.cache = tempfile.mkdtemp()
        self.store = Store(self.dbpath)
        self.http = HttpClient(self.cache)
        self.ing = Ingestor(self.store, self.http, verbose=False)

    def tearDown(self):
        self.store.close()

    def test_teams_fixture_loads_and_indexes(self):
        url = "https://api.nhle.com/stats/rest/en/team"
        prime_cache(self.http, url, load("nhl_stats_rest_team.json"))
        n = self.ing.teams()
        self.assertEqual(n, 32)
        self.assertEqual(self.store.one("SELECT abbrev FROM teams WHERE team_id=54")["abbrev"],
                         "VGK")
        # the Utah rename must come from the source, not be hard-coded
        self.assertEqual(self.store.one("SELECT full_name FROM teams WHERE team_id=68")["full_name"],
                         "Utah Mammoth")

    def test_scoreboard_fixture_parses_real_games(self):
        payload = load("nhl_scoreboard_20260919.json")
        games = parse_scoreboard_games(payload, season=20262027, game_types=(1,))
        self.assertEqual(len(games), 6)
        vgk = [g for g in games if g["game_id"] == 2026010002][0]
        self.assertEqual(vgk["away_id"], 54)
        self.assertEqual(vgk["home_id"], 26)
        self.assertEqual(vgk["away_score"], 4)
        self.assertEqual(vgk["home_score"], 2)
        self.assertEqual(vgk["state"], "FINAL")
        # the venue UTC offset is carried through, which is what the travel feature uses
        self.assertEqual(vgk["utc_offset"], "-07:00")

    def test_overtime_game_records_last_period_type(self):
        payload = load("nhl_scoreboard_20260919.json")
        games = parse_scoreboard_games(payload, season=20262027, game_types=(1,))
        ot = [g for g in games if g["game_id"] == 2026010007][0]
        self.assertEqual(ot["last_period_type"], "OT")

    def test_derive_team_games_produces_otl(self):
        payload = load("nhl_scoreboard_20260919.json")
        games = parse_scoreboard_games(payload, season=20262027, game_types=(1,))
        tg = {r["team_id"]: r for r in derive_team_games(games) if r["game_id"] == 2026010007}
        self.assertEqual(tg[8]["result"], "W")     # Montreal won 4-3 in OT
        self.assertEqual(tg[10]["result"], "OTL")  # Toronto lost in OT

    def test_game_referencing_an_unknown_team_is_skipped_and_flagged(self):
        """A game must never be attached to a guessed franchise, and must not crash ingest."""
        prime_cache(self.http, "https://api.nhle.com/stats/rest/en/team",
                    load("nhl_stats_rest_team.json"))
        self.ing.teams()
        bogus = {"_capture": {}, "gamesByDate": [{"date": "2026-09-19", "games": [
            {"id": 999, "season": 20262027, "gameType": 2, "gameDate": "2026-09-19",
             "startTimeUTC": "2026-09-19T23:00:00Z", "venueUTCOffset": "-05:00",
             "gameState": "FINAL",
             "awayTeam": {"id": 4242, "abbrev": "ZZZ", "score": 1},
             "homeTeam": {"id": 19, "abbrev": "STL", "score": 3},
             "periodDescriptor": {"number": 3, "periodType": "REG"}}]}]}
        prime_cache(self.http, "https://api-web.nhle.com/v1/scoreboard/2026-09-19", bogus)
        n = self.ing.scoreboard_window("2026-09-19")
        self.assertEqual(n, 1)                      # parsed...
        self.assertIsNone(self.store.one("SELECT * FROM games WHERE game_id=999"))  # ...not stored
        irr = self.store.one("SELECT * FROM irregularities WHERE kind='unknown_team'")
        self.assertIsNotNone(irr)
        self.assertIn("4242", irr["detail"])

    def test_venues_populated_without_invented_coordinates(self):
        url = "https://api-web.nhle.com/v1/scoreboard/2026-09-19"
        prime_cache(self.http, url, load("nhl_scoreboard_20260919.json"))
        prime_cache(self.http, "https://api.nhle.com/stats/rest/en/team",
                    load("nhl_stats_rest_team.json"))
        self.ing.teams()
        self.ing.scoreboard_window("2026-09-19")
        v = self.store.one("SELECT * FROM venues WHERE venue='Toyota Arena'")
        self.assertIsNotNone(v)
        self.assertEqual(v["utc_offset"], "-07:00")
        self.assertIsNone(v["lat"])
        self.assertIsNone(v["lon"])
        self.assertIn("no verified first-party coordinate source", v["notes"])


class TestFranchiseDisambiguation(unittest.TestCase):
    """The records API returns one row per franchise, and triCodes repeat across renames.

    Real examples from api.nhle.com/stats/rest/en/team (captured 2026-09-20):
    team 59 'Utah Hockey Club' and team 68 'Utah Mammoth' are both UTA;
    team 36 'Ottawa Senators (1917)' and team 9 'Ottawa Senators' are both SEN.
    """

    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        for tid, name, ab in ((59, "Utah Hockey Club", "UTA"), (68, "Utah Mammoth", "UTA"),
                              (36, "Ottawa Senators (1917)", "SEN"),
                              (9, "Ottawa Senators", "SEN")):
            self.store.execute("INSERT INTO teams(team_id, abbrev, full_name) VALUES(?,?,?)",
                               (tid, ab, name))
        self.store.commit()

    def tearDown(self):
        self.store.close()

    def test_duplicate_abbrevs_coexist(self):
        rows = self.store.query("SELECT team_id FROM teams WHERE abbrev='UTA'")
        self.assertEqual(sorted(r["team_id"] for r in rows), [59, 68])

    def test_full_name_disambiguates(self):
        self.assertEqual(self.store.team_id_for("UTA", "Utah Mammoth"), 68)
        self.assertEqual(self.store.team_id_for("UTA", "Utah Hockey Club"), 59)
        self.assertEqual(self.store.team_id_for("SEN", "Ottawa Senators"), 9)

    def test_ambiguous_without_a_full_name_returns_none(self):
        self.assertIsNone(self.store.team_id_for("UTA"))

    def test_active_flag_disambiguates_when_only_one_is_current(self):
        self.store.execute("UPDATE teams SET active=0")
        self.store.execute("UPDATE teams SET active=1 WHERE team_id=68")
        self.store.commit()
        self.assertEqual(self.store.team_id_for("UTA"), 68)


class TestKalshiMatching(unittest.TestCase):
    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        self.http = HttpClient(tempfile.mkdtemp())
        self.ing = Ingestor(self.store, self.http, verbose=False)
        prime_cache(self.http, "https://api.nhle.com/stats/rest/en/team",
                    load("nhl_stats_rest_team.json"))
        prime_cache(self.http, "https://api-web.nhle.com/v1/scoreboard/2026-09-19",
                    load("nhl_scoreboard_20260919.json"))
        self.ing.teams()
        self.ing.scoreboard_window("2026-09-19")

    def tearDown(self):
        self.store.close()

    def test_settled_contract_matches_a_real_nhl_game(self):
        m = load("kalshi_settled_kxnhlgame.json")["markets"][0]
        gid, abbrevs = self.ing._match_kalshi_game(m)
        self.assertEqual(abbrevs, ("VGK", "LAK"))
        self.assertEqual(gid, 2026010002)

    def test_unmatched_contract_is_flagged_not_guessed(self):
        m = load("kalshi_settled_kxnhlgame.json")["markets"][2]   # VAN@SEA has no game row
        gid, abbrevs = self.ing._match_kalshi_game(m)
        self.assertIsNone(gid)
        self.assertIsNotNone(self.store.one(
            "SELECT * FROM irregularities WHERE kind='unmatched_market'"))

    def test_kalshi_result_agrees_with_the_nhl_result(self):
        """Cross-validation: Kalshi settled VGK 'yes'; the NHL scoreboard has Vegas 4-2."""
        m = load("kalshi_settled_kxnhlgame.json")["markets"][0]
        gid, _ = self.ing._match_kalshi_game(m)
        g = self.store.one("SELECT * FROM games WHERE game_id=?", (gid,))
        kalshi_winner = m["yes_sub_title"] if m["result"] == "yes" else m["no_sub_title"]
        away = self.store.one("SELECT abbrev FROM teams WHERE team_id=?", (g["away_id"],))
        nhl_winner = away["abbrev"] if g["away_score"] > g["home_score"] else \
            self.store.one("SELECT abbrev FROM teams WHERE team_id=?", (g["home_id"],))["abbrev"]
        team_id = self.store.one("SELECT team_id FROM teams WHERE abbrev=?",
                                 (kalshi_winner if kalshi_winner != "Vegas" else "VGK",))
        self.assertEqual(nhl_winner, "VGK")
        self.assertIsNotNone(team_id)

    def test_settled_ingest_records_price_and_result(self):
        prime_cache(self.http,
                    "https://api.elections.kalshi.com/trade-api/v2/markets"
                    "?series_ticker=KXNHLGAME&limit=200&status=settled",
                    load("kalshi_settled_kxnhlgame.json"))
        n = self.ing.kalshi_settled(max_pages=1)
        self.assertEqual(n, 3)
        row = self.store.one("SELECT * FROM market_settlements WHERE contract=?",
                             ("KXNHLGAME-26SEP19VGKLA-VGK",))
        self.assertEqual(row["result"], "yes")
        self.assertAlmostEqual(float(row["ask_before"]), 0.47)
        self.assertAlmostEqual(float(row["volume"]), 30711.0)
        self.assertEqual(int(row["game_id"]), 2026010002)

    def test_normalize_market_emits_both_sides(self):
        m = load("kalshi_settled_kxnhlgame.json")["markets"][0]
        rows = normalize_market(m, ts_utc="2026-09-20T04:32:34Z",
                                retrieved_at="2026-09-20T12:00:00Z", source_url="u")
        self.assertEqual([r["side"] for r in rows], ["YES", "NO"])
        self.assertAlmostEqual(float(rows[0]["spread"]), 1.0 - 0.0)

    def test_binary_price_to_decimal(self):
        self.assertAlmostEqual(binary_price_to_decimal(0.5), 2.0)
        self.assertTrue(binary_price_to_decimal(0.0) != binary_price_to_decimal(0.0))  # nan


class TestKalshiStatusFilter(unittest.TestCase):
    """Kalshi rejects unknown status filters. Verified 2026-09-20:
    status=finalized -> {"error":{"code":"bad_request","details":"invalid status filter"}}.
    An empty result and a rejected request must not look the same to the caller."""

    def test_invalid_status_raises_instead_of_returning_empty(self):
        api = KalshiApi(HttpClient(tempfile.mkdtemp()))
        with self.assertRaises(KalshiApiError):
            api.markets("KXNHLGAME", status="active")
        with self.assertRaises(KalshiApiError):
            api.markets("KXNHLGAME", status="finalized")

    def test_valid_statuses_accepted(self):
        api = KalshiApi(HttpClient(tempfile.mkdtemp()))
        for st in ("open", "settled", "closed", "unopened"):
            self.assertIn(st, api.VALID_STATUSES)

    def test_api_error_body_is_surfaced_not_swallowed(self):
        http = HttpClient(tempfile.mkdtemp())
        http._store("https://api.elections.kalshi.com/trade-api/v2/markets"
                    "?series_ticker=KXNHLGAME&limit=200&status=open",
                    200, json.dumps({"error": {"code": "bad_request",
                                               "message": "bad request"}}))
        api = KalshiApi(http)
        with self.assertRaises(KalshiApiError):
            api.markets("KXNHLGAME", status="open")


class TestSideResolution(unittest.TestCase):
    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        self.http = HttpClient(tempfile.mkdtemp())
        ing = Ingestor(self.store, self.http, verbose=False)
        prime_cache(self.http, "https://api.nhle.com/stats/rest/en/team",
                    load("nhl_stats_rest_team.json"))
        ing.teams()
        self.names = build_team_names(self.store)

    def tearDown(self):
        self.store.close()

    def test_place_name_resolves_to_the_correct_side(self):
        g = GameRef(game_id=1, start=parse_iso("2026-09-20T04:00:00Z"), game_date="2026-09-19",
                    home_id=26, away_id=54, venue="Toyota Arena", venue_tz=None,
                    utc_offset="-07:00", season=20262027, game_type=1)
        self.assertEqual(_side_for_selection("Vegas wins", g, self.names), "away")
        self.assertEqual(_side_for_selection("Los Angeles wins", g, self.names), "home")
        self.assertEqual(_side_for_selection("Vegas", g, self.names), "away")

    def test_ambiguous_or_unknown_names_return_none(self):
        g = GameRef(game_id=1, start=parse_iso("2026-09-20T04:00:00Z"), game_date="2026-09-19",
                    home_id=26, away_id=54, venue="Toyota Arena", venue_tz=None,
                    utc_offset="-07:00", season=20262027, game_type=1)
        self.assertIsNone(_side_for_selection("Nowhere wins", g, self.names))
        self.assertIsNone(_side_for_selection(None, g, self.names))


class TestAmbiguousFranchiseMatching(unittest.TestCase):
    """Regression: two franchises share the UTA triCode (59 Utah Hockey Club, 68 Utah
    Mammoth). Resolving through a bare ``WHERE abbrev=?`` subquery returned whichever row
    SQLite hit first, so real Kalshi contracts for Utah failed to match their NHL game."""

    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        self.http = HttpClient(tempfile.mkdtemp())
        self.ing = Ingestor(self.store, self.http, verbose=False)
        for tid, name, active in ((59, "Utah Hockey Club", 0), (68, "Utah Mammoth", 1),
                                  (26, "Los Angeles Kings", 1), (21, "Colorado Avalanche", 1)):
            ab = "UTA" if tid in (59, 68) else ("LAK" if tid == 26 else "COL")
            self.store.execute("INSERT INTO teams(team_id, abbrev, full_name, active) "
                               "VALUES(?,?,?,?)", (tid, ab, name, active))
        for gid, gd, away, home in ((300, "2026-09-22", 68, 26), (301, "2026-09-20", 68, 21)):
            self.store.execute(
                "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
                " home_id, away_id, state) VALUES(?,20262027,1,?,?,?,?,?)",
                (gid, gd, f"{gd}T01:00:00Z", home, away, "FUTURE"))
        self.store.commit()

    def tearDown(self):
        self.store.close()

    def test_utah_contract_matches_the_active_franchise_game(self):
        m = {"event_ticker": "KXNHLGAME-26SEP22UTALA",
             "rules_primary": "If Utah wins the Utah vs Los Angeles NHL game originally "
                              "scheduled for Sep 22, 2026, then the market resolves to Yes."}
        gid, abbrevs = self.ing._match_kalshi_game(m)
        self.assertEqual(abbrevs, ("UTA", "LAK"))
        self.assertEqual(gid, 300)

    def test_second_utah_contract_also_matches(self):
        m = {"event_ticker": "KXNHLGAME-26SEP20UTACOL",
             "rules_primary": "If Utah wins the Utah vs Colorado NHL game originally "
                              "scheduled for Sep 20, 2026, then the market resolves to Yes."}
        gid, _ = self.ing._match_kalshi_game(m)
        self.assertEqual(gid, 301)

    def test_team_id_prefers_the_active_franchise(self):
        self.assertEqual(self.ing._team_id_for("Utah"), 68)


class TestRegistry(unittest.TestCase):
    def test_every_source_declares_the_required_fields(self):
        required = {"source_id", "name", "url", "data_type", "nhl_relevance", "historical_depth",
                    "live_available", "update_frequency", "api_available", "auth_required", "cost",
                    "genuinely_free", "usage_limits", "licensing", "provenance", "reliability",
                    "accuracy", "granularity", "automated_access", "known_limits"}
        for spec in SOURCES:
            missing = required - set(spec.__dict__)
            self.assertFalse(missing, f"{spec.source_id} missing {missing}")

    def test_free_tier_is_not_recorded_as_free(self):
        by_id = {s.source_id: s for s in SOURCES}
        self.assertEqual(by_id["odds.the_odds_api"].genuinely_free, "freemium")
        self.assertEqual(by_id["odds.the_odds_api"].status, "rejected")
        self.assertEqual(by_id["nst.money_puck_evolved"].status, "rejected")

    def test_candles_registry_entry_records_the_corrected_path(self):
        """The 2026-09-20 404s came from probing '/candles'; the endpoint is
        '/candlesticks'.  The registry must say so rather than keep a wrong rejection."""
        by_id = {s.source_id: s for s in SOURCES}
        spec = by_id["kalshi.candles"]
        self.assertNotEqual(spec.status, "rejected")
        self.assertIn("candlesticks", spec.url)
        self.assertIn("404", spec.known_limits)
        self.assertTrue(all("candlesticks" in u for u in spec.probe_urls))
        hist = by_id["kalshi.historical"]
        self.assertIn("/historical", hist.url)
        self.assertTrue(hist.probe_urls)

    def test_registry_still_rejects_scraping_only_sources(self):
        by_id = {s.source_id: s for s in SOURCES}
        self.assertEqual(by_id["nst.money_puck_evolved"].status, "rejected")
        self.assertEqual(by_id["odds.the_odds_api"].status, "rejected")
        self.assertEqual(by_id["nhl.legacy_statsapi"].status, "flagged")
        # MoneyPuck's listed downloads are permitted (non-commercial, credit) -> candidate
        self.assertEqual(by_id["moneypuck.downloads"].status, "candidate")
        self.assertIn("credit", by_id["moneypuck.downloads"].licensing.lower())

    def test_seeding_with_a_probe_satisfies_the_foreign_key(self):
        """Regression: the verification row was inserted before the registry row existed,
        which raised IntegrityError on any real probe run."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = Store(path)
        probes = {spec.source_id: (200, "reachable", "HTTP 200")
                  for spec in SOURCES if spec.probe_urls}
        n = seed_registry(s, verified_probe=probes)
        self.assertEqual(n, len(SOURCES))
        ver = s.query("SELECT * FROM source_verification")
        self.assertEqual(len(ver), len(probes))
        # a reachable source that is not already rejected becomes verified
        statuses = {r["source_id"]: r["status"] for r in
                    s.query("SELECT source_id, status FROM source_registry")}
        self.assertEqual(statuses["nhl.api_web"], "verified")
        # ...but a rejected source stays rejected even if a probe somehow succeeds
        s2 = Store(path)
        probes_all = {spec.source_id: (200, "reachable", "HTTP 200") for spec in SOURCES}
        seed_registry(s2, verified_probe=probes_all)
        statuses = {r["source_id"]: r["status"] for r in
                    s2.query("SELECT source_id, status FROM source_registry")}
        for spec in SOURCES:
            if spec.status == "rejected":
                self.assertEqual(statuses[spec.source_id], "rejected", spec.source_id)
        s2.close()
        s.close()

    def test_unreachable_probe_does_not_promote_a_source(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = Store(path)
        seed_registry(s, verified_probe={"nhl.api_web": (None, "unreachable", "DNS failure")})
        row = s.one("SELECT status FROM source_registry WHERE source_id='nhl.api_web'")
        self.assertNotEqual(row["status"], "verified")
        v = s.one("SELECT ok, verdict FROM source_verification")
        self.assertEqual(v["ok"], 0)
        self.assertEqual(v["verdict"], "unreachable")
        s.close()

    def test_seeding_without_a_probe_does_not_claim_verified(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = Store(path)
        seed_registry(s)
        rows = s.query("SELECT source_id, status FROM source_registry")
        self.assertTrue(rows)
        self.assertNotIn("verified", [r["status"] for r in rows])
        s.close()


class TestSettlementUpsertKeepsKnownFields(unittest.TestCase):
    """A settled contract can arrive more than once, and the later payload is not always the
    more complete one.  The upsert must let a newer value win without ever blanking a fact
    that is already on the record -- ``rung`` above all, because the ticker suffix is what
    tells the puck-line settlement code which line the contract was struck at."""

    #: fields that are genuinely absent on some Kalshi payloads -- the verified overtime
    #: capture carries no ``strike_type``/``floor_strike`` at all, and neither spread capture
    #: carries ``series_ticker``.  A page that leaves one out must not blank the stored value.
    DROPPED = ("strike_type", "floor_strike", "yes_sub_title", "title")

    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        self.http = HttpClient(tempfile.mkdtemp())
        self.ing = Ingestor(self.store, self.http, verbose=False)
        self.m = load("kalshi_historical_markets_kxnhlspread_limit2.json")["markets"][0]

    def tearDown(self):
        self.store.close()

    def _row(self):
        return self.store.one("SELECT * FROM market_settlements WHERE contract=? AND side='YES'",
                              (self.m["ticker"],))

    def test_real_payload_records_the_rung_and_line(self):
        self.ing._upsert_settlement(self.m, tier="historical", source_url="u1",
                                    ts="2026-09-21T00:00:00Z")
        row = self._row()
        self.assertIsNotNone(row)
        self.assertEqual(row["rung"], "VGK2")
        self.assertEqual(row["strike_type"], "greater")
        self.assertEqual(row["market_type"], "puck_line")
        self.assertEqual(row["selection"], "Vegas wins by over 2.5 goals")
        self.assertAlmostEqual(float(row["floor_strike"]), 2.5)

    def test_a_partial_payload_never_blanks_the_record(self):
        self.ing._upsert_settlement(self.m, tier="historical", source_url="u1",
                                    ts="2026-09-21T00:00:00Z")
        # a game match recorded by an earlier pass (when the game was inside the ingested
        # window) must survive a later pass that cannot re-derive it: this store has no game
        # rows at all, so the matcher returns None every time
        self.store.execute("UPDATE market_settlements SET game_id=2026030141 WHERE contract=?",
                           (self.m["ticker"],))
        self.store.commit()
        partial = {k: v for k, v in self.m.items() if k not in self.DROPPED}
        for k in self.DROPPED:
            self.assertIn(k, self.m, f"fixture must really carry {k}")
        self.ing._upsert_settlement(partial, tier="historical", source_url="u2",
                                    ts="2026-09-21T01:00:00Z")
        row = self._row()
        self.assertEqual(row["rung"], "VGK2")
        self.assertEqual(row["strike_type"], "greater")
        self.assertEqual(row["market_type"], "puck_line")
        self.assertEqual(row["selection"], "Vegas wins by over 2.5 goals")
        self.assertAlmostEqual(float(row["floor_strike"]), 2.5)
        self.assertEqual(int(row["game_id"]), 2026030141)
        # retrieval provenance does move forward: the row reports its most recent fetch
        self.assertEqual(row["source_url"], "u2")
        self.assertEqual(row["retrieved_at"], "2026-09-21T01:00:00Z")
        self.assertEqual(row["result"], "no")
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM market_settlements")["c"], 1, "still one row, not a duplicate")

    def test_a_newer_result_overrides_the_old_one(self):
        """COALESCE protects missing values; it must not freeze values that really changed."""
        self.ing._upsert_settlement(self.m, tier="historical", source_url="u1",
                                    ts="2026-09-21T00:00:00Z")
        self.assertEqual(self._row()["result"], "no")
        corrected = dict(self.m, result="yes", settlement_value_dollars="1.0000")
        self.ing._upsert_settlement(corrected, tier="historical", source_url="u2",
                                    ts="2026-09-21T02:00:00Z")
        row = self._row()
        self.assertEqual(row["result"], "yes")
        self.assertAlmostEqual(float(row["settle_price"]), 1.0)

    def test_overtime_contract_keeps_its_own_suffix(self):
        """The OT series' suffix is ``OT``, which is not a team code and must be stored
        verbatim rather than parsed into a line."""
        m = load("kalshi_historical_markets_kxnhlovertime_limit2.json")["markets"][0]
        self.ing._upsert_settlement(m, tier="historical", source_url="u1",
                                    ts="2026-09-21T00:00:00Z")
        row = self.store.one("SELECT * FROM market_settlements WHERE contract=?", (m["ticker"],))
        self.assertEqual(row["rung"], "OT")
        self.assertEqual(row["market_type"], "overtime")
        self.assertIsNone(row["floor_strike"])
        self.assertIsNone(row["strike_type"])


class TestMoneylineUpsertKeepsGameMatch(unittest.TestCase):
    """Same rule on the live-tier moneyline path in :mod:`nhlcomp.ingest`."""

    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        self.http = HttpClient(tempfile.mkdtemp())
        self.ing = Ingestor(self.store, self.http, verbose=False)
        prime_cache(self.http, "https://api.nhle.com/stats/rest/en/team",
                    load("nhl_stats_rest_team.json"))
        prime_cache(self.http, "https://api-web.nhle.com/v1/scoreboard/2026-09-19",
                    load("nhl_scoreboard_20260919.json"))
        self.ing.teams()
        self.ing.scoreboard_window("2026-09-19")

    def tearDown(self):
        self.store.close()

    URL = ("https://api.elections.kalshi.com/trade-api/v2/markets"
           "?series_ticker=KXNHLGAME&limit=200&status=settled")

    def test_reingest_that_cannot_rematch_keeps_the_match(self):
        payload = load("kalshi_settled_kxnhlgame.json")
        prime_cache(self.http, self.URL, payload)
        self.assertEqual(self.ing.kalshi_settled(max_pages=1), 3)
        matched = self.store.one("SELECT * FROM market_settlements WHERE contract=?",
                                 ("KXNHLGAME-26SEP19VGKLA-VGK",))
        self.assertEqual(int(matched["game_id"]), 2026010002)
        # VAN@SEA is the contract this window genuinely cannot match (no game row), so it
        # stands in for "a match recorded earlier, un-derivable now"
        un = "KXNHLGAME-26SEP19VANSEA-VAN"
        self.assertIsNone(self.store.one(
            "SELECT game_id FROM market_settlements WHERE contract=?", (un,))["game_id"])
        self.store.execute("UPDATE market_settlements SET game_id=2026010003 WHERE contract=?",
                           (un,))
        self.store.commit()
        self.ing.kalshi_settled(max_pages=1)
        row = self.store.one("SELECT * FROM market_settlements WHERE contract=?", (un,))
        self.assertEqual(int(row["game_id"]), 2026010003,
                         "a failed re-match must not erase an existing game match")
        self.assertEqual(row["result"], "no")
        # the matched contract is untouched and its result still overrides on re-ingest
        self.assertEqual(int(self.store.one(
            "SELECT game_id FROM market_settlements WHERE contract=?",
            ("KXNHLGAME-26SEP19VGKLA-VGK",))["game_id"]), 2026010002)
        self.assertEqual(self.store.one(
            "SELECT COUNT(*) c FROM market_settlements WHERE contract=?", (un,))["c"], 1)



if __name__ == "__main__":
    unittest.main()
