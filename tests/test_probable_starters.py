"""The probable-starter feed (espn.nhl_probables), end to end.

On 2026-09-22 the project found its first verified public source that names a starting
goalie BEFORE puck drop: ESPN's scoreboard publishes a ``probableStartingGoalie`` per
competitor of a scheduled game, with the feed's own status object (observed:
``type='expected'``).  Everything downstream is tested here:

* the parsers, against the REAL captured payloads (dates rewritten where a test needs a
  future slate -- the shape is what matters and the capture proves it);
* identity resolution (ESPN name -> NHL player id through the club roster), which is
  DERIVED and carries its basis;
* the append-only ``goalie_starts`` book and its point-in-time reads;
* the feature family, which is absent by construction on decided rows so the discovery
  engine can never mine a feed with no history;
* the forward path: a probable-gated rule trades on 'expected', a CONFIRMED-gated rule
  names the probable it refuses to treat as confirmed, and an out-of-scope game records
  the signal it was not allowed to trade;
* the post-game conversion check that measures the feed instead of asserting it.

Nothing in this file invents an NHL or ESPN fact: the fixtures are the captured payloads
plus synthetic rows shaped exactly like them, clearly marked as synthetic.
"""

import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from nhlcomp.features_ext import ExtendedFeatures, load_extended
from nhlcomp.http import HttpClient
from nhlcomp.ingest import Ingestor
from nhlcomp.pipeline import Pipeline
from nhlcomp.sources.nhl import (parse_espn_events, parse_roster, parse_schedule_games)
from nhlcomp.store import Store, utcnow
from nhlcomp.strategies import (DecisionContext, Quote, ThresholdStrategy)

HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURED = os.path.normpath(os.path.join(HERE, "..", "data", "captured"))
TOR_ID, OTT_ID = 10, 9


def load(name: str) -> dict:
    with open(os.path.join(CAPTURED, name), encoding="utf-8") as fh:
        return json.load(fh)


def prime_cache(http: HttpClient, url: str, payload: dict) -> None:
    http._store(url, 200, json.dumps(payload))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def espn_payload_tomorrow() -> dict:
    """The real 2026-09-23 capture with its dates shifted to tomorrow.

    The captured slate is fixed in time; a join test needs a slate that is still in the
    future whenever this suite runs.  Only the date strings are rewritten -- every field
    the parser reads is the real payload's.
    """
    payload = copy.deepcopy(load("espn_scoreboard_probable_goalies_20260923.json"))
    day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    for ev in payload.get("events", []):
        ev["date"] = f"{day}T23:00Z"
        for comp in ev.get("competitions", []):
            comp["date"] = f"{day}T23:00Z"
    return payload


def nhl_schedule_tomorrow() -> dict:
    """The real schedule capture, shifted to the same tomorrow as the ESPN payload."""
    payload = copy.deepcopy(load("nhl_schedule_20260923_excerpt.json"))
    day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    for week in payload.get("gameWeek", []):
        week["date"] = day
        for g in week.get("games", []):
            g["gameDate"] = day
            if "startTimeUTC" in g:
                g["startTimeUTC"] = f"{day}T23:00:00Z"
    return payload


class TestParsersAgainstRealCaptures(unittest.TestCase):
    def test_espn_events_keep_the_probables_and_their_status(self):
        evs = parse_espn_events(load("espn_scoreboard_probable_goalies_20260923.json"))
        self.assertEqual(len(evs), 2)
        first = [e for e in evs if e["event_id"] == "401879650"][0]
        self.assertEqual(first["venue"], "Canadian Tire Centre")
        probs = {p["abbrev"]: p for p in first["probables"]}
        self.assertEqual(probs["OTT"]["goalie_name"], "Linus Ullmark")
        self.assertEqual(probs["TOR"]["goalie_name"], "Anthony Stolarz")
        self.assertEqual(probs["OTT"]["status_type"], "expected")
        self.assertEqual(probs["OTT"]["position"], "G")
        self.assertEqual(probs["OTT"]["espn_player_id"], "3069285")
        # home/away tags survive: the join depends on them
        self.assertEqual(probs["OTT"]["home_away"], "home")
        self.assertEqual(probs["TOR"]["home_away"], "away")

    def test_roster_parser_tolerates_optional_fields(self):
        rows = parse_roster(load("nhl_roster_DAL_20262027_excerpt.json"),
                            abbrev="DAL", season=20262027, retrieved_at="R")
        self.assertEqual(len(rows), 8)
        goalies = {r["full_name"]: r for r in rows if r["roster_group"] == "goalies"}
        self.assertEqual(set(goalies), {"Casey DeSmith", "Brandon Halverson",
                                        "Jake Oettinger", "Rémi Poirier"})
        # sweaterNumber and shootsCatches are optional in the real payload
        anderson = [r for r in rows if r["full_name"] == "Jack Anderson"][0]
        self.assertIsNone(anderson["shoots"])
        self.assertEqual(anderson["position"], "D")

    def test_schedule_parser_carries_split_squad_flags(self):
        rows = parse_schedule_games(load("nhl_schedule_20260923_excerpt.json"))
        by_id = {r["game_id"]: r for r in rows}
        self.assertEqual(by_id[2026010035]["split_squad_home"], 1)
        self.assertEqual(by_id[2026010035]["split_squad_away"], 1)
        self.assertEqual(by_id[2026010034]["split_squad_home"], 0)
        self.assertEqual(by_id[2026010033]["split_squad_away"], 0)


class TestIdentityResolution(unittest.TestCase):
    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)
        for pid, name in ((1, "Linus Ullmark"), (2, "Rémi Poirier"), (3, "Jake Oettinger")):
            self.store.upsert_player({"player_id": pid, "full_name": name, "position": "G",
                                      "team_abbrev": "DAL", "season": 20262027,
                                      "roster_group": "goalies"})

    def tearDown(self):
        self.store.close()

    def test_exact_match(self):
        pid, basis = self.store.resolve_player("Linus Ullmark", "DAL")
        self.assertEqual(pid, 1)
        self.assertIn("exact", basis)

    def test_normalised_match_is_recorded_as_derived(self):
        pid, basis = self.store.resolve_player("Remi Poirier", "DAL")
        self.assertEqual(pid, 2)
        self.assertIn("normalised", basis)
        self.assertIn("DERIVED", basis)

    def test_unknown_name_is_not_resolved(self):
        pid, basis = self.store.resolve_player("Nobody Here", "DAL")
        self.assertIsNone(pid)
        self.assertIn("no roster row", basis)

    def test_position_scope(self):
        self.store.upsert_player({"player_id": 4, "full_name": "Jake Oettinger",
                                  "position": "G", "team_abbrev": "TOR"})
        pid, _ = self.store.resolve_player("Jake Oettinger", "TOR", position="D")
        # the TOR roster has no D named Jake Oettinger; the DAL goalie must not be borrowed
        self.assertIsNone(pid)


class TestGoalieStartsBook(unittest.TestCase):
    def setUp(self):
        fd, self.dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = Store(self.dbpath)

    def tearDown(self):
        self.store.close()

    def _row(self, **kw):
        base = dict(game_id=100, game_date="2026-09-23", team_id=TOR_ID, team_abbrev="TOR",
                    goalie_name="Anthony Stolarz", goalie_id=None, announced_at=None,
                    source_id="espn.nhl_probables", retrieved_at="2026-09-22T21:00:00Z",
                    is_confirmed=False, provenance="SOURCE", status_type="expected",
                    status_name="Expected", source_player_id="3067313",
                    snapshot_ts="2026-09-22T21:00:00Z", venue="Arena",
                    source_event_id="401879650", match_basis="unique date+home+away match",
                    side="home")
        base.update(kw)
        return base

    def test_changed_announcement_appends_and_asof_reads_the_latest(self):
        self.assertTrue(self.store.record_goalie_start(self._row()))
        # same source, same game/team, LATER snapshot, different name: a second row
        self.assertTrue(self.store.record_goalie_start(
            self._row(goalie_name="Joseph Woll", snapshot_ts="2026-09-23T02:00:00Z",
                      retrieved_at="2026-09-23T02:00:00Z")))
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM goalie_starts")["c"], 2)
        books = self.store.starters_asof("2026-09-23T10:00:00Z")
        self.assertEqual(books[100][TOR_ID]["goalie_name"], "Joseph Woll")
        # a decision stamped BEFORE the change must not see it
        early = self.store.starters_asof("2026-09-22T22:00:00Z")
        self.assertEqual(early[100][TOR_ID]["goalie_name"], "Anthony Stolarz")

    def test_confirmed_and_expected_are_indexed_apart(self):
        self.store.record_goalie_start(self._row())
        self.store.record_goalie_start(self._row(game_id=101, goalie_name="Confirmed Guy",
                                                 is_confirmed=True, status_type="confirmed"))
        # the store indexes rows with their status intact; the point-in-time read can
        # restrict to confirmed, and the pipeline splits the two books apart
        all_books = self.store.starters_asof()
        self.assertIn(100, all_books)
        confirmed_only = self.store.starters_asof(confirmed_only=True)
        self.assertIn(101, confirmed_only)
        self.assertNotIn(100, confirmed_only)
        from nhlcomp.pipeline import Pipeline
        from nhlcomp.http import HttpClient
        pipe = Pipeline(self.store, HttpClient(tempfile.mkdtemp(), offline=True), verbose=False)
        confirmed, expected = pipe._starter_books()
        self.assertIn(101, confirmed)
        self.assertIn(100, expected)
        self.assertNotIn(100, confirmed)


class ProbableHarness(unittest.TestCase):
    """Teams, two NHL games tomorrow (a split-squad home-and-home), rosters, and the
    ESPN scoreboard primed into the http cache."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "t.db"))
        self.addCleanup(self.store.close)
        self.http = HttpClient(cache_dir=os.path.join(self.tmp.name, "cache"))
        self.ing = Ingestor(self.store, self.http, verbose=False)
        self.day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
        self.store.execute("INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                           (TOR_ID, "TOR", "Toronto Maple Leafs"))
        self.store.execute("INSERT INTO teams(team_id, abbrev, full_name, active) VALUES(?,?,?,1)",
                           (OTT_ID, "OTT", "Ottawa Senators"))
        # the real double-header: same date, same pair of clubs, two venues
        self.gid_ott, self.gid_tor = 2026010036, 2026010035
        for gid, home, away, venue in ((self.gid_ott, OTT_ID, TOR_ID, "Canadian Tire Centre"),
                                       (self.gid_tor, TOR_ID, OTT_ID, "Scotiabank Arena")):
            self.store.execute(
                "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
                " home_id, away_id, venue, utc_offset, state, source_id, provenance)"
                " VALUES(?,?,?,?,?,?,?,?,?,'FUT','test','SOURCE')",
                (gid, 20262027, 1, self.day, f"{self.day}T23:00:00Z", home, away, venue,
                 "-04:00"))
        self.store.commit()

    def prime_espn(self):
        prime_cache(self.http,
                    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                    f"?dates={self.day.replace('-', '')}",
                    espn_payload_tomorrow())


class TestProbablesIngest(ProbableHarness):
    def test_probables_join_resolve_and_record(self):
        # rosters first: the names must resolve to NHL ids through them
        prime_cache(self.http, "https://api-web.nhle.com/v1/roster/TOR/20262027",
                    {"forwards": [], "defensemen": [],
                     "goalies": [{"id": 3067313, "firstName": {"default": "Anthony"},
                                  "lastName": {"default": "Stolarz"}, "positionCode": "G"},
                                 {"id": 1, "firstName": {"default": "Joseph"},
                                  "lastName": {"default": "Woll"}, "positionCode": "G"}]})
        self.ing.rosters(["TOR"], 20262027)
        self.prime_espn()
        n = self.ing.espn_probables([self.day])
        self.assertGreater(n, 0)
        rows = self.store.query("SELECT * FROM goalie_starts ORDER BY game_id")
        by_pair = {(r["game_id"], r["team_abbrev"]): r for r in rows}
        # the venue tie-break matched each event to its own NHL game: every game carries
        # BOTH its clubs' probables, under the right side
        ott_row = by_pair[(self.gid_ott, "OTT")]
        self.assertEqual(ott_row["goalie_name"], "Linus Ullmark")
        self.assertEqual(ott_row["side"], "home")
        tor = by_pair[(self.gid_ott, "TOR")]
        self.assertEqual(tor["goalie_name"], "Anthony Stolarz")
        self.assertEqual(tor["side"], "away")
        # the second captured event carries no probables (field-preserving excerpt), so
        # exactly these two rows exist
        self.assertEqual(len(rows), 2)
        self.assertEqual(tor["goalie_id"], 3067313)
        self.assertIn("exact", tor["match_basis"])
        self.assertEqual(tor["status_type"], "expected")
        self.assertEqual(tor["is_confirmed"], 0, "'expected' is never upgraded")
        self.assertIsNotNone(tor["snapshot_ts"])
        self.assertEqual(tor["source_event_id"], "401879650")
        self.assertIsNotNone(tor["match_basis"])

    def test_venue_tie_breaks_a_true_double_header(self):
        """Two NHL games, same date, same home/away pair, two venues: the ESPN venue is
        the only disambiguator, and each event must land on its own game."""
        self.store.execute(
            "INSERT INTO games(game_id, season, game_type, game_date, start_time_utc,"
            " home_id, away_id, venue, utc_offset, state, source_id, provenance)"
            " VALUES(?,?,?,?,?,?,?,?,?,'FUT','test','SOURCE')",
            (2026010099, 20262027, 1, self.day, f"{self.day}T23:00:00Z",
             OTT_ID, TOR_ID, "Some Third Arena", "-04:00"))
        self.store.commit()
        payload = espn_payload_tomorrow()
        # a synthetic SECOND ESPN event: same teams, same date, different venue
        twin = copy.deepcopy(payload["events"][0])
        twin["id"] = "409999999"
        twin["competitions"][0]["venue"]["fullName"] = "Some Third Arena"
        for c in twin["competitions"][0]["competitors"]:
            for pr in c.get("probables", []):
                pr["athlete"]["fullName"] = "Twin Squad Goalie"
                pr["athlete"]["id"] = "9999"
        payload["events"].append(twin)
        prime_cache(self.http,
                    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                    f"?dates={self.day.replace('-', '')}", payload)
        self.ing.espn_probables([self.day])
        # the venue named by the real event resolved to the real game
        real = self.store.one(
            "SELECT * FROM goalie_starts WHERE venue='Canadian Tire Centre'")
        self.assertIsNotNone(real)
        self.assertTrue(real["match_basis"].startswith(
            "date+home+away ambiguous; resolved by venue"), real["match_basis"])
        twin_rows = self.store.query(
            "SELECT * FROM goalie_starts WHERE venue='Some Third Arena'")
        self.assertEqual(len(twin_rows), 2)
        self.assertTrue(all(r["game_id"] == 2026010099 for r in twin_rows),
                        "the twin event must land on its own game, never on the real one")

    def test_unresolvable_goalie_name_is_stored_without_an_id(self):
        payload = espn_payload_tomorrow()
        prob = payload["events"][0]["competitions"][0]["competitors"][0]["probables"][0]
        prob["athlete"]["fullName"] = "Call Up From The Minors"
        prime_cache(self.http,
                    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                    f"?dates={self.day.replace('-', '')}", payload)
        self.ing.espn_probables([self.day])
        row = self.store.one("SELECT * FROM goalie_starts WHERE goalie_name LIKE 'Call Up%'")
        self.assertIsNotNone(row)
        self.assertIsNone(row["goalie_id"], "an unresolved name must not borrow an id")
        self.assertIn("no roster row", row["match_basis"])

    def test_non_goalie_probable_is_flagged_not_recorded(self):
        payload = espn_payload_tomorrow()
        prob = payload["events"][0]["competitions"][0]["competitors"][0]["probables"][0]
        prob["athlete"]["position"] = "C"
        prime_cache(self.http,
                    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                    f"?dates={self.day.replace('-', '')}", payload)
        self.ing.espn_probables([self.day])
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM goalie_starts")["c"],
                         1, "only the real goalie probable is recorded")
        flagged = self.store.one(
            "SELECT COUNT(*) c FROM irregularities WHERE kind='probable_goalie_position_mismatch'")
        self.assertEqual(flagged["c"], 1)

    def test_started_games_are_not_pre_game_observations(self):
        payload = espn_payload_tomorrow()
        payload["events"][0]["competitions"][0]["date"] = "2020-01-01T00:00Z"
        payload["events"][0]["date"] = "2020-01-01T00:00Z"
        prime_cache(self.http,
                    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                    f"?dates={self.day.replace('-', '')}", payload)
        self.ing.espn_probables([self.day])
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM goalie_starts")["c"], 0)

    def test_conversion_check_measures_the_feed(self):
        # record a probable for a game that is ALREADY final, then let the check compare
        self.store.execute(
            "UPDATE games SET state='FINAL', home_score=3, away_score=2 WHERE game_id=?",
            (self.gid_ott,))
        self.store.execute(
            """INSERT INTO goalie_game_stats(game_id, player_id, goalie_name, team_abbrev,
                 opponent_abbrev, home_road, game_date, started, saves, shots_against,
                 goals_against, save_pct, toi_seconds, decision, source_id, retrieved_at)
               VALUES(?, 500, 'Linus Ullmark', 'OTT', 'TOR', 'home', ?, 1, 30, 32, 2,
                      0.9375, 3600, 'W', 'test', ?)""",
            (self.gid_ott, self.day, utcnow()))
        self.store.record_goalie_start(dict(
            game_id=self.gid_ott, game_date=self.day, team_id=OTT_ID, team_abbrev="OTT",
            goalie_name="Linus Ullmark", goalie_id=None, source_id="espn.nhl_probables",
            retrieved_at=utcnow(), is_confirmed=False, status_type="expected",
            snapshot_ts=utcnow(), side="home"))
        self.store.commit()
        out = Pipeline(self.store, self.http, verbose=False).stage_goalie_conversion()
        self.assertEqual(out["checked"], 1)
        self.assertEqual(out["matched"], 1)
        row = self.store.one("SELECT * FROM goalie_starts")
        self.assertEqual(row["actual_goalie_name"], "Linus Ullmark")
        self.assertIsNotNone(row["conversion_checked_at"])
        finding = self.store.one("SELECT * FROM findings WHERE finding_id='FIND_STARTER_CONVERSION'")
        self.assertIsNotNone(finding)
        # idempotent: a second pass re-checks nothing
        self.assertEqual(Pipeline(self.store, self.http, verbose=False)
                         .stage_goalie_conversion()["checked"], 0)


class TestProbableFeatures(unittest.TestCase):
    W = 10

    @classmethod
    def setUpClass(cls):
        # dates relative to 'now' so the featured game is always in the future and the
        # goalie log always strictly before it, whatever day the suite runs.  The home
        # probable's LAST start is the day before the featured game -> b2b == 1.
        now = datetime.now(timezone.utc)
        cls.day = (now + timedelta(days=1)).date().isoformat()
        cls.last_start = now.date().isoformat()
        cls.two_ago = (now - timedelta(days=1)).date().isoformat()
        cls.start_iso = f"{cls.day}T23:00:00Z"

    def _ext(self):
        goalie_stats = [
            # the home probable started YESTERDAY (b2b) and two nights before that;
            # the away probable started yesterday only
            {"game_id": 1, "player_id": 11, "goalie_name": "Home Guy", "team_abbrev": "TOR",
             "game_date": self.last_start, "started": 1, "saves": 30, "shots_against": 32,
             "goals_against": 2, "toi_seconds": 3600},
            {"game_id": 2, "player_id": 11, "goalie_name": "Home Guy", "team_abbrev": "TOR",
             "game_date": self.two_ago, "started": 1, "saves": 25, "shots_against": 30,
             "goals_against": 5, "toi_seconds": 3600},
            {"game_id": 3, "player_id": 12, "goalie_name": "Away Guy", "team_abbrev": "OTT",
             "game_date": self.two_ago, "started": 1, "saves": 33, "shots_against": 34,
             "goals_against": 1, "toi_seconds": 3600},
        ]
        probables = [
            {"game_id": 100, "team_abbrev": "TOR", "goalie_name": "Home Guy",
             "goalie_id": 11, "is_confirmed": 0, "status_type": "expected",
             "snapshot_ts": "2026-09-22T21:00:00Z"},
            {"game_id": 100, "team_abbrev": "OTT", "goalie_name": "Away Guy",
             "goalie_id": 12, "is_confirmed": 0, "status_type": "expected",
             "snapshot_ts": "2026-09-22T21:00:00Z"},
        ]
        return ExtendedFeatures(team_stats=[], goalie_stats=goalie_stats, price_points=[],
                                abbrev_by_id={TOR_ID: "TOR", OTT_ID: "OTT"},
                                probable_starts=probables, window=self.W)

    def _row(self):
        return {"game_id": 100, "game_date": self.day, "start_time_utc": self.start_iso,
                "home_id": TOR_ID, "away_id": OTT_ID}

    def test_probable_features_exist_for_a_future_game_with_form(self):
        ext = self._ext()
        row = self._row()
        ext.augment(row)
        self.assertEqual(row["home_probable_starter_known"], 1)
        self.assertEqual(row["home_probable_starter_confirmed"], 0)
        self.assertEqual(row["home_probable_starter_name"], "Home Guy")
        self.assertEqual(row["home_probable_starter_status"], "expected")
        # form comes from the goalie log, strictly before the game date
        self.assertAlmostEqual(row["home_probable_starter_sv_pct_l10"], round(55 / 62, 4))
        self.assertEqual(row["home_probable_starter_b2b"], 1)
        self.assertEqual(row["away_probable_starter_b2b"], 0)
        self.assertAlmostEqual(row["diff_probable_starter_sv_pct_l10"],
                               round(row["home_probable_starter_sv_pct_l10"]
                                     - row["away_probable_starter_sv_pct_l10"], 4))

    def test_probable_features_are_absent_for_a_decided_game(self):
        ext = self._ext()
        row = self._row()
        row["game_date"] = "2020-01-01"
        row["start_time_utc"] = "2020-01-01T23:00:00Z"
        ext.augment(row)
        self.assertNotIn("home_probable_starter_known", row,
                         "a feed with no history must not be mineable on decided rows")

    def test_probable_keyed_by_snapshot_not_by_game_side_confusion(self):
        ext = self._ext()
        f = ext.probable_goalie_features(100, OTT_ID, self.day, game_started=False)
        self.assertEqual(f["probable_starter_name"], "Away Guy")


class ForwardHarness(ProbableHarness):
    """Adds the quote/feature plumbing the forward stage needs.  The two games are
    REGULAR SEASON (game_type=2) so default-scoped rules are in scope here; the
    out-of-scope test flips one back to preseason explicitly."""

    def setUp(self):
        super().setUp()
        self.store.execute("UPDATE games SET game_type=2 WHERE game_id IN (?,?)",
                           (self.gid_ott, self.gid_tor))
        self.store.commit()
        self.pipe = Pipeline(self.store, self.http, verbose=False)

    def register(self, strat) -> None:
        d = strat.describe()
        d.update(status="active", created_at=utcnow(), test_mode="FORWARD TEST",
                 leakage_checked=1)
        self.store.upsert_strategy(d)
        self.store.set_strategy_status(strat.strategy_id, strat.version, "active",
                                       reason="test", evidence="test")

    def quote_row(self, *, contract, market_key, gid, market_type="moneyline", title="",
                  side="YES", bid=0.44, ask=0.46, ask_size=200.0):
        self.store.execute(
            "INSERT INTO market_quotes(provider, market_key, contract, game_id, game_date,"
            " market_type, selection, side, bid, ask, spread, bid_size, ask_size, volume,"
            " liquidity, last_price, ts_utc, retrieved_at, source_url, strike, strike_type)"
            " VALUES('kalshi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'https://kalshi.example',?,?)",
            (market_key, contract, gid, self.day, market_type, title, side,
             bid, ask, round(ask - bid, 4), ask_size, ask_size, 10, 100, ask,
             utcnow(), utcnow(), None, None))
        self.store.commit()

    def feature_row(self, gid, **extra):
        row = {"game_id": gid, "game_date": self.day, "start_time_utc": f"{self.day}T23:00:00Z",
               "home_id": self.store.one("SELECT home_id FROM games WHERE game_id=?", (gid,))[0],
               "away_id": self.store.one("SELECT away_id FROM games WHERE game_id=?", (gid,))[0],
               "home_n_prior": 40, "away_n_prior": 40, "home_exp_goals": 3.0,
               "away_exp_goals": 2.8, "lam_home": 2.975, "lam_away": 2.775,
               "exp_total": 5.75, "p_home_ml": 0.541, "p_away_ml": 0.459,
               "p_home_elo": 0.55, "p_away_elo": 0.45, "p_home_logit": 0.54,
               "p_away_logit": 0.46}
        row.update(extra)
        return row


class TestForwardPath(ForwardHarness):
    STRAT = dict(strategy_id="NHL_PROB_TEST", username="NHL_PROB_TEST_001",
                 category="goaltending", feature="diff_probable_starter_sv_pct_l10",
                 operator=">=", threshold=0.0, bet_side="home", min_edge=0.03)

    def _seed_probables(self):
        # goalie form for both ids, strictly before tomorrow: the home probable saved more
        # (0.9375) than the away one (0.90), so the diff trigger passes for bet_side home
        yst = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        for pid, ab, opp, sv, sa, name in ((500, "OTT", "TOR", 30, 32, "Linus Ullmark"),
                                           (501, "TOR", "OTT", 27, 30, "Anthony Stolarz")):
            self.store.execute(
                """INSERT INTO goalie_game_stats(game_id, player_id, goalie_name, team_abbrev,
                     opponent_abbrev, home_road, game_date, started, saves, shots_against,
                     goals_against, save_pct, toi_seconds, decision, source_id, retrieved_at)
                   VALUES(9001, ?, ?, ?, ?, 'home', ?, 1, ?, ?, ?, ?, 3600, 'W', 'test', ?)""",
                (pid, name, ab, opp, yst, sv, sa, sa - sv, sv / sa, utcnow()))
        for tid, ab, pid, name in ((OTT_ID, "OTT", 500, "Linus Ullmark"),
                                   (TOR_ID, "TOR", 501, "Anthony Stolarz")):
            self.store.record_goalie_start(dict(
                game_id=self.gid_ott, game_date=self.day, team_id=tid, team_abbrev=ab,
                goalie_name=name, goalie_id=pid, source_id="espn.nhl_probables",
                retrieved_at=utcnow(), is_confirmed=False, status_type="expected",
                snapshot_ts=utcnow(), side="home" if tid == OTT_ID else "away"))
        self.store.commit()

    def _augmented(self, gid):
        from nhlcomp.features_ext import load_extended
        self.pipe.ext = load_extended(self.store)
        row = self.feature_row(gid)
        self.pipe.ext.augment(row)
        return row

    def test_probable_gated_rule_trades_on_expected_status(self):
        self._seed_probables()
        strat = ThresholdStrategy(**self.STRAT)
        self.register(strat)
        self.quote_row(contract=f"KXNHLGAME-X-OTT", market_key="KXNHLGAME-X",
                       gid=self.gid_ott, title="Ottawa Senators wins", bid=0.44, ask=0.46)
        self.quote_row(contract=f"KXNHLGAME-X-TOR", market_key="KXNHLGAME-X",
                       gid=self.gid_ott, title="Toronto Maple Leafs wins", bid=0.50, ask=0.52)
        rows = [self._augmented(self.gid_ott)]
        self.assertEqual(rows[0]["home_probable_starter_status"], "expected")
        # probable-gated rules are NOT priced-backtestable
        placed = self.pipe.stage_forward(rows)
        self.assertEqual(placed["BACKTEST"], 0)
        bet = self.store.one("SELECT * FROM bets WHERE test_mode='FORWARD TEST'")
        self.assertIsNotNone(bet, "an expected-status probable must unlock the rule")
        feats = json.loads(bet["features_json"])
        self.assertEqual(feats["home_probable_starter"]["name"], "Linus Ullmark")
        self.assertEqual(feats["home_probable_starter"]["status"], "expected")
        self.assertEqual(feats["away_probable_starter"]["name"], "Anthony Stolarz")

    def test_confirmed_gated_rule_names_the_probable_it_refuses(self):
        self._seed_probables()
        strat = ThresholdStrategy(**dict(self.STRAT, requires_goalie=True))
        self.register(strat)
        self.quote_row(contract="KXNHLGAME-X-OTT", market_key="KXNHLGAME-X",
                       gid=self.gid_ott, title="Ottawa Senators wins", bid=0.44, ask=0.46)
        rows = [self._augmented(self.gid_ott)]
        self.pipe.stage_forward(rows)
        up = self.store.one("SELECT * FROM upcoming_bets WHERE strategy_id=?",
                            (strat.strategy_id,))
        self.assertEqual(up["status"], "WAITING FOR GOALIE")
        self.assertIn("Linus Ullmark", up["blocking_reason"])
        self.assertIn("'expected'", up["blocking_reason"])
        self.assertIsNone(self.store.one("SELECT * FROM bets WHERE test_mode='FORWARD TEST'"))

    def test_preseason_signal_is_recorded_out_of_scope_and_not_traded(self):
        # flip the game back to preseason (game_type=1): a rule scoped to [2,3] must record
        # what it wanted and place nothing
        self.store.execute("UPDATE games SET game_type=1 WHERE game_id=?", (self.gid_ott,))
        self.store.commit()
        strat = ThresholdStrategy(strategy_id="NHL_SCOPE_TEST", username="NHL_SCOPE_TEST_001",
                                  category="t", feature="home_exp_goals", operator=">=",
                                  threshold=1.0, bet_side="home", min_edge=0.03)
        self.register(strat)  # default game_types [2, 3]
        self.quote_row(contract="KXNHLGAME-X-OTT", market_key="KXNHLGAME-X",
                       gid=self.gid_ott, title="Ottawa Senators wins", bid=0.44, ask=0.46)
        rows = [self.feature_row(self.gid_ott)]
        before = self.store.one("SELECT COUNT(*) c FROM bets")["c"]
        self.pipe.stage_forward(rows)
        up = self.store.one("SELECT * FROM upcoming_bets WHERE strategy_id=? AND game_id=?",
                            (strat.strategy_id, self.gid_ott))
        self.assertIsNotNone(up)
        statuses = {r["status"] for r in self.store.query(
            "SELECT status FROM upcoming_bets WHERE strategy_id=?", (strat.strategy_id,))}
        self.assertIn("OUT OF SCOPE", statuses)
        self.assertIn("gameType 1", up["blocking_reason"])
        self.assertEqual(self.store.one("SELECT COUNT(*) c FROM bets")["c"], before)


class TestDiscoveryOutcomesRecorded(unittest.TestCase):
    def test_every_scanned_trigger_leaves_an_experiments_row(self):
        fd, dbpath = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        store = Store(dbpath)
        try:
            store.execute("INSERT INTO teams(team_id, abbrev, full_name, active)"
                          " VALUES(1,'A','Alpha',1),(2,'B','Beta',1)")
            rows = []
            day0 = datetime(2025, 10, 1, tzinfo=timezone.utc)
            for i in range(120):
                d = day0 + timedelta(days=i)
                home, away = (1, 2) if i % 2 == 0 else (2, 1)
                hw = i % 3
                rest = (1, 2, 3)[i % 3]
                rows.append({
                    "game_id": 9000 + i, "game_date": d.date().isoformat(),
                    "start_time_utc": _iso(d), "home_id": home, "away_id": away,
                    "season": 20252026, "venue": "V", "utc_offset": "-05:00",
                    "home_n_prior": 5, "away_n_prior": 5,
                    "home_rest_days": rest, "away_rest_days": rest,
                    "rest_diff": 0.0,
                    "home_only_b2b": i % 4 == 0, "away_only_b2b": i % 4 == 1,
                    "home_back_to_back": i % 4 == 0, "away_back_to_back": i % 4 == 1,
                    "home_prev_ot": i % 5 == 0, "away_prev_ot": i % 5 == 2,
                    "home_streak_wins": i % 3, "away_streak_losses": (i + 1) % 3,
                    "home_games_last_7": 2 + (i % 2), "away_games_last_7": 2 + ((i + 1) % 2),
                    "home_road_trip_len": i % 3, "away_road_trip_len": (i + 2) % 3,
                    "_winner": home if hw else away,
                    "p_home_ml": 0.5, "p_away_ml": 0.5,
                })
            http = HttpClient(tempfile.mkdtemp(), offline=True)
            pipe = Pipeline(store, http, verbose=False)
            pipe.stage_strategies(rows)
            n1 = store.one("SELECT COUNT(*) c FROM experiments WHERE exp_id LIKE 'EXP_SCAN_%'")["c"]
            self.assertGreater(n1, 0, "every scanned trigger should land in experiments")
            kinds = {r["verdict"] for r in store.query(
                "SELECT verdict FROM experiments WHERE exp_id LIKE 'EXP_SCAN_%'")}
            self.assertTrue(kinds <= {"edge", "no_edge", "inconclusive", "rejected"}, kinds)
            # idempotent: a second run must not duplicate the outcome rows
            pipe.stage_strategies(rows)
            n2 = store.one("SELECT COUNT(*) c FROM experiments WHERE exp_id LIKE 'EXP_SCAN_%'")["c"]
            self.assertEqual(n1, n2)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
