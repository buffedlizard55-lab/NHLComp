"""Data acquisition.

Every fetch is stored twice: once as the normalized rows, and once as the raw payload in
``raw_response`` with a SHA-256 digest, so any number in this project can be traced to the
bytes it came from.  When two independent sources describe the same game, both are loaded
and compared; disagreements go to the irregularity queue instead of being resolved silently.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .http import HttpClient, NetworkUnavailable, parse_iso
from .sources.kalshi import KalshiApi, normalize_market
from .sources.nhl import (NhlApi, NhlStatsRest, daterange, normalize_game,
                          parse_scoreboard_games, parse_standings)
from .sources.registry import SOURCES, probe_urls_for, seed_registry
from .store import Store, utcnow

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _num(v: object) -> float | None:
    """Kalshi sends numbers as strings; return None rather than 0.0 when absent."""
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Ingestor:
    def __init__(self, store: Store, http: HttpClient, *, verbose: bool = True):
        self.store = store
        self.http = http
        self.nhl = NhlApi(http)
        self.rest = NhlStatsRest(http)
        self.kalshi = KalshiApi(http)
        self.verbose = verbose
        self.stats: dict[str, Any] = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ingest] {msg}", flush=True)

    # ------------------------------------------------------------------ sources
    def verify_sources(self) -> dict[str, Any]:
        probes: dict[str, tuple[int | None, str, str]] = {}
        results = {}
        for spec in SOURCES:
            if not spec.probe_urls:
                results[spec.source_id] = {"status": spec.status, "probe": "skipped (no probe url)"}
                continue
            url = spec.probe_urls[0]
            status, verdict = self.http.probe(url)
            evidence = f"HTTP {status} from {url}"
            probes[spec.source_id] = (status, verdict, evidence)
            results[spec.source_id] = {"status_code": status, "verdict": verdict, "url": url}
            self.log(f"{spec.source_id}: {verdict} ({status}) {url}")
        n = seed_registry(self.store, verified_probe=probes)
        self.stats["sources_seeded"] = n
        self.stats["source_probes"] = results
        return results

    # ------------------------------------------------------------------ teams
    def teams(self) -> int:
        rows = self.rest.teams()
        n = 0
        for r in rows:
            self.store.execute(
                """INSERT INTO teams(team_id, abbrev, full_name, franchise_id, active, provenance)
                   VALUES(?,?,?,?,?, 'SOURCE')
                   ON CONFLICT(team_id) DO UPDATE SET abbrev=excluded.abbrev,
                     full_name=excluded.full_name, franchise_id=excluded.franchise_id""",
                (int(r["id"]), r.get("triCode") or r.get("rawTricode"), r.get("fullName"),
                 r.get("franchiseId"), 0 if r.get("fullName") == "To be determined" else 1),
            )
            n += 1
        self.store.commit()
        self.log(f"teams: {n} franchise rows from api.nhle.com/stats/rest/en/team")
        return n

    def standings(self, day: str | None = None) -> int:
        payload = self.nhl.standings(day)
        url = f"https://api-web.nhle.com/v1/standings/{day or 'now'}"
        self.store.record_raw(url, json.dumps(payload), source_id="nhl.api_web")
        rows = parse_standings(payload)
        n = 0
        for r in rows:
            tid = self.store.team_id_for(r["abbrev"], r.get("full_name"))
            if tid is None:
                self.store.flag(
                    "ambiguous_team",
                    f"standings row for '{r['abbrev']}' ({r.get('full_name')}) cannot be resolved "
                    f"to a single franchise",
                    entity_type="team", entity_id=r["abbrev"], severity="warn")
                continue
            t = {"team_id": tid}
            # conference/division and venue-free metadata come along for the ride
            self.store.execute(
                "UPDATE teams SET conference=?, division=?, full_name=? WHERE team_id=?",
                (r.get("conference"), r.get("division"), r.get("full_name") or "", t["team_id"]))
            self.store.execute("UPDATE teams SET active=1 WHERE team_id=?", (t["team_id"],))
            cols = ["as_of", "season", "team_id", "gp", "wins", "losses", "otl", "points", "gf",
                    "ga", "home_wins", "home_losses", "home_otl", "home_gf", "home_ga",
                    "road_wins", "road_losses", "road_otl", "road_gf", "road_ga", "l10_wins",
                    "l10_losses", "l10_otl", "l10_gf", "l10_ga", "streak_code", "streak_count",
                    "shootout_wins", "shootout_losses", "clinch", "source_id"]
            vals = [r["as_of"], r["season"], t["team_id"]] + [r.get(c) for c in cols[3:]]
            self.store.execute(
                f"""INSERT INTO standings_snapshot({','.join(cols)})
                    VALUES({','.join('?' * len(cols))})
                    ON CONFLICT(as_of, season, team_id) DO UPDATE SET
                      {','.join(f'{c}=excluded.{c}' for c in cols[3:])}""", vals)
            n += 1
        self.store.commit()
        self.log(f"standings: {n} team rows as of {rows[0]['as_of'] if rows else '?'}")
        return n

    # ------------------------------------------------------------------ games
    def _insert_games(self, games: Sequence[dict], source_id: str) -> tuple[int, int]:
        inserted = updated = 0
        for g in games:
            existing = self.store.one("SELECT * FROM games WHERE game_id=?", (g["game_id"],))
            row = dict(g)
            row.pop("home_abbrev", None)
            row.pop("away_abbrev", None)
            row.pop("winning_goalie_id", None)
            row["source_id"] = source_id
            row["provenance"] = "SOURCE"
            cols = list(row.keys())
            if existing is None:
                self.store.execute(
                    f"INSERT INTO games({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                    tuple(row[c] for c in cols))
                inserted += 1
            else:
                sets = ",".join(f"{c}=excluded.{c}" for c in cols if c != "game_id")
                self.store.execute(
                    f"""INSERT INTO games({','.join(cols)}) VALUES({','.join('?' * len(cols))})
                        ON CONFLICT(game_id) DO UPDATE SET {sets}""",
                    tuple(row[c] for c in cols))
                updated += 1
            # venue dimension table, populated only from published fields
            if g.get("venue"):
                self.store.execute(
                    """INSERT INTO venues(venue, timezone, utc_offset, provenance, notes)
                       VALUES(?,?,?,?,?)
                       ON CONFLICT(venue) DO UPDATE SET utc_offset=excluded.utc_offset""",
                    (g["venue"], g.get("venue_tz"), g.get("utc_offset"), "SOURCE",
                     "lat/lon intentionally NULL: no verified first-party coordinate source"))
        self.store.commit()
        return inserted, updated

    def scoreboard_window(self, day: str, *, season: int | None = None,
                          game_types: tuple[int, ...] = (1, 2, 3)) -> int:
        url = f"https://api-web.nhle.com/v1/scoreboard/{day}"
        try:
            payload = self.nhl.scoreboard(day)
        except NetworkUnavailable as exc:
            self.store.flag("broken_api", f"scoreboard {day}: {exc}", severity="error",
                            entity_type="source", entity_id="nhl.api_web")
            return 0
        self.store.record_raw(url, json.dumps(payload), source_id="nhl.api_web")
        games = parse_scoreboard_games(payload, season=season, game_types=game_types)
        ins, upd = self._insert_games(games, "nhl.api_web")
        self.log(f"scoreboard {day}: {len(games)} games ({ins} new, {upd} updated)")
        return len(games)

    def club_season(self, abbrev: str, season: int) -> int:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{abbrev}/{season}"
        try:
            payload = self.nhl.club_schedule_season(abbrev, season)
        except NetworkUnavailable as exc:
            self.store.flag("broken_api", f"club schedule {abbrev} {season}: {exc}",
                            severity="error", entity_type="source", entity_id="nhl.api_web")
            return 0
        self.store.record_raw(url, json.dumps(payload), source_id="nhl.api_web")
        games = [normalize_game(g) for g in payload.get("games", [])]
        games = [g for g in games if g]
        ins, upd = self._insert_games(games, "nhl.api_web")
        self.log(f"club schedule {abbrev} {season}: {len(games)} games ({ins} new, {upd} updated)")
        return len(games)

    def sweep_scoreboards(self, start: str, end: str, *, step_days: int = 10,
                          season: int | None = None, game_types: tuple[int, ...] = (1, 2, 3)) -> int:
        total = 0
        for day in daterange(start, end, step_days):
            total += self.scoreboard_window(day, season=season, game_types=game_types)
        return total

    def rebuild_team_games(self) -> int:
        """Rebuild the team-level view from ``games``.  Pure derivation, no new facts."""
        from .sources.nhl import derive_team_games
        rows = [dict(r) for r in self.store.query("SELECT * FROM games")]
        tg = derive_team_games(rows)
        self.store.execute("DELETE FROM team_game")
        for r in tg:
            self.store.execute(
                """INSERT OR REPLACE INTO team_game(game_id, team_id, opp_id, is_home, gf, ga,
                                                    result, decided_in)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (r["game_id"], r["team_id"], r["opp_id"], r["is_home"], r["gf"], r["ga"],
                 r["result"], r["decided_in"]))
        self.store.commit()
        self.log(f"team_game rebuilt: {len(tg)} rows from {len(rows)} games")
        return len(tg)

    # ------------------------------------------------------------------ markets
    def kalshi_nhl(self, *, status: str = "active", series: str = "KXNHLGAME") -> int:
        markets = self.kalshi.markets(series, status=status)
        if not markets:
            self.store.flag("no_markets", f"no {series} markets with status={status}",
                            severity="info", entity_type="source", entity_id="kalshi.trade_api")
            return 0
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series}"
        self.store.record_raw(url, json.dumps(markets), source_id="kalshi.trade_api")
        ts = utcnow()
        n = 0
        for m in markets:
            game_id, abbrevs = self._match_kalshi_game(m)
            for row in normalize_market(m, ts_utc=m.get("updated_time") or ts,
                                        retrieved_at=ts, source_url=url):
                row["game_id"] = game_id
                cols = [c for c in row if c not in ("status", "settlement", "rules")]
                self.store.execute(
                    f"""INSERT INTO market_quotes({','.join(cols)})
                        VALUES({','.join('?' * len(cols))})
                        ON CONFLICT(provider, contract, side, ts_utc) DO UPDATE SET
                          bid=excluded.bid, ask=excluded.ask, spread=excluded.spread,
                          bid_size=excluded.bid_size, ask_size=excluded.ask_size,
                          volume=excluded.volume, liquidity=excluded.liquidity,
                          last_price=excluded.last_price, game_id=excluded.game_id""",
                    tuple(row[c] for c in cols))
        self.store.commit()
        self.log(f"kalshi: {len(markets)} contracts -> {n} quote rows")
        return len(markets)

    def _match_kalshi_game(self, m: dict) -> tuple[int | None, tuple[str | None, str | None]]:
        """Map a Kalshi contract to an NHL game_id using the rules text, then the ticker.

        Returns (game_id, (away_abbrev, home_abbrev)).  A None game_id is kept as None and
        flagged rather than being attached to a plausible-looking game.
        """
        rules = m.get("rules_primary") or ""
        away_name = home_name = None
        mm = re.search(r"the (.+?) vs (.+?) NHL game originally scheduled for "
                       r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", rules)
        game_date = None
        if mm:
            away_name, home_name, mon, d, y = mm.groups()
            game_date = f"{y}-{MONTHS.get(mon.upper(), 0):02d}-{int(d):02d}"
        if not away_name:
            for key in ("yes_sub_title", "no_sub_title"):
                pass
            ticker = (m.get("event_ticker") or "").replace("KXNHLGAME-", "")
            dm = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(.+)$", ticker)
            if dm:
                yy, mon, dd, rest = dm.groups()
                y = 2000 + int(yy)
                game_date = f"{y}-{MONTHS.get(mon, 0):02d}-{int(dd):02d}"
                for la, lb in ((3, 3), (2, 3), (3, 2), (2, 2)):
                    if len(rest) == la + lb:
                        away_name, home_name = rest[:la], rest[lb:]
                        break
        abbrevs = (self._abbrev_for(away_name), self._abbrev_for(home_name))
        if not (game_date and abbrevs[0] and abbrevs[1]):
            return None, abbrevs
        g = self.store.one(
            """SELECT game_id FROM games WHERE game_date=? AND
               ((away_id=(SELECT team_id FROM teams WHERE abbrev=?) AND
                 home_id=(SELECT team_id FROM teams WHERE abbrev=?)) OR
                (away_id=(SELECT team_id FROM teams WHERE abbrev=?) AND
                 home_id=(SELECT team_id FROM teams WHERE abbrev=?)))""",
            (game_date, abbrevs[0], abbrevs[1], abbrevs[1], abbrevs[0]))
        if g is None:
            self.store.flag("unmatched_market",
                            f"kalshi {m.get('event_ticker')} could not be matched to an NHL game "
                            f"({game_date} {abbrevs})", severity="warn",
                            entity_type="quote", entity_id=m.get("event_ticker"))
            return None, abbrevs
        return int(g["game_id"]), abbrevs

    def _abbrev_for(self, name: str | None) -> str | None:
        if not name:
            return None
        name = name.strip()
        t = self.store.one(
            "SELECT abbrev FROM teams WHERE active=1 AND upper(full_name)=upper(?)", (name,))
        if t:
            return t["abbrev"]
        cands = self.store.query(
            "SELECT abbrev, full_name FROM teams WHERE active=1 AND upper(full_name) LIKE ?",
            (name.upper() + "%",))
        if not cands:
            cands = self.store.query(
                "SELECT abbrev, full_name FROM teams WHERE upper(full_name) LIKE ?",
                (name.upper() + "%",))
        if len(cands) == 1:
            return cands[0]["abbrev"]
        # handle short codes like "LA" -> "LAK"
        cands = self.store.query(
            "SELECT abbrev FROM teams WHERE active=1 AND abbrev LIKE ? AND length(abbrev)<=3",
            (name.upper() + "%",))
        if len(cands) == 1:
            return cands[0]["abbrev"]
        if len(cands) > 1:
            self.store.flag("ambiguous_team",
                            f"kalshi team name '{name}' matches {len(cands)} NHL abbrevs",
                            severity="warn", entity_type="team", entity_id=name)
        return None

    def kalshi_settled(self, *, series: str = "KXNHLGAME", max_pages: int = 10) -> int:
        """Load settled contracts: real historical price + real outcome.

        This is the feed that makes a price-based BACKTEST legitimate.  The entry price used
        is ``previous_yes_ask_dollars`` (buying at the offer, never the mid), and because
        Kalshi does not document that field's timestamp every bet derived from it is stamped
        ``single_source_timing_unverified`` instead of being presented as precisely timed.
        """
        markets = self.kalshi.settled_markets(series, max_pages=max_pages)
        if not markets:
            self.store.flag("no_markets", f"no settled {series} markets returned",
                            severity="warn", entity_type="source",
                            entity_id="kalshi.settled_markets")
            return 0
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series}&status=settled"
        self.store.record_raw(url, json.dumps(markets), source_id="kalshi.settled_markets")
        ts = utcnow()
        n = 0
        for m in markets:
            game_id, _ = self._match_kalshi_game(m)
            ticker = m.get("ticker") or ""
            result = (m.get("result") or "").lower()
            for side, bid_k, ask_k in (("YES", "yes_bid_dollars", "yes_ask_dollars"),):
                self.store.execute(
                    """INSERT INTO market_settlements(provider, event_ticker, contract, game_id,
                                                      game_date, selection, side, result,
                                                      settle_price, price_before, bid_before,
                                                      ask_before, volume, open_interest, open_time,
                                                      settlement_ts, retrieved_at, source_url)
                       VALUES('kalshi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(provider, contract, side) DO UPDATE SET
                         result=excluded.result, settle_price=excluded.settle_price,
                         price_before=excluded.price_before, bid_before=excluded.bid_before,
                         ask_before=excluded.ask_before, volume=excluded.volume,
                         open_interest=excluded.open_interest, game_id=excluded.game_id""",
                    (m.get("event_ticker"), ticker, game_id,
                     (m.get("occurrence_datetime") or "")[:10] or None,
                     m.get("yes_sub_title") or ticker, side, result,
                     _num(m.get("settlement_value_dollars")),
                     _num(m.get("previous_price_dollars")),
                     _num(m.get("previous_yes_bid_dollars")),
                     _num(m.get("previous_yes_ask_dollars")),
                     _num(m.get("volume_fp")), _num(m.get("open_interest_fp")),
                     m.get("open_time"), m.get("settlement_ts"), ts, url))
                n += 1
        self.store.commit()
        unmatched = self.store.one(
            "SELECT COUNT(*) c FROM market_settlements WHERE game_id IS NULL")["c"]
        self.log(f"kalshi settled: {n} contracts ({unmatched} unmatched to an NHL game)")
        return n

    # ------------------------------------------------------------------ injuries
    def espn_injuries(self) -> int:
        url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"
        try:
            payload = self.http.get(url).json
        except NetworkUnavailable as exc:
            self.store.flag("broken_api", f"espn injuries: {exc}", severity="error",
                            entity_type="source", entity_id="espn.nhl_api")
            return 0
        self.store.record_raw(url, json.dumps(payload), source_id="espn.nhl_api")
        ts = payload.get("timestamp") or utcnow()
        n = 0
        for team in payload.get("injuries", []):
            for inj in team.get("injuries", []):
                ath = inj.get("athlete") or {}
                name = ath.get("displayName")
                if not name:
                    continue
                tm = (ath.get("team") or {}).get("abbreviation") or team.get("displayName")
                self.store.execute(
                    """INSERT OR IGNORE INTO injuries(source_id, player_name, team_abbrev, status,
                                                      detail, reported_at, retrieved_at, provenance)
                       VALUES('espn.nhl_api',?,?,?,?,?,?, 'SOURCE')""",
                    (name, tm, inj.get("status"), inj.get("shortComment") or inj.get("longComment"),
                     inj.get("date") or ts, ts))
                n += 1
        self.store.commit()
        self.log(f"espn injuries: {n} entries at {ts}")
        return n

    # ------------------------------------------------------------------ cross-check
    def cross_validate_scoreboard_vs_club(self, abbrevs: Sequence[str], season: int) -> int:
        """Independent re-read of the same games through a different endpoint."""
        board: dict[int, tuple[int, int]] = {}
        for g in self.store.query(
                "SELECT game_id, home_score, away_score FROM games WHERE season=? "
                "AND state IN ('FINAL','OFF')", (season,)):
            if g["home_score"] is not None and g["away_score"] is not None:
                board[int(g["game_id"])] = (int(g["home_score"]), int(g["away_score"]))
        from .verify import Verifier
        second: dict[int, tuple[int, int]] = {}
        for ab in abbrevs:
            url = f"https://api-web.nhle.com/v1/club-schedule-season/{ab}/{season}"
            try:
                payload = self.nhl.club_schedule_season(ab, season)
            except NetworkUnavailable:
                continue
            self.store.record_raw(url, json.dumps(payload), source_id="nhl.api_web",
                                  notes="cross-validation re-read")
            for g in payload.get("games", []):
                ng = normalize_game(g)
                if ng and ng["home_score"] is not None and ng["away_score"] is not None:
                    second[int(ng["game_id"])] = (int(ng["home_score"]), int(ng["away_score"]))
        return Verifier(self.store).cross_validate_games(
            [("games_table", board), ("club_schedule", second)])
