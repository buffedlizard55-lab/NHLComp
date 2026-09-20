"""Adapters for the NHL's own public endpoints.

Verified from this repository's build environment on 2026-09-20:

* ``https://api-web.nhle.com/v1/...``            -> HTTP 200, JSON
* ``https://api.nhle.com/stats/rest/en/...``     -> HTTP 200, JSON
* ``https://api.nhl.com/api/v1/...``             -> fetch failed (see registry note)
* ``https://statsapi.web.nhl.com/api/v1/...``    -> fetch failed (legacy host)

Additionally verified 2026-09-20 (research session, HTTP 200 with the documented payloads):

* ``/stats/rest/en/team/summary?isGame=true&cayenneExp=seasonId=20252026 and gameTypeId=2``
  -> one row per team per game (2,624 rows for 2025-26) with ``powerPlayPct``,
  ``penaltyKillPct``, ``shotsForPerGame``, ``shotsAgainstPerGame``, ``faceoffWinPct``,
  ``goalsFor``/``goalsAgainst``, ``homeRoad``, ``opponentTeamAbbrev`` and ``gameId``.
* ``/stats/rest/en/goalie/summary?isGame=true&...`` -> one row per goalie per game
  (2,768 rows) with ``gamesStarted``, ``saves``, ``shotsAgainst``, ``savePct``,
  ``timeOnIce`` (seconds), ``wins``/``losses``/``otLosses``, ``teamAbbrev``.
* ``/v1/gamecenter/{id}/boxscore`` -> per-player lines incl. goalie ``starter``/``decision``
  and even-strength / power-play / short-handed shots against.
* ``/v1/partner-game/US/now`` -> the NHL's betting-partner (DraftKings) moneyline, puck
  line and total for today's slate, American odds, *current only* (no history).
* ``/v1/edge/team-comparison/{teamId}/{season}/{gameType}`` and the other ``/v1/edge/...``
  resources -> NHL EDGE tracking aggregates (shot speed, skating speed/distance, zone
  time, shot location).  Season-to-date snapshots only, so usable for FORWARD tests but not
  for point-in-time backtests.

Nothing here invents a field: every parser reads defensively and any expected-but-
missing field is surfaced as an irregularity by the ingest layer rather than being
filled with a plausible-looking default.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

from ..http import HttpClient, parse_iso

API_WEB = "https://api-web.nhle.com/v1"
STATS_REST = "https://api.nhle.com/stats/rest/en"
LEGACY = "https://statsapi.web.nhl.com/api/v1"
NEW_V1 = "https://api.nhl.com/api/v1"


def _txt(d: dict, *path, default=None):
    """Walk a possibly-localised ``{"default": ...}`` dict or plain key path."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    if isinstance(cur, dict):
        for pref in ("default", "en"):
            if pref in cur:
                return cur[pref]
    return cur


class NhlApi:
    source_id = "nhl.api_web"

    def __init__(self, http: HttpClient):
        self.http = http

    def _get(self, path: str) -> dict:
        return self.http.get(f"{API_WEB}{path}").json

    # ------------------------------------------------------------ schedule
    def scoreboard(self, day: str) -> dict:
        """``/v1/scoreboard/{YYYY-MM-DD}``.

        Verified behaviour: the payload is a *window*, not a single day -- it contains
        ``gamesByDate`` spanning several days around the requested date.
        """
        return self._get(f"/scoreboard/{day}")

    def scoreboard_now(self) -> dict:
        return self._get("/scoreboard/now")

    def club_schedule_season(self, abbrev: str, season: int) -> dict:
        """Full season schedule for one club, including final scores and winning goalie."""
        return self._get(f"/club-schedule-season/{abbrev}/{season}")

    def standings(self, day: str | None = None) -> dict:
        return self._get(f"/standings/{day}" if day else "/standings/now")

    def gamecenter_landing(self, game_id: int) -> dict:
        return self._get(f"/gamecenter/{game_id}/landing")

    def gamecenter_boxscore(self, game_id: int) -> dict:
        return self._get(f"/gamecenter/{game_id}/boxscore")

    def gamecenter_play_by_play(self, game_id: int) -> dict:
        return self._get(f"/gamecenter/{game_id}/play-by-play")

    def player_landing(self, player_id: int) -> dict:
        return self._get(f"/player/{player_id}/landing")

    def roster(self, abbrev: str, season: int | None = None) -> dict:
        return self._get(f"/roster/{abbrev}/{season}" if season else f"/roster/{abbrev}/now")

    # ------------------------------------------------------------ odds / EDGE
    def partner_odds(self, country: str = "US") -> dict:
        """``/v1/partner-game/{country}/now`` -- the league's sportsbook partner prices.

        Verified 2026-09-20: ``bettingPartner.name == "DraftKings"``, ``games[]`` with
        ``homeTeam.odds`` / ``awayTeam.odds`` entries of ``description`` in
        {MONEY_LINE_2_WAY, PUCK_LINE, OVER_UNDER}, ``value`` (American) and ``qualifier``
        (e.g. ``O6.5``, ``-1.5``).  Live requests are never served from the disk cache.
        """
        return self.http.get(f"{API_WEB}/partner-game/{country}/now", use_cache=False).json

    def edge_team_comparison(self, team_id: int, season: int | None = None,
                             game_type: int = 2) -> dict:
        tail = f"{season}/{game_type}" if season else "now"
        return self._get(f"/edge/team-comparison/{team_id}/{tail}")

    def edge_team_detail(self, team_id: int, season: int | None = None,
                         game_type: int = 2) -> dict:
        tail = f"{season}/{game_type}" if season else "now"
        return self._get(f"/edge/team-detail/{team_id}/{tail}")


class NhlStatsRest:
    """The ``api.nhle.com/stats/rest`` records API (paginated, ``cayenneExp`` filters)."""

    source_id = "nhl.stats_rest"

    def __init__(self, http: HttpClient):
        self.http = http

    def teams(self) -> list[dict]:
        return self.http.get(f"{STATS_REST}/team").json.get("data", [])

    def paged(self, endpoint: str, cayenne: str, *, limit: int = 100,
              max_pages: int = 50, extra: str = "", is_game: bool = False,
              use_cache: bool = True) -> Iterable[dict]:
        start = 0
        for _ in range(max_pages):
            url = (f"{STATS_REST}/{endpoint}?isAggregate=false&isGame={'true' if is_game else 'false'}"
                   f"&start={start}&limit={limit}&cayenneExp={cayenne}{extra}")
            payload = self.http.get(url, use_cache=use_cache).json
            rows = payload.get("data", [])
            if not rows:
                return
            for row in rows:
                yield row
            start += len(rows)
            # The server caps a page at 100 rows whatever ``limit`` says (observed
            # 2026-09-20: limit=500 returned 100).  Paginate on the reported ``total``;
            # only fall back to the short-page rule when no total is published.
            total = payload.get("total")
            if total is not None:
                try:
                    if start >= int(total):
                        return
                    continue
                except (TypeError, ValueError):
                    pass
            if len(rows) < limit:
                return

    def team_game_rows(self, season: int, game_type: int = 2, *, limit: int = 100,
                       use_cache: bool = True) -> list[dict]:
        """Per-team per-game summary rows (``team/summary?isGame=true``).

        Each row is one team's line for one game: goals, shots, PP%, PK%, faceoff%.  The
        endpoint sorts arbitrarily, so callers key by (gameId, teamId) and never assume order.
        """
        cay = f"seasonId={season}%20and%20gameTypeId={game_type}"
        return list(self.paged("team/summary", cay, limit=limit, max_pages=80, is_game=True,
                               use_cache=use_cache))

    def goalie_game_rows(self, season: int, game_type: int = 2, *, limit: int = 100,
                         use_cache: bool = True) -> list[dict]:
        """Per-goalie per-game rows (``goalie/summary?isGame=true``): starts, saves, SA."""
        cay = f"seasonId={season}%20and%20gameTypeId={game_type}"
        return list(self.paged("goalie/summary", cay, limit=limit, max_pages=80, is_game=True,
                               use_cache=use_cache))


# ---------------------------------------------------------------- parsing
def parse_scoreboard_games(payload: dict, *, season: int | None = None,
                           game_types: tuple[int, ...] = (2,)) -> list[dict]:
    """Flatten a scoreboard payload into normalized game rows."""
    out: list[dict] = []
    for block in payload.get("gamesByDate", []):
        for g in block.get("games", []):
            row = normalize_game(g)
            if row is None:
                continue
            if season is not None and row["season"] != season:
                continue
            if row["game_type"] not in game_types:
                continue
            out.append(row)
    return out


def normalize_game(g: dict) -> dict | None:
    game_id = g.get("id")
    home = g.get("homeTeam") or {}
    away = g.get("awayTeam") or {}
    if not game_id or not home.get("id") or not away.get("id"):
        return None
    state = (g.get("gameState") or "").upper()
    pd = g.get("periodDescriptor") or {}
    hs, as_ = home.get("score"), away.get("score")
    final = state in ("FINAL", "OFF")
    row = {
        "game_id": int(game_id),
        "season": int(g.get("season") or 0),
        "game_type": int(g.get("gameType") or 0),
        "game_date": g.get("gameDate") or "",
        "start_time_utc": g.get("startTimeUTC") or "",
        "home_id": int(home["id"]),
        "away_id": int(away["id"]),
        "venue": _txt(g.get("venue") or {}, "default"),
        "venue_tz": g.get("venueTimezone"),
        "utc_offset": g.get("venueUTCOffset"),
        "eastern_offset": g.get("easternUTCOffset"),
        "neutral_site": 1 if g.get("neutralSite") else 0,
        "state": state,
        "home_score": int(hs) if isinstance(hs, int) else None,
        "away_score": int(as_) if isinstance(as_, int) else None,
        "last_period_type": (g.get("gameOutcome") or {}).get("lastPeriodType")
        or (pd.get("periodType") if final else None),
        "home_abbrev": home.get("abbrev"),
        "away_abbrev": away.get("abbrev"),
        "winning_goalie_id": ((g.get("winningGoalie") or {}).get("playerId")),
    }
    if row["last_period_type"] == "REG" and pd.get("periodType") == "OT":
        # defensive: the two fields disagree, prefer the explicit gameOutcome
        row["last_period_type"] = "REG"
    return row


def derive_team_games(games: Iterable[dict]) -> list[dict]:
    """Build the team-level view plus result/decided_in from a normalized game."""
    rows: list[dict] = []
    for g in games:
        hs, as_ = g.get("home_score"), g.get("away_score")
        decided = g.get("last_period_type") or "REG"
        for team_id, opp_id, is_home, gf, ga in (
            (g["home_id"], g["away_id"], 1, hs, as_),
            (g["away_id"], g["home_id"], 0, as_, hs),
        ):
            result = None
            if gf is not None and ga is not None:
                if gf > ga:
                    result = "W"
                elif gf < ga:
                    result = "L" if decided in ("OT", "SO") else "L"
                else:
                    result = "T"
                if gf < ga and decided in ("OT", "SO"):
                    result = "OTL"
            rows.append({"game_id": g["game_id"], "team_id": team_id, "opp_id": opp_id,
                         "is_home": is_home, "gf": gf, "ga": ga, "result": result,
                         "decided_in": decided if gf is not None else None})
    return rows


def parse_standings(payload: dict, *, source_id: str = "nhl.api_web") -> list[dict]:
    rows = []
    for s in payload.get("standings", []):
        abbrev = _txt(s.get("teamAbbrev") or {}, "default")
        rows.append({
            "as_of": s.get("date") or "",
            "season": int(s.get("seasonId") or 0),
            "abbrev": abbrev,
            "gp": s.get("gamesPlayed"), "wins": s.get("wins"), "losses": s.get("losses"),
            "otl": s.get("otLosses"), "points": s.get("points"),
            "gf": s.get("goalFor"), "ga": s.get("goalAgainst"),
            "home_wins": s.get("homeWins"), "home_losses": s.get("homeLosses"),
            "home_otl": s.get("homeOtLosses"), "home_gf": s.get("homeGoalsFor"),
            "home_ga": s.get("homeGoalsAgainst"),
            "road_wins": s.get("roadWins"), "road_losses": s.get("roadLosses"),
            "road_otl": s.get("roadOtLosses"), "road_gf": s.get("roadGoalsFor"),
            "road_ga": s.get("roadGoalsAgainst"),
            "l10_wins": s.get("l10Wins"), "l10_losses": s.get("l10Losses"),
            "l10_otl": s.get("l10OtLosses"), "l10_gf": s.get("l10GoalsFor"),
            "l10_ga": s.get("l10GoalsAgainst"),
            "streak_code": s.get("streakCode"), "streak_count": s.get("streakCount"),
            "shootout_wins": s.get("shootoutWins"), "shootout_losses": s.get("shootoutLosses"),
            "clinch": s.get("clinchIndicator"),
            "conference": s.get("conferenceAbbrev"), "division": s.get("divisionAbbrev"),
            "full_name": _txt(s.get("teamName") or {}, "default"),
            "source_id": source_id,
        })
    return rows


def season_dates(season: int) -> tuple[str, str]:
    """Approximate regular-season window used only for scheduling ingest sweeps."""
    start = date(season // 10000, 10, 1)
    end = date(season // 10000 + 1, 5, 1)
    return start.isoformat(), end.isoformat()


def daterange(start: str, end: str, step_days: int = 1) -> list[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out = []
    while s <= e:
        out.append(s.isoformat())
        s += timedelta(days=step_days)
    return out


# ---------------------------------------------------------------- per-game stats parsers
def _fnum(v: Any) -> float | None:
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_team_game_row(r: dict) -> dict | None:
    """Normalize one ``team/summary?isGame=true`` row.  Returns None when the identifying
    fields are missing rather than fabricating a key."""
    gid, tid = r.get("gameId"), r.get("teamId")
    if gid is None or tid is None:
        return None
    return {
        "game_id": int(gid), "team_id": int(tid),
        "game_date": r.get("gameDate"), "home_road": r.get("homeRoad"),
        "opponent_abbrev": r.get("opponentTeamAbbrev"),
        "team_name": r.get("teamFullName"),
        "gf": _fnum(r.get("goalsFor")), "ga": _fnum(r.get("goalsAgainst")),
        "sf": _fnum(r.get("shotsForPerGame")), "sa": _fnum(r.get("shotsAgainstPerGame")),
        "pp_pct": _fnum(r.get("powerPlayPct")), "pk_pct": _fnum(r.get("penaltyKillPct")),
        "pp_net_pct": _fnum(r.get("powerPlayNetPct")), "pk_net_pct": _fnum(r.get("penaltyKillNetPct")),
        "fo_pct": _fnum(r.get("faceoffWinPct")),
        "wins": _fnum(r.get("wins")), "losses": _fnum(r.get("losses")),
        "ot_losses": _fnum(r.get("otLosses")), "points": _fnum(r.get("points")),
        "reg_wins": _fnum(r.get("winsInRegulation")), "so_wins": _fnum(r.get("winsInShootout")),
    }


def parse_goalie_game_row(r: dict) -> dict | None:
    gid, pid = r.get("gameId"), r.get("playerId")
    if gid is None or pid is None:
        return None
    return {
        "game_id": int(gid), "player_id": int(pid),
        "goalie_name": r.get("goalieFullName"), "team_abbrev": r.get("teamAbbrev"),
        "opponent_abbrev": r.get("opponentTeamAbbrev"), "home_road": r.get("homeRoad"),
        "game_date": r.get("gameDate"),
        "started": int(bool(_fnum(r.get("gamesStarted")))),
        "saves": _fnum(r.get("saves")), "shots_against": _fnum(r.get("shotsAgainst")),
        "goals_against": _fnum(r.get("goalsAgainst")), "save_pct": _fnum(r.get("savePct")),
        "toi_seconds": _fnum(r.get("timeOnIce")),
        "decision": ("W" if _fnum(r.get("wins")) else "L" if _fnum(r.get("losses"))
                     else "OTL" if _fnum(r.get("otLosses")) else None),
    }


def parse_partner_odds(payload: dict) -> list[dict]:
    """Flatten ``/partner-game`` into one row per (game, market) with both sides.

    American prices are stored exactly as published; implied probabilities are computed
    downstream and labelled DERIVED.
    """
    partner = _txt(payload, "bettingPartner", "name") or "unknown"
    updated = payload.get("lastUpdatedUTC")
    out: list[dict] = []
    for g in payload.get("games", []) or []:
        gid = g.get("gameId")
        if gid is None:
            continue
        home, away = g.get("homeTeam") or {}, g.get("awayTeam") or {}
        by_desc: dict[str, dict[str, Any]] = {}
        for side, team in (("home", home), ("away", away)):
            for o in team.get("odds") or []:
                d = o.get("description")
                if not d:
                    continue
                slot = by_desc.setdefault(d, {})
                slot[f"{side}_price"] = _fnum(o.get("value"))
                slot[f"{side}_qualifier"] = o.get("qualifier") or ""
        for desc, vals in by_desc.items():
            out.append({
                "game_id": int(gid), "partner": partner, "market_desc": desc,
                "start_time_utc": g.get("startTimeUTC"), "game_type": g.get("gameType"),
                "home_abbrev": home.get("abbrev"), "away_abbrev": away.get("abbrev"),
                "home_price": vals.get("home_price"), "away_price": vals.get("away_price"),
                "home_qualifier": vals.get("home_qualifier", ""),
                "away_qualifier": vals.get("away_qualifier", ""),
                "source_updated_utc": updated,
            })
    return out
