"""Point-in-time feature construction.

The single most important rule in this module: for game *g*, every feature is derived
only from games whose ``start_time_utc`` is strictly earlier than ``g.start_time_utc``.
``as_of`` is threaded through every function and ``_prior()`` enforces the cutoff, so a
future result cannot leak into a historical feature even by accident.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .http import parse_iso


def _dt(s: str) -> datetime:
    return parse_iso(s)


@dataclass
class GameRef:
    game_id: int
    start: datetime
    game_date: str
    home_id: int
    away_id: int
    venue: str | None
    venue_tz: str | None
    utc_offset: str | None
    season: int
    game_type: int
    home_score: int | None = None
    away_score: int | None = None
    last_period_type: str | None = None

    @property
    def decided(self) -> bool:
        return self.home_score is not None and self.away_score is not None

    @property
    def winner(self) -> int | None:
        if not self.decided or self.home_score == self.away_score:
            return None
        return self.home_id if self.home_score > self.away_score else self.away_id

    @property
    def total_goals(self) -> int | None:
        return None if not self.decided else int(self.home_score) + int(self.away_score)

    def team_gf(self, team_id: int) -> int | None:
        if not self.decided:
            return None
        return self.home_score if team_id == self.home_id else self.away_score


def _days_between(earlier: str, later: str) -> int | None:
    """Whole calendar days between two ISO dates.  None if either is missing."""
    try:
        from datetime import date as _date
        return (_date.fromisoformat(later) - _date.fromisoformat(earlier)).days
    except (TypeError, ValueError):
        return None


def _offset_hours(off: str | None) -> float | None:
    """``"-05:00"`` -> -5.0.  Returns None rather than guessing."""
    if not off:
        return None
    try:
        sign = -1 if off.startswith("-") else 1
        hh, mm = off.lstrip("+-").split(":")[:2]
        return sign * (int(hh) + int(mm) / 60.0)
    except (ValueError, AttributeError):
        return None


class FeatureBuilder:
    """Builds team-game features from an ordered list of decided games."""

    def __init__(self, games: Sequence[GameRef]):
        self.games = sorted(games, key=lambda g: (g.start, g.game_id))
        self.by_team: dict[int, list[GameRef]] = defaultdict(list)
        for g in self.games:
            self.by_team[g.home_id].append(g)
            self.by_team[g.away_id].append(g)

    # ------------------------------------------------------------------ core
    def _prior(self, team_id: int, as_of: datetime, exclude_game_id: int | None = None
               ) -> list[GameRef]:
        out = []
        for g in self.by_team[team_id]:
            if exclude_game_id is not None and g.game_id == exclude_game_id:
                continue
            if g.start < as_of and g.decided:
                out.append(g)
        return out

    def team_features(self, team_id: int, as_of: datetime, *, is_home: bool,
                      opp_id: int, exclude_game_id: int | None = None,
                      window: int = 10) -> dict[str, Any]:
        prior = self._prior(team_id, as_of, exclude_game_id)
        prior.sort(key=lambda g: g.start)
        f: dict[str, Any] = {
            "team_id": team_id, "is_home": int(is_home), "n_prior": len(prior),
            "rest_days": None, "calendar_rest_days": None, "back_to_back": 0, "g3_in_4": 0, "g4_in_6": 0, "g5_in_7": 0,
            "games_last_7": 0, "prev_ot": 0, "prev_so": 0, "prev_result": None,
            "streak_wins": 0, "streak_losses": 0, "road_trip_len": 0,
            "tz_shift": None, "same_venue_as_prev": None,
            "gf_avg": None, "ga_avg": None, "goal_diff_avg": None,
            "total_avg": None, "ot_rate": None, "so_rate": None,
            f"gf_avg_l{window}": None, f"ga_avg_l{window}": None,
            f"total_avg_l{window}": None, f"win_pct_l{window}": None,
            "home_win_pct": None, "season_points_pct": None,
        }
        if not prior:
            return f

        last = prior[-1]
        rest = (as_of - last.start).total_seconds() / 86400.0
        f["rest_days"] = round(rest, 3)
        # Back-to-back is a calendar-day property, not an elapsed-hours one: two games on
        # consecutive dates can be 23 or 26 hours apart depending on puck-drop times, so an
        # exact 24h test would misclassify real back-to-backs.
        f["back_to_back"] = int(_days_between(last.game_date, as_of.date().isoformat()) == 1)
        f["calendar_rest_days"] = _days_between(last.game_date, as_of.date().isoformat())

        # Density windows.  "Three games in four nights" counts the game being played, so
        # the threshold applies to (prior games in window + 1).  Forgetting the current game
        # would systematically undercount and blunt every fatigue signal.
        def in_window(days: float) -> int:
            return len([g for g in prior if (as_of - g.start).total_seconds() <= days * 86400])

        f["games_last_7"] = in_window(7) + 1
        f["g3_in_4"] = int(in_window(4) + 1 >= 3)
        f["g4_in_6"] = int(in_window(6) + 1 >= 4)
        f["g5_in_7"] = int(in_window(7) + 1 >= 5)

        f["prev_ot"] = int(last.last_period_type == "OT")
        f["prev_so"] = int(last.last_period_type == "SO")

        def _res(g: GameRef) -> str | None:
            gf = g.team_gf(team_id)
            ga = g.home_score + g.away_score - gf if gf is not None else None
            if gf is None:
                return None
            if gf > ga:
                return "W"
            if gf < ga:
                return "OTL" if g.last_period_type in ("OT", "SO") else "L"
            return "T"

        f["prev_result"] = _res(last)
        for g in reversed(prior):
            r = _res(g)
            if r == "W":
                if f["streak_losses"]:
                    break
                f["streak_wins"] += 1
            elif r in ("L", "OTL"):
                if f["streak_wins"]:
                    break
                f["streak_losses"] += 1
            else:
                break

        # travel: derived from published venue UTC offsets only (no invented coordinates)
        prev_home = last.home_id == team_id
        prev_off = _offset_hours(last.utc_offset)
        f["tz_shift"] = None
        f["same_venue_as_prev"] = int(prev_home == is_home) if (last.venue and prev_home is not None) else None
        # tz_shift is only meaningful across a location change; the offset we know for the
        # *upcoming* game is passed in by the caller via set_upcoming_offset.
        f["_prev_offset"] = prev_off
        f["_prev_is_home"] = prev_home

        # road trip length = consecutive away games up to and including this one
        trip = 0
        for g in reversed(prior):
            if g.home_id == team_id:
                break
            trip += 1
        f["road_trip_len"] = trip + (0 if is_home else 1)

        gfs = [g.team_gf(team_id) for g in prior]
        gas = [(g.home_score + g.away_score - g.team_gf(team_id)) for g in prior]
        f["gf_avg"] = round(sum(gfs) / len(gfs), 4)
        f["ga_avg"] = round(sum(gas) / len(gas), 4)
        f["goal_diff_avg"] = round(f["gf_avg"] - f["ga_avg"], 4)
        f["total_avg"] = round(f["gf_avg"] + f["ga_avg"], 4)
        f["ot_rate"] = round(sum(1 for g in prior if g.last_period_type == "OT") / len(prior), 4)
        f["so_rate"] = round(sum(1 for g in prior if g.last_period_type == "SO") / len(prior), 4)

        w = prior[-window:]
        wgfs = [g.team_gf(team_id) for g in w]
        wgas = [(g.home_score + g.away_score - g.team_gf(team_id)) for g in w]
        f[f"gf_avg_l{window}"] = round(sum(wgfs) / len(wgfs), 4)
        f[f"ga_avg_l{window}"] = round(sum(wgas) / len(wgas), 4)
        f[f"total_avg_l{window}"] = round(
            sum(wgfs) / len(wgfs) + sum(wgas) / len(wgas), 4)
        pts = 0
        for g in w:
            r = _res(g)
            pts += 2 if r == "W" else (1 if r == "OTL" else 0)
        f[f"win_pct_l{window}"] = round(pts / (2 * len(w)), 4)

        home_games = [g for g in prior if g.home_id == team_id]
        if home_games:
            hw = sum(1 for g in home_games
                     if (g.home_score or 0) > (g.away_score or 0))
            f["home_win_pct"] = round(hw / len(home_games), 4)

        spts = 0
        for g in prior:
            r = _res(g)
            spts += 2 if r == "W" else (1 if r == "OTL" else 0)
        f["season_points_pct"] = round(spts / (2 * len(prior)), 4)
        return f

    def finalize_travel(self, feat: dict[str, Any], upcoming_offset: str | None) -> dict[str, Any]:
        """Add the time-zone shift for the upcoming game, using published offsets only."""
        cur = _offset_hours(upcoming_offset)
        prev = feat.get("_prev_offset")
        feat.pop("_prev_offset", None)
        feat.pop("_prev_is_home", None)
        if cur is None or prev is None:
            feat["tz_shift"] = None
            feat["travel_direction"] = None
        else:
            shift = round(cur - prev, 3)
            feat["tz_shift"] = shift
            feat["travel_direction"] = ("none" if shift == 0 else
                                        ("east" if shift > 0 else "west"))
        return feat

    def pair_features(self, g: GameRef) -> dict[str, Any]:
        """Both teams' features for one game, plus matchup interactions."""
        home = self.team_features(g.home_id, g.start, is_home=True, opp_id=g.away_id,
                                  exclude_game_id=g.game_id)
        away = self.team_features(g.away_id, g.start, is_home=False, opp_id=g.home_id,
                                  exclude_game_id=g.game_id)
        home = self.finalize_travel(home, g.utc_offset)
        away = self.finalize_travel(away, g.utc_offset)
        f: dict[str, Any] = {
            "game_id": g.game_id, "game_date": g.game_date, "season": g.season,
            "home_id": g.home_id, "away_id": g.away_id,
            "start_time_utc": g.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "venue": g.venue, "utc_offset": g.utc_offset,
        }
        f.update({f"home_{k}": v for k, v in home.items()})
        f.update({f"away_{k}": v for k, v in away.items()})

        # matchup interactions (brief sections 16-20)
        f["rest_diff"] = (round(home["rest_days"] - away["rest_days"], 3)
                          if (home["rest_days"] is not None and away["rest_days"] is not None)
                          else None)
        f["home_only_b2b"] = int(home["back_to_back"] and not away["back_to_back"])
        f["away_only_b2b"] = int(away["back_to_back"] and not home["back_to_back"])
        f["both_b2b"] = int(home["back_to_back"] and away["back_to_back"])
        f["fatigue_adv"] = ("home" if f["home_only_b2b"] == 0 and f["away_only_b2b"] == 1 else
                            ("away" if f["home_only_b2b"] == 1 and f["away_only_b2b"] == 0 else "none"))
        if home.get("total_avg") is not None and away.get("total_avg") is not None:
            f["pace_sum"] = round(home["total_avg"] + away["total_avg"], 4)
            f["pace_diff"] = round(home["total_avg"] - away["total_avg"], 4)
        else:
            f["pace_sum"] = f["pace_diff"] = None
        if home.get("gf_avg") is not None and away.get("ga_avg") is not None:
            f["home_exp_goals"] = round((home["gf_avg"] + away["ga_avg"]) / 2.0, 4)
            f["away_exp_goals"] = round((away["gf_avg"] + home["ga_avg"]) / 2.0, 4)
        else:
            f["home_exp_goals"] = f["away_exp_goals"] = None
        return f

    def build_all(self, games: Iterable[GameRef] | None = None) -> list[dict[str, Any]]:
        return [self.pair_features(g) for g in (games if games is not None else self.games)]
