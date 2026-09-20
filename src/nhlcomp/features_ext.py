"""Point-in-time features from the per-game stat lines, goalie logs and market history.

Every value here is computed from rows whose ``game_date`` is strictly *before* the game
being featured (the stats REST rows carry dates, not puck-drop times, and the NHL never
schedules a team twice on one date), so nothing from the game itself or later leaks in.

Goalie caveat (ASSUMPTION, stated on every strategy that uses it): for historical games
the starter is read from the post-game goalie log.  Starters are normally confirmed at the
morning skate, several hours before puck drop, and the Kalshi *closing* price already
reflects them, so pairing the confirmed starter with the closing price is consistent.  For
upcoming games no verified public feed of confirmed starters is ingested, which is why
goalie-gated strategies stay ``WAITING FOR GOALIE`` in forward tests.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import date
from typing import Any, Iterable, Sequence

from .market import implied_prob, two_way_vig


def _mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _days(a: str, b: str) -> int | None:
    try:
        return (date.fromisoformat(b) - date.fromisoformat(a)).days
    except (TypeError, ValueError):
        return None


class ExtendedFeatures:
    """Augments the schedule-based feature rows produced by :class:`FeatureBuilder`."""

    def __init__(self, *, team_stats: Iterable[dict], goalie_stats: Iterable[dict],
                 price_points: Iterable[dict], abbrev_by_id: dict[int, str],
                 window: int = 10):
        self.window = window
        self.abbrev_by_id = abbrev_by_id
        self.id_by_abbrev = {v: k for k, v in abbrev_by_id.items()}
        # team stats indexed by team_id, sorted by date
        self.ts: dict[int, list[dict]] = defaultdict(list)
        for r in team_stats:
            if r.get("game_date"):
                self.ts[int(r["team_id"])].append(dict(r))
        for rows in self.ts.values():
            rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
        self.ts_dates = {t: [r["game_date"] for r in rows] for t, rows in self.ts.items()}
        # goalie logs indexed by player and by (game_id, team_abbrev)
        self.gl: dict[int, list[dict]] = defaultdict(list)
        self.starter_for: dict[tuple[int, str], dict] = {}
        for r in goalie_stats:
            if not r.get("game_date"):
                continue
            r = dict(r)
            self.gl[int(r["player_id"])].append(r)
            if int(r.get("started") or 0) == 1:
                self.starter_for[(int(r["game_id"]), str(r.get("team_abbrev")))] = r
        for rows in self.gl.values():
            rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
        self.gl_dates = {p: [r["game_date"] for r in rows] for p, rows in self.gl.items()}
        # market price points indexed by (game_id, team_abbrev)
        self.pp: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
        for r in price_points:
            if r.get("game_id") is None or not r.get("team_abbrev"):
                continue
            self.pp[(int(r["game_id"]), str(r["team_abbrev"]))][r["point"]] = dict(r)

    # ------------------------------------------------------------------ team stats
    def _prior_team_rows(self, team_id: int, game_date: str) -> list[dict]:
        rows = self.ts.get(team_id)
        if not rows:
            return []
        i = bisect_left(self.ts_dates[team_id], game_date)
        return rows[:i]

    def team_stat_features(self, team_id: int, game_date: str) -> dict[str, Any]:
        prior = self._prior_team_rows(team_id, game_date)
        f: dict[str, Any] = {"n_stat_games": len(prior)}
        keys = ("pp_pct", "pk_pct", "sf", "sa", "fo_pct", "shot_share", "sh_pct", "sv_pct", "pdo")
        for k in keys:
            f[f"{k}_l{self.window}"] = None
            f[f"{k}_std"] = None
        if not prior:
            return f
        for label, rows in ((f"l{self.window}", prior[-self.window:]), ("std", prior)):
            gf = sum(r["gf"] or 0 for r in rows)
            ga = sum(r["ga"] or 0 for r in rows)
            sf = sum(r["sf"] or 0 for r in rows)
            sa = sum(r["sa"] or 0 for r in rows)
            f[f"pp_pct_{label}"] = _mean([r["pp_pct"] for r in rows])
            f[f"pk_pct_{label}"] = _mean([r["pk_pct"] for r in rows])
            f[f"sf_{label}"] = _ratio(sf, len(rows))
            f[f"sa_{label}"] = _ratio(sa, len(rows))
            f[f"fo_pct_{label}"] = _mean([r["fo_pct"] for r in rows])
            f[f"shot_share_{label}"] = _ratio(sf, sf + sa)
            sh = _ratio(gf, sf)
            sv = (round(1.0 - ga / sa, 4) if sa else None)
            f[f"sh_pct_{label}"] = sh
            f[f"sv_pct_{label}"] = sv
            f[f"pdo_{label}"] = round(sh + sv, 4) if (sh is not None and sv is not None) else None
        return f

    # ------------------------------------------------------------------ goalies
    def starter(self, game_id: int, team_id: int) -> dict | None:
        ab = self.abbrev_by_id.get(team_id)
        return self.starter_for.get((game_id, ab)) if ab else None

    def goalie_features(self, game_id: int, team_id: int, game_date: str) -> dict[str, Any]:
        f: dict[str, Any] = {
            "starter_known": 0, "starter_id": None, "starter_name": None,
            f"starter_sv_pct_l{self.window}": None, "starter_sv_pct_std": None,
            f"starter_gaa_l{self.window}": None, "starter_n_prior": None,
            "starter_rest_days": None, "starter_b2b": None, "starter_starts_l7d": None,
            "starter_sv_delta_team": None,
        }
        st = self.starter(game_id, team_id)
        if st is None:
            return f
        pid = int(st["player_id"])
        f.update(starter_known=1, starter_id=pid, starter_name=st.get("goalie_name"))
        rows = self.gl.get(pid, [])
        i = bisect_left(self.gl_dates[pid], game_date)
        prior = [r for r in rows[:i] if int(r.get("started") or 0) == 1 or (r.get("toi_seconds") or 0) > 0]
        f["starter_n_prior"] = len(prior)
        if not prior:
            return f
        for label, rs in ((f"l{self.window}", prior[-self.window:]), ("std", prior)):
            sa = sum(r["shots_against"] or 0 for r in rs)
            sv = sum(r["saves"] or 0 for r in rs)
            f[f"starter_sv_pct_{label}"] = _ratio(sv, sa)
        toi = sum(r["toi_seconds"] or 0 for r in prior[-self.window:])
        ga = sum(r["goals_against"] or 0 for r in prior[-self.window:])
        f[f"starter_gaa_l{self.window}"] = round(ga * 3600.0 / toi, 4) if toi else None
        last = prior[-1]
        gap = _days(last["game_date"], game_date)
        f["starter_rest_days"] = gap
        f["starter_b2b"] = int(gap == 1) if gap is not None else None
        f["starter_starts_l7d"] = len([r for r in prior if (_days(r["game_date"], game_date) or 99) <= 7
                                        and int(r.get("started") or 0) == 1])
        return f

    # ------------------------------------------------------------------ market
    def market_features(self, game_id: int, home_id: int, away_id: int) -> dict[str, Any]:
        hab, aab = self.abbrev_by_id.get(home_id), self.abbrev_by_id.get(away_id)
        hp = self.pp.get((game_id, hab), {}) if hab else {}
        ap = self.pp.get((game_id, aab), {}) if aab else {}
        f: dict[str, Any] = {}
        for pt in ("open", "t24h", "t6h", "t1h", "close", "ig60", "ig120", "latest"):
            h, a = hp.get(pt), ap.get(pt)
            f[f"mkt_{pt}_home_ask"] = h.get("ask") if h else None
            f[f"mkt_{pt}_home_bid"] = h.get("bid") if h else None
            f[f"mkt_{pt}_away_ask"] = a.get("ask") if a else None
            f[f"mkt_{pt}_away_bid"] = a.get("bid") if a else None
            f[f"mkt_{pt}_home_mid"] = implied_prob(h.get("ask"), h.get("bid")) if h else None
            f[f"mkt_{pt}_away_mid"] = implied_prob(a.get("ask"), a.get("bid")) if a else None
            f[f"mkt_{pt}_ts"] = h.get("end_period_ts") if h else (a.get("end_period_ts") if a else None)
        # For a game that has not started, the newest live candle stands in for the close so
        # that the same trigger definitions work in FORWARD tests.  The flag makes the
        # substitution visible; priced BACKTESTS exclude such rows.
        f["mkt_close_is_latest"] = 0
        if f["mkt_close_home_ask"] is None and f["mkt_close_away_ask"] is None and \
                (f.get("mkt_latest_home_ask") is not None or f.get("mkt_latest_away_ask") is not None):
            for suffix in ("home_ask", "home_bid", "away_ask", "away_bid", "home_mid", "away_mid", "ts"):
                f[f"mkt_close_{suffix}"] = f.get(f"mkt_latest_{suffix}")
            f["mkt_close_is_latest"] = 1
        f["mkt_close_vig"] = two_way_vig(f["mkt_close_home_ask"], f["mkt_close_away_ask"])
        f["mkt_close_volume"] = ((hp.get("close") or {}).get("open_interest") or 0) + \
                                ((ap.get("close") or {}).get("open_interest") or 0) or None
        # line movement, in probability points of the home side (mid to mid).  Before puck
        # drop the reference is the newest candle ('latest'); afterwards it is the close.
        ref = "close"
        for a_pt, name in (("open", "mkt_move_home"), ("t24h", "mkt_move24_home"), ("t6h", "mkt_move6_home")):
            a, b = f.get(f"mkt_{a_pt}_home_mid"), f.get(f"mkt_{ref}_home_mid")
            f[name] = round(b - a, 4) if (a is not None and b is not None) else None
        f["mkt_move_ref"] = "latest" if f["mkt_close_is_latest"] else "close"
        f["mkt_has_close"] = int(f["mkt_close_home_ask"] is not None or f["mkt_close_away_ask"] is not None)
        # home favourite flag by the closing mid
        hm = f.get("mkt_close_home_mid")
        f["mkt_home_fav"] = (int(hm > 0.5) if hm is not None else None)
        f["mkt_close_home_prob"] = hm
        return f

    # ------------------------------------------------------------------ compose
    def augment(self, row: dict[str, Any]) -> dict[str, Any]:
        gid, hid, aid, gd = row["game_id"], row["home_id"], row["away_id"], row["game_date"]
        for prefix, tid in (("home", hid), ("away", aid)):
            for k, v in self.team_stat_features(tid, gd).items():
                row[f"{prefix}_{k}"] = v
            for k, v in self.goalie_features(gid, tid, gd).items():
                row[f"{prefix}_{k}"] = v
        row.update(self.market_features(gid, hid, aid))
        # matchup differentials (home minus away)
        for k in (f"pp_pct_l{self.window}", f"pk_pct_l{self.window}", f"shot_share_l{self.window}",
                  f"pdo_l{self.window}", f"starter_sv_pct_l{self.window}", "starter_sv_pct_std",
                  f"fo_pct_l{self.window}"):
            h, a = row.get(f"home_{k}"), row.get(f"away_{k}")
            row[f"diff_{k}"] = round(h - a, 4) if (h is not None and a is not None) else None
        # special-teams matchup: home PP vs away PK and vice versa
        hpp, apk = row.get(f"home_pp_pct_l{self.window}"), row.get(f"away_pk_pct_l{self.window}")
        app, hpk = row.get(f"away_pp_pct_l{self.window}"), row.get(f"home_pk_pct_l{self.window}")
        row["st_edge_home"] = (round((hpp - (1 - apk)) - (app - (1 - hpk)), 4)
                               if None not in (hpp, apk, app, hpk) else None)
        row["both_starters_known"] = int(bool(row.get("home_starter_known")) and bool(row.get("away_starter_known")))
        return row

    def augment_all(self, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.augment(r) for r in rows]

    def coverage(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """How many feature rows actually carry each family of extended data.  Reported
        on the site so a reader can see what the strategies could and could not use."""
        w = self.window
        n = len(rows)
        def cnt(pred):
            return sum(1 for r in rows if pred(r))
        return {
            "rows": n,
            "team_stats_both": cnt(lambda r: r.get(f"home_pp_pct_l{w}") is not None
                                   and r.get(f"away_pp_pct_l{w}") is not None),
            "starters_known_both": cnt(lambda r: r.get("both_starters_known")),
            "market_close": cnt(lambda r: r.get("mkt_has_close") and not r.get("mkt_close_is_latest")),
            "market_open": cnt(lambda r: r.get("mkt_open_home_ask") is not None),
            "market_latest_only": cnt(lambda r: r.get("mkt_close_is_latest")),
            "market_ig60": cnt(lambda r: r.get("mkt_ig60_home_mid") is not None),
            "team_stat_rows": sum(len(v) for v in self.ts.values()),
            "goalie_stat_rows": sum(len(v) for v in self.gl.values()),
            "price_point_rows": sum(len(v) for v in self.pp.values()),
        }


def load_extended(store, *, window: int = 10) -> ExtendedFeatures:
    """Build the augmenter from the ledger."""
    abbrev_by_id = {int(r["team_id"]): r["abbrev"] for r in store.query(
        "SELECT team_id, abbrev FROM teams")}
    team_stats = [dict(r) for r in store.query("SELECT * FROM team_game_stats")]
    goalie_stats = [dict(r) for r in store.query("SELECT * FROM goalie_game_stats")]
    price_points = [dict(r) for r in store.query(
        """SELECT contract, point, end_period_ts, bid, ask, last, mean, volume, open_interest,
                  game_id, team_abbrev
             FROM market_price_points
            WHERE series_ticker='KXNHLGAME' AND game_id IS NOT NULL AND team_abbrev IS NOT NULL""")]
    return ExtendedFeatures(team_stats=team_stats, goalie_stats=goalie_stats,
                            price_points=price_points, abbrev_by_id=abbrev_by_id, window=window)
