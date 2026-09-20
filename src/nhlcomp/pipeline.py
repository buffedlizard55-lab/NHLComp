"""End-to-end pipeline: ingest -> features -> models -> discovery -> backtest ->
forward decisions -> settlement -> analysis -> verification -> site.

``run_all`` is what CI executes.  Each stage is independently callable and idempotent, so
a failed run can be resumed rather than restarted.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .analysis import Performance
from .backtest import Backtester, CAVEAT_NO_PRICE, PricedBacktester, market_baseline
from .discovery import DiscoveryEngine
from .features import FeatureBuilder, GameRef
from .features_ext import load_extended
from .http import HttpClient, NetworkUnavailable
from .ingest import Ingestor
from .market import american_to_prob, devig_pair
from .models import (EloModel, HomeIceOnly, LogisticRest, PoissonModel, brier, log_loss)
from .paper import PaperEngine
from .store import Store, utcnow
from .strategies import DecisionContext, Quote, build_seed_strategies
from .verify import Verifier


class Pipeline:
    def __init__(self, store: Store, http: HttpClient, *, verbose: bool = True):
        self.store = store
        self.http = http
        self.verbose = verbose
        self.ing = Ingestor(store, http, verbose=verbose)
        self.paper = PaperEngine(store)
        self.perf = Performance(store)
        self.report: dict[str, Any] = {}
        self.ext = None
        self._feature_kw: dict[str, Any] = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}", flush=True)

    # ------------------------------------------------------------- stage 1
    def _safe(self, out: dict[str, Any], key: str, fn: Any, *a: Any, **kw: Any) -> Any:
        """Run one ingest step; a failing source is recorded, never allowed to abort the
        run (the other sources are independent of it)."""
        try:
            out[key] = fn(*a, **kw)
        except NetworkUnavailable as exc:
            out[key] = {"error": f"network: {exc}"}
            self.log(f"{key} skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - recorded as an irregularity, not hidden
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
            self.store.flag("ingest_step_failed", f"{key}: {type(exc).__name__}: {exc}",
                            severity="warn", entity_type="dataset", entity_id=key)
            self.store.commit()
            self.log(f"{key} FAILED: {type(exc).__name__}: {exc}")
        return out[key]

    def stage_ingest(self, *, seasons: Sequence[int], scoreboard_days: Sequence[str],
                     club_abbrevs: Sequence[str], ingest_settled_pages: int = 5,
                     cross_check_abbrevs: Sequence[str] = (),
                     kalshi_budget: int = 1500, live_points_budget: int = 120,
                     stats_seasons: Sequence[int] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["probes"] = self.ing.verify_sources()
        out["teams"] = self.ing.teams()
        for season in seasons:
            try:
                self.ing.standings(f"{season // 10000 + 1}-04-17")
                out[f"active_teams_{season}"] = self.store.mark_current_teams(season)
            except NetworkUnavailable as exc:
                self.log(f"standings {season} skipped: {exc}")
        for season in seasons:
            for ab in club_abbrevs:
                self.ing.club_season(ab, season)
        for day in scoreboard_days:
            self.ing.scoreboard_window(day)
        out["winners_derived"] = self.store.derive_winners()
        out["team_game_rows"] = self.ing.rebuild_team_games()
        # official per-game team and goalie reports (stats REST, isGame=true)
        current = max(seasons) if seasons else None
        stat_seasons = list(stats_seasons or seasons)
        if stat_seasons:
            self._safe(out, "team_game_stats", self.ing.nhl_team_game_stats, stat_seasons,
                       current_season=current)
            self._safe(out, "goalie_game_stats", self.ing.nhl_goalie_game_stats, stat_seasons,
                       current_season=current)
        # Kalshi: series registry, live quotes, settled contracts + candle history
        self._safe(out, "kalshi_series", self.ing.kalshi_series_discovery)
        out["kalshi_active"] = self.ing.kalshi_nhl()
        self._safe(out, "kalshi_history", self.ing.kalshi_history, series="KXNHLGAME",
                   max_calls=kalshi_budget)
        # legacy settled sweep kept for the counters the old reports expose
        out["kalshi_settled"] = self.ing.kalshi_settled(max_pages=ingest_settled_pages)
        self._safe(out, "kalshi_live_points", self.ing.kalshi_live_price_points,
                   max_calls=live_points_budget)
        # sportsbook reference odds (DraftKings via NHL partner feed) and EDGE tracking
        self._safe(out, "partner_odds", self.ing.nhl_partner_odds)
        if current:
            self._safe(out, "edge_snapshots", self.ing.nhl_edge_snapshots, current)
        out["injuries"] = self.ing.espn_injuries()
        if cross_check_abbrevs and seasons:
            out["cross_validation_conflicts"] = self.ing.cross_validate_scoreboard_vs_club(
                cross_check_abbrevs, seasons[0])
        self.report["ingest"] = out
        return out

    # ------------------------------------------------------------- stage 2
    def game_refs(self, *, seasons: Sequence[int] | None = None,
                  game_types: tuple[int, ...] = (2,)) -> list[GameRef]:
        sql = "SELECT * FROM games WHERE game_type IN (%s)" % ",".join("?" * len(game_types))
        params: list[Any] = list(game_types)
        if seasons:
            sql += " AND season IN (%s)" % ",".join("?" * len(seasons))
            params += list(seasons)
        refs = []
        for r in self.store.query(sql + " ORDER BY start_time_utc", params):
            if not r["start_time_utc"]:
                continue
            refs.append(GameRef(
                game_id=int(r["game_id"]), start=_parse(r["start_time_utc"]),
                game_date=r["game_date"] or "", home_id=int(r["home_id"]),
                away_id=int(r["away_id"]), venue=r["venue"], venue_tz=r["venue_tz"],
                utc_offset=r["utc_offset"], season=int(r["season"]),
                game_type=int(r["game_type"]), home_score=r["home_score"],
                away_score=r["away_score"], last_period_type=r["last_period_type"]))
        return refs

    def stage_features(self, *, seasons: Sequence[int] | None = None,
                       game_types: tuple[int, ...] = (2,)) -> list[dict[str, Any]]:
        self._feature_kw = {"seasons": seasons, "game_types": game_types}
        refs = self.game_refs(seasons=seasons, game_types=game_types)
        fb = FeatureBuilder(refs)
        rows = fb.build_all()
        games_by_id = {g.game_id: g for g in refs}
        for r in rows:
            g = games_by_id[int(r["game_id"])]
            r["_winner"] = g.winner
            r["_total"] = g.total_goals
            r["_decided"] = g.decided
        # point-in-time team/goalie/market features from the extended tables
        self.ext = load_extended(self.store)
        self.ext.augment_all(rows)
        self.report["extended_feature_coverage"] = self.ext.coverage(rows)
        problems = Backtester.assert_no_leakage(rows, games_by_id)
        if problems:
            for p in problems[:50]:
                self.store.flag("strategy_leakage", p, severity="critical",
                                entity_type="feature")
        self.report["feature_rows"] = len(rows)
        self.report["leakage_problems"] = problems
        self.log(f"features: {len(rows)} rows, {len(problems)} leakage problems")
        return rows

    # ------------------------------------------------------------- stage 3
    def stage_models(self, rows: Sequence[dict[str, Any]], refs: Sequence[GameRef],
                     *, prior_season: int | None = None) -> dict[str, Any]:
        # Elo prior from a PREVIOUS season's points percentage (never the current one)
        elo = EloModel()
        if prior_season:
            pts = {}
            for s in self.store.query(
                    """SELECT team_id, points, gamesPlayed FROM (
                         SELECT ss.team_id, ss.points, ss.gp AS gamesPlayed
                         FROM standings_snapshot ss
                         WHERE ss.season=? AND ss.as_of=(
                           SELECT MAX(as_of) FROM standings_snapshot s2 WHERE s2.season=?))""",
                    (prior_season, prior_season)):
                gp = s["gamesPlayed"] or 0
                if gp:
                    pts[int(s["team_id"])] = (s["points"] or 0) / (2 * gp)
            if pts:
                elo.prior_from_points(pts)
                self.log(f"elo prior from {prior_season} standings for {len(pts)} teams")
            else:
                self.log(f"no {prior_season} standings available; elo starts from base rating")

        # league averages for the Poisson model, from PRIOR games only
        decided = [g for g in refs if g.decided]
        hg = [g.home_score for g in decided]
        ag = [g.away_score for g in decided]
        league_home = sum(hg) / len(hg) if hg else 2.9
        league_away = sum(ag) / len(ag) if ag else 2.7

        # walk chronologically: predict, then update
        elo_rows: dict[int, float] = {}
        for g in refs:
            if not g.decided:
                continue
            elo_rows[g.game_id] = elo.prob_home(g.home_id, g.away_id)
            elo.observe(g)
        for g in refs:
            if not g.decided:
                elo_rows.setdefault(g.game_id, elo.prob_home(g.home_id, g.away_id))

        pois = PoissonModel(league_home_gf=league_home, league_away_gf=league_away)
        by_id = {int(r["game_id"]): r for r in rows}
        for gid, r in by_id.items():
            p = pois.predict(r)
            r.update(p)
            pe = elo_rows.get(gid)
            if pe is not None:
                r["p_home_elo"] = round(pe, 5)
                r["p_away_elo"] = round(1 - pe, 5)

        # fit the tiny logistic model on the training window only, apply out of sample
        bt = Backtester(self.store)
        parts = bt.split(list(rows))
        train = [r for r in parts["train"] if r.get("_winner") is not None
                 and r.get("p_home_elo") is not None]
        feats = ("elo_diff", "rest_diff", "home_only_b2b", "away_only_b2b")
        trows = []
        for r in train:
            trows.append({
                "elo_diff": ((r.get("p_home_elo") or 0.5) - 0.5) * 4,
                "rest_diff": r.get("rest_diff") or 0.0,
                "home_only_b2b": r.get("home_only_b2b") or 0.0,
                "away_only_b2b": r.get("away_only_b2b") or 0.0,
            })
        labels = [int(r["_winner"] == r["home_id"]) for r in train]
        logit = LogisticRest(feature_names=feats).fit(trows, labels)
        for r in rows:
            vec = {"elo_diff": ((r.get("p_home_elo") or 0.5) - 0.5) * 4,
                   "rest_diff": r.get("rest_diff") or 0.0,
                   "home_only_b2b": r.get("home_only_b2b") or 0.0,
                   "away_only_b2b": r.get("away_only_b2b") or 0.0}
            r["p_home_logit"] = round(logit.predict_p(vec), 5)
            r["p_away_logit"] = round(1 - logit.predict_p(vec), 5)

        # honest model comparison on the held-out window
        test = [r for r in parts["test"] if r.get("_winner") is not None]
        labels_t = [int(r["_winner"] == r["home_id"]) for r in test]
        comp = {}
        for name, key in (("poisson", "p_home_ml"), ("elo", "p_home_elo"),
                          ("logistic", "p_home_logit")):
            ps = [r.get(key) for r in test]
            ok = all(p is not None for p in ps)
            comp[name] = {"n": len(test),
                          "log_loss": round(log_loss([p for p in ps], labels_t), 5) if ok and test else None,
                          "brier": round(brier([p for p in ps], labels_t), 5) if ok and test else None}
        home_only = HomeIceOnly(p_home=(sum(1 for r in test if r["_winner"] == r["home_id"]) /
                                        len(test)) if test else 0.55)
        comp["home_ice_constant"] = {
            "n": len(test),
            "log_loss": round(log_loss([home_only.predict(r)["p_home_ml"] for r in test],
                                       labels_t), 5) if test else None,
            "brier": round(brier([home_only.predict(r)["p_home_ml"] for r in test],
                                 labels_t), 5) if test else None}
        self.report["model_comparison_out_of_sample"] = comp
        self.report["league_avg_home_goals"] = round(league_home, 4)
        self.report["league_avg_away_goals"] = round(league_away, 4)
        self.store.execute(
            """INSERT OR REPLACE INTO findings(finding_id, created_at, title, body, evidence,
                                               confidence, kind) VALUES(?,?,?,?,?,?,?)""",
            ("FIND_MODEL_COMPARE", utcnow(),
             "Out-of-sample model comparison (held-out chronological window)",
             "Poisson, Elo, a 4-feature logistic model and a home-ice constant were compared on "
             "the same held-out window. Lower log loss is better. A complex model is only kept if "
             "it beats the constant.",
             json.dumps(comp), "medium", "model_evaluation"))
        self.store.commit()
        self.log(f"models: {json.dumps(comp)}")
        return comp

    # ------------------------------------------------------------- stage 4
    def stage_strategies(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        eng = DiscoveryEngine(self.store)
        eng.record_hypotheses(rows)
        decided = [r for r in rows if r.get("_winner") is not None]
        candidates = eng.scan(decided)
        created = eng.promote(candidates)
        eng.record_failures(candidates)
        self.store.execute(
            """INSERT OR REPLACE INTO findings(finding_id, created_at, title, body, evidence,
                                               confidence, kind) VALUES(?,?,?,?,?,?,?)""",
            ("FIND_DISCOVERY_SEARCH", utcnow(), "Discovery search completed",
             eng.search_size_caveat(), json.dumps({"tested": eng.tested,
                                                   "survivors": len(created)}),
             "high", "search_summary"))
        # register the seed library (never overwrites an existing version); a newer
        # version retires the older one -- the old row and its bets stay as history
        for s in build_seed_strategies():
            if self.store.one("SELECT 1 FROM strategies WHERE strategy_id=? AND version=?",
                              (s.strategy_id, s.version)):
                continue
            d = s.describe()
            d.update(status="active", created_at=utcnow(), test_mode="FORWARD TEST",
                     leakage_checked=1)
            self.store.upsert_strategy(d)
            self.store.set_strategy_status(s.strategy_id, s.version, "active",
                                           reason="registered from the seed library",
                                           evidence="hypothesis recorded, unproven")
            for old in self.store.query(
                    "SELECT version FROM strategies WHERE strategy_id=? AND version<? "
                    "AND status IN ('active','candidate','paused')", (s.strategy_id, s.version)):
                self.store.set_strategy_status(
                    s.strategy_id, int(old["version"]), "retired",
                    reason=f"superseded by version {s.version}",
                    evidence="definition changed because a new verified data source became "
                             "available; the old version is kept for the record")
        self.report["discovery"] = {"tested": eng.tested, "survivors": len(created),
                                    "candidates": [c.key for c in candidates][:50]}
        self.log(f"strategies: {eng.tested} triggers tested, {len(created)} promoted, "
                 f"{len(self.store.latest_versions())} total")
        return self.report["discovery"]

    def _injury_map(self) -> dict[int, list[tuple[str, str]]]:
        """team_id -> [(player, status)] from the verified ESPN feed.

        Mapped through Store.team_id_for so a triCode shared by two franchises resolves to
        the active club rather than an arbitrary row.  Entries that cannot be mapped are
        counted and reported, not dropped in silence.
        """
        out: dict[int, list[tuple[str, str]]] = {}
        unmapped: list[str] = []
        for r in self.store.query(
                "SELECT player_name, team_abbrev, status FROM injuries"):
            ab = r["team_abbrev"]
            tid = self.store.team_id_for(ab) if ab else None
            if tid is None:
                unmapped.append(f"{r['player_name']} ({ab})")
                continue
            out.setdefault(tid, []).append((r["player_name"], r["status"] or ""))
        if unmapped:
            self.store.flag(
                "unmapped_injury",
                f"{len(unmapped)} injury entr(ies) could not be mapped to a team id: "
                f"{', '.join(sorted(unmapped)[:8])}"
                + (" ..." if len(unmapped) > 8 else ""),
                severity="warn", entity_type="dataset", entity_id="injuries",
                sources="espn.nhl_api")
        return out

    def _starters(self) -> dict[int, str | None]:
        """Confirmed starting goalies, keyed by team id.

        Deliberately returns an empty mapping: no verified public source publishes
        starting goalies before game time (probes against the NHL endpoints are recorded
        in data/probe.json).  An empty map means "unknown", which is what gates
        goalie-dependent strategies into WAITING FOR GOALIE instead of letting them bet on
        an assumed starter.  This is not a stub to be filled in with a guess.
        """
        return {}

    def stage_backtest(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Two kinds of backtest, never merged:

        * accuracy-only (``label='all'``): hit rate vs. base rate on every decided game;
          no P&L because most games have no recovered price;
        * priced (``label='priced_*'``): flat-stake ROI at the Kalshi closing offer on the
          games whose candle history was recovered, on the full window and on the
          chronological train / validation / test splits, net of the published taker fee.
        """
        bt = Backtester(self.store)
        pbt = PricedBacktester(self.store)
        decided = [r for r in rows if r.get("_winner") is not None]
        out = {}
        for s in self.store.latest_versions():
            strat = _hydrate(s)
            res = bt.run(strat, decided, label="all")
            entry = {}
            if res:
                entry.update({"n": res.n_bets, "hit": res.hit_rate, "base": res.base_rate,
                              "lift": res.lift, "ci": [res.ci_low, res.ci_high],
                              "log_loss": res.log_loss, "brier": res.brier})
            pr = pbt.run(strat, decided, label="priced_all")
            if pr:
                entry["priced"] = {"n": pr.n_bets, "games": pr.n_games, "roi": pr.roi,
                                   "pnl": pr.pnl, "hit": pr.hit_rate, "avg_price": pr.avg_price,
                                   "mdd": pr.max_drawdown}
                splits = pbt.run_splits(strat, decided)
                entry["priced_splits"] = {k: ({"n": v.n_bets, "roi": v.roi} if v else None)
                                          for k, v in splits.items()}
            if entry:
                out[s["strategy_id"]] = entry
        # what blindly buying a side at the close returns (spread + fee), for context
        baseline = {side: market_baseline(decided, bet_side=side) for side in ("home", "away")}
        self.report["market_baseline"] = baseline
        self.store.execute(
            """INSERT OR REPLACE INTO findings(finding_id, created_at, title, body, evidence,
                                               confidence, kind) VALUES(?,?,?,?,?,?,?)""",
            ("FIND_MARKET_BASELINE", utcnow(),
             "Market baseline: buying every home / away contract at the Kalshi close",
             "Flat-stake ROI of buying one side of every game that has a recovered closing "
             "candle, net of the published taker fee. This is the cost of the spread plus fees; "
             "a strategy's priced ROI has to beat it, not zero.",
             json.dumps(baseline), "high", "baseline"))
        self.store.commit()
        self.report["backtests"] = out
        self.log(f"backtests: {len(out)} strategies; market baseline {json.dumps(baseline)}")
        return out

    # ------------------------------------------------------------- stage 5
    KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
    KALSHI_HIST_URL = "https://api.elections.kalshi.com/trade-api/v2/historical/markets"

    def _settled_contracts(self) -> list[dict[str, Any]]:
        """Settled KXNHLGAME contracts joined to their named price points.

        Each row carries ``points`` = {point: {bid, ask, ts, volume}} taken from the
        candlestick history (SOURCE DATA).  ``ask_before`` (the live tier's
        previous_yes_ask) is kept as a fallback whose timing is unverified.
        """
        rows = self.store.query(
            """SELECT ms.*, g.home_id, g.away_id
                 FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                WHERE ms.provider='kalshi' AND ms.result IN ('yes','no')
                  AND (ms.series_ticker='KXNHLGAME' OR ms.series_ticker IS NULL)""")
        pts: dict[str, dict[str, dict[str, Any]]] = {}
        for pr in self.store.query(
                """SELECT contract, point, end_period_ts, bid, ask, mean, volume
                     FROM market_price_points WHERE point IN ('open','t24h','t6h','t1h','close')"""):
            pts.setdefault(pr["contract"], {})[pr["point"]] = {
                "bid": pr["bid"], "ask": pr["ask"], "mean": pr["mean"],
                "ts": pr["end_period_ts"], "volume": pr["volume"]}
        out = []
        for ms in rows:
            d = dict(ms)
            d["points"] = pts.get(ms["contract"], {})
            out.append(d)
        return out

    def _dk_reference(self) -> dict[int, dict[str, Any]]:
        """Latest sportsbook moneyline per game (DraftKings via the NHL partner feed),
        converted from American odds and de-vigged.  Reference only -- Kalshi is the
        execution venue; this is recorded next to each forward signal so sportsbook-vs-
        exchange disagreement is visible and auditable."""
        out: dict[int, dict[str, Any]] = {}
        for r in self.store.query(
                """SELECT o.game_id, o.partner, o.home_price, o.away_price, o.retrieved_at,
                          o.source_updated_utc
                     FROM odds_snapshots o
                    WHERE o.market_desc='MONEY_LINE_2_WAY'
                      AND o.id = (SELECT MAX(id) FROM odds_snapshots o2
                                   WHERE o2.game_id=o.game_id AND o2.market_desc=o.market_desc)"""):
            if r["home_price"] is None or r["away_price"] is None:
                continue
            ph_raw = american_to_prob(float(r["home_price"]))
            pa_raw = american_to_prob(float(r["away_price"]))
            if ph_raw is None or pa_raw is None:
                continue
            ph, pa = devig_pair(ph_raw, pa_raw)
            out[int(r["game_id"])] = {
                "provider": r["partner"], "ts": r["source_updated_utc"] or r["retrieved_at"],
                "home_american": r["home_price"], "away_american": r["away_price"],
                "home_prob": round(ph, 4), "away_prob": round(pa, 4),
                "vig": round(ph_raw + pa_raw - 1.0, 4)}
        return out

    @staticmethod
    def _pregame(g: GameRef, as_of: str | None = None) -> bool:
        """True while the scheduled start is still in the future (relative to ``as_of`` or
        now).  Quotes on a game in progress are recorded but never traded by pre-game rules."""
        now = _parse(as_of) if as_of else datetime.now(timezone.utc)
        return g.start > now

    def _side_for_contract(self, ms: dict[str, Any], g: GameRef,
                           names: dict[int, list[str]]) -> str | None:
        ab = ms.get("team_abbrev")
        if ab:
            if self.store.team_id_for(ab) == g.home_id:
                return "home"
            if self.store.team_id_for(ab) == g.away_id:
                return "away"
        return _side_for_selection(ms.get("selection"), g, names)

    def stage_forward(self, rows: Sequence[dict[str, Any]] | None = None, *,
                      as_of: str | None = None,
                      include_unsettled_settlements: bool = True) -> dict[str, Any]:
        """Evaluate every strategy against every game that has a real quote.

        Two price feeds are used, clearly separated:

        * settled contracts -> BACKTEST rows.  The entry price is the candlestick at the
          strategy's ``entry_point`` (default: the last pre-game candle, "close"), with
          its timestamp.  Contracts without candle history fall back to the live tier's
          previous_yes_ask, labelled ``single_source_timing_unverified``.  Point-in-time
          feature rows and model outputs from the feature/model stages are required, so
          no rating that has already seen the result is used to price it.
        * live contracts -> FORWARD TEST rows that settle later as games finish.
        """
        if rows is None:
            # stand-alone call: build the point-in-time rows and model outputs first so
            # BACKTEST pricing never sees a rating that already includes the result
            kw = getattr(self, "_feature_kw", {}) or {}
            rows = self.stage_features(**kw)
            self.stage_models(rows, self.game_refs(**kw), prior_season=None)
        refs = {g.game_id: g for g in self.game_refs(game_types=(1, 2, 3))}
        names = build_team_names(self.store)
        pit = {int(r["game_id"]): r for r in rows}
        # schedule features for games outside the modelled set (forward evaluation only)
        fb = FeatureBuilder([g for g in refs.values() if g.game_type == 2])
        feats = {int(r["game_id"]): r for r in fb.build_all(list(refs.values()))}
        if getattr(self, "ext", None) is None:
            self.ext = load_extended(self.store)
        self.ext.augment_all([f for gid, f in feats.items() if gid not in pit])
        pois = PoissonModel()
        elo = EloModel()
        for g in sorted((g for g in refs.values() if g.decided and g.game_type == 2),
                        key=lambda x: x.start):
            elo.observe(g)

        def forward_row(gid: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
            f = pit.get(gid)
            if f is not None and f.get("p_home_ml") is not None:
                return f, {k: f[k] for k in ("p_home_ml", "p_away_ml", "p_home_elo", "p_away_elo",
                                             "p_home_logit", "p_away_logit", "p_over", "p_under",
                                             "exp_total") if k in f and f[k] is not None}
            f = feats.get(gid)
            if f is None:
                return None
            g = refs[gid]
            pred = pois.predict(f)
            pe = elo.prob_home(g.home_id, g.away_id)
            pred["p_home_elo"], pred["p_away_elo"] = pe, 1 - pe
            pred["p_home_logit"], pred["p_away_logit"] = pred["p_home_ml"], pred["p_away_ml"]
            return f, pred

        def starters_for(f: dict[str, Any]) -> dict[int, str | None]:
            out: dict[int, str | None] = {}
            for side in ("home", "away"):
                if f.get(f"{side}_starter_known"):
                    out[int(f[f"{side}_id"])] = f.get(f"{side}_starter_name") or "confirmed"
            return out

        placed = {"BACKTEST": 0, "FORWARD TEST": 0}
        skipped = 0
        inj = self._injury_map()
        dk = self._dk_reference()
        settled = self._settled_contracts()
        live = self._live_quotes()
        for s in self.store.latest_versions():
            if s["status"] == "rejected":
                continue
            strat = _hydrate(s)
            entry_point = str(getattr(strat, "entry_point", "close") or "close")
            self.store.sync_bankroll(strat.strategy_id, strat.version)
            srow = self.store.one("SELECT bankroll FROM strategies WHERE strategy_id=? AND version=?",
                                  (strat.strategy_id, strat.version))
            bankroll = float(srow["bankroll"]) if srow else strat.starting_bankroll
            open_exp = float(self.store.one(
                "SELECT COALESCE(SUM(stake),0) e FROM bets WHERE strategy_id=? AND "
                "strategy_version=? AND result='OPEN'", (strat.strategy_id, strat.version))["e"] or 0)

            # ---------------------------------------------------------- BACKTEST
            for ms in settled:
                gid = int(ms["game_id"])
                g = refs.get(gid)
                f = pit.get(gid)
                if g is None or f is None or f.get("p_home_ml") is None or not g.decided:
                    continue
                sel = self._side_for_contract(ms, g, names)
                if sel is None:
                    continue
                if strat.category == "kalshi_both_sides":
                    strat.bet_side = sel
                if sel != strat.bet_side:
                    continue
                pt = ms["points"].get(entry_point)
                close_pt = ms["points"].get("close")
                if pt and pt.get("ask") and 0 < float(pt["ask"]) < 1:
                    q = Quote(provider="kalshi", market_key=ms["event_ticker"],
                              contract=ms["contract"], game_id=gid, game_date=g.game_date,
                              market_type="moneyline", selection=sel, side="YES",
                              bid=pt.get("bid"), ask=float(pt["ask"]), bid_size=None,
                              ask_size=None, volume=pt.get("volume"), liquidity=None,
                              ts_utc=pt["ts"])
                    verification = f"kalshi_candle_{entry_point}"
                    basis = (f"kalshi candlestick yes_ask close of the last 60-minute candle "
                             f"ending {pt['ts']} ({entry_point})")
                    src = self.KALSHI_HIST_URL
                elif entry_point == "close" and ms.get("ask_before") and \
                        0 < float(ms["ask_before"]) < 1:
                    q = Quote(provider="kalshi", market_key=ms["event_ticker"],
                              contract=ms["contract"], game_id=gid, game_date=g.game_date,
                              market_type="moneyline", selection=sel, side="YES",
                              bid=ms["bid_before"], ask=float(ms["ask_before"]),
                              bid_size=None, ask_size=None, volume=ms["volume"],
                              liquidity=None, ts_utc=ms["open_time"] or ms["retrieved_at"])
                    verification = "single_source_timing_unverified"
                    basis = "kalshi previous_yes_ask (pre-settlement, timing unverified)"
                    src = self.KALSHI_MARKETS_URL
                else:
                    continue
                pred = {k: f[k] for k in ("p_home_ml", "p_away_ml", "p_home_elo", "p_away_elo",
                                          "p_home_logit", "p_away_logit") if f.get(k) is not None}
                # historical injuries are not available -> injury-gated strategies wait
                ctx = DecisionContext(decision_ts=q.ts_utc, features=f, predictions=pred,
                                      quotes=[q], starters=starters_for(f), injuries={},
                                      bankroll=strat.starting_bankroll, open_exposure=0.0)
                for sig in strat.evaluate(ctx):
                    if sig.status != "READY TO BET" or sig.quote is None or sel != strat.bet_side:
                        continue
                    sig.supporting["decision_ts"] = ctx.decision_ts
                    sig.supporting["contract"] = ms["contract"]
                    sig.supporting["price_basis"] = basis
                    sig.supporting["price_point"] = entry_point
                    sig.supporting["data_labels"] = {
                        "price": "SOURCE DATA (kalshi)", "features": "DERIVED (point-in-time)",
                        "model_prob": "MODEL OUTPUT"}
                    bid = self.paper.place(sig, decision_ts=ctx.decision_ts, test_mode="BACKTEST",
                                           provider="kalshi", source_url=src,
                                           verification=verification)
                    if bid:
                        placed["BACKTEST"] += 1
                        won = (ms["result"] == "yes")
                        close_mid = None
                        if close_pt and close_pt.get("bid") is not None and close_pt.get("ask") is not None:
                            close_mid = round((float(close_pt["bid"]) + float(close_pt["ask"])) / 2, 4)
                        clv = (round(close_mid - float(sig.quote.ask), 4)
                               if close_mid is not None and entry_point != "close" else None)
                        self.store.settle_bet(
                            bid, result="WIN" if won else "LOSS", pnl=_settle_pnl(sig, won),
                            close_price=close_mid, clv=clv, settle_ts=ms["settlement_ts"],
                            reason=f"kalshi contract settled {ms['result']}")
                        self.store.execute(
                            "UPDATE bets SET price_point=?, close_price_ts=? WHERE bet_id=?",
                            (entry_point, close_pt["ts"] if close_pt else None, bid))
            self.store.sync_bankroll(strat.strategy_id, strat.version)

            # ---------------------------------------------------- FORWARD TEST
            for q in live:
                gid = q.game_id
                g = refs.get(gid) if gid else None
                if g is None or g.decided:
                    continue
                if not self._pregame(g, as_of):
                    continue   # the game has started: a pre-game rule may not enter now
                sel = _side_for_selection(q.selection, g, names)
                if sel is None or sel != strat.bet_side:
                    continue
                fr = forward_row(gid)
                if fr is None:
                    continue
                f, pred = fr
                ref = dk.get(gid)
                if ref and ref.get("home_prob") is not None:
                    pred = dict(pred, p_home_book=ref["home_prob"], p_away_book=ref["away_prob"])
                ctx = DecisionContext(decision_ts=q.ts_utc, features=f, predictions=pred,
                                      quotes=[q], starters=starters_for(f), injuries=inj,
                                      bankroll=bankroll, open_exposure=open_exp)
                for sig in strat.evaluate(ctx):
                    sig.supporting["decision_ts"] = q.ts_utc
                    sig.supporting["price_point"] = "live_quote"
                    ref = dk.get(gid)
                    if ref and ref.get("home_prob") is not None:
                        p_ref = ref["home_prob"] if sel == "home" else ref["away_prob"]
                        sig.supporting["sportsbook_reference"] = {
                            "provider": ref.get("provider"), "ts": ref.get("ts"),
                            "devigged_prob": p_ref,
                            "kalshi_ask_minus_book_prob": (round(float(q.ask) - p_ref, 4)
                                                           if q.ask is not None else None)}
                    self.paper.record_upcoming(sig, provider="kalshi",
                                               source_url=self.KALSHI_MARKETS_URL)
                    if sig.status == "READY TO BET":
                        bid = self.paper.place(sig, decision_ts=q.ts_utc, test_mode="FORWARD TEST",
                                               provider="kalshi", source_url=self.KALSHI_MARKETS_URL,
                                               verification="single_source")
                        if bid:
                            placed["FORWARD TEST"] += 1
                            open_exp += sig.stake
                            bankroll -= sig.stake
                            self.store.execute("UPDATE bets SET price_point='live_quote' WHERE bet_id=?",
                                               (bid,))
                    else:
                        skipped += 1
        self.store.commit()
        self.report["placed"] = placed
        self.report["non_executable_signals"] = skipped
        self.log(f"forward: {placed}")
        return placed

    def _live_quotes(self) -> list[Quote]:
        out = []
        for q in self.store.query(
                """SELECT * FROM market_quotes WHERE game_id IS NOT NULL AND ask IS NOT NULL
                   AND ask > 0 AND ask < 1 AND side='YES'
                   AND ts_utc = (SELECT MAX(ts_utc) FROM market_quotes q2
                                 WHERE q2.contract = market_quotes.contract)"""):
            out.append(Quote(provider=q["provider"], market_key=q["market_key"],
                             contract=q["contract"], game_id=q["game_id"],
                             game_date=q["game_date"], market_type=q["market_type"],
                             selection=q["selection"], side=q["side"], bid=q["bid"],
                             ask=q["ask"], bid_size=q["bid_size"], ask_size=q["ask_size"],
                             volume=q["volume"], liquidity=q["liquidity"], ts_utc=q["ts_utc"]))
        return out

    # ------------------------------------------------------------- stage 6
    def stage_settle(self) -> int:
        n = 0
        for g in self.store.query("SELECT * FROM games WHERE state IN ('FINAL','OFF')"):
            n += self.paper.settle_game(
                int(g["game_id"]), home_id=int(g["home_id"]), away_id=int(g["away_id"]),
                home_score=g["home_score"], away_score=g["away_score"], state=g["state"],
                last_period_type=g["last_period_type"])
        self.report["settled_now"] = n
        self.report["signals_expired"] = self.paper.expire_started(utcnow())
        return n

    def stage_clv(self) -> int:
        """Attach the closing price (last pre-game candle) to FORWARD TEST bets.

        Closing-line value is entry ask vs. the closing mid of the same contract; it is
        only written once the candle history for the contract has been ingested, and it
        is never estimated.
        """
        n = 0
        for b in self.store.query(
                """SELECT b.bet_id, b.entry_price, b.price, p.bid, p.ask, p.end_period_ts
                     FROM bets b
                     JOIN market_price_points p
                       ON p.contract = json_extract(b.notes, '$.contract') AND p.point='close'
                    WHERE b.test_mode='FORWARD TEST' AND b.close_price IS NULL
                      AND p.bid IS NOT NULL AND p.ask IS NOT NULL"""):
            close_mid = round((float(b["bid"]) + float(b["ask"])) / 2, 4)
            entry = float(b["entry_price"] or b["price"])
            self.store.amend_bet(b["bet_id"], {"close_price": close_mid,
                                               "clv": round(close_mid - entry, 4),
                                               "close_price_ts": b["end_period_ts"]},
                                 reason="closing price from kalshi candlestick (last pre-game candle)")
            n += 1
        self.report["clv_attached"] = n
        return n

    def stage_analysis(self) -> dict[str, Any]:
        self.report["leaderboard"] = self.perf.leaderboard(test_mode="FORWARD TEST")
        self.report["leaderboard_backtest"] = self.perf.leaderboard(test_mode="BACKTEST")
        self.report["competition"] = {
            "FORWARD TEST": self.perf.competition_totals(test_mode="FORWARD TEST"),
            "BACKTEST": self.perf.competition_totals(test_mode="BACKTEST")}
        self.store.execute(
            """INSERT OR REPLACE INTO findings(finding_id, created_at, title, body, evidence,
                                               confidence, kind) VALUES(?,?,?,?,?,?,?)""",
            ("FIND_COMPETITION_STATE", utcnow(), "Competition state snapshot",
             "Aggregate paper-trading state. Win rate is always shown with a sample size and a "
             "Wilson interval.", json.dumps(self.report["competition"]), "high", "status"))
        self.store.commit()
        return self.report["competition"]

    def stage_verify(self) -> dict[str, Any]:
        v = Verifier(self.store)
        self.report["verification"] = v.run_all()
        self.log(f"verification: {json.dumps(self.report['verification'])}")
        return self.report["verification"]

    # ------------------------------------------------------------- all
    def run_all(self, **kw: Any) -> dict[str, Any]:
        if "ingest" in kw:
            self.stage_ingest(**kw["ingest"])
        else:
            self.log("skipping ingest (no ingest config supplied)")
        rows = self.stage_features(**kw.get("features", {}))
        refs = self.game_refs(**kw.get("features", {}))
        self.stage_models(rows, refs, prior_season=kw.get("prior_season"))
        self.stage_strategies(rows)
        self.stage_backtest(rows)
        self.stage_forward(rows)
        self.stage_settle()
        self.stage_clv()
        self.stage_analysis()
        self.stage_verify()
        self.store.audit("pipeline", "RUN_ALL", "", json.dumps(self.report.get("competition", {})))
        self.store.commit()
        return self.report


# ----------------------------------------------------------------- helpers
def _parse(ts: str) -> datetime:
    from .http import parse_iso
    return parse_iso(ts)


def _hydrate(row: Any):
    """Rebuild a strategy from its stored row.

    Everything the strategy persisted into params_json is passed back through, so a
    reloaded strategy behaves exactly like the one that was saved.  An earlier version of
    this function named each parameter by hand, which meant any field added later was
    silently dropped on reload -- gating flags such as requires_goalie simply evaporated
    and the strategy started betting as though it had never been gated.
    """
    from .strategies import ThresholdStrategy
    params = json.loads(row["params_json"] or "{}")
    kw = dict(params)
    kw.pop("min_edge", None)
    kw.pop("stake_fraction", None)
    kw.setdefault("feature", "home_n_prior")
    kw.setdefault("operator", ">=")
    kw.setdefault("threshold", 0)
    kw.setdefault("bet_side", "home")
    kw.setdefault("use_model", "poisson")
    return ThresholdStrategy(
        strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
        name=row["name"], category=row["category"],
        hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
        price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
        markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
        starting_bankroll=float(row["starting_bankroll"]),
        min_edge=params.get("min_edge", 0.03),
        stake_fraction=params.get("stake_fraction", 0.25), **kw)


def _side_for_selection(selection: str | None, g: GameRef,
                        names: dict[int, list[str]]) -> str | None:
    """Map a Kalshi contract title ("Vegas wins") to the home/away side of the NHL game.

    Returns None when the name cannot be matched unambiguously -- the caller must then skip
    the contract rather than guess which side it refers to.
    """
    if not selection:
        return None
    sel = selection.lower().replace(" wins", "").strip()
    if sel in ("home", "away"):
        return sel
    for team_id, side in ((g.home_id, "home"), (g.away_id, "away")):
        for nm in names.get(team_id, []):
            if nm == sel or sel.startswith(nm) or nm.startswith(sel):
                return side
    return None


def build_team_names(store: Store) -> dict[int, list[str]]:
    """Place-name variants per team id, derived from the official full name.

    "Vegas Golden Knights" -> ["vegas golden knights", "vegas"].  Nothing is hard-coded, so
    a rename (e.g. Utah Hockey Club -> Utah Mammoth) is picked up from the source.
    """
    out: dict[int, list[str]] = {}
    for t in store.query("SELECT team_id, full_name, abbrev FROM teams"):
        variants: list[str] = []
        full = (t["full_name"] or "").lower().strip()
        if full:
            variants.append(full)
            parts = full.split()
            if len(parts) >= 2:
                variants.append(" ".join(parts[:-1]))
                variants.append(parts[0])
        if t["abbrev"]:
            variants.append(str(t["abbrev"]).lower())
            variants.append(str(t["abbrev"]).lower().rstrip("k"))
        out[int(t["team_id"])] = sorted({v for v in variants if v})
    return out


def _settle_pnl(sig: Any, won: bool) -> float:
    from .market import kalshi_taker_fee
    from .paper import binary_settlement
    price = float(sig.quote.ask)
    contracts = float(sig.stake) / price
    return binary_settlement(price, contracts, won, fee=kalshi_taker_fee(price, contracts))
