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
from .market import (american_to_prob, devig_pair, margin_side_and_strike,
                     team_total_side_and_strike, period_total_side_and_strike,
                     total_side_and_strike, total_side_from_text,
                     normalize_market_type)
from .models import (EloModel, HomeIceOnly, LogisticRest, OtCalibration, PoissonModel,
                     brier, log_loss)
from .paper import PaperEngine
from .store import Store, utcnow
from .sources.kalshi import KALSHI_TEAM_ALIASES, suffix_team_code
from .strategies import (PLAYOFFS, REGULAR_SEASON, DecisionContext, Quote,
                          build_seed_strategies)
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
        # fitted in stage_models; an empty calibration is scale=1.0 / sufficient=False, which
        # makes an overtime rule refuse to trade rather than price off an unfitted ratio
        self.ot_calibration = OtCalibration()
        self._ot_cal_rows: list[dict[str, Any]] = []

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
                     totals_budget: int = 400, spread_budget: int = 400,
                     overtime_budget: int = 250, polymarket_limit: int = 100,
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
        # totals ("Over k.5 goals"): same exchange, its own series, its own budget so a
        # long totals walk can never starve the moneyline walk
        self._safe(out, "kalshi_totals_live", self.ing.kalshi_nhl, series="KXNHLTOTAL")
        self._safe(out, "kalshi_history", self.ing.kalshi_history, series="KXNHLGAME",
                   max_calls=kalshi_budget)
        self._safe(out, "kalshi_totals_history", self.ing.kalshi_history, series="KXNHLTOTAL",
                   max_calls=totals_budget)
        # puck line ("<team> wins by over k.5 goals") and overtime ("will there be
        # overtime"): each its own series, its own live read and its own history budget, so
        # a long walk in one market can never starve another.  Both were verified on
        # 2026-09-21 -- the spread series on the live tier (quoting the 2026-09-24 slate)
        # and on the historical tier back to the 2026 Stanley Cup Final, the overtime series
        # on the historical tier only (about 100 settled playoff events, zero open contracts).
        self._safe(out, "kalshi_spread_live", self.ing.kalshi_nhl, series="KXNHLSPREAD")
        self._safe(out, "kalshi_overtime_live", self.ing.kalshi_nhl, series="KXNHLOVERTIME")
        self._safe(out, "kalshi_spread_history", self.ing.kalshi_history, series="KXNHLSPREAD",
                   max_calls=spread_budget)
        self._safe(out, "kalshi_overtime_history", self.ing.kalshi_history,
                   series="KXNHLOVERTIME", max_calls=overtime_budget)
        # series the exchange lists but has no contracts for yet: polled every run so the
        # day one is listed its prices are captured from that day, and so "nothing listed"
        # stays a recorded verified negative instead of a silent gap
        self._safe(out, "kalshi_series_watch", self.ing.kalshi_series_watch)
        # legacy settled sweep kept for the counters the old reports expose
        out["kalshi_settled"] = self.ing.kalshi_settled(max_pages=ingest_settled_pages)
        self._safe(out, "kalshi_live_points", self.ing.kalshi_live_price_points,
                   max_calls=live_points_budget)
        # official period-by-period scoring from a second NHL endpoint: what a period market
        # would settle on, and an independent reading of the final score and lastPeriodType
        # to cross-check the scoreboard feed against
        self._safe(out, "period_goals", self.ing.nhl_period_goals, scoreboard_days)
        # a second prediction market, as reference only -- Kalshi remains the execution
        # venue and no Polymarket price funds or settles a wager here
        self._safe(out, "polymarket_nhl", self.ing.polymarket_nhl, limit=polymarket_limit)
        # sportsbook reference odds (DraftKings via NHL partner feed) and EDGE tracking
        self._safe(out, "partner_odds", self.ing.nhl_partner_odds)
        if current:
            n_edge = self._safe(out, "edge_snapshots", self.ing.nhl_edge_snapshots, current)
            if not n_edge and len(seasons) > 1:
                # before the first game of a season the EDGE feed has nothing for it;
                # keep snapshotting the most recent completed season instead
                prev = sorted(seasons)[-2]
                self._safe(out, f"edge_snapshots_{prev}", self.ing.nhl_edge_snapshots, prev)
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
        # stamp every pre-existing wager with the season phase of its game before anything
        # reads the ledger per phase.  DERIVED from games.game_type (SOURCE DATA); audited;
        # no price, stake, result or P&L is touched.
        filled = self.store.backfill_bet_game_type()
        if filled:
            self.report["bet_game_type_backfilled"] = filled
            self.log(f"backfilled game_type on {filled} bet row(s) from games.game_type "
                     "(derived; audited)")
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

        # fit the tiny logistic model on the training window only, apply out of sample.
        # Split the DECIDED games chronologically: with next season's schedule already
        # loaded, splitting all rows would put nothing but unplayed games in the test window.
        bt = Backtester(self.store)
        parts = bt.split([r for r in rows if r.get("_winner") is not None])
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

        # ---- overtime calibration -------------------------------------------------
        # The Poisson tie mass is a model output that has never been checked against what
        # actually happened, so it is not used raw to price a KXNHLOVERTIME contract.  The
        # ratio observed_past_regulation / mean_modelled_tie is fitted on the earlier
        # chronological window (the same 60% split the model comparison uses) and applied to
        # later games only.  Preseason games are excluded: their lineups and effort differ
        # materially and their observed past-regulation rate in this ledger is 17/104 and
        # 22/104 against 271/1312 and 326/1312 in the regular season.
        cal_rows = self._ot_calibration_rows(rows, refs)
        self.ot_calibration = self.ot_calibration_asof(
            cal_rows, train_split=bt.split(cal_rows)["train"] if cal_rows else None)
        for r in rows:
            if r.get("p_overtime") is not None:
                r["p_ot_cal"] = self.ot_calibration.apply(float(r["p_overtime"]))
        self._ot_cal_rows = cal_rows
        ev = self.ot_calibration.as_evidence()
        ev["scope"] = ("regular season + playoff games decided before the fitted window's end; "
                       "preseason excluded")
        ev["label"] = ("NHL lastPeriodType in (OT, SO) -- SOURCE DATA, not a model output")
        ev["settlement_note"] = (
            "Kalshi's own settled KXNHLOVERTIME history verified 2026-09-21 is playoff-only "
            "(~100 events, 2026-04-28..2026-06-14), where no shootout exists, so whether a "
            "shootout counts as 'going to overtime' is UNVERIFIED. Wagers on such a game are "
            "left OPEN and settled from the exchange's own result rather than from a guess.")
        self.report["ot_calibration"] = ev
        self.log(f"ot calibration: scale={ev['scale']} on n={ev['n']} games "
                 f"(observed {ev['observed_rate']} vs model mean {ev['model_mean_tie_prob']}) "
                 f"sufficient={ev['sufficient']}")
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
        # the strikes the exchange actually listed per game, so a totals accuracy run is
        # scored at real lines rather than at strikes we chose ourselves
        strikes_by_game: dict[int, set[float]] = {}
        for r in self.store.query(
                """SELECT game_id, floor_strike FROM market_settlements
                    WHERE series_ticker='KXNHLTOTAL' AND floor_strike IS NOT NULL
                      AND game_id IS NOT NULL"""):
            strikes_by_game.setdefault(int(r["game_id"]), set()).add(float(r["floor_strike"]))
        strikes_by_game = {k: sorted(v) for k, v in strikes_by_game.items()}
        out = {}
        for s in self.store.latest_versions():
            strat = _hydrate(s)
            if getattr(strat, "market", "moneyline") == "total":
                tr = bt.run_totals_accuracy(strat, decided,
                                            strikes_by_game=strikes_by_game)
                if tr:
                    out[s["strategy_id"]] = {
                        "market": "total", "n": tr.n_bets, "hit": tr.hit_rate,
                        "base": tr.base_rate, "lift": tr.lift, "ci": [tr.ci_low, tr.ci_high],
                        "log_loss": tr.log_loss, "brier": tr.brier,
                        "games": getattr(tr, "n_games", None),
                        "note": "accuracy only; priced totals bets are written by the "
                                "forward/backtest stage from KXNHLTOTAL candles"}
                continue
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
        pts = self._candle_points()
        out = []
        for ms in rows:
            d = dict(ms)
            d["points"] = pts.get(ms["contract"], {})
            out.append(d)
        return out

    def _settled_totals(self) -> list[dict[str, Any]]:
        """Settled KXNHLTOTAL contracts ("Over k.5 goals") with their named price points.

        Each row gains ``direction``/``strike`` taken from the exchange's own
        ``strike_type``/``floor_strike`` (falling back to the sub-title text of the same
        payload when the columns predate the strike_type addition), and ``points`` from the
        candlestick history.  The candle feed publishes the YES offer only, so a settled
        totals row can price an **Over** entry; the Under side has no historical offer and
        is forward-test only (see :class:`TotalsStrategy`).
        """
        rows = self.store.query(
            """SELECT ms.*, g.home_id, g.away_id
                 FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                WHERE ms.provider='kalshi' AND ms.result IN ('yes','no')
                  AND ms.series_ticker='KXNHLTOTAL' AND ms.floor_strike IS NOT NULL""")
        pts: dict[str, dict[str, dict[str, Any]]] = {}
        for pr in self.store.query(
                """SELECT contract, point, end_period_ts, bid, ask, mean, volume
                     FROM market_price_points WHERE point IN ('open','t24h','t6h','t1h','close')"""):
            pts.setdefault(pr["contract"], {})[pr["point"]] = {
                "bid": pr["bid"], "ask": pr["ask"], "mean": pr["mean"],
                "ts": pr["end_period_ts"], "volume": pr["volume"]}
        out = []
        for ms in rows:
            line = total_side_and_strike(ms["strike_type"], ms["floor_strike"])
            basis = "strike_type+floor_strike"
            if line is None:
                line = total_side_from_text(ms["selection"])
                basis = "yes_sub_title text"
            if line is None:
                self.store.flag("unmapped_totals_line",
                                f"totals contract {ms['contract']} has no readable line "
                                f"(strike_type={ms['strike_type']}, floor_strike="
                                f"{ms['floor_strike']}, selection={ms['selection']!r}); skipped "
                                f"rather than assumed", severity="warn",
                                entity_type="market", entity_id=ms["contract"])
                continue
            d = dict(ms)
            d["direction"], d["strike"] = line[0], line[1]
            d["line_basis"] = basis
            d["points"] = pts.get(ms["contract"], {})
            out.append(d)
        return out

    def _totals_coverage(self, totals: Sequence[dict[str, Any]], refs: dict[int, GameRef],
                         pit: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """Record how much totals price history is actually usable, and why the rest is not.

        A priced totals BACKTEST needs two things for the same game: a candlestick offer from
        Kalshi and a point-in-time feature row from the NHL stats feed.  The two walks are
        budgeted separately and move in opposite directions -- the candle walk starts from the
        most recent contracts, the stats walk works forward through a season -- so they can
        hold a lot of data each and still not overlap.  On the 2026-09-21 CI ledger they did
        not: 577 settled contracts had a pre-puck-drop offer across 73 games (64 of them
        2025-26 playoff games, 9 pre-season), while ``team_game_stats`` covered only the
        2024-25 season, so **no** game had both.  That is a coverage gap, not a result, and
        it is written to the ledger as one rather than being left as an empty backtest that
        looks like "the totals rules found nothing".
        """
        with_offer = [t for t in totals if t.get("points", {}).get(self.entry_point_default)]
        games_offer = {int(t["game_id"]) for t in with_offer}
        scored = {g for g in games_offer
                  if pit.get(g) is not None and pit[g].get("lam_home") is not None}
        decided = {g for g in games_offer
                   if refs.get(g) is not None and refs[g].decided}
        gap = {"settled_totals_contracts": len(totals),
               "contracts_with_a_pregame_offer": len(with_offer),
               "games_with_a_pregame_offer": len(games_offer),
               "of_those_games_decided_in_the_ledger": len(decided),
               "of_those_games_with_point_in_time_features": len(scored),
               "backtestable_games": len(scored & decided)}
        if gap["backtestable_games"] == 0 and gap["contracts_with_a_pregame_offer"]:
            gap["reason"] = (
                "The Kalshi totals candle walk and the NHL season-stats walk have not reached "
                "the same games yet: every contract with a pre-puck-drop offer belongs to a "
                "game with no point-in-time feature row, so no totals entry can be priced "
                "honestly. Totals are therefore accuracy-only (BACKTEST) and forward-test at "
                "live offers until the two walks overlap. Nothing was inferred to fill the gap.")
            self.store.flag("totals_price_coverage",
                            "Totals price history and feature history do not overlap yet, so no "
                            "totals entry can be priced honestly. " + json.dumps(gap, indent=1),
                            severity="warn", entity_type="market", entity_id="KXNHLTOTAL",
                            sources="kalshi.candles,nhl.stats_rest")
        self.report["totals_price_coverage"] = gap
        self.log(f"totals price coverage: {gap['contracts_with_a_pregame_offer']} contracts with "
                 f"a pre-game offer across {gap['games_with_a_pregame_offer']} games; "
                 f"{gap['backtestable_games']} backtestable")
        return gap

    entry_point_default = "close"

    def _backtest_totals(self, strat: Any, *, refs: dict[int, GameRef],
                         pit: dict[int, dict[str, Any]], totals: Sequence[dict[str, Any]],
                         entry_point: str) -> int:
        """Price a totals rule against settled KXNHLTOTAL candle offers (BACKTEST mode).

        Entry price is the ``yes_ask`` of the last candle ending at or before puck drop --
        a real, timestamped offer -- and settlement is the contract's own Kalshi result.
        Under rules are skipped: the candle feed publishes no NO-side offer, and
        ``no_ask = 1 - yes_bid`` is contradicted by this ledger's own quotes, so there is
        no honest historical Under price to buy at.

        Fixed 2026-09-21: previously looped per-contract, placing one bet per strike
        rung (e.g. 5.5 and 7.5 for the same game), which violated the one-bet-per-game
        invariant and triggered 64 duplicate_bet irregularities. Now groups by game and
        hands the whole ladder to the strategy once, so _pick_strike chooses a single
        rung (nearest target, lower on ties) exactly as forward-test does.
        """
        if getattr(strat, "market", "moneyline") != "total":
            return 0
        if getattr(strat, "direction", "over") != "over":
            return 0
        # group by game, as puck-line does
        ladders: dict[int, list[dict[str, Any]]] = {}
        for ms in totals:
            gid = int(ms["game_id"])
            ladders.setdefault(gid, []).append(ms)
        for rungs in ladders.values():
            rungs.sort(key=lambda z: float(z["strike"]) if z.get("strike") is not None else 0.0)
        placed = 0
        for gid, rungs in sorted(ladders.items()):
            g = refs.get(gid)
            f = pit.get(gid)
            if g is None or f is None or f.get("lam_home") is None or not g.decided:
                continue
            if not self._in_scope(strat, g):
                continue
            quotes: list[Quote] = []
            settle_by: dict[str, dict[str, Any]] = {}
            for ms in rungs:
                pt = ms["points"].get(entry_point)
                if not (pt and pt.get("ask") and 0 < float(pt["ask"]) < 1):
                    continue
                q = Quote(provider="kalshi", market_key=ms["event_ticker"], contract=ms["contract"],
                          game_id=gid, game_date=g.game_date, market_type="total",
                          selection=ms["direction"], side="YES", bid=pt.get("bid"),
                          ask=float(pt["ask"]), bid_size=None, ask_size=None, volume=pt.get("volume"),
                          liquidity=None, ts_utc=pt["ts"],
                          label=ms["selection"], strike=float(ms["strike"]))
                quotes.append(q)
                settle_by[q.contract] = ms
            if not quotes:
                continue
            # decision_ts is the earliest of the ladder's points (all should be same entry_point ts, but use first)
            ts = quotes[0].ts_utc
            pred = {k: f[k] for k in ("p_home_ml", "p_away_ml", "lam_home", "lam_away",
                                      "exp_total") if f.get(k) is not None}
            ctx = DecisionContext(decision_ts=ts, features=f, predictions=pred, quotes=quotes,
                                  starters={}, injuries={}, bankroll=strat.starting_bankroll,
                                  open_exposure=0.0)
            for sig in strat.evaluate(ctx):
                if sig.status != "READY TO BET" or sig.quote is None:
                    continue
                ms = settle_by.get(sig.quote.contract)
                if ms is None:
                    continue
                pt = ms["points"].get(entry_point)
                close_pt = ms["points"].get("close")
                sig.supporting["decision_ts"] = ctx.decision_ts
                sig.supporting["contract"] = ms["contract"]
                sig.supporting["line_basis"] = ms["line_basis"]
                sig.supporting["price_basis"] = (
                    f"kalshi candlestick yes_ask close of the last 60-minute candle ending "
                    f"{pt['ts']} ({entry_point})")
                sig.supporting["price_point"] = entry_point
                bid = self.paper.place(sig, decision_ts=ctx.decision_ts, test_mode="BACKTEST",
                                       provider="kalshi", source_url=self.KALSHI_HIST_URL,
                                       verification=f"kalshi_candle_{entry_point}",
                                       depth_volume=(float(pt["volume"])
                                                     if pt.get("volume") is not None else None))
                if not bid:
                    continue
                placed += 1
                won = (ms["result"] == "yes")
                close_mid = None
                if close_pt and close_pt.get("bid") is not None and close_pt.get("ask") is not None:
                    close_mid = round((float(close_pt["bid"]) + float(close_pt["ask"])) / 2, 4)
                clv = (round(close_mid - float(sig.quote.ask), 4)
                       if close_mid is not None and entry_point != "close" else None)
                self.store.settle_bet(
                    bid, result="WIN" if won else "LOSS", pnl=_settle_pnl(self.store, bid, won),
                    close_price=close_mid, clv=clv, settle_ts=ms["settlement_ts"],
                    reason=f"kalshi totals contract settled {ms['result']} "
                           f"(strike {float(ms['strike']):g})")
                self.store.execute(
                    "UPDATE bets SET price_point=?, close_price_ts=? WHERE bet_id=?",
                    (entry_point, close_pt["ts"] if close_pt else None, bid))
        return placed

    #: Which NHL ``lastPeriodType`` values mean the game went past regulation.  SOURCE DATA:
    #: the league's own field, not an inference.  A shootout can only be reached through
    #: overtime, so it is included in the *label* used to fit the calibration; whether Kalshi
    #: settles a shootout game as YES is a separate question and is handled at settlement
    #: (see :attr:`PaperEngine.OT_VERIFIED_YES`).
    PAST_REGULATION_PERIODS = ("OT", "SO")

    def _ot_calibration_rows(self, rows: Sequence[dict[str, Any]],
                             refs: Sequence[GameRef] | dict[int, GameRef]
                             ) -> list[dict[str, Any]]:
        """Chronologically sorted ``(model tie probability, official outcome)`` pairs.

        Each returned dict is the point-in-time feature row itself, plus ``_start`` (the
        scheduled UTC start, so a caller can cut the window without another lookup) and
        ``_went_past_regulation`` (0/1 from the NHL's ``lastPeriodType``).  Rows missing
        either the model output or the official outcome are dropped and counted, never
        coerced into a label.
        """
        by_ref = ({int(g.game_id): g for g in refs.values()}
                  if isinstance(refs, dict) else {int(g.game_id): g for g in refs})
        out: list[dict[str, Any]] = []
        dropped = 0
        for r in rows:
            g = by_ref.get(int(r.get("game_id") or 0))
            if g is None or not g.decided or g.game_type not in (2, 3):
                continue
            if r.get("p_overtime") is None or r.get("p_tie_reg") is None:
                dropped += 1
                continue
            lpt = (g.last_period_type or "").upper()
            if lpt not in ("OT", "SO", "REG"):
                dropped += 1
                continue        # an outcome this project cannot read is not a label
            r["_went_past_regulation"] = int(lpt in self.PAST_REGULATION_PERIODS)
            r["_start"] = g.start.isoformat()
            out.append(r)
        out.sort(key=lambda r: str(r.get("_start")))
        if dropped:
            self.log(f"ot calibration: {dropped} decided game(s) dropped for a missing model "
                     f"tie probability or an unreadable lastPeriodType")
        return out

    def ot_calibration_asof(self, cal_rows: Sequence[dict[str, Any]], *,
                            cutoff: str | None = None,
                            train_split: Sequence[dict[str, Any]] | None = None
                            ) -> OtCalibration:
        """The calibration fitted on games that started before ``cutoff``.

        ``train_split`` may be given instead, to reuse the same chronological split the model
        comparison uses.  Either way the fitted window is strictly earlier than the games the
        resulting ratio is applied to, so a wager is never priced with a number that has seen
        its own outcome.
        """
        if train_split is not None:
            use = list(train_split)
            window = (f"the earliest {len(use)} of {len(cal_rows)} decided regular/playoff games "
                      f"in chronological order" if use else "no games")
        elif cutoff:
            use = [r for r in cal_rows if str(r.get("_start")) < cutoff]
            window = f"decided games starting before {cutoff}"
        else:
            use = list(cal_rows)
            window = f"all {len(use)} decided regular/playoff games in the ledger"
        return OtCalibration().fit(use, window=window)

    def _candle_points(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Named price points per contract from Kalshi's candlestick history.

        One read shared by every strike-priced market so the totals, puck-line and overtime
        backtests are all priced off exactly the same feed and the same point names.
        """
        pts: dict[str, dict[str, dict[str, Any]]] = {}
        for pr in self.store.query(
                """SELECT contract, point, end_period_ts, bid, ask, mean, volume
                     FROM market_price_points WHERE point IN ('open','t24h','t6h','t1h','close')"""):
            pts.setdefault(pr["contract"], {})[pr["point"]] = {
                "bid": pr["bid"], "ask": pr["ask"], "mean": pr["mean"],
                "ts": pr["end_period_ts"], "volume": pr["volume"]}
        return pts

    def _settled_puck_line(self) -> list[dict[str, Any]]:
        """Settled KXNHLSPREAD contracts ("<team> wins by over k.5 goals") with line and side.

        Two things are read, from two different places, and they are not interchangeable:

        * **which team** the contract names -- the exchange's own ticker suffix, stored
          verbatim in ``market_settlements.rung`` with the rung index discarded
          (``-VGK3`` -> VGK);
        * **the line** -- only ``strike_type``/``floor_strike``.  The suffix digit is a rung
          index, not a line: on 2026-09-21 the live tier's ``-VGK3`` carried
          ``floor_strike=2.5`` while the historical tier's ``-VGK2`` carried the same 2.5.

        A contract whose team cannot be matched to that game's home/away is skipped and
        flagged, because guessing would silently turn a wager on Vegas -1.5 into one on Utah
        -1.5.  As with totals, the candle feed publishes the YES offer only, so these rows
        can price a **cover** entry; the NO side (the opponent's +1.5) is forward-test only.
        """
        rows = self.store.query(
            """SELECT ms.*, g.home_id, g.away_id
                 FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                WHERE ms.provider='kalshi' AND ms.result IN ('yes','no')
                  AND ms.series_ticker='KXNHLSPREAD'""")
        pts = self._candle_points()
        out: list[dict[str, Any]] = []
        for ms in rows:
            line = margin_side_and_strike(ms["strike_type"], ms["floor_strike"])
            if line is None:
                self.store.flag(
                    "unreadable_puck_line",
                    f"puck-line contract {ms['contract']} has no readable line "
                    f"(strike_type={ms['strike_type']}, floor_strike={ms['floor_strike']}); "
                    "skipped rather than assumed", severity="warn",
                    entity_type="market", entity_id=ms["contract"])
                continue
            comparison, strike = line
            code = suffix_team_code(ms["rung"] if "rung" in ms.keys() else None)
            team = None
            if code:
                tid = self.store.team_id_for(KALSHI_TEAM_ALIASES.get(code, code))
                if tid is not None and tid == int(ms["home_id"]):
                    team = "home"
                elif tid is not None and tid == int(ms["away_id"]):
                    team = "away"
            if team is None:
                self.store.flag(
                    "unmapped_puck_line_side",
                    f"puck-line contract {ms['contract']} names suffix {ms['rung']!r} "
                    f"(team code {code!r}), which is not either side of game {ms['game_id']} "
                    f"(home {ms['home_id']}, away {ms['away_id']}); skipped rather than "
                    "assigned to a side by guesswork", severity="warn",
                    entity_type="market", entity_id=ms["contract"])
                continue
            d = dict(ms)
            d["team"] = team
            d["strike"] = strike
            d["comparison"] = comparison
            d["line_basis"] = "strike_type+floor_strike (team from ticker suffix)"
            d["points"] = pts.get(ms["contract"], {})
            out.append(d)
        return out

    def _settled_overtime(self) -> list[dict[str, Any]]:
        """Settled KXNHLOVERTIME contracts ("will there be overtime in this game").

        One contract per game and no strike at all -- verified 2026-09-21: the settled rows
        carry no ``floor_strike``/``strike_type`` and their suffix is ``-OT``, which is not a
        team code.  The settled history this project has fetched spans 2026-04-28 to
        2026-06-14 and is entirely playoff games, so the exchange's treatment of a
        regular-season game decided in a shootout is not covered by it; that is recorded on
        the row (``settlement_scope``) and handled at settlement, never assumed here.
        """
        rows = self.store.query(
            """SELECT ms.*, g.home_id, g.away_id, g.last_period_type, g.game_type
                 FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                WHERE ms.provider='kalshi' AND ms.result IN ('yes','no')
                  AND ms.series_ticker='KXNHLOVERTIME'""")
        pts = self._candle_points()
        out: list[dict[str, Any]] = []
        for ms in rows:
            d = dict(ms)
            d["selection_side"] = "ot"
            d["strike"] = None
            d["line_basis"] = "strike-less contract (verified: no floor_strike/strike_type)"
            d["points"] = pts.get(ms["contract"], {})
            out.append(d)
        return out

    def _strike_market_coverage(self, rows: Sequence[dict[str, Any]], refs: dict[int, GameRef],
                                pit: dict[int, dict[str, Any]], *, series: str,
                                report_key: str, flag_kind: str) -> dict[str, Any]:
        """How much of a strike-priced market's history is actually usable, and why not.

        Same question the totals coverage note answers, asked of KXNHLSPREAD and
        KXNHLOVERTIME: a priced BACKTEST needs a pre-puck-drop offer from the exchange and a
        point-in-time feature row for the *same* game, and the two ingest walks are budgeted
        separately so they can each hold a lot of data and still not overlap.  When they do
        not, the gap is written to the ledger as a gap.  Nothing is inferred to fill it.
        """
        with_offer = [r for r in rows if r.get("points", {}).get(self.entry_point_default)]
        games_offer = {int(r["game_id"]) for r in with_offer}
        scored = {g for g in games_offer
                  if pit.get(g) is not None and pit[g].get("lam_home") is not None}
        decided = {g for g in games_offer if refs.get(g) is not None and refs[g].decided}
        cov = {"series": series,
               "settled_contracts": len(rows),
               "contracts_with_a_pregame_offer": len(with_offer),
               "games_with_a_pregame_offer": len(games_offer),
               "of_those_games_decided_in_the_ledger": len(decided),
               "of_those_games_with_point_in_time_features": len(scored),
               "backtestable_games": len(scored & decided)}
        if cov["backtestable_games"] == 0 and cov["contracts_with_a_pregame_offer"]:
            cov["reason"] = (
                f"Every {series} contract with a pre-puck-drop offer belongs to a game with no "
                "point-in-time feature row, so no entry in this market can be priced honestly "
                "yet. The market is forward-tested at live offers until the exchange-price walk "
                "and the NHL feature walk reach the same games. Nothing was inferred to fill "
                "the gap.")
            self.store.flag(flag_kind, cov["reason"] + " " + json.dumps(cov, indent=1),
                            severity="warn", entity_type="market", entity_id=series,
                            sources="kalshi.candles,nhl.stats_rest")
        self.report[report_key] = cov
        self.log(f"{series} price coverage: {cov['contracts_with_a_pregame_offer']} contracts "
                 f"with a pre-game offer across {cov['games_with_a_pregame_offer']} games; "
                 f"{cov['backtestable_games']} backtestable")
        return cov

    def _backtest_strike_market(self, strat: Any, *, refs: dict[int, GameRef],
                                pit: dict[int, dict[str, Any]],
                                rows: Sequence[dict[str, Any]], entry_point: str,
                                market: str, ot_calibrations: Any = None) -> int:
        """Price a puck-line or overtime rule against settled candle offers (BACKTEST mode).

        Entry price is the ``yes_ask`` of the last candle ending at or before puck drop -- a
        real, timestamped offer -- and settlement is the contract's own Kalshi result, never
        this project's reading of it.  Rules that buy the **NO** side are skipped: the candle
        feed publishes no NO-side offer, and ``no_ask = 1 - yes_bid`` is contradicted by this
        ledger's own quotes, so there is no honest historical price for them to buy at.  Each
        game is offered to the rule as a whole ladder of rungs, exactly as the live book is,
        so the strike it trades is chosen by its own declared target and not by ingest order.

        For the overtime market the calibration handed to the model is the one fitted on
        games decided **before** the game being priced (``ot_calibrations``), so a wager is
        never priced with a ratio fitted on its own outcome.
        """
        if getattr(strat, "market", "moneyline") != market:
            return 0
        if getattr(strat, "quote_side", "YES") != "YES":
            self.report.setdefault(f"{market}_no_history", []).append(
                f"{strat.strategy_id} v{strat.version} buys the NO side, which kalshi's "
                "candlestick history does not publish an offer for; FORWARD TEST ONLY "
                f"({getattr(strat, 'no_history_reason', None) or 'no historical NO-side price'})")
            return 0
        need = "lam_home" if market == "puck_line" else "p_overtime"
        ladders: dict[tuple, list[dict[str, Any]]] = {}
        for r in rows:
            key = (int(r["game_id"]), r.get("team") if market == "puck_line" else "ot")
            ladders.setdefault(key, []).append(r)
        for rungs in ladders.values():
            rungs.sort(key=lambda z: float(z["strike"]) if z.get("strike") is not None else 0.0)
        placed = 0
        for (gid, sel_key), rungs in sorted(ladders.items()):
            g = refs.get(gid)
            f = pit.get(gid)
            if g is None or f is None or f.get(need) is None or not g.decided:
                continue
            if not self._in_scope(strat, g):
                continue
            quotes: list[Quote] = []
            settle_by: dict[str, dict[str, Any]] = {}
            for ms in rungs:
                pt = ms["points"].get(entry_point)
                if not (pt and pt.get("ask") and 0 < float(pt["ask"]) < 1):
                    continue
                q = Quote(provider="kalshi", market_key=ms["event_ticker"],
                          contract=ms["contract"], game_id=gid, game_date=g.game_date,
                          market_type=market,
                          selection=(ms["team"] if market == "puck_line" else "ot"),
                          side="YES", bid=pt.get("bid"), ask=float(pt["ask"]),
                          bid_size=None, ask_size=None, volume=pt.get("volume"),
                          liquidity=None, ts_utc=pt["ts"],
                          label=ms["selection"],
                          strike=(float(ms["strike"]) if ms.get("strike") is not None else None),
                          strike_type=(ms.get("comparison") or None),
                          price_basis=f"kalshi_candle_{entry_point}")
                quotes.append(q)
                settle_by[q.contract] = ms
            if not quotes:
                continue
            pred = {k: f[k] for k in ("p_home_ml", "p_away_ml", "p_home_elo", "p_away_elo",
                                      "p_home_logit", "p_away_logit", "lam_home", "lam_away",
                                      "exp_total", "p_overtime", "p_tie_reg")
                    if f.get(k) is not None}
            if market == "overtime" and ot_calibrations is not None:
                cal = ot_calibrations(g)
                pred["ot_calibration"] = cal.as_evidence()
                if pred.get("p_overtime") is not None:
                    pred["p_ot_cal"] = cal.apply(float(pred["p_overtime"]))
            ts = quotes[0].ts_utc
            ctx = DecisionContext(decision_ts=ts, features=f, predictions=pred, quotes=quotes,
                                  starters={}, injuries={}, bankroll=strat.starting_bankroll,
                                  open_exposure=0.0)
            for sig in strat.evaluate(ctx):
                if sig.status != "READY TO BET" or sig.quote is None:
                    continue
                ms = settle_by.get(sig.quote.contract)
                if ms is None:
                    continue
                pt = ms["points"].get(entry_point)
                close_pt = ms["points"].get("close")
                sig.supporting["decision_ts"] = ts
                sig.supporting["contract"] = ms["contract"]
                sig.supporting["line_basis"] = ms["line_basis"]
                sig.supporting["price_basis"] = (
                    f"kalshi candlestick yes_ask close of the last 60-minute candle ending "
                    f"{pt['ts']} ({entry_point})")
                sig.supporting["price_point"] = entry_point
                sig.supporting["data_labels"] = {
                    "price": f"SOURCE DATA (kalshi candle {entry_point})",
                    "features": "DERIVED (point-in-time)",
                    "model_prob": "MODEL OUTPUT",
                    "settlement": "SOURCE DATA (kalshi's own contract result)"}
                bid = self.paper.place(sig, decision_ts=ts, test_mode="BACKTEST",
                                       provider="kalshi", source_url=self.KALSHI_HIST_URL,
                                       verification=f"kalshi_candle_{entry_point}",
                                       depth_volume=(float(pt["volume"])
                                                     if pt.get("volume") is not None else None))
                if not bid:
                    continue
                placed += 1
                won = (str(ms["result"]).lower() == "yes")   # the exchange's own settlement
                close_mid = None
                if close_pt and close_pt.get("bid") is not None and close_pt.get("ask") is not None:
                    close_mid = round((float(close_pt["bid"]) + float(close_pt["ask"])) / 2, 4)
                clv = (round(close_mid - float(sig.quote.ask), 4)
                       if close_mid is not None and entry_point != "close" else None)
                strike_txt = (f" (strike {float(ms['strike']):g})"
                              if ms.get("strike") is not None else "")
                self.store.settle_bet(
                    bid, result="WIN" if won else "LOSS",
                    pnl=_settle_pnl(self.store, bid, won),
                    close_price=close_mid, clv=clv, settle_ts=ms["settlement_ts"],
                    reason=f"kalshi {market} contract settled {ms['result']}{strike_txt}")
                self.store.execute(
                    "UPDATE bets SET price_point=?, close_price_ts=? WHERE bet_id=?",
                    (entry_point, close_pt["ts"] if close_pt else None, bid))
        return placed

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
    def _in_scope(strat: Any, g: GameRef | None) -> bool:
        """Is this strategy allowed to trade this game's season phase?

        A rule declares the NHL ``gameTypeId`` values it was built for (``game_types``,
        default regular season + playoffs).  Without this check the pipeline let every
        regular-season model loose on September preseason games, where the roster on the ice
        is not the roster the model rates: on the 2026-09-21 ledger **all 102** forward-test
        wagers were preseason games, so the headline competition was measuring a rule in a
        setting it had no evidence for.  See ``Strategy.game_types``.
        """
        if g is None:
            return False
        scope = getattr(strat, "game_types", None) or (REGULAR_SEASON, PLAYOFFS)
        return int(g.game_type) in {int(x) for x in scope}

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

        PRED_KEYS = ("p_home_ml", "p_away_ml", "p_home_elo", "p_away_elo", "p_home_logit",
                     "p_away_logit", "p_over", "p_under", "exp_total", "lam_home", "lam_away",
                     "p_overtime", "p_tie_reg")

        def forward_row(gid: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
            f = pit.get(gid)
            if f is not None and f.get("p_home_ml") is not None:
                pred = {k: f[k] for k in PRED_KEYS if k in f and f[k] is not None}
            else:
                f = feats.get(gid)
                if f is None:
                    return None
                g = refs[gid]
                pred = pois.predict(f)
                pe = elo.prob_home(g.home_id, g.away_id)
                pred["p_home_elo"], pred["p_away_elo"] = pe, 1 - pe
                pred["p_home_logit"], pred["p_away_logit"] = pred["p_home_ml"], pred["p_away_ml"]
            # The tie mass is handed over with the calibration that was fitted on strictly
            # earlier games, plus that calibration's own evidence, so an overtime rule can
            # see how much history stands behind the number it is being asked to trade on.
            pred["ot_calibration"] = self.ot_calibration.as_evidence()
            if pred.get("p_overtime") is not None:
                pred["p_ot_cal"] = self.ot_calibration.apply(float(pred["p_overtime"]))
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
        totals = self._settled_totals()
        self._totals_coverage(totals, refs, pit)
        puck_lines = self._settled_puck_line()
        self._strike_market_coverage(puck_lines, refs, pit, series="KXNHLSPREAD",
                                     report_key="puck_line_price_coverage",
                                     flag_kind="puck_line_price_coverage")
        overtimes = self._settled_overtime()
        self._strike_market_coverage(overtimes, refs, pit, series="KXNHLOVERTIME",
                                     report_key="overtime_price_coverage",
                                     flag_kind="overtime_price_coverage")
        live = self._live_quotes(refs, names)
        # Kalshi lists one "Over k.5" contract per strike, so a totals rule is handed the
        # whole ladder for a game and chooses the rung it declares.
        ladder: dict[tuple, list[Quote]] = {}
        puck_ladder: dict[tuple, list[Quote]] = {}
        team_total_ladder: dict[tuple, list[Quote]] = {}
        period_ladder: dict[tuple, list[Quote]] = {}
        period_total_ladder: dict[tuple, list[Quote]] = {}
        regulation_ladder: dict[tuple, list[Quote]] = {}
        for q in live:
            mtype = q.market_type or "moneyline"
            if mtype == "total":
                ladder.setdefault((q.game_id, mtype, q.selection, q.side), []).append(q)
            elif mtype == "puck_line":
                puck_ladder.setdefault((q.game_id, q.side), []).append(q)
            elif mtype == "team_total":
                team_total_ladder.setdefault((q.game_id, q.selection, q.side), []).append(q)
            elif "period_total" in mtype:
                # first/second/third period total or generic period_total
                period_total_ladder.setdefault((q.game_id, mtype, q.selection, q.side), []).append(q)
            elif mtype in ("period", "first_period", "second_period", "third_period"):
                period_ladder.setdefault((q.game_id, mtype, q.side), []).append(q)
            elif mtype == "regulation":
                regulation_ladder.setdefault((q.game_id, q.selection, q.side), []).append(q)
        for rungs in (list(ladder.values()) + list(puck_ladder.values()) +
                      list(team_total_ladder.values()) + list(period_total_ladder.values()) +
                      list(period_ladder.values()) + list(regulation_ladder.values())):
            rungs.sort(key=lambda z: float(z.strike) if z.strike is not None else 0.0)
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
            # No historical injury feed exists, so an injury-sensitive rule cannot be
            # replayed honestly: "no pending injuries" would really mean "unknown".
            backtestable = not getattr(strat, "injury_sensitive", False)
            mkt = getattr(strat, "market", "moneyline")
            if backtestable and mkt == "total":
                placed["BACKTEST"] += self._backtest_totals(
                    strat, refs=refs, pit=pit, totals=totals, entry_point=entry_point)
                self.store.sync_bankroll(strat.strategy_id, strat.version)
            elif backtestable and mkt == "puck_line":
                placed["BACKTEST"] += self._backtest_strike_market(
                    strat, refs=refs, pit=pit, rows=puck_lines, entry_point=entry_point,
                    market="puck_line")
                self.store.sync_bankroll(strat.strategy_id, strat.version)
            elif backtestable and mkt == "overtime":
                # the calibration is re-fitted per game on strictly earlier games only
                def _cal_for(g: GameRef, _rows=self._ot_cal_rows) -> OtCalibration:
                    return self.ot_calibration_asof(_rows, cutoff=g.start.isoformat())
                placed["BACKTEST"] += self._backtest_strike_market(
                    strat, refs=refs, pit=pit, rows=overtimes, entry_point=entry_point,
                    market="overtime", ot_calibrations=_cal_for)
                self.store.sync_bankroll(strat.strategy_id, strat.version)
            for ms in (settled if (backtestable and mkt == "moneyline") else ()):
                gid = int(ms["game_id"])
                g = refs.get(gid)
                f = pit.get(gid)
                if g is None or f is None or f.get("p_home_ml") is None or not g.decided:
                    continue
                if not self._in_scope(strat, g):
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
                                          "p_home_logit", "p_away_logit", "lam_home", "lam_away",
                                          "exp_total") if f.get(k) is not None}
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
                                           verification=verification,
                                           # a candle publishes no offer size, but it does
                                           # publish how many contracts traded in that hour:
                                           # real depth evidence, and a taker cannot have been
                                           # filled more than the hour traded
                                           depth_volume=(float(pt["volume"])
                                                         if pt and pt.get("volume") is not None
                                                         else None))
                    if bid:
                        placed["BACKTEST"] += 1
                        won = (ms["result"] == "yes")
                        close_mid = None
                        if close_pt and close_pt.get("bid") is not None and close_pt.get("ask") is not None:
                            close_mid = round((float(close_pt["bid"]) + float(close_pt["ask"])) / 2, 4)
                        clv = (round(close_mid - float(sig.quote.ask), 4)
                               if close_mid is not None and entry_point != "close" else None)
                        self.store.settle_bet(
                            bid, result="WIN" if won else "LOSS",
                            pnl=_settle_pnl(self.store, bid, won),
                            close_price=close_mid, clv=clv, settle_ts=ms["settlement_ts"],
                            reason=f"kalshi contract settled {ms['result']}")
                        self.store.execute(
                            "UPDATE bets SET price_point=?, close_price_ts=? WHERE bet_id=?",
                            (entry_point, close_pt["ts"] if close_pt else None, bid))
            self.store.sync_bankroll(strat.strategy_id, strat.version)

            # ---------------------------------------------------- FORWARD TEST
            seen_ladders: set[tuple] = set()
            out_of_scope = 0
            for q in live:
                gid = q.game_id
                g = refs.get(gid) if gid else None
                if g is None or g.decided:
                    continue
                if not self._in_scope(strat, g):
                    out_of_scope += 1
                    continue
                if not self._pregame(g, as_of):
                    continue
                qtype = q.market_type or "moneyline"
                strat_mkt = getattr(strat, "market", "moneyline")
                # allow period variants to match generic period / period_total markets
                market_match = (strat_mkt == qtype)
                if not market_match:
                    if strat_mkt == "period" and qtype in ("period", "first_period", "second_period", "third_period"):
                        market_match = True
                    elif strat_mkt == "period_total" and "period_total" in qtype:
                        market_match = True
                if not market_match:
                    continue
                if q.side != getattr(strat, "quote_side", "YES"):
                    continue
                if qtype == "total":
                    sel = q.selection
                    if sel != getattr(strat, "direction", None):
                        continue
                    key = (gid, qtype, sel, q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = ladder.get(key, [q])
                elif qtype == "puck_line":
                    sel = "puck_line"
                    key = (gid, qtype, "*", q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = puck_ladder.get((gid, q.side), [q])
                elif qtype == "team_total":
                    # team totals: selection is team home/away
                    sel = q.selection
                    if sel != getattr(strat, "team", None):
                        # also allow bet_side matching for legacy
                        if sel != getattr(strat, "bet_side", None):
                            continue
                    key = (gid, sel, q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = team_total_ladder.get(key, [q])
                elif "period_total" in qtype:
                    sel = q.selection  # over/under
                    if sel != getattr(strat, "direction", None):
                        continue
                    # period number check for period_total strategies
                    strat_period = getattr(strat, "period", None)
                    if strat_period is not None:
                        # map qtype to period number
                        q_period = None
                        if "first" in qtype:
                            q_period = 1
                        elif "second" in qtype:
                            q_period = 2
                        elif "third" in qtype:
                            q_period = 3
                        if q_period is not None and q_period != strat_period:
                            continue
                    key = (gid, qtype, sel, q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = period_total_ladder.get(key, [q])
                elif qtype in ("period", "first_period", "second_period", "third_period"):
                    # period moneyline: selection is team
                    sel = q.selection
                    strat_team = getattr(strat, "team", None) or getattr(strat, "bet_side", None)
                    if sel != strat_team:
                        # need to map via _side_for_selection if selection is not home/away token
                        mapped = _side_for_selection(q.selection, g, names)
                        if mapped is None or mapped != strat_team:
                            # allow if q.selection already equals team token
                            if q.selection != strat_team:
                                continue
                            sel = q.selection
                        else:
                            sel = mapped
                    strat_period = getattr(strat, "period", None)
                    if strat_period is not None:
                        q_period = None
                        if "first" in qtype:
                            q_period = 1
                        elif "second" in qtype:
                            q_period = 2
                        elif "third" in qtype:
                            q_period = 3
                        if q_period is not None and q_period != strat_period:
                            continue
                    key = (gid, qtype, q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = period_ladder.get(key, [q])
                elif qtype == "overtime":
                    sel = "ot"
                    key = (gid, qtype, q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = [q]
                elif qtype == "regulation":
                    sel = q.selection
                    strat_team = getattr(strat, "team", None) or getattr(strat, "bet_side", None)
                    if sel != strat_team:
                        mapped = _side_for_selection(q.selection, g, names)
                        if mapped is None or mapped != strat_team:
                            if q.selection != strat_team:
                                continue
                        else:
                            sel = mapped
                    key = (gid, sel, q.side)
                    if key in seen_ladders:
                        continue
                    seen_ladders.add(key)
                    rungs = regulation_ladder.get(key, [q])
                else:
                    sel = _side_for_selection(q.selection, g, names)
                    if sel is None or sel != strat.bet_side:
                        continue
                    rungs = [q]
                fr = forward_row(gid)
                if fr is None:
                    continue
                f, pred = fr
                ref = dk.get(gid) if (q.market_type or "moneyline") == "moneyline" else None
                if ref and ref.get("home_prob") is not None:
                    pred = dict(pred, p_home_book=ref["home_prob"], p_away_book=ref["away_prob"])
                ctx = DecisionContext(decision_ts=q.ts_utc, features=f, predictions=pred,
                                      quotes=rungs, starters=starters_for(f), injuries=inj,
                                      bankroll=bankroll, open_exposure=open_exp)
                for sig in strat.evaluate(ctx):
                    traded = sig.quote if sig.quote is not None else q
                    sig.supporting["decision_ts"] = q.ts_utc
                    sig.supporting["price_point"] = "live_quote"
                    sig.supporting["contract"] = traded.contract
                    sig.supporting["contract_label"] = traded.label
                    sig.supporting["strike"] = traded.strike
                    sig.supporting["quote_side"] = traded.side
                    ref = dk.get(gid) if (q.market_type or "moneyline") == "moneyline" else None
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
        # A signal the venue's own book could not have filled is not a silent non-event.
        # The published depth of a book level is shared by every rule that trades it, so a
        # rule that asks for more contracts than are left is *blocked by finite liquidity*,
        # and that is recorded once per contract with the size that was already claimed.
        for (contract, ts), info in sorted(self.paper.depth_blocked.items()):
            # Two different findings wear the same "no wager" shape and must not be reported
            # as one: an entry point that published no depth at all is a gap in what the
            # venue publishes, while a level that earlier wagers took is the shared-book
            # effect this engine exists to model.  The flag kind names which one it is.
            blocked_by = ", ".join(info["strategies"][:6]) + (
                ", …" if len(info["strategies"]) > 6 else "")
            if info.get("reason") == "no_depth_published":
                self.store.flag(
                    "no_depth_evidence_at_entry",
                    f"{contract}: the entry point at {ts} publishes no depth at all (capacity "
                    f"{info['capacity']:g} contract(s): no offer size and no traded volume), so "
                    f"{len(info['strategies'])} signal(s) could not be filled ({blocked_by}). "
                    "No wager is recorded for them, and nothing was assumed in its place: with "
                    "no published size and no traded volume there is no evidence any contract "
                    "could have been bought at that price.",
                    severity="warn", entity_type="market", entity_id=f"{contract}@{ts}",
                    sources="kalshi.candles")
            else:
                self.store.flag(
                    "depth_exhausted_by_earlier_wagers",
                    f"{contract}: published depth {info['capacity']:g} contract(s) at {ts} was "
                    f"already fully claimed ({info['already_claimed']:g}) by earlier wagers, so "
                    f"{len(info['strategies'])} further signal(s) could not be filled "
                    f"({blocked_by}). No wager is recorded for them; the price was real, the "
                    "size was not there.",
                    severity="warn", entity_type="market", entity_id=f"{contract}@{ts}",
                    sources="kalshi.trade_api")
        for contract, info in sorted(self.paper.wide_book_skipped.items()):
            self.store.flag(
                "wide_book_no_entry",
                f"{contract}: bid {info['bid']:g} / ask {info['ask']:g} is a "
                f"{info['spread']:g}-wide two-sided book (declared maximum "
                f"{info['max_spread']:g}), so the published ask is not an offer a participant "
                f"would have met. ASSUMPTION, not source data: the threshold is this project's, "
                f"and the entry is refused with the numbers recorded rather than filled.",
                severity="warn", entity_type="market", entity_id=contract,
                sources="kalshi.trade_api")
        self.store.commit()
        self.report["placed"] = placed
        self.report["non_executable_signals"] = skipped
        levels = list(self.paper.depth_blocked.values())
        self.report["depth_blocked_levels"] = len(levels)
        self.report["depth_blocked_no_evidence_levels"] = sum(
            1 for v in levels if v.get("reason") == "no_depth_published")
        self.report["depth_blocked_shared_levels"] = sum(
            1 for v in levels if v.get("reason") == "already_claimed")
        self.report["wide_book_refusals"] = len(self.paper.wide_book_skipped)
        self.report["forward_quotes_out_of_season_scope"] = out_of_scope
        self.log(f"forward: {placed}"
                 + (f" ({out_of_scope} quote(s) skipped as outside the rule's season scope)"
                    if out_of_scope else ""))
        return placed

    def _live_quotes(self, refs: dict[int, GameRef] | None = None,
                     names: dict[int, list[str]] | None = None) -> list[Quote]:
        """Latest quote per contract *and side*, normalized into this project's vocabulary.

        Extended to support regulation, team_total, period and period_total markets.
        Each market reuses the same strike parsing as totals/puck_line where applicable,
        never inventing a line.  Unknown lines are skipped, not assumed.
        """
        if refs is None:
            refs = {g.game_id: g for g in self.game_refs(game_types=(1, 2, 3))}
        if names is None:
            names = build_team_names(self.store)
        out: list[Quote] = []
        for q in self.store.query(
                """SELECT * FROM market_quotes mq
                    WHERE mq.game_id IS NOT NULL AND mq.ask IS NOT NULL
                      AND mq.ask > 0 AND mq.ask < 1
                      AND mq.ts_utc = (SELECT MAX(q2.ts_utc) FROM market_quotes q2
                                        WHERE q2.contract = mq.contract AND q2.side = mq.side)"""):
            raw_mtype = q["market_type"] or "moneyline"
            mtype = normalize_market_type(raw_mtype)
            # keep original raw for display but normalized for matching
            # if normalize returns empty, fallback to raw lower
            if not mtype:
                mtype = (raw_mtype or "moneyline").lower()
            gid = int(q["game_id"]) if q["game_id"] is not None else None
            g = refs.get(gid) if gid else None

            # ---- totals and team_totals and period_totals share same shape ----
            if mtype in ("total", "team_total", "first_period_total", "second_period_total",
                         "third_period_total", "period_total"):
                # normalize to canonical types
                if mtype.startswith("first"):
                    canon = "period_total"
                    period_num = 1
                elif mtype.startswith("second"):
                    canon = "period_total"
                    period_num = 2
                elif mtype.startswith("third"):
                    canon = "period_total"
                    period_num = 3
                else:
                    canon = mtype if mtype in ("total", "team_total", "period_total") else "total"
                    period_num = None

                if canon == "team_total":
                    line = team_total_side_and_strike(q["strike_type"], q["strike"])
                elif canon == "period_total":
                    line = period_total_side_and_strike(q["strike_type"], q["strike"])
                else:
                    line = total_side_and_strike(q["strike_type"], q["strike"])

                if line is None:
                    # try text fallback for totals only (source re-read)
                    if canon == "total":
                        line = total_side_from_text(q["selection"])
                    if line is None:
                        continue
                yes_dir, strike = line
                direction = yes_dir if q["side"] == "YES" else ("under" if yes_dir == "over" else "over")

                if canon == "team_total":
                    team = None
                    if g is not None and q["team_abbrev"]:
                        tid = self.store.team_id_for(q["team_abbrev"])
                        if tid == g.home_id:
                            team = "home"
                        elif tid == g.away_id:
                            team = "away"
                    if team is None and g is not None:
                        team = _side_for_selection(q["selection"], g, names)
                    sel = team or q["selection"] or direction
                    # normalize selection to home/away token for strategy matching
                    if sel not in ("home", "away"):
                        # if selection contains team name, try to resolve again via lower
                        low = str(sel).lower()
                        if "home" in low:
                            sel = "home"
                        elif "away" in low:
                            sel = "away"
                    out.append(Quote(provider=q["provider"], market_key=q["market_key"],
                                     contract=q["contract"], game_id=gid, game_date=q["game_date"],
                                     market_type="team_total", selection=sel, side=q["side"],
                                     bid=q["bid"], ask=q["ask"], bid_size=q["bid_size"],
                                     ask_size=q["ask_size"], volume=q["volume"],
                                     liquidity=q["liquidity"], ts_utc=q["ts_utc"],
                                     label=q["selection"], strike=strike,
                                     strike_type=q["strike_type"]))
                elif canon == "period_total":
                    # preserve period distinction in market_type when known
                    if period_num == 1:
                        pt_mtype = "first_period_total"
                    elif period_num == 2:
                        pt_mtype = "second_period_total"
                    elif period_num == 3:
                        pt_mtype = "third_period_total"
                    else:
                        pt_mtype = "period_total"
                    out.append(Quote(provider=q["provider"], market_key=q["market_key"],
                                     contract=q["contract"], game_id=gid, game_date=q["game_date"],
                                     market_type=pt_mtype, selection=direction,
                                     side=q["side"], bid=q["bid"], ask=q["ask"],
                                     bid_size=q["bid_size"], ask_size=q["ask_size"],
                                     volume=q["volume"], liquidity=q["liquidity"],
                                     ts_utc=q["ts_utc"], label=q["selection"],
                                     strike=strike, strike_type=q["strike_type"]))
                else:
                    out.append(Quote(provider=q["provider"], market_key=q["market_key"],
                                     contract=q["contract"], game_id=gid, game_date=q["game_date"],
                                     market_type="total", selection=direction, side=q["side"],
                                     bid=q["bid"], ask=q["ask"], bid_size=q["bid_size"],
                                     ask_size=q["ask_size"], volume=q["volume"],
                                     liquidity=q["liquidity"], ts_utc=q["ts_utc"],
                                     label=q["selection"], strike=strike,
                                     strike_type=q["strike_type"]))
                continue

            if mtype == "puck_line":
                team = None
                if g is not None and q["team_abbrev"]:
                    tid = self.store.team_id_for(q["team_abbrev"])
                    if tid is not None and tid == g.home_id:
                        team = "home"
                    elif tid is not None and tid == g.away_id:
                        team = "away"
                if team is None and g is not None:
                    team = _side_for_selection(q["selection"], g, names)
                line = margin_side_and_strike(q["strike_type"], q["strike"])
                if team is None or line is None:
                    self.store.flag("unreadable_puck_line",
                                    f"Kalshi contract {q['contract']} could not be read as a "
                                    f"puck line: team={team or 'unresolved'}, strike_type="
                                    f"{q['strike_type']!r}, floor_strike={q['strike']!r}. It is "
                                    "not traded; the line and the side are never assumed.")
                    continue
                comparison, strike = line
                if comparison != "greater":
                    self.store.flag("unverified_contract_shape",
                                    f"Kalshi contract {q['contract']} carries strike_type "
                                    f"'{comparison}', which this engine has never verified "
                                    "against a settled outcome. Refused rather than assumed.")
                    continue
                out.append(Quote(
                    provider=q["provider"], market_key=q["market_key"], contract=q["contract"],
                    game_id=gid, game_date=q["game_date"], market_type="puck_line",
                    selection=team, side=q["side"], bid=q["bid"], ask=q["ask"],
                    bid_size=q["bid_size"], ask_size=q["ask_size"], volume=q["volume"],
                    liquidity=q["liquidity"], ts_utc=q["ts_utc"], label=q["selection"],
                    strike=strike, strike_type=comparison))
                continue

            if mtype == "overtime":
                out.append(Quote(
                    provider=q["provider"], market_key=q["market_key"], contract=q["contract"],
                    game_id=gid, game_date=q["game_date"], market_type="overtime",
                    selection="ot", side=q["side"], bid=q["bid"], ask=q["ask"],
                    bid_size=q["bid_size"], ask_size=q["ask_size"], volume=q["volume"],
                    liquidity=q["liquidity"], ts_utc=q["ts_utc"], label=q["selection"],
                    strike=None, strike_type=None))
                continue

            if mtype in ("regulation",):
                # regulation win: team from selection
                team = None
                if g is not None and q["team_abbrev"]:
                    tid = self.store.team_id_for(q["team_abbrev"])
                    if tid == g.home_id:
                        team = "home"
                    elif tid == g.away_id:
                        team = "away"
                if team is None and g is not None:
                    team = _side_for_selection(q["selection"], g, names)
                if team is None:
                    team = (q["selection"] or "").lower()
                out.append(Quote(
                    provider=q["provider"], market_key=q["market_key"], contract=q["contract"],
                    game_id=gid, game_date=q["game_date"], market_type="regulation",
                    selection=team or "home", side=q["side"], bid=q["bid"], ask=q["ask"],
                    bid_size=q["bid_size"], ask_size=q["ask_size"], volume=q["volume"],
                    liquidity=q["liquidity"], ts_utc=q["ts_utc"], label=q["selection"],
                    strike=None, strike_type=None))
                continue

            if mtype in ("first_period", "second_period", "third_period", "period"):
                # period moneyline
                if mtype == "first_period":
                    period_num = 1
                elif mtype == "second_period":
                    period_num = 2
                elif mtype == "third_period":
                    period_num = 3
                else:
                    period_num = None
                team = None
                if g is not None and q["team_abbrev"]:
                    tid = self.store.team_id_for(q["team_abbrev"])
                    if tid == g.home_id:
                        team = "home"
                    elif tid == g.away_id:
                        team = "away"
                if team is None and g is not None:
                    team = _side_for_selection(q["selection"], g, names)
                if team is None:
                    team = (q["selection"] or "").lower()
                # encode period in market_type if known, else generic period
                mt = f"period" if period_num is None else (
                    "first_period" if period_num == 1 else (
                        "second_period" if period_num == 2 else "third_period"))
                out.append(Quote(
                    provider=q["provider"], market_key=q["market_key"], contract=q["contract"],
                    game_id=gid, game_date=q["game_date"], market_type=mt,
                    selection=team or "home", side=q["side"], bid=q["bid"], ask=q["ask"],
                    bid_size=q["bid_size"], ask_size=q["ask_size"], volume=q["volume"],
                    liquidity=q["liquidity"], ts_utc=q["ts_utc"], label=q["selection"],
                    strike=None, strike_type=None))
                continue

            # default: moneyline or unknown - require YES side
            if q["side"] != "YES":
                continue
            sel = _side_for_selection(q["selection"], g, names) if g is not None else None
            if sel is None:
                continue
            out.append(Quote(provider=q["provider"], market_key=q["market_key"],
                             contract=q["contract"], game_id=gid, game_date=q["game_date"],
                             market_type=mtype, selection=sel, side=q["side"], bid=q["bid"],
                             ask=q["ask"], bid_size=q["bid_size"], ask_size=q["ask_size"],
                             volume=q["volume"], liquidity=q["liquidity"], ts_utc=q["ts_utc"],
                             label=q["selection"]))
        return out

    # ------------------------------------------------------------- stage 6
    def stage_settle(self) -> int:
        # the exchange's own settled result first: for a contract Kalshi has finalized, that
        # result is what the instrument paid, and it is also the only honest way to settle a
        # payoff this project has not verified itself (an overtime contract on a shootout
        # game).  Whatever is still OPEN afterwards is settled from the official NHL result.
        from_exchange = self.paper.settle_from_exchange()
        if from_exchange:
            self.report["settled_from_exchange_result"] = from_exchange
            self.log(f"settle: {from_exchange} wager(s) settled on kalshi's own contract result")
        n = 0
        for g in self.store.query("SELECT * FROM games WHERE state IN ('FINAL','OFF')"):
            n += self.paper.settle_game(
                int(g["game_id"]), home_id=int(g["home_id"]), away_id=int(g["away_id"]),
                home_score=g["home_score"], away_score=g["away_score"], state=g["state"],
                last_period_type=g["last_period_type"])
        self.report["settled_now"] = n
        self.report["settled_total"] = n + from_exchange
        self.report["signals_expired"] = self.paper.expire_started(utcnow())
        return n + from_exchange

    def stage_clv(self) -> int:
        """Attach the closing price (last pre-game candle) to FORWARD TEST bets.

        Closing-line value is entry ask vs. the closing mid of the same contract; it is
        only written once the candle history for the contract has been ingested, and it
        is never estimated.
        """
        n = 0
        # Kalshi's candlestick feed publishes the YES bid/ask only.  A closing *mid of the
        # YES side* is the closing price of a YES-side wager; comparing a NO-side entry (an
        # Under, a +1.5 puck line, a "no overtime") against it produces a number with the
        # wrong sign and the wrong instrument, and ``no_close = 1 - yes_close`` is not
        # assumed here for the same reason it is not assumed anywhere else in this project:
        # on the quotes in this ledger ``no_ask != 1 - yes_bid``.  So NO-side wagers get no
        # closing price and no CLV, and that absence is recorded rather than papered over.
        for b in self.store.query(
                """SELECT b.bet_id, b.entry_price, b.price, p.bid, p.ask, p.end_period_ts,
                          COALESCE(b.exchange_side, json_extract(b.notes, '$.exchange_side'),
                                   'YES') AS xs
                     FROM bets b
                     JOIN market_price_points p
                       ON p.contract = json_extract(b.notes, '$.contract') AND p.point='close'
                    WHERE b.test_mode='FORWARD TEST' AND b.close_price IS NULL
                      AND p.bid IS NOT NULL AND p.ask IS NOT NULL"""):
            if str(b["xs"]).upper() != "YES":
                self.store.flag(
                    "clv_unavailable_no_side",
                    f"bet {b['bet_id']} bought the {str(b['xs']).upper()} side of its contract; "
                    "kalshi's candlestick history publishes the YES bid/ask only, so there is "
                    "no timestamped closing price for the side that was bought and no CLV is "
                    "reported for it (1 - yes_close is not assumed)",
                    severity="info", entity_type="bet", entity_id=b["bet_id"])
                continue
            close_mid = round((float(b["bid"]) + float(b["ask"])) / 2, 4)
            entry = float(b["entry_price"] or b["price"])
            self.store.amend_bet(b["bet_id"], {"close_price": close_mid,
                                               "clv": round(close_mid - entry, 4),
                                               "close_price_ts": b["end_period_ts"]},
                                 reason="closing price from kalshi candlestick (last pre-game candle)")
            n += 1
        self.report["clv_attached"] = n
        return n

    INJURY_CONTEXT_NOTE = "backtest_without_injury_context"

    def stage_reconcile(self) -> int:
        """Annotate BACKTEST rows that a stricter rule would not have written.

        An injury-sensitive strategy cannot be replayed (no historical injury feed), so any
        BACKTEST wager it holds was placed without knowing whether an injury was pending.
        Those rows are never deleted; their verification_status gains a suffix, through
        ``amend_bet`` so the before/after audit row exists.  Idempotent.
        """
        n = 0
        for s in self.store.query("SELECT strategy_id, version, params_json FROM strategies"):
            try:
                params = json.loads(s["params_json"] or "{}")
            except (TypeError, ValueError):
                continue
            if not params.get("injury_sensitive"):
                continue
            for b in self.store.query(
                    """SELECT bet_id, verification_status FROM bets
                        WHERE strategy_id=? AND strategy_version=? AND test_mode='BACKTEST'
                          AND verification_status NOT LIKE ?""",
                    (s["strategy_id"], int(s["version"]), f"%{self.INJURY_CONTEXT_NOTE}%")):
                self.store.amend_bet(
                    b["bet_id"],
                    {"verification_status": f"{b['verification_status']};{self.INJURY_CONTEXT_NOTE}"},
                    reason="injury-sensitive rule replayed without historical injury data; "
                           "row kept, annotated as not a valid test of the rule")
                n += 1
        self.report["reconciled_bets"] = n
        if n:
            self.log(f"reconcile: {n} BACKTEST rows annotated '{self.INJURY_CONTEXT_NOTE}'")
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

    def stage_master_site_review(self) -> dict[str, Any]:
        """Record the review of the owner's MasterSite directory in the findings ledger.

        The brief asks for the master directory to be reviewed and for infrastructure to be
        reused only after it is verified as relevant and functional.  The review itself
        lives in :mod:`nhlcomp.mastersite` (fetched sources, per-project verdicts, and the
        one reuse candidate that was tested and rejected); this stage writes it to the
        ledger so it is auditable next to every other finding instead of living only in a
        docstring.
        """
        from . import mastersite
        summary = mastersite.summary()
        self.store.execute(
            """INSERT OR REPLACE INTO findings(finding_id, created_at, title, body, evidence,
                                               confidence, kind) VALUES(?,?,?,?,?,?,?)""",
            ("FIND_MASTER_SITE_REVIEW", utcnow(),
             f"MasterSite directory reviewed {mastersite.REVIEW_DATE} "
             f"({summary['projects_reviewed']} projects)",
             "The owner's MasterSite index was fetched, every named project was checked against "
             "the GitHub API, candidate READMEs were read, and the one data claim that mattered "
             "was tested against the live endpoint before anything was reused. "
             f"Repositories that do not exist: {', '.join(summary['missing_repositories']) or 'none'}. "
             "One reuse candidate (NHL prices from the ESPN scoreboard odds block) was tested and "
             "rejected for historical use; the feed itself was kept as a registered source for "
             "the results and post-game goalie identification it verifiably supplies.",
             json.dumps(summary, indent=1), "high", "source_review"))
        self.store.commit()
        self.report["master_site_review"] = summary
        self.log(f"master site review: {summary['projects_reviewed']} projects, "
                 f"missing repos {summary['missing_repositories']}")
        return summary

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
        self.stage_reconcile()
        self.stage_analysis()
        self.stage_master_site_review()
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
    from .strategies import (OvertimeStrategy, PeriodStrategy, PeriodTotalStrategy,
                             PuckLineStrategy, RegulationStrategy, TeamTotalStrategy,
                             ThresholdStrategy, TotalsStrategy)
    params = json.loads(row["params_json"] or "{}")
    # Fall back to the stored column for the season scope: params_json may predate it (and a
    # hand-built legacy row in a test has no column at all, hence the guard).
    stored_scope = (row["game_types"] if "game_types" in row.keys() else None)
    if stored_scope is not None and "game_types" not in params:
        try:
            params["game_types"] = json.loads(stored_scope)
        except (TypeError, ValueError):
            pass
    # the class is part of the stored parameter set: reloading a totals rule as a
    # threshold rule would silently drop its strike logic and let it bet a moneyline
    cls_map = {
        "totals": TotalsStrategy,
        "puck_line": PuckLineStrategy,
        "overtime": OvertimeStrategy,
        "regulation": RegulationStrategy,
        "team_total": TeamTotalStrategy,
        "period": PeriodStrategy,
        "period_total": PeriodTotalStrategy,
    }
    cls = cls_map.get(params.get("kind"), ThresholdStrategy)
    kw = dict(params)
    kw.pop("min_edge", None)
    kw.pop("stake_fraction", None)
    kw.pop("kind", None)
    # derived in each rule's __init__ from its own direction/exchange_side: it is published
    # in describe() for the audit trail, but reloading must not let a stale blob override it
    kw.pop("quote_side", None)
    if cls is PuckLineStrategy:
        kw.setdefault("contract_team", "model_stronger")
        kw.setdefault("exchange_side", "YES")
        kw.setdefault("target_strike", 1.5)
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.04),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
    if cls is OvertimeStrategy:
        kw.setdefault("direction", "yes")
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.03),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
    if cls is TotalsStrategy:
        kw.setdefault("direction", "over")
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.04),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
    if cls is RegulationStrategy:
        kw.setdefault("team", "home")
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.04),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
    if cls is TeamTotalStrategy:
        kw.setdefault("team", "home")
        kw.setdefault("direction", "over")
        kw.setdefault("target_strike", 2.5)
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.04),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
    if cls is PeriodStrategy:
        kw.setdefault("period", 1)
        kw.setdefault("team", "home")
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.04),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
    if cls is PeriodTotalStrategy:
        kw.setdefault("period", 1)
        kw.setdefault("direction", "over")
        kw.setdefault("target_strike", 1.5)
        return cls(
            strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
            name=row["name"], category=row["category"],
            hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
            price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
            markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
            starting_bankroll=float(row["starting_bankroll"]),
            min_edge=params.get("min_edge", 0.04),
            stake_fraction=params.get("stake_fraction", 0.25), **kw)
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


def _settle_pnl(store: Store, bet_id: str, won: bool) -> float:
    """P&L from the fill actually recorded on the bet row (contracts, price, fee), so the
    settlement and the verifier's recomputation agree to the cent."""
    from .paper import binary_settlement
    b = store.one("SELECT entry_price, price, filled_size, stake, fee FROM bets WHERE bet_id=?",
                  (bet_id,))
    price = float(b["entry_price"] or b["price"])
    contracts = float(b["filled_size"] or (float(b["stake"]) / price if price else 0))
    return binary_settlement(price, contracts, won, fee=float(b["fee"] or 0.0))
