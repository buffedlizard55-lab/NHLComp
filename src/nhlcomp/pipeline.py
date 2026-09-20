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
from .backtest import Backtester, CAVEAT_NO_PRICE
from .discovery import DiscoveryEngine
from .features import FeatureBuilder, GameRef
from .http import HttpClient, NetworkUnavailable
from .ingest import Ingestor
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

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}", flush=True)

    # ------------------------------------------------------------- stage 1
    def stage_ingest(self, *, seasons: Sequence[int], scoreboard_days: Sequence[str],
                     club_abbrevs: Sequence[str], ingest_settled_pages: int = 5,
                     cross_check_abbrevs: Sequence[str] = ()) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["probes"] = self.ing.verify_sources()
        out["teams"] = self.ing.teams()
        for season in seasons:
            try:
                self.ing.standings(f"{season // 10000 + 1}-04-17")
            except NetworkUnavailable as exc:
                self.log(f"standings {season} skipped: {exc}")
        for season in seasons:
            for ab in club_abbrevs:
                self.ing.club_season(ab, season)
        for day in scoreboard_days:
            self.ing.scoreboard_window(day)
        out["team_game_rows"] = self.ing.rebuild_team_games()
        out["kalshi_active"] = self.ing.kalshi_nhl()
        out["kalshi_settled"] = self.ing.kalshi_settled(max_pages=ingest_settled_pages)
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
        refs = self.game_refs(seasons=seasons, game_types=game_types)
        fb = FeatureBuilder(refs)
        rows = fb.build_all()
        games_by_id = {g.game_id: g for g in refs}
        for r in rows:
            g = games_by_id[int(r["game_id"])]
            r["_winner"] = g.winner
            r["_total"] = g.total_goals
            r["_decided"] = g.decided
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
        # register the seed library (never overwrites an existing version)
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
        self.report["discovery"] = {"tested": eng.tested, "survivors": len(created),
                                    "candidates": [c.key for c in candidates][:50]}
        self.log(f"strategies: {eng.tested} triggers tested, {len(created)} promoted, "
                 f"{len(self.store.latest_versions())} total")
        return self.report["discovery"]

    def stage_backtest(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        bt = Backtester(self.store)
        out = {}
        for s in self.store.latest_versions():
            strat = _hydrate(s)
            res = bt.run(strat, [r for r in rows if r.get("_winner") is not None], label="all")
            if res:
                out[s["strategy_id"]] = {"n": res.n_bets, "hit": res.hit_rate,
                                         "base": res.base_rate, "lift": res.lift,
                                         "ci": [res.ci_low, res.ci_high],
                                         "log_loss": res.log_loss, "brier": res.brier}
        self.report["backtests"] = out
        return out

    # ------------------------------------------------------------- stage 5
    def stage_forward(self, *, as_of: str | None = None,
                      include_unsettled_settlements: bool = True) -> dict[str, Any]:
        """Evaluate every strategy against every game that has a real quote.

        Two price feeds are used, clearly separated:

        * settled contracts -> BACKTEST rows (real price, real outcome, entry timestamp
          documented only as "pre-settlement")
        * live contracts     -> FORWARD TEST rows that settle later as games finish
        """
        refs = {g.game_id: g for g in self.game_refs(game_types=(1, 2, 3))}
        names = build_team_names(self.store)
        fb = FeatureBuilder([g for g in refs.values() if g.game_type == 2])
        feats = {int(r["game_id"]): r for r in fb.build_all(list(refs.values()))}
        pois = PoissonModel()
        elo = EloModel()
        for g in sorted((g for g in refs.values() if g.decided and g.game_type == 2),
                        key=lambda x: x.start):
            elo.observe(g)

        placed = {"BACKTEST": 0, "FORWARD TEST": 0}
        skipped = 0
        for s in self.store.latest_versions():
            if s["status"] == "rejected":
                continue
            strat = _hydrate(s)
            self.store.sync_bankroll(strat.strategy_id, strat.version)
            srow = self.store.one("SELECT bankroll FROM strategies WHERE strategy_id=? AND version=?",
                                  (strat.strategy_id, strat.version))
            bankroll = float(srow["bankroll"]) if srow else strat.starting_bankroll
            open_exp = float(self.store.one(
                "SELECT COALESCE(SUM(stake),0) e FROM bets WHERE strategy_id=? AND "
                "strategy_version=? AND result='OPEN'", (strat.strategy_id, strat.version))["e"] or 0)

            for ms in self.store.query(
                    """SELECT * FROM market_settlements WHERE game_id IS NOT NULL
                       AND result IN ('yes','no') AND ask_before IS NOT NULL
                       AND ask_before > 0 AND ask_before < 1"""):
                gid = int(ms["game_id"])
                g = refs.get(gid)
                if g is None:
                    continue
                sel = _side_for_selection(ms["selection"], g, names)
                if sel is None:
                    continue
                f = feats.get(gid)
                if f is None:
                    continue
                pred = pois.predict(f)
                pe = elo.prob_home(g.home_id, g.away_id)
                pred["p_home_elo"], pred["p_away_elo"] = pe, 1 - pe
                pred["p_home_logit"] = pred["p_home_ml"]
                pred["p_away_logit"] = pred["p_away_ml"]
                q = Quote(provider="kalshi", market_key=ms["event_ticker"],
                          contract=ms["contract"], game_id=gid, game_date=g.game_date,
                          market_type="moneyline", selection=sel, side="YES",
                          bid=ms["bid_before"], ask=ms["ask_before"],
                          bid_size=None, ask_size=None, volume=ms["volume"],
                          liquidity=None, ts_utc=ms["open_time"] or ms["retrieved_at"])
                ctx = DecisionContext(decision_ts=ms["open_time"] or ms["retrieved_at"],
                                      features=f, predictions=pred, quotes=[q],
                                      bankroll=bankroll, open_exposure=open_exp)
                strat.bet_side = sel if strat.category == "kalshi_both_sides" else strat.bet_side
                for sig in strat.evaluate(ctx):
                    sig.supporting["decision_ts"] = ctx.decision_ts
                    sig.supporting["contract"] = ms["contract"]
                    sig.supporting["price_basis"] = "kalshi previous_yes_ask (pre-settlement)"
                    if sig.status != "READY TO BET" or sig.quote is None or sel != strat.bet_side:
                        continue
                    bid = self.paper.place(
                        sig, decision_ts=ctx.decision_ts, test_mode="BACKTEST",
                        provider="kalshi",
                        source_url="https://api.elections.kalshi.com/trade-api/v2/markets",
                        verification="single_source_timing_unverified")
                    if bid:
                        placed["BACKTEST"] += 1
                        won = (ms["result"] == "yes")
                        self.store.settle_bet(
                            bid, result="WIN" if won else "LOSS",
                            pnl=_settle_pnl(sig, won),
                            settle_ts=ms["settlement_ts"],
                            reason=f"kalshi contract settled {ms['result']}")
                        self.store.sync_bankroll(strat.strategy_id, strat.version)
                        bankroll = float(self.store.one(
                            "SELECT bankroll FROM strategies WHERE strategy_id=? AND version=?",
                            (strat.strategy_id, strat.version))["bankroll"])

            # live quotes -> open forward-test positions
            for q in self._live_quotes():
                gid = q.game_id
                g = refs.get(gid) if gid else None
                if g is None or g.decided:
                    continue
                sel = _side_for_selection(q.selection, g, names)
                if sel is None or sel != strat.bet_side:
                    continue
                f = feats.get(gid)
                if f is None:
                    continue
                pred = pois.predict(f)
                pe = elo.prob_home(g.home_id, g.away_id)
                pred["p_home_elo"], pred["p_away_elo"] = pe, 1 - pe
                pred["p_home_logit"], pred["p_away_logit"] = pred["p_home_ml"], pred["p_away_ml"]
                ctx = DecisionContext(decision_ts=q.ts_utc, features=f, predictions=pred,
                                      quotes=[q], bankroll=bankroll, open_exposure=open_exp)
                for sig in strat.evaluate(ctx):
                    sig.supporting["decision_ts"] = q.ts_utc
                    self.paper.record_upcoming(
                        sig, provider="kalshi",
                        source_url="https://api.elections.kalshi.com/trade-api/v2/markets")
                    if sig.status == "READY TO BET":
                        bid = self.paper.place(
                            sig, decision_ts=q.ts_utc, test_mode="FORWARD TEST",
                            provider="kalshi",
                            source_url="https://api.elections.kalshi.com/trade-api/v2/markets",
                            verification="single_source")
                        if bid:
                            placed["FORWARD TEST"] += 1
                            open_exp += sig.stake
                            bankroll -= sig.stake
                    else:
                        skipped += 1
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
        return n

    def stage_analysis(self) -> dict[str, Any]:
        self.report["leaderboard"] = self.perf.leaderboard()
        self.report["competition"] = self.perf.competition_totals()
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
        self.stage_ingest(**kw.get("ingest", {}))
        rows = self.stage_features(**kw.get("features", {}))
        refs = self.game_refs(**kw.get("features", {}))
        self.stage_models(rows, refs, prior_season=kw.get("prior_season"))
        self.stage_strategies(rows)
        self.stage_backtest(rows)
        self.stage_forward()
        self.stage_settle()
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
    from .strategies import ThresholdStrategy
    params = json.loads(row["params_json"] or "{}")
    return ThresholdStrategy(
        strategy_id=row["strategy_id"], version=int(row["version"]), username=row["username"],
        name=row["name"], category=row["category"],
        hypothesis=row["hypothesis"], data_used=row["data_used"], entry_rule=row["entry_rule"],
        price_rule=row["price_rule"], settlement_rule=row["settlement_rule"],
        markets=row["markets"], origin=row["origin"], origin_ref=row["origin_ref"],
        starting_bankroll=float(row["starting_bankroll"]),
        feature=params.get("feature", "home_n_prior"), operator=params.get("operator", ">="),
        threshold=params.get("threshold", 0), bet_side=params.get("bet_side", "home"),
        use_model=params.get("use_model", "poisson"), min_edge=params.get("min_edge", 0.03),
        stake_fraction=params.get("stake_fraction", 0.25))


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
    from .paper import binary_settlement
    price = float(sig.quote.ask)
    contracts = float(sig.stake) / price
    return binary_settlement(price, contracts, won)
