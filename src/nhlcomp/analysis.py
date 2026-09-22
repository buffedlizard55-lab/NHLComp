"""Performance measurement.

Win rate is never reported on its own: every headline number ships with a sample size and
a Wilson interval, and drawdown / volatility / closing-line value are computed alongside so
an aggressive strategy cannot hide behind a lucky run.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

from .models import wilson_interval
from .store import Store


def _safe_div(a: float, b: float) -> float | None:
    return (a / b) if b else None


#: NHL ``gameTypeId`` values, kept local so this module has no strategy import.
PHASE_LABELS = {1: "preseason", 2: "regular season", 3: "playoffs"}
#: The phase the competition is scored on.  A rule fitted on regular-season form has a
#: verified claim on regular-season and playoff games and none at all on a September
#: preseason roster, so preseason wagers -- which is what the ledger held before the scope
#: gate of 2026-09-22 -- are reported on their own line and never merged into these totals.
COMPETITION_PHASES = (2, 3)
#: Every phase, for a caller that explicitly wants the whole ledger in one number.
ALL_PHASES = (1, 2, 3)


def phase_of(game_type: Any) -> str:
    try:
        return PHASE_LABELS.get(int(game_type), f"game_type {game_type}")
    except (TypeError, ValueError):
        return "unknown phase"


class Performance:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------------ ledger
    def strategy_bets(self, strategy_id: str, version: int,
                      test_mode: str | None = None) -> list[dict[str, Any]]:
        sql = ("SELECT * FROM bets WHERE strategy_id=? AND strategy_version=?"
               + (" AND test_mode=?" if test_mode else "") + " ORDER BY bet_ts, bet_id")
        params = [strategy_id, version] + ([test_mode] if test_mode else [])
        return [dict(r) for r in self.store.query(sql, params)]

    def equity_curve(self, bets: Sequence[dict[str, Any]], start: float) -> list[float]:
        eq, out = start, [start]
        for b in bets:
            eq += float(b.get("pnl") or 0.0)
            out.append(round(eq, 4))
        return out

    @staticmethod
    def max_drawdown(curve: Sequence[float]) -> float:
        peak = -math.inf
        dd = 0.0
        for v in curve:
            peak = max(peak, v)
            dd = min(dd, v - peak)
        return round(dd, 4) if dd else 0.0

    @staticmethod
    def volatility(bets: Sequence[dict[str, Any]]) -> float | None:
        xs = [float(b["pnl"]) for b in bets if b.get("pnl") is not None]
        if len(xs) < 2:
            return None
        mu = sum(xs) / len(xs)
        return round(math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)), 4)

    @staticmethod
    def longest_losing_streak(bets: Sequence[dict[str, Any]]) -> int:
        best = run = 0
        for b in bets:
            if b.get("result") == "LOSS":
                run += 1
                best = max(best, run)
            elif b.get("result") in ("WIN", "PUSH", "VOID"):
                run = 0
        return best

    def summarize(self, strategy_id: str, version: int) -> dict[str, Any]:
        s = self.store.one("SELECT * FROM strategies WHERE strategy_id=? AND version=?",
                           (strategy_id, version))
        if s is None:
            return {}
        start = float(s["starting_bankroll"])
        out: dict[str, Any] = {}
        for mode in ("ALL", "FORWARD TEST", "BACKTEST"):
            bets = (self.strategy_bets(strategy_id, version) if mode == "ALL"
                    else self.strategy_bets(strategy_id, version, mode))
            settled = [b for b in bets if b.get("result") in ("WIN", "LOSS", "PUSH", "VOID")]
            wins = sum(1 for b in settled if b["result"] == "WIN")
            losses = sum(1 for b in settled if b["result"] == "LOSS")
            pushes = sum(1 for b in settled if b["result"] in ("PUSH", "VOID"))
            pnl = sum(float(b["pnl"] or 0) for b in settled)
            staked = sum(float(b["stake"] or 0) for b in settled)
            open_stake = sum(float(b["stake"] or 0) for b in bets if b.get("result") == "OPEN")
            curve = self.equity_curve(settled, start)
            ci = wilson_interval(wins, max(wins + losses, 0))
            clvs = [float(b["clv"]) for b in settled if b.get("clv") is not None]
            out[mode] = {
                "n_bets": len(bets), "n_settled": len(settled), "wins": wins, "losses": losses,
                "pushes": pushes, "open": len(bets) - len(settled),
                "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else None,
                "win_rate_ci": [round(ci[0], 4), round(ci[1], 4)],
                "pnl": round(pnl, 2), "staked": round(staked, 2),
                "roi": round(pnl / staked, 4) if staked else None,
                "bankroll": round(start + pnl - open_stake, 2),
                "open_exposure": round(open_stake, 2),
                "max_drawdown": self.max_drawdown(curve),
                "volatility": self.volatility(settled),
                "longest_losing_streak": self.longest_losing_streak(settled),
                "avg_price": round(sum(float(b["price"]) for b in settled) / len(settled), 4)
                if settled else None,
                "avg_edge": round(sum(float(b["edge"]) for b in settled
                                      if b.get("edge") is not None) /
                                  max(sum(1 for b in settled if b.get("edge") is not None), 1), 4)
                if settled else None,
                "clv_beat_close": round(sum(1 for c in clvs if c > 0) / len(clvs), 4) if clvs else None,
                "clv_avg": round(sum(clvs) / len(clvs), 5) if clvs else None,
                "n_clv": len(clvs),
            }
        out["strategy"] = dict(s)
        out["breakdowns"] = self.breakdowns(strategy_id, version)
        # The mode totals above are the strategy's *own* ledger: its bankroll has to account
        # for every wager it actually holds, preseason included.  Which means they must be
        # read next to the phase split, or a strategy whose only wagers were preseason games
        # looks like it is competing when it is not.
        phases: dict[str, Any] = {}
        for gtype, label in sorted(PHASE_LABELS.items()):
            bets = [b for b in self.strategy_bets(strategy_id, version)
                    if b.get("game_type") == gtype]
            if not bets:
                continue
            settled_p = [b for b in bets if b.get("result") in ("WIN", "LOSS", "PUSH", "VOID")]
            wins_p = sum(1 for b in settled_p if b["result"] == "WIN")
            losses_p = sum(1 for b in settled_p if b["result"] == "LOSS")
            pnl_p = sum(float(b["pnl"] or 0) for b in settled_p)
            staked_p = sum(float(b["stake"] or 0) for b in settled_p)
            phases[label] = {
                "game_type": gtype, "bets": len(bets), "settled": len(settled_p),
                "open": len(bets) - len(settled_p), "wins": wins_p, "losses": losses_p,
                "pnl": round(pnl_p, 2), "staked": round(staked_p, 2),
                "roi": round(pnl_p / staked_p, 4) if staked_p else None,
                "win_rate": round(wins_p / (wins_p + losses_p), 4) if (wins_p + losses_p) else None,
                "scored_in_competition": gtype in COMPETITION_PHASES,
            }
        unlabelled = [b for b in self.strategy_bets(strategy_id, version)
                      if b.get("game_type") is None]
        if unlabelled:
            phases["phase not recorded (row predates bets.game_type)"] = {
                "game_type": None, "bets": len(unlabelled),
                "settled": sum(1 for b in unlabelled
                               if b.get("result") in ("WIN", "LOSS", "PUSH", "VOID")),
                "open": sum(1 for b in unlabelled if b.get("result") == "OPEN"),
                "wins": sum(1 for b in unlabelled if b["result"] == "WIN"),
                "losses": sum(1 for b in unlabelled if b["result"] == "LOSS"),
                "pnl": round(sum(float(b["pnl"] or 0) for b in unlabelled), 2),
                "staked": round(sum(float(b["stake"] or 0) for b in unlabelled), 2),
                "roi": None, "win_rate": None, "scored_in_competition": False,
            }
        out["phases"] = phases
        out["preseason_bets"] = sum(1 for b in self.strategy_bets(strategy_id, version)
                                    if b.get("game_type") == 1)
        return out

    def breakdowns(self, strategy_id: str, version: int) -> dict[str, dict[str, Any]]:
        bets = self.strategy_bets(strategy_id, version)
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {
            "market": defaultdict(list), "month": defaultdict(list),
            "season": defaultdict(list), "team": defaultdict(list),
            "category": defaultdict(list), "timing": defaultdict(list),
            "provider": defaultdict(list),
        }
        cat = self.store.one("SELECT category FROM strategies WHERE strategy_id=? AND version=?",
                             (strategy_id, version))
        category = cat["category"] if cat else "unknown"
        for b in bets:
            if b.get("result") not in ("WIN", "LOSS", "PUSH", "VOID"):
                continue
            groups["market"][b["market"] or "?"].append(b)
            groups["month"][(b["game_date"] or "?")[:7]].append(b)
            groups["season"][str(b["season"] or "?")].append(b)
            groups["team"][b["selection"] or "?"].append(b)
            groups["category"][category].append(b)
            groups["timing"][(b["bet_ts"] or "?")[:13] + ":00"].append(b)
            groups["provider"][b["provider"] or "?"].append(b)
        out: dict[str, dict[str, Any]] = {}
        for gname, buckets in groups.items():
            out[gname] = {}
            for key, rows in buckets.items():
                pnl = sum(float(r["pnl"] or 0) for r in rows)
                staked = sum(float(r["stake"] or 0) for r in rows)
                wins = sum(1 for r in rows if r["result"] == "WIN")
                losses = sum(1 for r in rows if r["result"] == "LOSS")
                out[gname][key] = {"n": len(rows), "pnl": round(pnl, 2),
                                   "staked": round(staked, 2),
                                   "roi": round(pnl / staked, 4) if staked else None,
                                   "win_rate": round(wins / (wins + losses), 4)
                                   if (wins + losses) else None}
        return out

    def leaderboard(self, *, test_mode: str | None = None) -> list[dict[str, Any]]:
        rows = []
        for s in self.store.latest_versions():
            summ = self.summarize(s["strategy_id"], int(s["version"]))
            mode = test_mode or "ALL"
            if mode not in summ:
                continue
            m = summ[mode]
            rows.append({
                "strategy_id": s["strategy_id"], "version": int(s["version"]),
                "username": s["username"], "name": s["name"], "category": s["category"],
                "status": s["status"], "test_mode": s["test_mode"],
                # composition: how much of this strategy's ledger is preseason, which is
                # *not* scored.  A row with preseason_bets > 0 must not be read as a
                # competition result without that column in view.
                "preseason_bets": summ.get("preseason_bets", 0),
                "phases": summ.get("phases", {}),
                **{k: m[k] for k in ("n_bets", "n_settled", "wins", "losses", "pushes", "open",
                                     "win_rate", "win_rate_ci", "pnl", "staked", "roi",
                                     "bankroll", "open_exposure", "max_drawdown", "volatility",
                                     "longest_losing_streak", "avg_price", "avg_edge",
                                     "clv_beat_close", "clv_avg", "n_clv")},
                "starting_bankroll": float(s["starting_bankroll"]),
            })
        rows.sort(key=lambda r: (-(r["pnl"] or 0), -(r["n_settled"] or 0)))
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        return rows

    def competition_totals(self, *, test_mode: str | None = None,
                           phases: Sequence[int] | None = COMPETITION_PHASES) -> dict[str, Any]:
        """Aggregate ledger state.  ``test_mode`` restricts to BACKTEST or FORWARD TEST; the
        default aggregates both and is labelled ``mode='ALL'`` so a reader can tell.

        ``phases`` restricts to NHL ``gameTypeId`` values and **defaults to
        :data:`COMPETITION_PHASES` (regular season + playoffs)**, so a caller that forgets to
        think about the phase still gets the scored competition rather than a number quietly
        inflated by preseason wagers on rosters no model here has seen.  Pass
        :data:`ALL_PHASES`, or ``None`` for no filter at all, to aggregate everything.
        """
        clauses, params = [], []
        if test_mode:
            clauses.append("test_mode=?")
            params.append(test_mode)
        if phases is not None:
            clauses.append("COALESCE(game_type, 2) IN (%s)" % ",".join("?" * len(phases)))
            params.extend(int(p) for p in phases)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.store.query(
            f"""SELECT COALESCE(SUM(pnl),0) pnl, COALESCE(SUM(stake),0) staked, COUNT(*) n,
                      SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) wins,
                      SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) losses,
                      SUM(CASE WHEN result='OPEN' THEN 1 ELSE 0 END) open
               FROM bets{where}""", tuple(params))[0]
        settled = int(rows["wins"] or 0) + int(rows["losses"] or 0)
        return {
            "mode": test_mode or "ALL",
            "strategies": len(self.store.latest_versions()),
            "bets": int(rows["n"] or 0), "settled": settled, "open": int(rows["open"] or 0),
            "wins": int(rows["wins"] or 0), "losses": int(rows["losses"] or 0),
            "pnl": round(float(rows["pnl"] or 0), 2),
            "staked": round(float(rows["staked"] or 0), 2),
            "roi": round(float(rows["pnl"] or 0) / float(rows["staked"]), 4)
            if rows["staked"] else None,
            "win_rate": round(int(rows["wins"] or 0) / settled, 4) if settled else None,
        }

    def phase_breakdown(self, *, test_mode: str | None = None) -> list[dict[str, Any]]:
        """The ledger split by season phase, with the scored competition called out.

        Every wager carries the ``game_type`` of the game it was placed on (1 preseason,
        2 regular season, 3 playoffs).  Reporting the phases apart is not cosmetic: all 102
        forward-test wagers on the 2026-09-21 ledger were preseason games, so a single merged
        leaderboard was presenting a regular-season model's view of AHL-heavy September
        rosters as though it were the competition.
        """
        out: list[dict[str, Any]] = []
        for gtype, label in sorted(PHASE_LABELS.items()):
            for mode in (None, "BACKTEST", "FORWARD TEST"):
                t = self.competition_totals(test_mode=mode, phases=(gtype,))
                if not t["bets"]:
                    continue
                t["phase"] = label
                t["game_type"] = gtype
                t["scored_in_competition"] = gtype in COMPETITION_PHASES
                if test_mode and mode != test_mode:
                    continue
                out.append(t)
        # rows written before bets.game_type existed have no phase recorded; report them
        # explicitly rather than letting them fall into a phase they were never assigned
        for mode in (None, "BACKTEST", "FORWARD TEST"):
            if test_mode and mode != test_mode:
                continue
            clauses = ["game_type IS NULL"]
            params: list[Any] = []
            if mode:
                clauses.append("test_mode=?")
                params.append(mode)
            r = self.store.query(
                f"""SELECT COALESCE(SUM(pnl),0) pnl, COALESCE(SUM(stake),0) staked, COUNT(*) n,
                           SUM(CASE WHEN result IN ('WIN','LOSS') THEN 1 ELSE 0 END) settled,
                           SUM(CASE WHEN result='OPEN' THEN 1 ELSE 0 END) open,
                           SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) wins
                      FROM bets WHERE {' AND '.join(clauses)}""", tuple(params))[0]
            if not r["n"]:
                continue
            settled = int(r["settled"] or 0)
            out.append({
                "mode": mode or "ALL", "phase": "phase not recorded (row predates bets.game_type)",
                "game_type": None, "scored_in_competition": False,
                "strategies": len(self.store.latest_versions()),
                "bets": int(r["n"]), "settled": settled, "open": int(r["open"] or 0),
                "wins": int(r["wins"] or 0), "losses": settled - int(r["wins"] or 0),
                "pnl": round(float(r["pnl"] or 0), 2), "staked": round(float(r["staked"] or 0), 2),
                "roi": round(float(r["pnl"] or 0) / float(r["staked"]), 4) if r["staked"] else None,
                "win_rate": round(int(r["wins"] or 0) / settled, 4) if settled else None,
            })
        return out

    def post_settlement_analysis(self, bet_id: str) -> dict[str, Any]:
        """Brief section 46: explain each settled wager without over-claiming."""
        row = self.store.one("SELECT * FROM bets WHERE bet_id=?", (bet_id,))
        if row is None:
            return {}
        b = dict(row)
        bets = self.strategy_bets(b["strategy_id"], int(b["strategy_version"]))
        settled = [x for x in bets if x.get("result") in ("WIN", "LOSS")]
        lo, hi = wilson_interval(sum(1 for x in settled if x["result"] == "WIN"), len(settled))
        directional = (b["model_prob"] > 0.5) == (b["result"] == "WIN") if b.get("model_prob") else None
        notes = []
        if len(settled) < 30:
            notes.append(f"sample size {len(settled)} is far too small to judge the strategy")
        if b.get("clv") is not None:
            notes.append("closed better than entry (thesis supported by the market)"
                         if b["clv"] > 0 else
                         "closed worse than entry (the market disagreed with the thesis)")
        if b.get("edge") is not None and b.get("result") == "LOSS":
            notes.append("a loss on a positive-edge bet is expected variance, not evidence of "
                         "a broken model, at this sample size")
        return {
            "bet_id": bet_id, "result": b["result"], "pnl": b["pnl"],
            "model_prob": b["model_prob"], "price": b["price"], "edge": b["edge"],
            "clv": b["clv"], "directional_correct": directional,
            "strategy_n": len(settled),
            "strategy_win_rate_ci": [round(lo, 4), round(hi, 4)],
            "notes": notes,
            "recommend_modify": (len(settled) >= 100 and lo < 0.5 and hi < 0.55),
        }
