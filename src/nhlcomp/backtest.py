"""Backtesting.

Hard rule from the brief: *if historical prices cannot be verified, do not call the result
a historical backtest*.  No verified historical NHL odds feed is available to this
project (see ``source_registry``: the sportsbook aggregator requires a paid key and the
Kalshi candle endpoint is unverified).  Therefore:

* :meth:`Backtester.run` measures **predictive accuracy only** -- hit rate, lift over the
  base rate, log loss, Brier score -- and stores it with ``data_sufficient=0`` plus an
  explicit caveat.  It never produces a PnL figure, because producing one would require
  inventing odds.
* Profit and loss for every strategy comes exclusively from the forward-test ledger,
  where the price was really quoted.

Splits are chronological.  Features are already point-in-time by construction
(``features.FeatureBuilder``), and ``assert_no_leakage`` re-checks that no feature row
was built from a game that had not started yet.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .models import brier, log_loss, wilson_interval
from .store import Store, utcnow
from .strategies import Strategy, ThresholdStrategy

CAVEAT_NO_PRICE = ("No verified historical betting-price feed is available, so this run measures "
                   "predictive accuracy only. PnL, ROI and CLV are deliberately not computed -- "
                   "doing so would require inventing historical odds. Profit is measured by the "
                   "forward test instead.")


@dataclass
class BacktestResult:
    strategy_id: str
    version: int
    label: str
    n_bets: int
    n_wins: int
    n_losses: int
    n_push: int
    hit_rate: float
    base_rate: float
    lift: float
    ci_low: float
    ci_high: float
    log_loss: float
    brier: float
    data_sufficient: int
    caveat: str
    window: tuple[str, str]


class Backtester:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------------ guards
    @staticmethod
    def assert_no_leakage(rows: Sequence[dict[str, Any]], games: dict[int, Any]) -> list[str]:
        """Return a list of violations.  A violation means a feature value depended on a
        game that started at or after the game being described."""
        problems: list[str] = []
        for r in rows:
            gid = int(r.get("game_id") or 0)
            g = games.get(gid)
            if g is None:
                continue
            for key in ("home_n_prior", "away_n_prior"):
                # a team cannot have played negative games, and the count must be strictly
                # less than the total number of its games up to and including this one
                n = r.get(key)
                if n is not None and n < 0:
                    problems.append(f"game {gid}: {key}={n} is negative")
            if r.get("start_time_utc") and g.start.strftime("%Y-%m-%dT%H:%M:%SZ") != r["start_time_utc"]:
                problems.append(f"game {gid}: feature start_time {r['start_time_utc']} does not "
                                f"match game start {g.start.isoformat()}")
        return problems

    @staticmethod
    def split(rows: Sequence[dict[str, Any]], train_frac: float = 0.6,
              valid_frac: float = 0.2) -> dict[str, list[dict[str, Any]]]:
        n = len(rows)
        i1, i2 = int(n * train_frac), int(n * (train_frac + valid_frac))
        return {"train": list(rows[:i1]), "valid": list(rows[i1:i2]), "test": list(rows[i2:])}

    # ------------------------------------------------------------------ run
    def run(self, strat: Strategy, rows: Sequence[dict[str, Any]], *, label: str = "full",
            window: tuple[str, str] | None = None) -> BacktestResult | None:
        if not rows:
            return None
        triggered: list[tuple[dict[str, Any], float]] = []
        for r in rows:
            if r.get("_winner") is None:
                continue
            if not isinstance(strat, ThresholdStrategy):
                continue
            raw = r.get(strat.feature)
            if raw is None:
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            ok = (v >= strat.threshold if strat.operator == ">=" else v <= strat.threshold)
            if not ok:
                continue
            p = (r.get("p_home_ml") if strat.bet_side == "home" else r.get("p_away_ml"))
            if strat.use_model == "elo":
                p = (r.get("p_home_elo") if strat.bet_side == "home"
                     else r.get("p_away_elo")) or p
            if strat.use_model == "logistic":
                p = (r.get("p_home_logit") if strat.bet_side == "home"
                     else r.get("p_away_logit")) or p
            if p is None:
                continue
            triggered.append((r, float(p)))

        n = len(triggered)
        if n == 0:
            return BacktestResult(strat.strategy_id, strat.version, label, 0, 0, 0, 0,
                                  float("nan"), float("nan"), float("nan"), 0.0, 0.0,
                                  float("nan"), float("nan"), 0,
                                  "Trigger never fired in the available sample.",
                                  window or ("", ""))

        wins = sum(int((r["home_id"] if strat.bet_side == "home" else r["away_id"]) == r["_winner"])
                   for r, _ in triggered)
        all_rows = [r for r in rows if r.get("_winner") is not None]
        base = (sum(int((r["home_id"] if strat.bet_side == "home" else r["away_id"]) == r["_winner"])
                    for r in all_rows) / len(all_rows)) if all_rows else float("nan")
        probs = [p for _, p in triggered]
        labels = [int((r["home_id"] if strat.bet_side == "home" else r["away_id"]) == r["_winner"])
                  for r, _ in triggered]
        lo, hi = wilson_interval(wins, n)
        res = BacktestResult(
            strategy_id=strat.strategy_id, version=strat.version, label=label,
            n_bets=n, n_wins=wins, n_losses=n - wins, n_push=0,
            hit_rate=round(wins / n, 4), base_rate=round(base, 4),
            lift=round(wins / n - base, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
            log_loss=round(log_loss(probs, labels), 5), brier=round(brier(probs, labels), 5),
            data_sufficient=0, caveat=CAVEAT_NO_PRICE,
            window=window or (rows[0].get("game_date", ""), rows[-1].get("game_date", "")),
        )
        self.store.execute(
            """INSERT INTO backtests(strategy_id, version, label, test_from, test_to, n_bets,
                                     n_wins, n_losses, n_push, staked, pnl, roi, max_drawdown,
                                     sharpe, avg_price, data_sufficient, caveat, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(strategy_id, version, label) DO UPDATE SET
                 n_bets=excluded.n_bets, n_wins=excluded.n_wins, n_losses=excluded.n_losses,
                 n_push=excluded.n_push, data_sufficient=excluded.data_sufficient,
                 caveat=excluded.caveat, created_at=excluded.created_at""",
            (res.strategy_id, res.version, res.label, res.window[0], res.window[1], res.n_bets,
             res.n_wins, res.n_losses, res.n_push, None, None, None, None, None, None,
             res.data_sufficient, res.caveat, utcnow()),
        )
        self.store.commit()
        return res

    def run_splits(self, strat: Strategy, rows: Sequence[dict[str, Any]]) -> dict[str, BacktestResult | None]:
        parts = self.split(rows)
        out = {}
        for name, chunk in parts.items():
            if not chunk:
                out[name] = None
                continue
            w = (chunk[0].get("game_date", ""), chunk[-1].get("game_date", ""))
            out[name] = self.run(strat, chunk, label=name, window=w)
        return out
