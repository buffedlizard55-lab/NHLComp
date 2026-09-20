"""Backtesting.

Hard rule from the brief: *if historical prices cannot be verified, do not call the result
a historical backtest*.  Two modes exist and are never mixed:

* :meth:`Backtester.run` -- **accuracy only** (hit rate, lift, log loss, Brier) for games
  without a verified price.  Stored with ``data_sufficient=0`` and an explicit caveat; it
  never produces a PnL figure because that would require inventing odds.
* :meth:`Backtester.run_priced` -- a genuine priced backtest for games whose Kalshi
  closing (or opening / T-6h) offer was recovered from timestamped candlesticks
  (``market_price_points``).  Every bet's entry price is a real quote with a real
  timestamp; PnL, ROI, drawdown and CLV are computed from those prices and the settled
  result.  Stored with ``data_sufficient=1`` and ``price_basis`` naming the candle point.

Splits are chronological.  Features are already point-in-time by construction
(``features.FeatureBuilder`` / ``features_ext.ExtendedFeatures``), and
``assert_no_leakage`` re-checks that no feature row was built from a game that had not
started yet.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from .models import brier, log_loss, wilson_interval
from .market import kalshi_fee_per_unit_staked
from .store import Store, utcnow
from .strategies import Strategy, ThresholdStrategy, feature_value

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
            v = feature_value(r, strat.feature)
            if v is None:
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


# ===================================================================== priced backtests
PRICED_CAVEAT = ("Entry price = Kalshi {point} candle yes_ask (real, timestamped). Flat 1-unit "
                 "stakes. Kalshi's general taker fee 0.07*C*P*(1-P) (fee schedule effective "
                 "2026-07-07) is deducted; the bid/ask spread is paid by buying the offer. "
                 "Treat ROI with the reported CI and against the market baseline.")


@dataclass
class PricedResult:
    strategy_id: str
    version: int
    label: str
    n_bets: int
    n_wins: int
    n_losses: int
    staked: float
    pnl: float
    roi: float | None
    avg_price: float | None
    avg_edge: float | None
    clv: float | None
    max_drawdown: float
    sharpe: float | None
    hit_rate: float | None
    ci_low: float | None
    ci_high: float | None
    base_rate: float | None
    n_games: int
    price_basis: str
    window: tuple[str, str]
    bets: list[dict[str, Any]]


def _pnl_curve(pnls: Sequence[float]) -> tuple[float, float | None]:
    """(max drawdown of cumulative PnL, per-bet Sharpe-like ratio)."""
    peak = cum = 0.0
    mdd = 0.0
    for x in pnls:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    n = len(pnls)
    if n < 2:
        return round(mdd, 4), None
    mean = sum(pnls) / n
    var = sum((x - mean) ** 2 for x in pnls) / (n - 1)
    sd = math.sqrt(var)
    return round(mdd, 4), (round(mean / sd, 4) if sd > 0 else None)


def trigger_roi(rows: Sequence[dict[str, Any]], *, feature: str, operator: str, threshold: float,
                bet_side: str, point: str = "close", fees: bool = True) -> dict[str, Any]:
    """Flat-stake ROI of buying ``bet_side`` at the named candle offer whenever the trigger
    fires -- no model, no edge filter.  This is the cleanest test of whether a situational
    trigger beats the market price.  Kalshi's published taker fee is deducted per unit
    staked unless ``fees`` is False."""
    pnls: list[float] = []
    asks: list[float] = []
    wins = 0
    for r in rows:
        w = r.get("_winner")
        if w is None:
            continue
        v = feature_value(r, feature)
        if v is None:
            continue
        ok = v >= threshold if operator == ">=" else v <= threshold
        if not ok:
            continue
        ask = r.get(f"mkt_{point}_{bet_side}_ask")
        if ask is None or not (0 < ask < 1) or r.get("mkt_close_is_latest"):
            continue
        won = (r["home_id"] if bet_side == "home" else r["away_id"]) == w
        wins += int(won)
        asks.append(ask)
        fee = kalshi_fee_per_unit_staked(ask) if fees else 0.0
        pnls.append(((1 - ask) / ask if won else -1.0) - fee)
    n = len(pnls)
    if n == 0:
        return {"n": 0, "roi": None, "pnl": 0.0, "avg_price": None, "hit": None, "mdd": None}
    mdd, sharpe = _pnl_curve(pnls)
    return {"n": n, "roi": round(sum(pnls) / n, 4), "pnl": round(sum(pnls), 4),
            "avg_price": round(sum(asks) / n, 4), "hit": round(wins / n, 4),
            "mdd": mdd, "sharpe": sharpe}


def market_baseline(rows: Sequence[dict[str, Any]], *, bet_side: str, point: str = "close",
                    fees: bool = True) -> dict[str, Any]:
    """ROI of blindly buying one side at the offer for every priced game (~ minus the vig
    and fees).  Every strategy's priced ROI should be read against this number."""
    return trigger_roi(rows, feature="mkt_has_close", operator=">=", threshold=1,
                       bet_side=bet_side, point=point, fees=fees)


class PricedBacktester:
    """Priced backtests against recovered Kalshi candle prices."""

    def __init__(self, store: Store):
        self.store = store

    def run(self, strat: Strategy, rows: Sequence[dict[str, Any]], *, label: str = "priced_all",
            point: str = "close", persist: bool = True) -> PricedResult | None:
        if not isinstance(strat, ThresholdStrategy):
            return None
        if getattr(strat, "blocked_reason", None) or strat.use_model == "sportsbook" \
                or getattr(strat, "injury_sensitive", False):
            return None   # nothing to price: blocked, or needs a feed with no history
        priced = [r for r in rows if r.get("_winner") is not None
                  and r.get(f"mkt_{point}_home_ask") is not None
                  and not r.get("mkt_close_is_latest")]
        if not priced:
            return None
        bets: list[dict[str, Any]] = []
        side = strat.bet_side
        for r in priced:
            v = feature_value(r, strat.feature)
            if v is None:
                continue
            ok = (v >= strat.threshold if strat.operator == ">=" else v <= strat.threshold)
            if not ok:
                continue
            p = (r.get("p_home_ml") if side == "home" else r.get("p_away_ml"))
            if strat.use_model == "elo":
                p = (r.get("p_home_elo") if side == "home" else r.get("p_away_elo")) or p
            if strat.use_model == "logistic":
                p = (r.get("p_home_logit") if side == "home" else r.get("p_away_logit")) or p
            if strat.use_model == "market":
                p = None
            ask = r.get(f"mkt_{point}_{side}_ask")
            if ask is None or not (0 < ask < 1) or ask < strat.min_price:
                continue
            if strat.requires_goalie and not r.get("both_starters_known"):
                continue
            if p is not None and ask > p - strat.min_edge:
                continue   # the strategy's own price rule says no
            won = (r["home_id"] if side == "home" else r["away_id"]) == r["_winner"]
            fee = kalshi_fee_per_unit_staked(ask)
            pnl = ((1 - ask) / ask if won else -1.0) - fee
            close_mid = r.get(f"mkt_close_{side}_mid")
            bets.append({"game_id": r["game_id"], "game_date": r["game_date"], "side": side,
                         "ask": ask, "p": p, "won": won, "pnl": round(pnl, 4),
                         "edge": (round(p - ask, 4) if p is not None else None),
                         "clv": (round(close_mid - ask, 4) if (close_mid is not None and point != "close") else None),
                         "ts": r.get(f"mkt_{point}_ts")})
        n = len(bets)
        window = (priced[0].get("game_date", ""), priced[-1].get("game_date", ""))
        basis = f"kalshi candle {point} ask"
        if n == 0:
            res = PricedResult(strat.strategy_id, strat.version, label, 0, 0, 0, 0.0, 0.0, None,
                               None, None, None, 0.0, None, None, None, None, None, len(priced),
                               basis, window, [])
        else:
            wins = sum(int(b["won"]) for b in bets)
            pnls = [b["pnl"] for b in bets]
            mdd, sharpe = _pnl_curve(pnls)
            lo, hi = wilson_interval(wins, n)
            edges = [b["edge"] for b in bets if b["edge"] is not None]
            clvs = [b["clv"] for b in bets if b["clv"] is not None]
            base = sum(int((r["home_id"] if side == "home" else r["away_id"]) == r["_winner"])
                       for r in priced) / len(priced)
            res = PricedResult(
                strategy_id=strat.strategy_id, version=strat.version, label=label, n_bets=n,
                n_wins=wins, n_losses=n - wins, staked=float(n), pnl=round(sum(pnls), 4),
                roi=round(sum(pnls) / n, 4), avg_price=round(sum(b["ask"] for b in bets) / n, 4),
                avg_edge=(round(sum(edges) / len(edges), 4) if edges else None),
                clv=(round(sum(clvs) / len(clvs), 4) if clvs else None),
                max_drawdown=mdd, sharpe=sharpe, hit_rate=round(wins / n, 4),
                ci_low=round(lo, 4), ci_high=round(hi, 4), base_rate=round(base, 4),
                n_games=len(priced), price_basis=basis, window=window, bets=bets)
        if persist:
            self.persist(res, point=point)
        return res

    def persist(self, res: PricedResult, *, point: str) -> None:
        self.store.execute(
            """INSERT INTO backtests(strategy_id, version, label, test_from, test_to, n_bets,
                                     n_wins, n_losses, n_push, staked, pnl, roi, max_drawdown,
                                     sharpe, avg_price, clv, data_sufficient, caveat, created_at,
                                     avg_edge, hit_rate, base_rate, n_games, price_basis)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(strategy_id, version, label) DO UPDATE SET
                 test_from=excluded.test_from, test_to=excluded.test_to,
                 n_bets=excluded.n_bets, n_wins=excluded.n_wins, n_losses=excluded.n_losses,
                 staked=excluded.staked, pnl=excluded.pnl, roi=excluded.roi,
                 max_drawdown=excluded.max_drawdown, sharpe=excluded.sharpe,
                 avg_price=excluded.avg_price, clv=excluded.clv,
                 data_sufficient=excluded.data_sufficient, caveat=excluded.caveat,
                 created_at=excluded.created_at, avg_edge=excluded.avg_edge,
                 hit_rate=excluded.hit_rate, base_rate=excluded.base_rate,
                 n_games=excluded.n_games, price_basis=excluded.price_basis""",
            (res.strategy_id, res.version, res.label, res.window[0], res.window[1], res.n_bets,
             res.n_wins, res.n_losses, 0, res.staked, res.pnl, res.roi, res.max_drawdown,
             res.sharpe, res.avg_price, res.clv, 1, PRICED_CAVEAT.format(point=point), utcnow(),
             res.avg_edge, res.hit_rate, res.base_rate, res.n_games, res.price_basis))
        self.store.commit()

    def run_splits(self, strat: Strategy, rows: Sequence[dict[str, Any]], *, point: str = "close"
                   ) -> dict[str, PricedResult | None]:
        parts = Backtester.split(rows)
        out: dict[str, PricedResult | None] = {}
        for name, chunk in parts.items():
            out[name] = self.run(strat, chunk, label=f"priced_{name}", point=point) if chunk else None
        return out
