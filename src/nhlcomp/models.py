"""Prediction models.  Deliberately a spread of complexities, including trivial baselines,
because the brief insists complexity must earn its place rather than be assumed.

All models are strictly causal: ratings are updated game-by-game in chronological order
and only ever consume games that finished before the game being predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .features import GameRef


# --------------------------------------------------------------------- metrics
def log_loss(probs: Sequence[float], labels: Sequence[int], eps: float = 1e-9) -> float:
    if not probs:
        return float("nan")
    tot = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, eps), 1 - eps)
        tot += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return tot / len(probs)


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def calibration(probs: Sequence[float], labels: Sequence[int], bins: int = 10
                ) -> list[dict[str, float]]:
    out = []
    edges = [i / bins for i in range(bins + 1)]
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [j for j, p in enumerate(probs)
               if (lo <= p < hi) or (i == bins - 1 and p == 1.0)]
        if not idx:
            continue
        out.append({"bin_low": lo, "bin_high": hi, "n": len(idx),
                    "mean_prob": sum(probs[j] for j in idx) / len(idx),
                    "hit_rate": sum(labels[j] for j in idx) / len(idx)})
    return out


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Used everywhere in the UI so nobody reads a 3-bet win rate as signal."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------- Elo
@dataclass
class EloModel:
    k: float = 24.0
    home_adv: float = 55.0
    base: float = 1500.0
    margin_mult: float = 0.0
    ratings: dict[int, float] = field(default_factory=dict)

    def rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.base)

    def prob_home(self, home_id: int, away_id: int) -> float:
        diff = self.rating(home_id) - self.rating(away_id) + self.home_adv
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def observe(self, g: GameRef) -> None:
        """Update ratings using a finished game.  Must be called in chronological order."""
        if not g.decided or g.winner is None:
            return
        ph = self.prob_home(g.home_id, g.away_id)
        actual = 1.0 if g.winner == g.home_id else 0.0
        mult = 1.0
        if self.margin_mult:
            mov = abs((g.home_score or 0) - (g.away_score or 0))
            mult = math.log(max(mov, 1) + 1) * self.margin_mult + 1.0
        delta = self.k * mult * (actual - ph)
        self.ratings[g.home_id] = self.rating(g.home_id) + delta
        self.ratings[g.away_id] = self.rating(g.away_id) - delta

    def prior_from_points(self, points_pct: dict[int, float], *, scale: float = 400.0) -> None:
        """Seed ratings from a *previous* season's points percentage.

        Using a prior season is not leakage for the following season; using the same
        season's final standings would be, and ``features`` forbids that separately.
        """
        for team_id, pct in points_pct.items():
            self.ratings[team_id] = self.base + (pct - 0.5) * scale


# --------------------------------------------------------------------- Poisson
def poisson_pmf(lam: float, k: int) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


@dataclass
class PoissonModel:
    """Independent-Poisson goal model with an explicit OT/shootout mass.

    Returns the probability the *home team wins including overtime* as well as the
    regulation-win and total-goals distributions, which is what the moneyline,
    regulation-win and totals markets actually need.
    """
    league_home_gf: float = 2.9
    league_away_gf: float = 2.7
    max_goals: int = 12
    ot_share: float = 0.0  # fitted: fraction of ties pushed to OT

    def expected_goals(self, f: dict[str, Any]) -> tuple[float, float]:
        hg = f.get("home_exp_goals")
        ag = f.get("away_exp_goals")
        if hg is None or ag is None:
            return self.league_home_gf, self.league_away_gf
        # shrink toward the league average when the sample behind the feature is thin
        n = min(f.get("home_n_prior") or 0, f.get("away_n_prior") or 0)
        w = n / (n + 10.0)
        return (w * hg + (1 - w) * self.league_home_gf,
                w * ag + (1 - w) * self.league_away_gf)

    def score_matrix(self, lh: float, la: float) -> list[list[float]]:
        return [[poisson_pmf(lh, i) * poisson_pmf(la, j)
                 for j in range(self.max_goals + 1)] for i in range(self.max_goals + 1)]

    def predict(self, f: dict[str, Any]) -> dict[str, float]:
        lh, la = self.expected_goals(f)
        m = self.score_matrix(lh, la)
        p_home_reg = sum(m[i][j] for i in range(self.max_goals + 1)
                         for j in range(self.max_goals + 1) if i > j)
        p_away_reg = sum(m[i][j] for i in range(self.max_goals + 1)
                         for j in range(self.max_goals + 1) if j > i)
        p_tie = 1.0 - p_home_reg - p_away_reg
        # ties resolve in OT; NHL OT is close to a coin flip with a small home edge
        p_home_ot = p_tie * 0.55
        p_total = {t: sum(m[i][j] for i in range(self.max_goals + 1)
                          for j in range(self.max_goals + 1) if i + j == t)
                   for t in range(0, 2 * self.max_goals + 1)}
        # the truncated score matrix loses a little probability mass beyond max_goals; renormalize
        # so the over/under split is a proper partition instead of summing to 0.99998
        tot = sum(p_total.values())
        if tot > 0:
            p_total = {t: v / tot for t, v in p_total.items()}
        mean_total = lh + la
        return {
            "lam_home": lh, "lam_away": la,
            "p_home_reg": p_home_reg, "p_away_reg": p_away_reg, "p_tie_reg": p_tie,
            "p_home_ml": p_home_reg + p_home_ot,
            "p_away_ml": p_away_reg + (p_tie - p_home_ot),
            "p_overtime": p_tie,
            "p_over55": sum(v for t, v in p_total.items() if t >= 6),
            "p_under55": sum(v for t, v in p_total.items() if t <= 5),
            "mean_total": mean_total,
        }


# --------------------------------------------------------------------- baselines
class HomeIceOnly:
    """The cheapest possible baseline: a single fitted home-win constant."""

    def __init__(self, p_home: float = 0.55):
        self.p_home = p_home

    def predict(self, f: dict[str, Any]) -> dict[str, float]:
        return {"p_home_ml": self.p_home, "p_away_ml": 1 - self.p_home}


@dataclass
class LogisticRest:
    """Tiny logistic regression on a handful of interpretable features.

    Implemented from scratch (no numpy dependency) with plain gradient descent so the
    whole project runs on the standard library in CI.
    """
    feature_names: tuple[str, ...] = ("elo_diff", "rest_diff", "home_only_b2b", "away_only_b2b")
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    lr: float = 0.05
    epochs: int = 200
    l2: float = 1e-3

    def _vec(self, row: dict[str, Any]) -> dict[str, float]:
        return {k: float(row.get(k) or 0.0) for k in self.feature_names}

    def predict_p(self, row: dict[str, Any]) -> float:
        z = self.intercept + sum(self.weights.get(k, 0.0) * v for k, v in self._vec(row).items())
        return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))

    def fit(self, rows: Sequence[dict[str, Any]], labels: Sequence[int]) -> "LogisticRest":
        self.weights = {k: 0.0 for k in self.feature_names}
        self.intercept = 0.0
        if not rows:
            return self
        n = len(rows)
        for _ in range(self.epochs):
            grad = {k: 0.0 for k in self.feature_names}
            gi = 0.0
            for row, y in zip(rows, labels):
                p = self.predict_p(row)
                err = p - y
                vec = self._vec(row)
                for k in self.feature_names:
                    grad[k] += err * vec[k]
                gi += err
            for k in self.feature_names:
                self.weights[k] -= self.lr * (grad[k] / n + self.l2 * self.weights[k])
            self.intercept -= self.lr * gi / n
        return self


def decimal_to_prob(price: float) -> float:
    return 1.0 / price if price and price > 1 else float("nan")


def prob_to_decimal(p: float) -> float:
    return 1.0 / p if p and p > 0 else float("nan")


def american_to_decimal(odds: int) -> float:
    return 1 + 100 / odds if odds > 0 else 1 + abs(odds) / 100
