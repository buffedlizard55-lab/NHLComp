"""Autonomous strategy discovery.

The engine asks the brief's question -- *what was known before the market moved, and could
it have predicted the outcome?* -- mechanically:

1. Enumerate every point-in-time feature actually present in the feature table.
2. Enumerate candidate thresholds for each, plus pairwise interactions.
3. Split the data chronologically (train -> validation -> held-out test).  No shuffling.
4. Score each candidate on the training window, keep only those that survive on the
   validation window, and report the number of hypotheses tested so that a lucky hit in a
   large search is visible as such.
5. Emit the survivors as *candidate* strategies with a recorded rationale, and emit the
   failures as findings -- a rejected hypothesis is data too.

Two scores are kept apart.  *Accuracy* (hit rate vs. base rate) is computed on every decided
game.  *Priced ROI* is computed only on games whose Kalshi closing candle was recovered --
flat stakes at the offer, net of the published taker fee -- and a market-family trigger
(one that *is* the price) is judged on price alone.  Survivors are CANDIDATES; only forward
tests against live quotes can promote them.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .backtest import trigger_roi
from .features import GameRef
from .models import brier, log_loss, wilson_interval
from .store import Store, utcnow
from .strategies import feature_value

# Features that are legitimate *situational* triggers.  Anything beginning with "_" or
# containing a future-looking key is excluded by construction.
SITUATIONAL_FEATURES = (
    "home_back_to_back", "away_back_to_back", "home_g3_in_4", "away_g3_in_4",
    "home_g4_in_6", "away_g4_in_6", "home_g5_in_7", "away_g5_in_7",
    "home_rest_days", "away_rest_days", "rest_diff",
    "home_prev_ot", "away_prev_ot", "home_prev_so", "away_prev_so",
    "home_only_b2b", "away_only_b2b", "both_b2b",
    "home_road_trip_len", "away_road_trip_len",
    "home_streak_wins", "away_streak_wins", "home_streak_losses", "away_streak_losses",
    "home_games_last_7", "away_games_last_7",
)

NUMERIC_FEATURES = (
    "home_gf_avg", "away_gf_avg", "home_ga_avg", "away_ga_avg",
    "home_gf_avg_l10", "away_gf_avg_l10", "home_ga_avg_l10", "away_ga_avg_l10",
    "home_total_avg", "away_total_avg", "pace_diff", "pace_sum",
    "home_win_pct_l10", "away_win_pct_l10", "home_season_points_pct", "away_season_points_pct",
    "home_home_win_pct", "home_ot_rate", "away_ot_rate",
    "home_exp_goals", "away_exp_goals",
)

# Special teams / shot quality / goalie features from the NHL stats REST per-game lines
# (features_ext.ExtendedFeatures).  All computed from games strictly before the game date.
STATS_FEATURES = (
    "home_pp_pct_l10", "away_pp_pct_l10", "home_pk_pct_l10", "away_pk_pct_l10",
    "home_shot_share_l10", "away_shot_share_l10", "home_pdo_l10", "away_pdo_l10",
    "home_fo_pct_l10", "away_fo_pct_l10", "diff_shot_share_l10", "diff_pdo_l10",
    "diff_pp_pct_l10", "diff_pk_pct_l10", "st_edge_home",
    "home_starter_sv_pct_l10", "away_starter_sv_pct_l10", "diff_starter_sv_pct_l10",
    "home_starter_b2b", "away_starter_b2b", "home_starter_starts_l7d", "away_starter_starts_l7d",
    "home_starter_rest_days", "away_starter_rest_days",
)

# Market-derived triggers.  ``mkt_close_*`` is the closing Kalshi line (known at decision
# time when the decision is made at the close); ``mkt_move*`` is the movement from the
# opening / T-24h / T-6h line to the close ("steam" = follow, "fade" = go against).
MARKET_FEATURES = (
    "mkt_close_home_mid", "mkt_move_home", "mkt_move24_home", "mkt_move6_home", "mkt_close_vig",
)

THRESHOLD_STEPS = {
    "home_rest_days": (0.5, 1.5, 2.5), "away_rest_days": (0.5, 1.5, 2.5),
    "rest_diff": (-2, -1, 1, 2), "home_road_trip_len": (2, 3, 4), "away_road_trip_len": (2, 3, 4),
    "home_streak_wins": (2, 3), "away_streak_wins": (2, 3),
    "home_streak_losses": (2, 3), "away_streak_losses": (2, 3),
    "home_games_last_7": (3, 4), "away_games_last_7": (3, 4),
    "pace_diff": (-0.5, 0.5), "home_gf_avg_l10": (2.8, 3.2, 3.6), "away_gf_avg_l10": (2.4, 2.8, 3.2),
    "home_win_pct_l10": (0.4, 0.6, 0.7), "away_win_pct_l10": (0.3, 0.4, 0.5),
    "home_exp_goals": (2.8, 3.2), "away_exp_goals": (2.4, 2.8),
    # stats-based
    "home_pp_pct_l10": (0.15, 0.25, 0.30), "away_pp_pct_l10": (0.15, 0.25, 0.30),
    "home_pk_pct_l10": (0.75, 0.80, 0.85), "away_pk_pct_l10": (0.75, 0.80, 0.85),
    "home_shot_share_l10": (0.47, 0.50, 0.53), "away_shot_share_l10": (0.47, 0.50, 0.53),
    "home_pdo_l10": (0.98, 1.00, 1.02), "away_pdo_l10": (0.98, 1.00, 1.02),
    "diff_shot_share_l10": (-0.04, -0.02, 0.02, 0.04), "diff_pdo_l10": (-0.03, 0.03),
    "diff_pp_pct_l10": (-0.08, 0.08), "diff_pk_pct_l10": (-0.08, 0.08), "st_edge_home": (-0.1, 0.1),
    "home_fo_pct_l10": (0.48, 0.52), "away_fo_pct_l10": (0.48, 0.52),
    "home_starter_sv_pct_l10": (0.890, 0.905, 0.920), "away_starter_sv_pct_l10": (0.890, 0.905, 0.920),
    "diff_starter_sv_pct_l10": (-0.02, -0.01, 0.01, 0.02),
    "home_starter_b2b": (1,), "away_starter_b2b": (1,),
    "home_starter_starts_l7d": (3, 4), "away_starter_starts_l7d": (3, 4),
    "home_starter_rest_days": (1, 4), "away_starter_rest_days": (1, 4),
    # market-based
    "mkt_close_home_mid": (0.35, 0.40, 0.45, 0.55, 0.60, 0.65),
    "mkt_move_home": (-0.06, -0.03, 0.03, 0.06), "mkt_move24_home": (-0.04, -0.02, 0.02, 0.04),
    "mkt_move6_home": (-0.03, -0.015, 0.015, 0.03), "mkt_close_vig": (0.02, 0.04),
}

# Families let the site and the findings explain *what kind* of signal a candidate is.
FEATURE_FAMILY = {}
for _f in SITUATIONAL_FEATURES:
    FEATURE_FAMILY[_f] = "schedule"
for _f in NUMERIC_FEATURES:
    FEATURE_FAMILY[_f] = "form"
for _f in STATS_FEATURES:
    FEATURE_FAMILY[_f] = "goalie" if "starter" in _f else "special_teams" if ("pp_" in _f or "pk_" in _f or "st_edge" in _f) else "shot_quality"
for _f in MARKET_FEATURES:
    FEATURE_FAMILY[_f] = "market"


#: Family-wise error rate for the whole discovery search.  A scan evaluates thousands of
#: triggers, so a per-test 5% threshold guarantees false survivors: at 3,000 tests on games
#: the market prices fairly, ~150 triggers clear 5% by chance alone.  Every p-value this
#: module reports is therefore judged against a threshold corrected for the number of tests
#: actually run (Holm's step-down procedure over the same family-wise rate), and a candidate
#: that does not clear it is still allowed to *forward test* -- collecting evidence is the
#: point of the forward test -- but no edge is claimed for it anywhere in this project.
FAMILY_ALPHA = 0.05


@dataclass
class Candidate:
    feature: str
    operator: str
    threshold: float
    bet_side: str
    rationale: str
    train_n: int = 0
    train_hit: float = 0.0
    train_lift: float = 0.0
    valid_n: int = 0
    valid_hit: float = 0.0
    valid_lift: float = 0.0
    verdict: str = "inconclusive"
    # priced evaluation (flat stake at the Kalshi closing offer); None when unpriced
    train_priced_n: int = 0
    train_roi: float | None = None
    valid_priced_n: int = 0
    valid_roi: float | None = None
    valid_avg_price: float | None = None
    price_verdict: str = "unpriced"      # market_edge | market_no_edge | unpriced | thin
    # --- multiple-testing bookkeeping (filled in by annotate_significance) -------------
    #: z and one-sided p for the validation hit rate against that split's own base rate
    valid_z: float | None = None
    valid_p: float | None = None
    #: z and one-sided p for the validation win rate against the average closing offer paid
    priced_z: float | None = None
    priced_p: float | None = None
    #: how many triggers the scan evaluated, and the threshold that number implies
    n_tests: int = 0
    alpha_family: float = FAMILY_ALPHA
    alpha_holm: float | None = None
    significant_accuracy: bool = False
    significant_priced: bool = False
    #: the sentence that goes on the record with the candidate, either way
    significance_note: str = "not yet assessed for multiple testing"

    @property
    def family(self) -> str:
        base = self.feature.split("+")[0]
        return FEATURE_FAMILY.get(base, "interaction" if "+" in self.feature else "other")

    @property
    def key(self) -> str:
        return f"{self.feature}|{self.operator}|{self.threshold}|{self.bet_side}"


def binomial_z(hits: float | None, n: int | None, p0: float | None) -> float | None:
    """One-sample z for a hit rate against a reference probability (normal approximation).

    Used two ways, both against a number that is *not* fitted from the same sample:

    * accuracy -- hits vs the base rate of the same split (did the trigger move the outcome?);
    * price -- wins vs the average closing offer that was paid (did it beat the market?).

    Returns None when the test is undefined (no trials, or a reference probability of 0/1).
    """
    if not n or n <= 0 or p0 is None or hits is None:
        return None
    p = float(p0)
    if p <= 0.0 or p >= 1.0:
        return None
    return (float(hits) - n * p) / math.sqrt(n * p * (1.0 - p))


def p_value_upper(z: float | None) -> float | None:
    """One-sided upper-tail p-value: P(Z >= z) for a standard normal."""
    if z is None:
        return None
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def holm_significant(pairs: Sequence[tuple[str, float | None]], *,
                     alpha: float = FAMILY_ALPHA) -> dict[str, dict[str, Any]]:
    """Holm's step-down correction over one family of tests.

    ``pairs`` is ``(key, p_value)`` for every test in the family -- including the ones that
    failed earlier gates is not possible here, so the family size is taken from
    ``DiscoveryEngine.tested`` by the caller passing every evaluated p-value plus the count of
    tests that produced none (see :meth:`DiscoveryEngine.annotate_significance`).  Sorted
    ascending, the i-th p-value must clear ``alpha / (m - i)`` to survive, and the procedure
    stops at the first failure, so nothing after it is declared significant either.

    Uniformly at least as powerful as Bonferroni while controlling the same family-wise error
    rate, and it needs nothing but the p-values -- no resampling, no extra assumptions.
    """
    scored = [(k, float(pv)) for k, pv in pairs if pv is not None]
    scored.sort(key=lambda kp: kp[1])
    m = len(pairs)
    out: dict[str, dict[str, Any]] = {}
    for k, _ in pairs:
        out[k] = {"significant": False, "holm_threshold": None, "rank": None}
    stopped = False
    for i, (k, pv) in enumerate(scored):
        threshold = alpha / (m - i)
        out[k]["rank"] = i + 1
        out[k]["holm_threshold"] = threshold
        out[k]["p_value"] = pv
        if stopped:
            continue
        if pv <= threshold:
            out[k]["significant"] = True
        else:
            stopped = True        # step-down: nothing weaker is significant either
    return out


def _fmt_roi(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def _fmt_p(v: float | None) -> str:
    return "n/a (test undefined)" if v is None else f"{v:.3g}"


def _split_rows(rows: Sequence[dict[str, Any]], train_frac: float = 0.6,
                valid_frac: float = 0.2) -> tuple[list, list, list]:
    """Chronological split.  Rows must already be sorted by time."""
    n = len(rows)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + valid_frac))
    return list(rows[:i1]), list(rows[i1:i2]), list(rows[i2:])


class DiscoveryEngine:
    def __init__(self, store: Store, *, min_train_n: int = 30, min_valid_n: int = 15,
                 min_lift: float = 0.02, verbose: bool = True, min_priced_n: int = 20,
                 min_roi: float = 0.02):
        self.store = store
        self.min_train_n = min_train_n
        self.min_valid_n = min_valid_n
        self.min_lift = min_lift
        self.min_priced_n = min_priced_n
        self.min_roi = min_roi
        self.verbose = verbose
        self.tested = 0
        self.priced_games = 0
        self.significance: dict[str, Any] = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[discovery] {msg}", flush=True)

    # ------------------------------------------------------------------ scan
    def _hit_rate(self, rows: Sequence[dict[str, Any]], feature: str, operator: str,
                  threshold: float, bet_side: str) -> tuple[int, float, float]:
        """Returns (n, hit_rate, base_rate) for games matching the trigger."""
        matched, hits, all_games = 0, 0, 0
        for r in rows:
            winner = r.get("_winner")
            if winner is None:
                continue
            all_games += 1
            v = feature_value(r, feature)
            if v is None:
                continue
            ok = v >= threshold if operator == ">=" else v <= threshold
            if not ok:
                continue
            matched += 1
            want = r["home_id"] if bet_side == "home" else r["away_id"]
            hits += int(winner == want)
        if matched == 0 or all_games == 0:
            return 0, 0.0, 0.0
        base = sum(int((r["home_id"] if bet_side == "home" else r["away_id"]) == r["_winner"])
                   for r in rows if r.get("_winner") is not None) / all_games
        return matched, hits / matched, base

    def scan(self, rows: Sequence[dict[str, Any]]) -> list[Candidate]:
        train, valid, _test = _split_rows(rows)
        candidates: list[Candidate] = []

        triggers: list[tuple[str, str, float, str, str]] = []
        for feat in SITUATIONAL_FEATURES:
            if feat in ("home_rest_days", "away_rest_days", "rest_diff",
                        "home_road_trip_len", "away_road_trip_len",
                        "home_streak_wins", "away_streak_wins",
                        "home_streak_losses", "away_streak_losses",
                        "home_games_last_7", "away_games_last_7"):
                continue  # handled with explicit thresholds below
            for bet_side in ("home", "away"):
                triggers.append((feat, ">=", 1, bet_side,
                                 f"{feat} == 1 identifies a specific schedule situation"))

        for feat, thresholds in THRESHOLD_STEPS.items():
            for thr in thresholds:
                for op in (">=", "<="):
                    for bet_side in ("home", "away"):
                        triggers.append((feat, op, float(thr), bet_side,
                                         f"{feat} {op} {thr} as a situational trigger"))

        # pairwise interactions between two binary situational flags
        bin_feats = [f for f in SITUATIONAL_FEATURES if f.endswith(("b2b", "_ot", "_so", "in_4",
                                                                    "in_6", "in_7"))]
        for a, b in itertools.combinations(bin_feats, 2):
            for bet_side in ("home", "away"):
                triggers.append((f"{a}+{b}", ">=", 2, bet_side,
                                 f"interaction: both {a} and {b} are true"))

        priced_train = [r for r in train if r.get("mkt_close_home_ask") is not None]
        priced_valid = [r for r in valid if r.get("mkt_close_home_ask") is not None]
        self.priced_games = len(priced_train) + len(priced_valid)

        for feat, op, thr, bet_side, why in triggers:
            self.tested += 1
            tn, thit, tbase = self._hit_rate(train, feat, op, thr, bet_side)
            is_market = feat.split("+")[0] in MARKET_FEATURES
            # Market triggers are judged on price only: a hit-rate lift is meaningless for
            # a feature that *is* the price.  Everything else must first show a lift.
            if not is_market:
                if tn < self.min_train_n:
                    continue
                if thit - tbase < self.min_lift:
                    continue
            vn, vhit, vbase = self._hit_rate(valid, feat, op, thr, bet_side)
            if not is_market and vn < self.min_valid_n:
                continue
            c = Candidate(feature=feat, operator=op, threshold=thr, bet_side=bet_side,
                          rationale=why, train_n=tn, train_hit=round(thit, 4),
                          train_lift=round(thit - tbase, 4), valid_n=vn,
                          valid_hit=round(vhit, 4), valid_lift=round(vhit - vbase, 4))
            if vhit - vbase >= self.min_lift:
                c.verdict = "edge"
            elif vhit >= tbase:
                c.verdict = "inconclusive"
            else:
                c.verdict = "no_edge"
            # priced evaluation at the closing offer
            if not is_market and vn > 0 and vbase > 0:
                c.valid_z = binomial_z(round(vhit * vn), vn, vbase)
                c.valid_p = p_value_upper(c.valid_z)
            if priced_train or priced_valid:
                tr = trigger_roi(priced_train, feature=feat, operator=op, threshold=thr, bet_side=bet_side)
                va = trigger_roi(priced_valid, feature=feat, operator=op, threshold=thr, bet_side=bet_side)
                c.train_priced_n, c.train_roi = tr["n"], tr["roi"]
                c.valid_priced_n, c.valid_roi, c.valid_avg_price = va["n"], va["roi"], va["avg_price"]
                if va.get("n") and va.get("hit") is not None and va.get("avg_price"):
                    # beating the closing offer: wins against the price that was paid
                    c.priced_z = binomial_z(round(va["hit"] * va["n"]), va["n"], va["avg_price"])
                    c.priced_p = p_value_upper(c.priced_z)
                if tr["n"] >= self.min_priced_n and va["n"] >= self.min_priced_n:
                    if tr["roi"] >= self.min_roi and va["roi"] >= self.min_roi:
                        c.price_verdict = "market_edge"
                    elif va["roi"] is not None and va["roi"] < 0:
                        c.price_verdict = "market_no_edge"
                    else:
                        c.price_verdict = "inconclusive"
                else:
                    c.price_verdict = "thin"
            if is_market and c.price_verdict in ("unpriced", "thin"):
                continue
            candidates.append(c)

        self.annotate_significance(candidates)
        candidates.sort(key=lambda c: (c.price_verdict != "market_edge",
                                       -(c.valid_roi or -9), -c.valid_lift, -c.valid_n))
        return candidates

    def annotate_significance(self, candidates: Sequence[Candidate]) -> dict[str, Any]:
        """Judge every surviving candidate against a threshold corrected for the whole search.

        The correction is applied *after* the scan, because the family size is the number of
        triggers evaluated -- not the number that survived the earlier gates.  Judging each
        candidate against a per-test 5% threshold while having tried thousands of triggers is
        how a project ends up with a leaderboard of noise, so the p-values here are Holm-
        adjusted over ``self.tested`` tests and the result is written onto the candidate
        whether or not it clears.

        Two independent tests are recorded per candidate, because they answer different
        questions and either can be the one that matters:

        * **accuracy** -- the validation hit rate against that split's own base rate;
        * **price** -- the validation win rate against the average closing offer actually
          paid, which is the test of whether the trigger beat the market rather than merely
          being right often.

        A candidate that clears neither is still promotable to FORWARD TEST -- that is what
        forward testing is for -- but ``significance_note`` says plainly that no edge is
        claimed, and every place that reports it carries the same sentence.
        """
        if not candidates:
            m = max(self.tested, 0)
            empty = {"n_tests": m, "alpha_family": FAMILY_ALPHA, "candidates": 0,
                     "significant_accuracy": 0, "significant_priced": 0,
                     "holm_threshold_strongest": (FAMILY_ALPHA / m) if m else None,
                     "holm_threshold_first": None}
            self.significance = empty
            return empty
        m = max(self.tested, len(candidates))
        acc = holm_significant([(f"acc:{c.key}", c.valid_p) for c in candidates]
                               + [(f"pad:{i}", None) for i in range(m - len(candidates))],
                               alpha=FAMILY_ALPHA)
        pri = holm_significant([(f"pri:{c.key}", c.priced_p) for c in candidates]
                               + [(f"ppd:{i}", None) for i in range(m - len(candidates))],
                               alpha=FAMILY_ALPHA)
        # the strongest threshold any test in this family could have faced; used whenever a
        # candidate's own rank is undefined (no p-value) so the note still quotes a number
        strongest = FAMILY_ALPHA / m
        n_acc = n_pri = 0
        for c in candidates:
            a = acc.get(f"acc:{c.key}", {})
            pr = pri.get(f"pri:{c.key}", {})
            c.n_tests = m
            c.alpha_family = FAMILY_ALPHA
            c.alpha_holm = a.get("holm_threshold") or pr.get("holm_threshold") or strongest
            c.significant_accuracy = bool(a.get("significant"))
            c.significant_priced = bool(pr.get("significant"))
            if c.valid_p is None and c.priced_p is None:
                # too few validation games, or no closing price was ever recovered for them:
                # there is no test to correct, and no verdict to correct either
                c.significance_note = (
                    f"No significance test could be computed for this candidate "
                    f"(p_accuracy={_fmt_p(c.valid_p)}, p_price={_fmt_p(c.priced_p)} over {m} "
                    f"triggers evaluated), so NO EDGE IS CLAIMED: it is forward-tested to "
                    "collect the evidence the validation split could not supply.")
            elif c.significant_accuracy or c.significant_priced:
                n_acc += int(c.significant_accuracy)
                n_pri += int(c.significant_priced)
                which = " and ".join(
                    [w for w, ok in (("the accuracy test", c.significant_accuracy),
                                     ("the price test", c.significant_priced)) if ok])
                c.significance_note = (
                    f"Clears {which} after Holm correction for {m} triggers "
                    f"(p_accuracy={_fmt_p(c.valid_p)}, p_price={_fmt_p(c.priced_p)}, "
                    f"threshold {c.alpha_holm:.2e}). Still CANDIDATE status: only forward-test "
                    "results can confirm it.")
            else:
                c.significance_note = (
                    f"Does NOT survive multiple-testing correction: {m} triggers were "
                    f"evaluated, so the Holm threshold is {c.alpha_holm:.2e} and this "
                    f"candidate's p-values are p_accuracy={_fmt_p(c.valid_p)}, "
                    f"p_price={_fmt_p(c.priced_p)}. It is forward-tested to collect evidence; "
                    "NO EDGE IS CLAIMED.")
        summary = {"n_tests": m, "alpha_family": FAMILY_ALPHA, "candidates": len(candidates),
                   "significant_accuracy": n_acc, "significant_priced": n_pri,
                   "holm_threshold_strongest": strongest,
                   "holm_threshold_first": next(
                       (c.alpha_holm for c in candidates if c.alpha_holm), None)}
        self.significance = summary
        self.log(f"discovery significance: {len(candidates)} candidate(s) judged against a Holm "
                 f"threshold for {m} tests; {n_acc} significant on accuracy, {n_pri} on price")
        return summary

    # ------------------------------------------------------------------ persist
    def record_hypotheses(self, rows: Sequence[dict[str, Any]]) -> int:
        n = 0
        seen = {r["hyp_id"] for r in self.store.query("SELECT hyp_id FROM hypotheses")}
        have_stats = any(r.get("home_n_stat_games") for r in rows[-200:]) if rows else False
        have_prices = any(r.get("mkt_close_home_ask") is not None for r in rows) if rows else False
        families = (
            (SITUATIONAL_FEATURES + NUMERIC_FEATURES, "NHL schedule + results (api-web.nhle.com)",
             "Situational/positional features are computable before puck drop from the NHL "
             "schedule and results alone, so they cannot leak future information.", True),
            (STATS_FEATURES, "NHL stats REST per-game team + goalie lines (api.nhle.com/stats/rest)",
             "Special-teams, shot-share, PDO and goalie workload/quality are computed from games "
             "strictly before the game date. Starter identity for past games comes from the "
             "post-game log (ASSUMPTION: it was public at the morning skate).", have_stats),
            (MARKET_FEATURES, "Kalshi candlesticks (historical + live tier)",
             "Closing line, line movement and vig are real timestamped exchange prices; the test "
             "is whether following or fading them earns more than the vig costs.", have_prices),
        )
        for feats, req, why, avail in families:
            for feat in feats:
                hid = f"HYP_{feat.upper()}"
                if hid in seen:
                    continue
                self.store.execute(
                    """INSERT INTO hypotheses(hyp_id, created_at, question, rationale, required_data,
                                              data_available, testable, status, origin)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (hid, utcnow(),
                     f"Does {feat} move the true win probability away from the market price?",
                     why, req, int(bool(avail)), int(bool(avail)), "proposed", "self_generated"),
                )
                n += 1
        self.store.commit()
        return n

    def promote(self, candidates: Sequence[Candidate], *, top: int = 12,
                min_valid_n: int = 25) -> list[dict[str, Any]]:
        """Turn surviving candidates into versioned candidate strategies."""
        from .strategies import ThresholdStrategy   # local import to avoid a cycle

        created: list[dict[str, Any]] = []
        # Priced survivors first (they beat the actual market on both splits), then
        # accuracy-only survivors that were never priced (no candle for their games).
        priced = [c for c in candidates if c.price_verdict == "market_edge"]
        unpriced = [c for c in candidates if c.verdict == "edge" and c.valid_n >= min_valid_n
                    and c.price_verdict in ("unpriced", "thin")]
        taken = (priced + unpriced)[:top]
        existing = {r["origin_ref"] for r in self.store.query(
            "SELECT origin_ref FROM strategies WHERE origin_ref IS NOT NULL")}
        # the same trigger may already exist as a seed strategy (e.g. away_g3_in_4 >= 1 ->
        # home is NHL_G3IN4); a generated twin would just double-count its wagers
        seen_rules: set[tuple] = set()
        for r in self.store.query("SELECT params_json FROM strategies WHERE params_json IS NOT NULL"):
            try:
                pj = json.loads(r["params_json"])
            except (TypeError, ValueError):
                continue
            seen_rules.add((pj.get("feature"), pj.get("operator"), float(pj.get("threshold") or 0),
                            pj.get("bet_side")))
        for i, c in enumerate(taken, start=1):
            ref = f"discovery:{c.key}"
            if ref in existing:
                continue   # identical trigger already exists as a version; never duplicate
            if (c.feature, c.operator, float(c.threshold), c.bet_side) in seen_rules:
                continue   # a seed strategy already tests exactly this rule
            base_id = "NHL_GEN_" + c.feature.replace("+", "_AND_").upper()[:28]
            row = self.store.one(
                "SELECT MAX(version) AS v FROM strategies WHERE strategy_id=?", (base_id,))
            version = int(row["v"] or 0) + 1
            is_market = c.family == "market"
            strat = ThresholdStrategy(
                strategy_id=base_id, version=version,
                username=f"{base_id}_{version:03d}",
                name=f"Generated ({c.family}): {c.feature} {c.operator} {c.threshold} -> {c.bet_side}",
                category=f"generated_{c.family}", feature=c.feature, operator=c.operator,
                threshold=c.threshold, bet_side=c.bet_side, origin="self_generated",
                origin_ref=ref, min_edge=0.0 if is_market else 0.04,
                use_model="market" if is_market else "poisson",
                requires_goalie=("starter" in c.feature),
            )
            d = strat.describe()
            roi_txt = ""
            if c.price_verdict != "unpriced":
                roi_txt = (f" Priced at the Kalshi close: train ROI {_fmt_roi(c.train_roi)} "
                           f"(n={c.train_priced_n}), validation ROI {_fmt_roi(c.valid_roi)} "
                           f"(n={c.valid_priced_n}, avg price {c.valid_avg_price}). "
                           f"Verdict: {c.price_verdict}.")
            # the multiple-testing verdict travels with the candidate, in its own words, so
            # a reader of the strategy page sees the corrected p-values next to the lift
            # rather than having to take the promotion on trust
            d["hypothesis"] = (c.rationale + f" Train hit {c.train_hit:.3f} "
                               f"(+{c.train_lift:.3f}, n={c.train_n}); validation hit "
                               f"{c.valid_hit:.3f} (+{c.valid_lift:.3f}, n={c.valid_n})." + roi_txt
                               + f" MULTIPLE TESTING: {c.significance_note}")
            d["status"] = "candidate"
            d["created_at"] = utcnow()
            d["test_mode"] = "FORWARD TEST"
            d["leakage_checked"] = 1
            self.store.upsert_strategy(d)
            self.store.execute(
                """INSERT INTO experiments(exp_id, created_at, hypothesis, strategy_id, version,
                                           kind, payload_json, result_json, conclusion, verdict)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (f"EXP_{base_id}_v{version}", utcnow(), d["hypothesis"], base_id, version,
                 "feature_scan", d["params_json"],   # already a JSON blob from describe()
                 json.dumps({"train": [c.train_n, c.train_hit, c.train_lift],
                             "valid": [c.valid_n, c.valid_hit, c.valid_lift],
                             "priced": {"train_n": c.train_priced_n, "train_roi": c.train_roi,
                                        "valid_n": c.valid_priced_n, "valid_roi": c.valid_roi,
                                        "verdict": c.price_verdict},
                             "family": c.family,
                             "hypotheses_tested": self.tested, "priced_games": self.priced_games,
                             "multiple_testing": {
                                 "n_tests": c.n_tests, "alpha_family": c.alpha_family,
                                 "holm_threshold": c.alpha_holm,
                                 "z_accuracy": c.valid_z, "p_accuracy": c.valid_p,
                                 "z_price": c.priced_z, "p_price": c.priced_p,
                                 "significant_accuracy": c.significant_accuracy,
                                 "significant_priced": c.significant_priced,
                                 "note": c.significance_note}}),
                 (("Survived chronological train->validation split at the real closing price"
                   if c.price_verdict == "market_edge" else
                   "Survived chronological train->validation split (accuracy only; unpriced)")
                  + ("" if (c.significant_accuracy or c.significant_priced) else
                     "; does NOT survive Holm correction for the size of the search, so it is "
                     "forward-tested for evidence and no edge is claimed")),
                 c.price_verdict if c.price_verdict == "market_edge" else "edge"),
            )
            created.append(d)
            existing.add(ref)
        self.store.commit()
        return created

    def record_failures(self, candidates: Sequence[Candidate]) -> int:
        n = 0
        import hashlib
        for c in candidates:
            if c.verdict == "edge" and c.price_verdict not in ("market_no_edge",):
                continue
            fid = "FIND_FAIL_" + hashlib.sha1(c.key.encode()).hexdigest()[:12]
            if c.price_verdict == "market_no_edge":
                title = (f"Beats the base rate but not the price: {c.feature} {c.operator} "
                         f"{c.threshold} ({c.bet_side})")
                body = (f"Hit-rate lift train {c.train_lift:+.3f} / valid {c.valid_lift:+.3f}, but "
                        f"buying at the Kalshi closing offer returned {_fmt_roi(c.train_roi)} on train "
                        f"(n={c.train_priced_n}) and {_fmt_roi(c.valid_roi)} on validation "
                        f"(n={c.valid_priced_n}). The market already prices this situation.")
            else:
                title = f"No reproducible edge: {c.feature} {c.operator} {c.threshold} ({c.bet_side})"
                body = (f"Train hit rate {c.train_hit:.3f} (lift {c.train_lift:+.3f}, n={c.train_n}) did "
                        f"not reproduce on validation ({c.valid_hit:.3f}, lift {c.valid_lift:+.3f}, "
                        f"n={c.valid_n}). Rejected rather than tuned further.")
            self.store.execute(
                """INSERT OR IGNORE INTO findings(finding_id, created_at, title, body, evidence,
                                                  confidence, kind) VALUES(?,?,?,?,?,?,?)""",
                (fid, utcnow(), title, body,
                 json.dumps({"train": [c.train_n, c.train_hit], "valid": [c.valid_n, c.valid_hit],
                             "priced": [c.train_priced_n, c.train_roi, c.valid_priced_n, c.valid_roi],
                             "family": c.family}),
                 "medium", "rejection"),
            )
            n += 1
        self.store.commit()
        return n

    def record_candidate_outcomes(self, candidates: Sequence[Candidate]) -> int:
        """One experiments row per scanned trigger, whatever the outcome.

        ``promote`` records the survivors and ``record_failures`` records the loud
        rejections; a candidate that neither survived nor failed loudly (verdict
        'inconclusive', or a thin priced sample) used to leave no ledger trace at all --
        the search silently narrowed to whoever was reading the logs.  Requirement 8 wants
        every experiment permanently auditable, so the full scan lands in ``experiments``
        with its train/validation numbers, its Holm-adjusted significance and its verdict.
        Idempotent: the key is a hash of the trigger, so a re-run updates nothing.
        """
        import hashlib
        n = 0
        for c in candidates:
            exp_id = "EXP_SCAN_" + hashlib.sha1(c.key.encode()).hexdigest()[:12]
            verdict = ("edge" if c.verdict == "edge" and c.price_verdict == "market_edge"
                       else "rejected" if c.verdict == "edge"
                       else "no_edge" if c.verdict == "no_edge" else "inconclusive")
            conclusion = ("Survived the chronological split but not the price test."
                          if verdict == "rejected"
                          else f"{c.verdict} on the train->validation split"
                          if verdict in ("no_edge", "inconclusive")
                          else "Survived train->validation and the priced test")
            cur = self.store.execute(
                """INSERT OR IGNORE INTO experiments(exp_id, created_at, hypothesis,
                                                     strategy_id, version, kind,
                                                     payload_json, result_json,
                                                     conclusion, verdict)
                    VALUES(?,?,?,NULL,NULL,'feature_scan',?,?,?,?)""",
                (exp_id, utcnow(), c.key,
                 json.dumps({"operator": c.operator, "threshold": c.threshold,
                             "bet_side": c.bet_side, "family": c.family}),
                 json.dumps({"train": [c.train_n, round(c.train_hit, 4), round(c.train_lift, 4)],
                             "valid": [c.valid_n, round(c.valid_hit, 4), round(c.valid_lift, 4)],
                             "priced": [c.train_priced_n,
                                        round(c.train_roi, 4) if c.train_roi is not None else None,
                                        c.valid_priced_n,
                                        round(c.valid_roi, 4) if c.valid_roi is not None else None],
                             "valid_p": c.valid_p, "alpha_holm": c.alpha_holm,
                             "price_verdict": c.price_verdict}),
                 conclusion, verdict))
            n += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        self.store.commit()
        return n

    def search_size_caveat(self) -> str:
        sig = self.significance or {}
        thr = sig.get("holm_threshold_first") or sig.get("holm_threshold_strongest")
        return (f"{self.tested} candidate triggers were evaluated ({self.priced_games} games had "
                f"a recovered Kalshi closing price). With that many tests a per-test 5% "
                f"threshold guarantees that some triggers clear it by chance alone, so every "
                f"p-value is judged against "
                f"Holm's step-down correction for a {FAMILY_ALPHA:.0%} family-wise error rate"
                + (f" (the strongest survivor had to clear p <= {thr:.2e})" if thr else "")
                + f". {sig.get('significant_accuracy', 0)} candidate(s) clear it on accuracy and "
                f"{sig.get('significant_priced', 0)} on price. Everything promoted is labelled "
                f"CANDIDATE, carries its own corrected p-values, and only forward-test results "
                f"can promote it; a candidate that does not clear the corrected threshold is "
                f"forward-tested to collect evidence and NO EDGE IS CLAIMED for it.")
