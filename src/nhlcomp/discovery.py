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

    @property
    def family(self) -> str:
        base = self.feature.split("+")[0]
        return FEATURE_FAMILY.get(base, "interaction" if "+" in self.feature else "other")

    @property
    def key(self) -> str:
        return f"{self.feature}|{self.operator}|{self.threshold}|{self.bet_side}"


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
            if priced_train or priced_valid:
                tr = trigger_roi(priced_train, feature=feat, operator=op, threshold=thr, bet_side=bet_side)
                va = trigger_roi(priced_valid, feature=feat, operator=op, threshold=thr, bet_side=bet_side)
                c.train_priced_n, c.train_roi = tr["n"], tr["roi"]
                c.valid_priced_n, c.valid_roi, c.valid_avg_price = va["n"], va["roi"], va["avg_price"]
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

        candidates.sort(key=lambda c: (c.price_verdict != "market_edge",
                                       -(c.valid_roi or -9), -c.valid_lift, -c.valid_n))
        return candidates

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
        for i, c in enumerate(taken, start=1):
            ref = f"discovery:{c.key}"
            if ref in existing:
                continue   # identical trigger already exists as a version; never duplicate
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
                roi_txt = (f" Priced at the Kalshi close: train ROI {c.train_roi:+.3f} "
                           f"(n={c.train_priced_n}), validation ROI {c.valid_roi:+.3f} "
                           f"(n={c.valid_priced_n}, avg price {c.valid_avg_price}). "
                           f"Verdict: {c.price_verdict}.")
            d["hypothesis"] = (c.rationale + f" Train hit {c.train_hit:.3f} "
                               f"(+{c.train_lift:.3f}, n={c.train_n}); validation hit "
                               f"{c.valid_hit:.3f} (+{c.valid_lift:.3f}, n={c.valid_n})." + roi_txt)
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
                             "hypotheses_tested": self.tested, "priced_games": self.priced_games}),
                 ("Survived chronological train->validation split at the real closing price"
                  if c.price_verdict == "market_edge" else
                  "Survived chronological train->validation split (accuracy only; unpriced)"),
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
                        f"buying at the Kalshi closing offer returned {c.train_roi:+.3f} on train "
                        f"(n={c.train_priced_n}) and {c.valid_roi:+.3f} on validation "
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

    def search_size_caveat(self) -> str:
        return (f"{self.tested} candidate triggers were evaluated ({self.priced_games} games had a "
                f"recovered Kalshi closing price). With that many tests, a few will clear the "
                f"threshold by chance alone; survivors are labelled CANDIDATE and only "
                f"forward-test results can promote them.")
