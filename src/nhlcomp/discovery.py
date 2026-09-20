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

Nothing here is allowed to see a price, because no verified historical price feed exists.
Discovery therefore scores predictive accuracy, not profit; profit only ever comes from
forward tests against real quotes.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .features import GameRef
from .models import brier, log_loss, wilson_interval
from .store import Store, utcnow

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

THRESHOLD_STEPS = {
    "home_rest_days": (0.5, 1.5, 2.5), "away_rest_days": (0.5, 1.5, 2.5),
    "rest_diff": (-2, -1, 1, 2), "home_road_trip_len": (2, 3, 4), "away_road_trip_len": (2, 3, 4),
    "home_streak_wins": (2, 3), "away_streak_wins": (2, 3),
    "home_streak_losses": (2, 3), "away_streak_losses": (2, 3),
    "home_games_last_7": (3, 4), "away_games_last_7": (3, 4),
    "pace_diff": (-0.5, 0.5), "home_gf_avg_l10": (2.8, 3.2, 3.6), "away_gf_avg_l10": (2.4, 2.8, 3.2),
    "home_win_pct_l10": (0.4, 0.6, 0.7), "away_win_pct_l10": (0.3, 0.4, 0.5),
    "home_exp_goals": (2.8, 3.2), "away_exp_goals": (2.4, 2.8),
}


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
                 min_lift: float = 0.02, verbose: bool = True):
        self.store = store
        self.min_train_n = min_train_n
        self.min_valid_n = min_valid_n
        self.min_lift = min_lift
        self.verbose = verbose
        self.tested = 0

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
            v = r.get(feature)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
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

        for feat, op, thr, bet_side, why in triggers:
            self.tested += 1
            tn, thit, tbase = self._hit_rate(train, feat, op, thr, bet_side)
            if tn < self.min_train_n:
                continue
            if thit - tbase < self.min_lift:
                continue
            vn, vhit, vbase = self._hit_rate(valid, feat, op, thr, bet_side)
            if vn < self.min_valid_n:
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
            candidates.append(c)

        candidates.sort(key=lambda c: (-c.valid_lift, -c.valid_n))
        return candidates

    # ------------------------------------------------------------------ persist
    def record_hypotheses(self, rows: Sequence[dict[str, Any]]) -> int:
        n = 0
        seen = {r["hyp_id"] for r in self.store.query("SELECT hyp_id FROM hypotheses")}
        for feat in SITUATIONAL_FEATURES + NUMERIC_FEATURES:
            hid = f"HYP_{feat.upper()}"
            if hid in seen:
                continue
            self.store.execute(
                """INSERT INTO hypotheses(hyp_id, created_at, question, rationale, required_data,
                                          data_available, testable, status, origin)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (hid, utcnow(),
                 f"Does {feat} move the true win probability away from the market price?",
                 "Situational/positional features are computable before puck drop from the NHL "
                 "schedule and results alone, so they cannot leak future information.",
                 "NHL schedule + results (api-web.nhle.com)", 1, 1, "proposed", "self_generated"),
            )
            n += 1
        self.store.commit()
        return n

    def promote(self, candidates: Sequence[Candidate], *, top: int = 12,
                min_valid_n: int = 25) -> list[dict[str, Any]]:
        """Turn surviving candidates into versioned candidate strategies."""
        from .strategies import ThresholdStrategy   # local import to avoid a cycle

        created: list[dict[str, Any]] = []
        taken = [c for c in candidates if c.verdict == "edge" and c.valid_n >= min_valid_n][:top]
        for i, c in enumerate(taken, start=1):
            base_id = "NHL_GEN_" + c.feature.replace("+", "_AND_").upper()[:28]
            row = self.store.one(
                "SELECT MAX(version) AS v FROM strategies WHERE strategy_id=?", (base_id,))
            version = int(row["v"] or 0) + 1
            strat = ThresholdStrategy(
                strategy_id=base_id, version=version,
                username=f"{base_id}_{version:03d}",
                name=f"Generated: {c.feature} {c.operator} {c.threshold} -> {c.bet_side}",
                category="generated", feature=c.feature, operator=c.operator,
                threshold=c.threshold, bet_side=c.bet_side, origin="self_generated",
                origin_ref=f"discovery:{c.key}", min_edge=0.04,
            )
            d = strat.describe()
            d["hypothesis"] = (c.rationale + f" Train hit {c.train_hit:.3f} "
                               f"(+{c.train_lift:.3f}, n={c.train_n}); validation hit "
                               f"{c.valid_hit:.3f} (+{c.valid_lift:.3f}, n={c.valid_n}).")
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
                             "hypotheses_tested": self.tested}),
                 "Survived chronological train->validation split", "edge"),
            )
            created.append(d)
        self.store.commit()
        return created

    def record_failures(self, candidates: Sequence[Candidate]) -> int:
        n = 0
        for c in candidates:
            if c.verdict == "edge":
                continue
            self.store.execute(
                """INSERT OR IGNORE INTO findings(finding_id, created_at, title, body, evidence,
                                                  confidence, kind) VALUES(?,?,?,?,?,?,?)""",
                (f"FIND_FAIL_{abs(hash(c.key)) % 10**10}", utcnow(),
                 f"No reproducible edge: {c.feature} {c.operator} {c.threshold} ({c.bet_side})",
                 f"Train hit rate {c.train_hit:.3f} (lift {c.train_lift:+.3f}, n={c.train_n}) did "
                 f"not reproduce on validation ({c.valid_hit:.3f}, lift {c.valid_lift:+.3f}, "
                 f"n={c.valid_n}). Rejected rather than tuned further.",
                 json.dumps({"train": [c.train_n, c.train_hit], "valid": [c.valid_n, c.valid_hit]}),
                 "medium", "rejection"),
            )
            n += 1
        self.store.commit()
        return n

    def search_size_caveat(self) -> str:
        return (f"{self.tested} candidate triggers were evaluated. With that many tests, a few "
                f"will clear the threshold by chance alone; survivors are labelled CANDIDATE and "
                f"only forward-test results can promote them.")
