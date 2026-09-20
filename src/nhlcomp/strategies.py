"""Strategy definitions, the execution/sizing layer, and the bet-decision pipeline.

A strategy is a *rule*, not a hand-tuned list of picks: it receives a
:class:`DecisionContext` containing only information whose timestamp precedes the
simulated decision moment, and returns zero or more :class:`Signal` objects.  Anything
the strategy cannot see (a starting goalie that has not been announced, a price that has
not been quoted yet) is reported as a blocking status rather than guessed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

# ------------------------------------------------------------ status vocabulary
STATUSES = (
    "WATCHING", "QUALIFIED", "READY TO BET", "PRICE TOO HIGH", "PRICE TOO LOW",
    "WAITING FOR GOALIE", "WAITING FOR LINEUP", "WAITING FOR INJURY",
    "WAITING FOR OTHER INFORMATION", "EXECUTED", "CANCELLED", "EXPIRED",
)


@dataclass
class Quote:
    provider: str
    market_key: str
    contract: str | None
    game_id: int | None
    game_date: str | None
    market_type: str
    selection: str
    side: str
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    volume: float | None
    liquidity: float | None
    ts_utc: str

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass
class DecisionContext:
    decision_ts: str
    features: dict[str, Any]                 # one game's point-in-time features
    predictions: dict[str, float]            # model outputs for that game
    quotes: list[Quote] = field(default_factory=list)
    starters: dict[int, str | None] = field(default_factory=dict)   # team_id -> goalie name/None
    injuries: dict[int, list[str]] = field(default_factory=dict)
    bankroll: float = 0.0
    open_exposure: float = 0.0
    league: dict[str, float] = field(default_factory=dict)


@dataclass
class Signal:
    strategy_id: str
    version: int
    username: str
    game_id: int
    game_date: str
    matchup: str
    market: str
    selection: str
    side: str
    model_prob: float
    fair_price: float
    required_price: float
    stake: float
    status: str
    blocking_reason: str = ""
    supporting: dict[str, Any] = field(default_factory=dict)
    quote: Quote | None = None
    odds_format: str = "binary"

    @property
    def edge(self) -> float:
        return self.model_prob - self.fair_price if self.quote is None else \
            self.model_prob - (self.quote.ask or self.quote.mid or 0.0)


class Strategy:
    """Base class.  Subclasses implement :meth:`evaluate`."""

    strategy_id = "NHL_BASE"
    version = 1
    username = "NHL_BASE_000"
    name = "Baseline"
    category = "baseline"
    hypothesis = "None."
    data_used = "None."
    entry_rule = "None."
    price_rule = "None."
    settlement_rule = "NHL official final result."
    markets = "moneyline"
    origin = "self_generated"
    origin_ref = ""
    params: dict[str, Any] = {}
    starting_bankroll = 1000.0
    stake_fraction = 0.25     # Kelly fraction; 1.0 = full Kelly (aggressive, allowed)
    max_stake_pct = 0.10
    min_edge = 0.03
    min_sample_for_bet = 0

    def __init__(self, **overrides: Any):
        for k, v in overrides.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ helpers
    def kelly(self, p: float, price: float) -> float:
        """Fractional Kelly for a binary contract costing ``price`` that pays 1."""
        if price <= 0 or price >= 1:
            return 0.0
        b = (1.0 - price) / price          # net odds
        f = (b * p - (1 - p)) / b
        return max(0.0, f) * self.stake_fraction

    def size_stake(self, p: float, price: float, bankroll: float, open_exposure: float) -> float:
        f = self.kelly(p, price)
        cap = min(self.max_stake_pct, max(0.0, 0.25))
        f = min(f, cap)
        available = max(0.0, bankroll - open_exposure)
        return round(min(f * available, available), 2)

    def find_quote(self, ctx: DecisionContext, market_type: str, selection: str,
                   side: str) -> Quote | None:
        for q in ctx.quotes:
            if q.market_type != market_type or q.side != side:
                continue
            if selection.lower() in q.selection.lower() or q.selection.lower() in selection.lower():
                return q
        return None

    def evaluate(self, ctx: DecisionContext) -> list[Signal]:   # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id, "version": self.version,
            "username": self.username, "name": self.name, "category": self.category,
            "hypothesis": self.hypothesis, "data_used": self.data_used,
            "entry_rule": self.entry_rule, "price_rule": self.price_rule,
            "settlement_rule": self.settlement_rule, "markets": self.markets,
            # the DB column is params_json; keeping it serialized here means the same dict
            # round-trips into the ledger and back out of _hydrate unchanged
            "params_json": json.dumps(self.params, default=str, sort_keys=True),
            "origin": self.origin, "origin_ref": self.origin_ref,
            "starting_bankroll": self.starting_bankroll,
            # stake_fraction / min_edge live inside params_json rather than as columns, so
            # that a version's full parameter set is one comparable blob
        }


class ThresholdStrategy(Strategy):
    """The workhorse generated strategy.

    Bets one side of a moneyline when a named feature crosses a threshold and the
    available ask is at or below the model's fair price less the required edge.  Every
    parameter is explicit so the discovery engine can search the space and so two
    versions of a strategy are byte-comparable.
    """

    strategy_id = "NHL_THRESHOLD"
    category = "threshold"
    hypothesis = "A named situational feature shifts true win probability away from the market."
    data_used = "api-web.nhle.com schedule/results; kalshi.trade_api quotes"
    entry_rule = "Feature condition satisfied at decision time."
    price_rule = "Bet only when ask <= model_prob - min_edge."
    markets = "moneyline"

    def __init__(self, *, feature: str, side_feature: str = "home", operator: str = ">=",
                 threshold: float = 0.0, bet_side: str = "home", market: str = "moneyline",
                 use_model: str = "poisson", **kw: Any):
        super().__init__(**kw)
        self.feature = feature
        self.side_feature = side_feature
        self.operator = operator
        self.threshold = float(threshold)
        self.bet_side = bet_side
        self.market = market
        self.use_model = use_model
        self.params = {"feature": feature, "side_feature": side_feature, "operator": operator,
                       "threshold": threshold, "bet_side": bet_side, "market": market,
                       "use_model": use_model, "min_edge": self.min_edge,
                       "stake_fraction": self.stake_fraction}
        self.hypothesis = (f"When {feature} {operator} {threshold} the {bet_side} side wins more "
                           f"often than the market price implies.")
        self.entry_rule = (f"At decision time, compute {feature} from NHL schedule/results only; "
                           f"enter when it is {operator} {threshold}.")
        self.price_rule = (f"Require an executable ask <= model_prob - {self.min_edge:.3f}; "
                           f"never bet into a stale or missing quote.")
        self.data_used = "api-web.nhle.com schedule/results (point-in-time); kalshi.trade_api quotes"

    # ------------------------------------------------------------------ logic
    def _passes(self, ctx: DecisionContext) -> tuple[bool, str]:
        raw = ctx.features.get(self.feature)
        if raw is None:
            return False, f"feature {self.feature} unavailable"
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return False, f"feature {self.feature} non-numeric"
        ops = {">=": val >= self.threshold, "<=": val <= self.threshold,
               ">": val > self.threshold, "<": val < self.threshold,
               "==": abs(val - self.threshold) < 1e-9}
        return ops[self.operator], f"{self.feature}={val}"

    def _prob(self, ctx: DecisionContext) -> float | None:
        preds = ctx.predictions
        if self.use_model == "elo":
            key = "p_home_elo" if self.bet_side == "home" else "p_away_elo"
        elif self.use_model == "logistic":
            key = "p_home_logit" if self.bet_side == "home" else "p_away_logit"
        else:
            key = "p_home_ml" if self.bet_side == "home" else "p_away_ml"
        p = preds.get(key)
        if p is None:
            # fall back to the poisson moneyline if the requested model did not emit
            p = preds.get("p_home_ml" if self.bet_side == "home" else "p_away_ml")
        return None if p is None else float(p)

    def evaluate(self, ctx: DecisionContext) -> list[Signal]:
        ok, detail = self._passes(ctx)
        f = ctx.features
        matchup = f"{f.get('away_id')}@{f.get('home_id')}"
        p = self._prob(ctx)
        base = dict(strategy_id=self.strategy_id, version=self.version, username=self.username,
                    game_id=int(f.get("game_id") or 0), game_date=f.get("game_date") or "",
                    matchup=matchup, market=self.market, selection=self.bet_side,
                    side="YES", model_prob=p if p is not None else float("nan"),
                    fair_price=float("nan"), required_price=float("nan"), stake=0.0,
                    supporting={"trigger": detail, "feature": self.feature,
                                "feature_value": f.get(self.feature)})
        if p is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="no model probability available", **base)]
        if not ok:
            return [Signal(status="WATCHING", blocking_reason=f"condition not met ({detail})", **base)]

        quote = self.find_quote(ctx, self.market, self.bet_side, "YES")
        if quote is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="no executable quote found for this market",
                           **base)]
        ask = quote.ask
        if ask is None or ask <= 0:
            return [Signal(status="PRICE TOO HIGH", blocking_reason="no offer on the book",
                           quote=quote, **base)]
        if ask >= 1.0:
            return [Signal(status="PRICE TOO HIGH", blocking_reason="offer at or above par",
                           quote=quote, **base)]
        required = round(p - self.min_edge, 4)
        base.update(fair_price=round(ask, 4), required_price=required, quote=quote)
        if ask > required:
            return [Signal(status="PRICE TOO HIGH",
                           blocking_reason=f"ask {ask:.2f} > required {required:.2f}", **base)]
        size = quote.ask_size
        if size is not None and size <= 0:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="zero offer size at the quoted price", **base)]
        stake = self.size_stake(p, ask, ctx.bankroll, ctx.open_exposure)
        if stake <= 0:
            return [Signal(status="CANCELLED", blocking_reason="zero bankroll available", **base)]
        base.update(stake=stake)
        return [Signal(status="READY TO BET", **base)]


# --------------------------------------------------------------------- registry
def build_seed_strategies() -> list[Strategy]:
    """Seed library.

    These are the *starting* set.  The discovery engine adds generated variants; nothing
    here is presented as proven -- each one is a hypothesis with a recorded rationale.
    """
    out: list[Strategy] = []

    def add(**kw: Any) -> None:
        out.append(ThresholdStrategy(**kw))

    # -- fatigue / scheduling (brief 16-18)
    add(strategy_id="NHL_B2B_FADE", username="NHL_B2B_FADE_001", name="Fade the lone back-to-back team",
        category="fatigue", feature="away_only_b2b", operator=">=", threshold=1, bet_side="home",
        hypothesis="A team playing its second game in two nights, against a rested opponent, "
                   "is underpriced by the market because rest effects are partially public but "
                   "under-reacted to in the closing price.",
        min_edge=0.03)
    add(strategy_id="NHL_B2B_HOME_FADE", username="NHL_B2B_HOME_FADE_002",
        name="Fade a home team on a back-to-back", category="fatigue",
        feature="home_only_b2b", operator=">=", threshold=1, bet_side="away",
        hypothesis="Home-ice advantage shrinks when the home team is on the second night of a "
                   "back-to-back, more than the market discounts it.",
        min_edge=0.03)
    add(strategy_id="NHL_G3IN4", username="NHL_G3IN4_003", name="Third game in four nights",
        category="fatigue", feature="away_g3_in_4", operator=">=", threshold=1, bet_side="home",
        hypothesis="Cumulative fatigue in a 3-in-4 stretch depresses shot quality and goaltending "
                   "more than a single back-to-back does.",
        min_edge=0.04)
    add(strategy_id="NHL_ROADTRIP_END", username="NHL_ROADTRIP_END_004",
        name="Late road trip leg", category="travel",
        feature="away_road_trip_len", operator=">=", threshold=3, bet_side="home",
        hypothesis="Performance degrades on the third and later legs of a road trip.",
        min_edge=0.03)

    # -- pace / totals style, traded as moneyline because that is the verified market
    add(strategy_id="NHL_PACE_MISMATCH", username="NHL_PACE_MISMATCH_005",
        name="Pace mismatch favours the faster team", category="matchup",
        feature="pace_diff", operator=">=", threshold=0.6, bet_side="home",
        hypothesis="When the home team plays a materially higher-event game than its opponent, "
                   "the variance favours the higher-scoring side beyond what the price reflects.",
        min_edge=0.03)

    # -- rest advantage
    add(strategy_id="NHL_REST_EDGE", username="NHL_REST_EDGE_006", name="Two-day rest advantage",
        category="rest", feature="rest_diff", operator=">=", threshold=2, bet_side="home",
        hypothesis="A rest differential of two or more days is a real but small edge that the "
                   "market prices only partially.",
        min_edge=0.025)

    # -- extra-effort previous game
    add(strategy_id="NHL_OT_HANGOVER", username="NHL_OT_HANGOVER_007",
        name="Fade the team coming off overtime", category="fatigue",
        feature="away_prev_ot", operator=">=", threshold=1, bet_side="home",
        hypothesis="An away team that played overtime the previous night has less recovery time "
                   "than the raw rest-day count implies.",
        min_edge=0.03)

    # -- model-based
    add(strategy_id="NHL_POISSON_ML", username="NHL_POISSON_ML_008",
        name="Poisson moneyline value", category="statistical",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home",
        use_model="poisson",
        hypothesis="An independent-Poisson goal model built from point-in-time scoring rates is "
                   "better calibrated than the market on ordinary games.",
        min_edge=0.05)
    add(strategy_id="NHL_ELO_ML", username="NHL_ELO_ML_009", name="Elo moneyline value",
        category="statistical", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", use_model="elo",
        hypothesis="A simple Elo rating with home advantage captures most of the signal, and any "
                   "residual edge over the market is small.",
        min_edge=0.05)

    # -- home ice, team-specific per brief 19
    add(strategy_id="NHL_HOMEICE_STRONG", username="NHL_HOMEICE_STRONG_010",
        name="Strong team-specific home ice", category="home_ice",
        feature="home_home_win_pct", operator=">=", threshold=0.62, bet_side="home",
        hypothesis="Home advantage is not uniform across teams; a subset of teams holds a "
                   "persistently larger home edge than a single league constant implies.",
        min_edge=0.04)

    return out
