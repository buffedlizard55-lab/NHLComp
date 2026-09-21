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


def feature_value(features: dict[str, Any], feature: str) -> float | None:
    """Resolve a feature name to a number.

    ``a+b`` denotes an interaction: the sum of the two binary flags (2 == both true).  A
    missing or non-numeric component makes the whole value None -- never 0 -- so that an
    interaction is not silently treated as "false" when one input was simply unavailable.
    """
    parts = feature.split("+") if "+" in feature else [feature]
    total = 0.0
    for part in parts:
        raw = features.get(part)
        if raw is None:
            return None
        try:
            total += float(raw)
        except (TypeError, ValueError):
            return None
    return total


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
    #: The exchange's own wording for the contract ("Anaheim wins", "Over 6.5 goals
    #: scored").  ``selection`` is this project's side vocabulary ("home", "away",
    #: "over", "under") so a rule can match it; ``label`` keeps the original text so the
    #: ledger can show exactly which contract was quoted.  Never dropped, never edited.
    label: str | None = None
    #: Line for a totals/puck-line contract (6.5, 1.5); None for a moneyline.
    strike: float | None = None
    #: How ``ask`` was obtained: 'exchange' (the offer as quoted) or 'derived' (see
    #: pipeline._live_quotes for the NO-side identity).  Recorded on every bet.
    price_basis: str = "exchange"

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
    stake_mode = "kelly"      # kelly | flat
    flat_pct = 0.02           # bankroll fraction per bet when stake_mode == "flat"

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
        available = max(0.0, bankroll - open_exposure)
        if getattr(self, "stake_mode", "kelly") == "flat":
            return round(min(float(self.flat_pct) * bankroll, available), 2)
        f = self.kelly(p, price)
        cap = min(self.max_stake_pct, max(0.0, 0.25))
        f = min(f, cap)
        return round(min(f * available, available), 2)

    def find_quote(self, ctx: DecisionContext, market_type: str, selection: str,
                   side: str) -> Quote | None:
        """Match a quote to the side this rule trades.

        ``selection`` is this project's side token ("home"/"away"/"over"/"under").  A
        quote carries that token in ``Quote.selection``; ``Quote.label`` (the exchange's
        own wording, "Anaheim wins") is also tried, so a caller that hands over an
        un-normalized quote still matches on a team name instead of silently reporting
        "no quote published".  Substring matching is deliberately symmetric because the
        label may be a place name ("Vegas") or a full name ("Vegas Golden Knights").
        """
        for q in ctx.quotes:
            if q.market_type != market_type or q.side != side:
                continue
            for cand in (q.selection, getattr(q, "label", None)):
                if not cand:
                    continue
                c = cand.lower().strip()
                if selection.lower() in c or c in selection.lower():
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
                 use_model: str = "poisson", requires_goalie: bool = False,
                 requires_lineup: bool = False, injury_sensitive: bool = False,
                 min_price: float = 0.05, blocked_reason: str | None = None, **kw: Any):
        super().__init__(**kw)
        self.feature = feature
        self.side_feature = side_feature
        self.operator = operator
        self.threshold = float(threshold)
        self.bet_side = bet_side
        self.market = market
        self.use_model = use_model
        self.requires_goalie = bool(requires_goalie)
        self.requires_lineup = bool(requires_lineup)
        self.injury_sensitive = bool(injury_sensitive)
        self.min_price = float(min_price)
        self.blocked_reason = blocked_reason
        if use_model == "market":
            # a market-following rule has no probability model of its own, so Kelly is
            # undefined; it stakes a flat fraction and its evidence is the priced backtest.
            # It also has no "edge" to demand: entry is at the offered price when the
            # trigger fires.
            if "stake_mode" not in kw:
                self.stake_mode = "flat"
            if "min_edge" not in kw:
                self.min_edge = 0.0
        self.params = {"feature": feature, "side_feature": side_feature, "operator": operator,
                       "threshold": threshold, "bet_side": bet_side, "market": market,
                       "use_model": use_model, "min_edge": self.min_edge,
                       "requires_goalie": self.requires_goalie,
                       "requires_lineup": self.requires_lineup,
                       "injury_sensitive": self.injury_sensitive, "min_price": self.min_price,
                       "blocked_reason": self.blocked_reason,
                       "stake_fraction": self.stake_fraction,
                       "stake_mode": self.stake_mode, "flat_pct": self.flat_pct}
        self.hypothesis = (f"When {feature} {operator} {threshold} the {bet_side} side wins more "
                           f"often than the market price implies.")
        self.entry_rule = (f"At decision time, compute {feature} from NHL schedule/results only; "
                           f"enter when it is {operator} {threshold}.")
        self.price_rule = (f"Require an executable ask <= model_prob - {self.min_edge:.3f}; "
                           f"never bet into a stale or missing quote.")
        self.data_used = "api-web.nhle.com schedule/results (point-in-time); kalshi.trade_api quotes"

    # ------------------------------------------------------------------ logic
    def _passes(self, ctx: DecisionContext) -> tuple[bool, str]:
        val = feature_value(ctx.features, self.feature)
        if val is None:
            return False, f"feature {self.feature} unavailable"
        ops = {">=": val >= self.threshold, "<=": val <= self.threshold,
               ">": val > self.threshold, "<": val < self.threshold,
               "==": abs(val - self.threshold) < 1e-9}
        return ops[self.operator], f"{self.feature}={val}"

    def _prob(self, ctx: DecisionContext) -> float | None:
        preds = ctx.predictions
        if self.use_model == "market":
            # no model: the market's own offer is the reference probability
            q = self.find_quote(ctx, self.market, self.bet_side, "YES")
            if q is None:
                return None
            return float(q.ask) if q.ask is not None else (float(q.mid) if q.mid is not None else None)
        if self.use_model == "sportsbook":
            # de-vigged sportsbook moneyline (DraftKings via the NHL partner feed) as the
            # reference probability; absent for past games, so this is forward-only
            key = "p_home_book" if self.bet_side == "home" else "p_away_book"
            p = preds.get(key)
            return None if p is None else float(p)
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

    #: Injury statuses that are genuinely undecided.  "Out" and "Injured Reserve" are
    #: *resolved* facts -- the player will not play -- so they never justify waiting.
    #: Only a game-time-decision style status leaves the lineup actually unknown.
    UNRESOLVED_INJURY_STATUSES = ("day-to-day", "questionable", "game time decision", "gtd")

    def _information_gates(self, ctx: DecisionContext) -> tuple[str, str] | None:
        """Return (status, reason) when a required input is missing, else None.

        These gates are deliberately conservative: a strategy that declares a dependency
        refuses to bet rather than betting on an assumed lineup.  Where no verified public
        source supplies the input at all, the reason says so plainly instead of implying
        the data merely had not arrived yet.
        """
        if self.blocked_reason:
            return ("WAITING FOR OTHER INFORMATION", self.blocked_reason)

        f = ctx.features
        home, away = f.get("home_id"), f.get("away_id")
        sides = [t for t in (home, away) if t is not None]

        if self.requires_goalie:
            unknown = [t for t in sides if not ctx.starters.get(t)]
            if unknown:
                return ("WAITING FOR GOALIE",
                        f"confirmed starter unknown for team id(s) {unknown}; no verified "
                        f"public source publishes starting goalies before game time, so this "
                        f"strategy stays gated rather than assume a starter")

        if self.requires_lineup:
            unknown = [t for t in sides if not ctx.features.get(f"lines_confirmed_{t}")]
            if unknown:
                return ("WAITING FOR LINEUP",
                        f"forward lines unconfirmed for team id(s) {unknown}; no verified "
                        f"public source publishes confirmed line combinations")

        if self.injury_sensitive:
            pending: list[str] = []
            for t in sides:
                for name in ctx.injuries.get(t, []):
                    if isinstance(name, tuple):
                        nm, st = name
                    else:
                        nm, st = name, ""
                    if str(st).strip().lower() in self.UNRESOLVED_INJURY_STATUSES:
                        pending.append(f"{nm} ({st})")
            if pending:
                return ("WAITING FOR INJURY",
                        f"game-time-decision injury(ies) on the matchup: "
                        f"{', '.join(sorted(pending))}")

        return None

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
            if self.use_model == "market":
                if not ok:
                    return [Signal(status="WATCHING", blocking_reason=f"condition not met ({detail})", **base)]
                return [Signal(status="QUALIFIED",
                               blocking_reason="market-rule strategy: trigger met but no quote "
                                               "published for this market yet", **base)]
            if self.use_model == "sportsbook":
                return [Signal(status="WAITING FOR OTHER INFORMATION",
                               blocking_reason="no sportsbook reference line for this game: the "
                                               "NHL partner-odds feed covers the current slate "
                                               "only, so this rule is forward-test only", **base)]
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="no model probability available", **base)]
        if not ok:
            return [Signal(status="WATCHING", blocking_reason=f"condition not met ({detail})", **base)]

        blocked = self._information_gates(ctx)
        if blocked is not None:
            status, reason = blocked
            return [Signal(status=status, blocking_reason=reason, **base)]

        quote = self.find_quote(ctx, self.market, self.bet_side, "YES")
        if quote is None:
            # The trigger fired and the model has a number, so the opportunity has
            # qualified -- it is a price that is missing, not information.
            return [Signal(status="QUALIFIED",
                           blocking_reason="condition met and priced, but no quote published "
                                           "for this market yet", **base)]
        ask = quote.ask
        if ask is None or ask <= 0:
            return [Signal(status="PRICE TOO HIGH", blocking_reason="no offer on the book",
                           quote=quote, **base)]
        if ask >= 1.0:
            return [Signal(status="PRICE TOO HIGH", blocking_reason="offer at or above par",
                           quote=quote, **base)]
        if ask < self.min_price:
            return [Signal(status="PRICE TOO LOW",
                           blocking_reason=f"ask {ask:.2f} below floor {self.min_price:.2f}: the "
                                           f"remaining payoff does not justify the variance",
                           quote=quote, **base)]
        required = round(p - (0.0 if self.use_model == "market" else self.min_edge), 4)
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


# --------------------------------------------------------------------- totals
class TotalsStrategy(ThresholdStrategy):
    """Trade a Kalshi KXNHLTOTAL contract ("Over k.5 goals") with an expected-goals model.

    Why this class exists separately: a totals contract is priced by a *strike*, not by a
    team, so the rule needs the strike before it can ask its own model for a probability.
    The strike arrives with the quote (:attr:`Quote.strike`, taken from Kalshi's
    ``floor_strike``), never from a guess.

    Two directions are possible and they are NOT symmetric in what can be verified:

    * ``over`` buys the **YES** side.  Kalshi's candlestick history publishes
      ``yes_ask`` for every settled contract, so an ``over`` rule can be *backtested* at
      real, timestamped prices.
    * ``under`` buys the **NO** side.  The candlestick feed publishes only the YES bid/ask,
      so there is no historical NO-side offer to buy at.  ``no_ask = 1 - yes_bid`` is NOT
      assumed: on the 2026-09-20 quotes already in this ledger that identity fails on 2 of
      12 contracts (KXNHLGAME-26SEP20SJANA: yes_bid 0.19, no_ask 0.99).  An ``under`` rule
      therefore carries ``no_history_reason`` and is **forward-test only**, entered at the
      live ``no_ask_dollars`` Kalshi actually quotes.
    """

    strategy_id = "NHL_TOTAL_GOALS"
    category = "totals"
    market = "total"
    markets = "total"

    def __init__(self, *, direction: str = "over", min_edge: float = 0.04,
                 min_price: float = 0.05, min_strike: float = 4.5, max_strike: float = 8.5,
                 no_history_reason: str | None = None, **kw: Any):
        # the threshold machinery is inherited only for the shared information gates and
        # sizing; a totals rule has no feature trigger of its own, so the trigger is set
        # to a condition that is always satisfied and says so.
        kw.setdefault("feature", "home_n_prior")
        kw.setdefault("operator", ">=")
        kw.setdefault("threshold", 0)
        kw.setdefault("bet_side", direction)
        kw.setdefault("use_model", "poisson")
        kw["min_edge"] = min_edge
        kw["min_price"] = min_price
        super().__init__(**kw)
        self.direction = direction
        self.market = "total"
        self.markets = "total"
        self.min_strike = float(min_strike)
        self.max_strike = float(max_strike)
        self.no_history_reason = no_history_reason
        self.settlement_rule = ("Official NHL final score. Kalshi settles an Over k.5 contract "
                                "on regulation + overtime goals, with a shootout counted as one "
                                "goal for the winner -- which is what the official final score "
                                "already includes, so total = home_score + away_score.")
        self.params = {**self.params, "kind": "totals", "direction": direction,
                       "market": "total", "min_strike": self.min_strike,
                       "max_strike": self.max_strike, "no_history_reason": no_history_reason}
        side_word = "over" if direction == "over" else "under"
        self.hypothesis = (f"The independent-Poisson expected-goals model prices a {side_word} "
                           f"total more accurately than the exchange does, so buying the "
                           f"{side_word} side when the offer is at least {min_edge:.3f} below the "
                           f"model probability earns the difference.")
        self.entry_rule = (f"At decision time compute expected goals from prior games only; buy "
                           f"the {side_word} side of the quoted strike when the offer is at or "
                           f"below model P({side_word}) - {min_edge:.3f}. Strikes outside "
                           f"[{self.min_strike}, {self.max_strike}] are not traded.")
        self.price_rule = (f"Require an executable offer <= model P({side_word}) - "
                           f"{self.min_edge:.3f}; never bet into a stale or missing quote.")
        self.data_used = ("kalshi.trade_api KXNHLTOTAL contracts (floor_strike, yes/no offers, "
                          "candlesticks) + api-web.nhle.com results for expected goals.")
        if no_history_reason:
            self.data_used += f" BACKTEST NOTE: {no_history_reason}"

    # ------------------------------------------------------------------ logic
    def _lambdas(self, ctx: DecisionContext) -> tuple[float, float] | None:
        p = ctx.predictions
        lh, la = p.get("lam_home"), p.get("lam_away")
        if lh is None or la is None:
            return None
        try:
            return float(lh), float(la)
        except (TypeError, ValueError):
            return None

    def evaluate(self, ctx: DecisionContext) -> list[Signal]:
        from .models import p_total_over      # local import: strategies stay import-light
        f = ctx.features
        side = "YES" if self.direction == "over" else "NO"
        base: dict[str, Any] = dict(
            strategy_id=self.strategy_id, version=self.version, username=self.username,
            game_id=int(f.get("game_id") or 0), game_date=f.get("game_date") or "",
            matchup=f"{f.get('away_id')}@{f.get('home_id')}", market="total",
            selection=self.direction, side=side, model_prob=float("nan"),
            fair_price=float("nan"), required_price=float("nan"), stake=0.0,
            supporting={"market": "total", "direction": self.direction})

        q = self.find_quote(ctx, "total", self.direction, side)
        if q is None:
            return [Signal(status="WATCHING", quote=None,
                           blocking_reason="no KXNHLTOTAL contract quoted for this game", **base)]
        base["quote"] = q
        strike = getattr(q, "strike", None)
        if strike is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="contract carries no floor_strike; the line is not "
                                           "known, so it is not assumed", **base)]
        base["supporting"] = {**base["supporting"], "strike": strike,
                              "contract": q.contract, "contract_label": q.label,
                              "price_basis": q.price_basis}
        if not (self.min_strike <= float(strike) <= self.max_strike):
            return [Signal(status="WATCHING",
                           blocking_reason=f"strike {strike} outside the traded range "
                                           f"[{self.min_strike}, {self.max_strike}]", **base)]

        lam = self._lambdas(ctx)
        if lam is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="no expected-goals model output for this game", **base)]
        p_over = p_total_over(lam[0], lam[1], float(strike))
        p = p_over if self.direction == "over" else round(1.0 - p_over, 6)
        base["model_prob"] = p
        base["supporting"] = {**base["supporting"], "lam_home": lam[0], "lam_away": lam[1],
                              "p_over": p_over,
                              "data_labels": {"price": f"SOURCE DATA (kalshi, {q.price_basis})",
                                              "features": "DERIVED (point-in-time)",
                                              "model_prob": "MODEL OUTPUT (independent Poisson)"}}

        blocked = self._information_gates(ctx)
        if blocked is not None:
            status, reason = blocked
            return [Signal(status=status, blocking_reason=reason, **base)]

        ask = q.ask
        if ask is None or ask <= 0:
            return [Signal(status="PRICE TOO HIGH", blocking_reason="no offer on the book", **base)]
        if ask >= 1.0:
            return [Signal(status="PRICE TOO HIGH", blocking_reason="offer at or above par", **base)]
        if ask < self.min_price:
            return [Signal(status="PRICE TOO LOW",
                           blocking_reason=f"ask {ask:.2f} below floor {self.min_price:.2f}", **base)]
        required = round(p - self.min_edge, 4)
        base.update(fair_price=round(ask, 4), required_price=required)
        if ask > required:
            return [Signal(status="PRICE TOO HIGH",
                           blocking_reason=f"ask {ask:.2f} > required {required:.2f} "
                                           f"(model P({self.direction})={p:.4f})", **base)]
        if q.ask_size is not None and float(q.ask_size) <= 0:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="zero offer size at the quoted price", **base)]
        stake = self.size_stake(p, float(ask), ctx.bankroll, ctx.open_exposure)
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

    # -- categories the brief requires but whose inputs are NOT yet verifiable.
    # Each is registered with the real reason it cannot fire, rather than being quietly
    # omitted or being pointed at a guessed data feed.
    add(strategy_id="NHL_GOALIE_EDGE", username="NHL_GOALIE_EDGE_011", category="goaltending",
        name="Goaltending matchup edge", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", requires_goalie=True, min_edge=0.05,
        data_used="api-web.nhle.com schedule/results; kalshi.trade_api quotes. Starting-goalie "
                  "confirmation: NO VERIFIED SOURCE.",
        hypothesis="Save-percentage above expectation by the confirmed starter moves true win "
                   "probability more than the moneyline reflects. UNTESTED: cannot be "
                   "evaluated until confirmed starters are obtainable.",
        entry_rule="Blocked: requires the confirmed starting goalie for both teams.")
    # v2: the per-game goalie log (api.nhle.com/stats/rest goalie/summary isGame=true) is
    # verified, so the matchup can be built point-in-time.  Starter identity for PAST games
    # is the post-game log (ASSUMPTION: public at the morning skate); for UPCOMING games no
    # verified pre-game starter source exists, so the goalie gate still holds forward.
    add(strategy_id="NHL_GOALIE_EDGE", version=2, username="NHL_GOALIE_EDGE_011",
        category="goaltending", name="Goaltending matchup edge (starter SV% last 10 starts)",
        feature="diff_starter_sv_pct_l10", operator=">=", threshold=0.010, bet_side="home",
        requires_goalie=True, min_edge=0.04,
        data_used="api.nhle.com/stats/rest goalie/summary per-game (verified); api-web.nhle.com "
                  "schedule/results; kalshi candlesticks + quotes.",
        hypothesis="A home starter whose save percentage over his last 10 starts exceeds the "
                   "away starter's by a full point is under-priced because the market anchors "
                   "on team strength. BACKTESTABLE with post-game starter logs; FORWARD gated "
                   "until a verified pre-game starter source exists.",
        entry_rule="diff_starter_sv_pct_l10 >= 0.010 with both starters known.")
    add(strategy_id="NHL_GOALIE_FATIGUE", username="NHL_GOALIE_FATIGUE_022",
        category="goaltending", name="Fade the goalie on consecutive-night starts",
        feature="away_starter_b2b", operator=">=", threshold=1, bet_side="home",
        requires_goalie=True, min_edge=0.04,
        data_used="api.nhle.com/stats/rest goalie/summary per-game (verified); kalshi prices.",
        hypothesis="A goalie starting the second night of a back-to-back saves fewer shots than "
                   "his baseline, and the moneyline prices the team, not the goalie's workload.",
        entry_rule="Away starter also started the previous night (starter_b2b == 1).")

    add(strategy_id="NHL_GOALIE_NEWS", username="NHL_GOALIE_NEWS_012", category="goalie_news",
        name="Market reaction to a goalie announcement", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", requires_goalie=True,
        injury_sensitive=True, min_edge=0.06,
        data_used="espn.nhl_api injuries (verified, cross-check only); kalshi.trade_api quotes. "
                  "Goalie announcements: NO VERIFIED SOURCE.",
        hypothesis="A starter announcement moves the price, and the first mover is paid. "
                   "UNTESTED: neither the announcement feed nor intraday prices exist here.",
        entry_rule="Blocked: requires the confirmed starter and a pre-announcement price.")

    # v2: the post-game goalie log can identify past starters, but this rule is about the
    # *announcement* and the price before it -- neither exists in any verified feed -- so
    # it is blocked outright rather than allowed to replay as a plain model-value bet.
    add(strategy_id="NHL_GOALIE_NEWS", version=2, username="NHL_GOALIE_NEWS_012",
        category="goalie_news", name="Market reaction to a goalie announcement (blocked)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home",
        requires_goalie=True, injury_sensitive=True, min_edge=0.06,
        blocked_reason="requires a timestamped starting-goalie announcement feed and the "
                       "Kalshi price immediately before it; neither exists in a verified "
                       "source, and the post-game goalie log cannot stand in for an "
                       "announcement time",
        data_used="NO VERIFIED SOURCE for goalie announcements or announcement timing.",
        hypothesis="A starter announcement moves the price, and the first mover is paid. "
                   "UNTESTED and not backtestable with current sources.",
        entry_rule="Blocked: requires the announcement time and a pre-announcement price.")

    add(strategy_id="NHL_LINE_COMBO", username="NHL_LINE_COMBO_013", category="player_lines",
        name="Top-line deployment edge", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", requires_lineup=True, min_edge=0.05,
        data_used="NO VERIFIED SOURCE for forward-line combinations.",
        hypothesis="Line combinations change scoring rate independently of roster quality. "
                   "UNTESTED and unbacktestable with current sources.",
        entry_rule="Blocked: requires confirmed forward lines for both teams.")

    add(strategy_id="NHL_PROP_EDGE", username="NHL_PROP_EDGE_014", category="player_props",
        name="Player prop mispricing", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", requires_lineup=True, min_edge=0.05,
        data_used="NO VERIFIED SOURCE for player-level projections or prop markets. Kalshi "
                  "KXNHLGAME is game-level only.",
        hypothesis="Player props are softer than game markets. UNTESTED: no prop market and no "
                   "player-level feed is verified in this system.",
        entry_rule="Blocked: requires a player prop market and player-level data.")

    add(strategy_id="NHL_SPECIAL_TEAMS", username="NHL_SPECIAL_TEAMS_015",
        category="special_teams", name="Power-play / penalty-kill differential",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", min_edge=0.04,
        blocked_reason="no verified play-by-play source: PP% and PK% are not obtainable from "
                       "the endpoints recorded as reachable in data/probe.json, so special-teams "
                       "features cannot be built without inventing them",
        data_used="NONE VERIFIED for special teams.",
        hypothesis="A PP/PK differential is predictive beyond goal differential. UNTESTED.",
        entry_rule="Blocked: requires a verified play-by-play or special-teams feed.")
    # v2: team/summary isGame=true carries PP% / PK% per game, so the block is lifted.
    add(strategy_id="NHL_SPECIAL_TEAMS", version=2, username="NHL_SPECIAL_TEAMS_015",
        category="special_teams", name="Special-teams matchup edge (PP vs PK, last 10)",
        feature="st_edge_home", operator=">=", threshold=0.10, bet_side="home", min_edge=0.04,
        data_used="api.nhle.com/stats/rest team/summary per-game PP%/PK% (verified); kalshi prices.",
        hypothesis="When the home power play against the away penalty kill is at least ten "
                   "points stronger than the reverse matchup, the moneyline under-weights it. "
                   "Tested, not assumed: the priced backtest decides.",
        entry_rule="st_edge_home >= 0.10 computed from games strictly before the game date.")
    add(strategy_id="NHL_SHOT_SHARE", username="NHL_SHOT_SHARE_023", category="shot_quality",
        name="Shot-share dominance", feature="diff_shot_share_l10", operator=">=",
        threshold=0.04, bet_side="home", min_edge=0.04,
        data_used="api.nhle.com/stats/rest team/summary per-game shots for/against (verified).",
        hypothesis="Shot share is a stabler skill signal than goal differential over ten games; "
                   "a four-point shot-share gap is under-priced when recent results hid it.",
        entry_rule="home shot share minus away shot share (last 10) >= 0.04.")
    add(strategy_id="NHL_PDO_REGRESSION", username="NHL_PDO_REGRESSION_024",
        category="shot_quality", name="Fade the PDO-inflated visitor", feature="away_pdo_l10",
        operator=">=", threshold=1.03, bet_side="home", min_edge=0.04,
        data_used="api.nhle.com/stats/rest team/summary per-game (verified).",
        hypothesis="A visitor whose shooting + save percentage (PDO) over the last ten games is "
                   "above 1.03 is riding variance; the market extrapolates the results.",
        entry_rule="away_pdo_l10 >= 1.03.")

    add(strategy_id="NHL_EDGE_TRACKING", username="NHL_EDGE_TRACKING_016",
        category="edge_tracking", name="NHL EDGE tracking variables",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", min_edge=0.04,
        blocked_reason="nhl.com/stats/edge is not reachable as a plain GET and has no published "
                       "API; recorded unreachable in data/probe.json. EDGE variables are not "
                       "assumed predictive and cannot be sourced, so nothing is fabricated",
        data_used="NONE VERIFIED for NHL EDGE.",
        hypothesis="Skate distance / shot speed / high-danger chances are predictive. UNTESTED "
                   "and, per the brief, not assumed predictive in advance.",
        entry_rule="Blocked: requires a verified NHL EDGE feed.")
    # v2: the JSON EDGE API (api-web.nhle.com/v1/edge/team-comparison) IS reachable and is
    # snapshotted every run, but it is a season-to-date aggregate with no per-game history,
    # so nothing can be backtested yet.  Blocked until a forward history has accumulated.
    add(strategy_id="NHL_EDGE_TRACKING", version=2, username="NHL_EDGE_TRACKING_016",
        category="edge_tracking", name="NHL EDGE tracking variables (snapshot collection)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", min_edge=0.04,
        blocked_reason="EDGE team snapshots (skating distance, speed bursts, shot speed, zone "
                       "time) are collected forward from the verified api-web.nhle.com/v1/edge "
                       "API, but the feed is season-to-date with no per-game history; no test "
                       "is possible until enough dated snapshots exist, and predictiveness is "
                       "not assumed",
        data_used="api-web.nhle.com/v1/edge/team-comparison (verified JSON; snapshots only).",
        hypothesis="Skating distance / speed bursts / shot speed are predictive of results "
                   "beyond the scoreboard. UNTESTED until the snapshot history is long enough.",
        entry_rule="Blocked: requires a dated EDGE history for point-in-time features.")

    add(strategy_id="NHL_WEATHER_OUTDOOR", username="NHL_WEATHER_OUTDOOR_017",
        category="weather_arena", name="Outdoor-game weather effect",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", min_edge=0.04,
        blocked_reason="open-meteo archive is verified reachable, but arena coordinates are not: "
                       "statsapi.web.nhl.com (the only first-party lat/lon source) is "
                       "unreachable from the CI runner, so venues.lat/lon are NULL by design "
                       "and outdoor games cannot be identified without guessing",
        data_used="archive-api.open-meteo.com (verified, CC BY 4.0); arena coordinates NONE.",
        hypothesis="Weather affects outdoor games. UNTESTED: outdoor games cannot be identified.",
        entry_rule="Blocked: requires arena coordinates and an outdoor-game flag.")

    add(strategy_id="NHL_LIVE_INGAME", username="NHL_LIVE_INGAME_018", category="live_in_game",
        name="In-game momentum reaction", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", min_edge=0.06,
        blocked_reason="Kalshi candle history returns 404 on both documented path shapes "
                       "(confirmed from CI, recorded in data/probe.json), so there is no "
                       "intraday price series to react to; the scoreboard does carry LIVE state "
                       "but no verified live price feed exists to trade against",
        data_used="api-web.nhle.com scoreboard (verified, LIVE state available); "
                  "kalshi.trade_api candles NOT AVAILABLE.",
        hypothesis="Live prices overreact to goals. UNTESTED: no intraday price history.",
        entry_rule="Blocked: requires an intraday price feed.")
    # v2: candlesticks are available (the endpoint is /candlesticks, not /candles) and the
    # 60- and 120-minute in-game points are stored for research; but this pipeline runs on
    # a batch schedule and cannot act during a game, so in-game rules stay research-only.
    add(strategy_id="NHL_LIVE_INGAME", version=2, username="NHL_LIVE_INGAME_018",
        category="live_in_game", name="In-game momentum reaction (research only)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", min_edge=0.06,
        blocked_reason="intraday Kalshi candlesticks are now ingested (ig60/ig120 points) and "
                       "can be studied, but the paper-trading loop runs on a batch schedule "
                       "and cannot observe or act on a price during a game; no in-game bet is "
                       "simulated because its execution time cannot be honoured",
        data_used="kalshi candlesticks (verified, 60-minute periods); api-web.nhle.com scoreboard.",
        hypothesis="Live prices overreact to goals. Research finding only; not tradeable here.",
        entry_rule="Blocked: batch pipeline cannot execute intraday.")

    add(strategy_id="NHL_PERIOD_BET", username="NHL_PERIOD_BET_019", category="period_betting",
        name="Period-specific value", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", min_edge=0.05,
        blocked_reason="Kalshi KXNHLGAME exposes game-level moneyline contracts only; no "
                       "verified first-period or period-specific market was found, and none is "
                       "assumed to exist",
        data_used="kalshi.trade_api KXNHLGAME (game-level only, verified).",
        hypothesis="Period markets are less efficient than game markets. UNTESTED.",
        entry_rule="Blocked: requires a verified period-level market.")
    # v2: the series listing shows KXNHL1P (first-period) and KXNHLOVERTIME exist on Kalshi.
    # They are registered in kalshi_series but not priced or modelled here yet.
    add(strategy_id="NHL_PERIOD_BET", version=2, username="NHL_PERIOD_BET_019",
        category="period_betting", name="Period-specific value (market found, unmodelled)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", min_edge=0.05,
        blocked_reason="Kalshi lists first-period (KXNHL1P) and overtime (KXNHLOVERTIME) series "
                       "(verified via /series?category=Sports&tags=Hockey), but this system has "
                       "no period-level model and does not ingest those contracts' prices yet; "
                       "nothing is bet until both exist",
        data_used="kalshi series registry (verified); no period-level prices ingested.",
        hypothesis="Period markets are less efficient than game markets. UNTESTED.",
        entry_rule="Blocked: requires period-level prices and a period-level model.")

    # -- market-structure rules (priced backtests decide; flat stakes, no model)
    add(strategy_id="NHL_STEAM_FOLLOW", username="NHL_STEAM_FOLLOW_025", category="market",
        name="Follow the late move toward the home side", feature="mkt_move6_home",
        operator=">=", threshold=0.03, bet_side="home", use_model="market",
        data_used="kalshi candlesticks: T-6h mid vs. closing mid (verified, timestamped).",
        hypothesis="A three-cent move toward the home side in the last six hours reflects "
                   "informed order flow ('sharp money' made measurable), and the close still "
                   "under-reacts. Measured, not assumed: the priced backtest decides.",
        entry_rule="closing mid minus T-6h mid >= +0.03 for the home contract; buy home at the ask.")
    add(strategy_id="NHL_STEAM_FADE", username="NHL_STEAM_FADE_026", category="market",
        name="Fade the late move against the home side", feature="mkt_move6_home",
        operator="<=", threshold=-0.03, bet_side="home", use_model="market",
        data_used="kalshi candlesticks: T-6h mid vs. closing mid (verified, timestamped).",
        hypothesis="The opposite of STEAM_FOLLOW: a late move against the home side overshoots "
                   "on a thin book and the home contract is cheap at the close. One of the two "
                   "rules must lose; keeping both makes the test honest.",
        entry_rule="closing mid minus T-6h mid <= -0.03 for the home contract; buy home at the ask.")
    add(strategy_id="NHL_HOME_FAV", username="NHL_HOME_FAV_027", category="market",
        name="Buy home favourites at the close", feature="mkt_close_home_mid", operator=">=",
        threshold=0.60, bet_side="home", use_model="market",
        data_used="kalshi candlesticks closing mid (verified).",
        hypothesis="Favourite-longshot bias: favourites priced at 60c+ win more often than the "
                   "price implies once the exchange vig is paid. Simple baseline the model rules "
                   "must beat.",
        entry_rule="closing home mid >= 0.60; buy home at the ask.")
    add(strategy_id="NHL_HOME_DOG", username="NHL_HOME_DOG_028", category="market",
        name="Buy home underdogs at the close", feature="mkt_close_home_mid", operator="<=",
        threshold=0.42, bet_side="home", use_model="market",
        data_used="kalshi candlesticks closing mid (verified).",
        hypothesis="Home underdogs below 42c are over-faded by the public. The mirror image of "
                   "HOME_FAV; at most one of them can be right about the same prices.",
        entry_rule="closing home mid <= 0.42; buy home at the ask.")
    add(strategy_id="NHL_BOOK_VS_EXCHANGE", username="NHL_BOOK_VS_EXCHANGE_029",
        category="market", name="Exchange cheaper than the de-vigged sportsbook line",
        feature="home_n_prior", operator=">=", threshold=0, bet_side="home",
        use_model="sportsbook", min_edge=0.03, stake_mode="flat",
        data_used="api-web.nhle.com/v1/partner-game (DraftKings moneyline, verified, current "
                  "slate only); kalshi.trade_api quotes.",
        hypothesis="When the Kalshi ask is at least three cents below the de-vigged DraftKings "
                   "probability for the same side, the exchange is slow and the book is the "
                   "better estimate. FORWARD TEST ONLY: no sportsbook history is available, so "
                   "no backtest is claimed.",
        entry_rule="kalshi ask <= devigged DK probability - 0.03; buy home at the ask.")

    add(strategy_id="NHL_OT_SURVIVAL", username="NHL_OT_SURVIVAL_020", category="ot_shootout",
        name="Extra-time specialists are underpriced in close games", feature="ot_rate",
        operator=">=", threshold=0.20, bet_side="home", min_edge=0.05,
        data_used="api-web.nhle.com gameOutcome.lastPeriodType (REG/OT/SO) -- real, verified, "
                  "3,008 decided games. kalshi.trade_api quotes.",
        hypothesis="Teams that reach extra time often have above-average expected points, and a "
                   "moneyline priced for regulation-only outcomes under-values them. This one "
                   "uses real data and is testable, unlike the blocked categories above.",
        entry_rule="home ot_rate >= 0.20 over the prior window.")

    add(strategy_id="NHL_INJURY_IMPACT", username="NHL_INJURY_IMPACT_021", category="injury",
        name="Fade the side with an unresolved key injury", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", injury_sensitive=True, min_edge=0.05,
        data_used="site.api.espn.com NHL injuries (verified 200, no key); cross-check only, "
                  "not treated as authoritative.",
        hypothesis="A game-time-decision injury is not yet priced in. Held for confirmation "
                   "rather than bet on assumption.",
        entry_rule="Condition met, but any day-to-day injury on either team blocks entry.")
    # v2: identical rule, FORWARD TEST only.  There is no historical injury feed, so a
    # replay cannot know whether an injury was pending; v1's BACKTEST rows were written
    # without that context and are annotated (never deleted) by Pipeline.stage_reconcile.
    add(strategy_id="NHL_INJURY_IMPACT", version=2, username="NHL_INJURY_IMPACT_021",
        category="injury", name="Fade the side with an unresolved key injury (forward only)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home",
        injury_sensitive=True, min_edge=0.05,
        data_used="site.api.espn.com NHL injuries (verified, current list only; no history) -> "
                  "FORWARD TEST only. kalshi.trade_api quotes.",
        hypothesis="A game-time-decision injury is not yet priced in. Held for confirmation "
                   "rather than bet on assumption. Not backtestable: no historical injury data.",
        entry_rule="Condition met, but any day-to-day injury on either team blocks entry.")

    # -- totals (KXNHLTOTAL).  The exchange lists one contract per strike ("Over k.5");
    # buying YES is an Over bet and buying NO is an Under bet.  Only the Over side has a
    # verified historical price (the candlestick feed publishes the YES ask), so the Under
    # rule declares why it cannot be backtested instead of inventing a NO-side history.
    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_OVER", username="NHL_TOTALS_OVER_030", category="totals",
        name="Expected-goals model over the exchange total", direction="over", min_edge=0.04,
        data_used="kalshi.trade_api KXNHLTOTAL floor_strike + yes_ask candlesticks (verified "
                  "2026-09-21, historical tier back to the 2026 Stanley Cup Final); "
                  "api-web.nhle.com results for expected goals."))
    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_UNDER", username="NHL_TOTALS_UNDER_031", category="totals",
        name="Expected-goals model under the exchange total (forward only)", direction="under",
        min_edge=0.04,
        no_history_reason="Kalshi's candlestick feed publishes the YES bid/ask only, so no "
                          "historical NO-side offer exists to buy an Under at, and "
                          "no_ask = 1 - yes_bid does not hold on the quotes in this ledger "
                          "(2 of 12 contracts on 2026-09-20). FORWARD TEST ONLY, entered at the "
                          "live no_ask_dollars."))
    # A tighter-edge variant of the Over rule.  Keeping both makes the edge threshold itself
    # an experiment rather than a tuned constant: only one of them can be right about the
    # same prices, and the pair shows how much of the result is the threshold's doing.
    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_OVER_TIGHT", username="NHL_TOTALS_OVER_TIGHT_032",
        category="totals", name="Expected-goals model over the exchange total (2c edge)",
        direction="over", min_edge=0.02,
        data_used="kalshi.trade_api KXNHLTOTAL floor_strike + yes_ask candlesticks (verified); "
                  "api-web.nhle.com results for expected goals."))

    return out
