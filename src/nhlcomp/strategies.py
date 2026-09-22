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
    #: Line for a totals/puck-line contract (6.5, 1.5); None for a moneyline or for the
    #: strike-less overtime contract.
    strike: float | None = None
    #: The exchange's own comparison for that line (``greater`` on every KXNHLTOTAL and
    #: KXNHLSPREAD contract verified 2026-09-21).  Kept on the quote because it is what
    #: makes the payoff well defined: ``greater`` + 1.5 pays on a margin of 2 or more, and
    #: an instrument whose comparison this project has not verified is refused, not assumed.
    strike_type: str | None = None
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


#: NHL ``gameTypeId`` values, spelled out because a rule's scope has to be readable.
PRESEASON = 1
REGULAR_SEASON = 2
PLAYOFFS = 3
GAME_TYPE_LABELS = {PRESEASON: "preseason", REGULAR_SEASON: "regular season",
                    PLAYOFFS: "playoffs"}


class Strategy:
    """Base class.  Subclasses implement :meth:`evaluate`."""

    strategy_id = "NHL_BASE"
    version = 1
    username = "NHL_BASE_000"
    name = "Baseline"
    category = "baseline"
    hypothesis = "None."
    #: Which NHL season phases this rule may trade.  Default is regular season + playoffs --
    #: the games every input in this project describes.  Preseason is excluded on evidence,
    #: not on taste: on the 2025-26 preseason there is no priced edge to claim (62 games,
    #: 124 settled contracts, blind buy at the close offer -9.15% ROI, favourite -1.33%,
    #: underdog -20.0%), and the 2026-27 preseason books are largely empty (102 of 170 close
    #: candles traded nothing at all; mean bid-ask spread 0.314 against 0.103 a year earlier).
    #: A rule fitted on regular-season Elo and last season's team stats would be claiming a
    #: view on a September roster it has never seen, so it does not get to trade one.
    game_types: tuple[int, ...] = (REGULAR_SEASON, PLAYOFFS)
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
        matches = self.find_quotes(ctx, market_type, selection, side)
        return matches[0] if matches else None

    def find_quotes(self, ctx: DecisionContext, market_type: str, selection: str,
                    side: str) -> list[Quote]:
        """Every quote that matches the side this rule trades, in book order.

        A moneyline has one contract per team, so the first match is the only match.  A
        totals market does not: Kalshi lists one "Over k.5" contract per strike (the
        2026-09-24 slate carried eight, 1.5 through 8.5), and every one of them is a
        legitimate quote for the side "over".  A caller that takes only the first match
        silently trades whichever strike happened to sort first -- on the real ledger that
        was 1.5, which every totals rule then refused, so the totals market never traded at
        all.  Totals rules therefore ask for the whole list and choose the strike
        themselves, from a target declared before any price is seen.
        """
        out: list[Quote] = []
        for q in ctx.quotes:
            if q.market_type != market_type or q.side != side:
                continue
            for cand in (q.selection, getattr(q, "label", None)):
                if not cand:
                    continue
                c = cand.lower().strip()
                if selection.lower() in c or c in selection.lower():
                    out.append(q)
                    break
        return out

    def evaluate(self, ctx: DecisionContext) -> list[Signal]:   # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id, "version": self.version,
            "username": self.username, "name": self.name, "category": self.category,
            "hypothesis": self.hypothesis, "data_used": self.data_used,
            "entry_rule": self.entry_rule, "price_rule": self.price_rule,
            "settlement_rule": self.settlement_rule, "markets": self.markets,
            # which season phases this rule is allowed to trade, as a stored column so the
            # scope is queryable and a reader can see the rule was never let near a game it
            # was not fitted for
            "game_types": json.dumps(list(self.game_types)),
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
    #: which side of the exchange's book this rule buys.  A moneyline is entered on the YES
    #: side of the chosen team's contract; the totals, puck-line and overtime rules set their
    #: own.  The engine uses it to hand a rule only the quotes it could actually have bought,
    #: so a YES-only rule is never evaluated against a NO quote (and can never overwrite the
    #: status of a wager the other side already recorded).
    quote_side = "YES"
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
                       "stake_mode": self.stake_mode, "flat_pct": self.flat_pct,
                       "quote_side": self.quote_side,
                       "game_types": list(self.game_types)}
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
                 target_strike: float = 6.5, no_history_reason: str | None = None, **kw: Any):
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
        self.quote_side = "YES" if direction == "over" else "NO"
        self.market = "total"
        self.markets = "total"
        self.min_strike = float(min_strike)
        self.max_strike = float(max_strike)
        # The line this rule means by "the total".  Declared here, before any price is
        # looked at, so which strike gets traded is part of the rule's definition and not a
        # choice made after seeing the book.  The exchange offers many strikes; the rule
        # trades the offered one nearest to this, and refuses if none is in range.
        self.target_strike = float(target_strike)
        self.no_history_reason = no_history_reason
        self.settlement_rule = ("Official NHL final score. Kalshi settles an Over k.5 contract "
                                "on regulation + overtime goals, with a shootout counted as one "
                                "goal for the winner -- which is what the official final score "
                                "already includes, so total = home_score + away_score.")
        self.params = {**self.params, "kind": "totals", "direction": direction,
                       "market": "total", "min_strike": self.min_strike,
                       "max_strike": self.max_strike, "target_strike": self.target_strike,
                       "no_history_reason": no_history_reason, "quote_side": self.quote_side}
        side_word = "over" if direction == "over" else "under"
        self.hypothesis = (f"The independent-Poisson expected-goals model prices a {side_word} "
                           f"total more accurately than the exchange does, so buying the "
                           f"{side_word} side when the offer is at least {min_edge:.3f} below the "
                           f"model probability earns the difference.  The exchange lists one "
                           f"contract per strike; this rule trades the offered strike nearest "
                           f"{self.target_strike:g}.")
        self.entry_rule = (f"At decision time compute expected goals from prior games only; buy "
                           f"the {side_word} side of the quoted strike when the offer is at or "
                           f"below model P({side_word}) - {min_edge:.3f}. The offered strike "
                           f"nearest {self.target_strike:g} is traded; strikes outside "
                           f"[{self.min_strike}, {self.max_strike}] are never traded, and a game "
                           f"whose offered strikes are all outside that range is skipped.")
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

    def _pick_strike(self, ctx: DecisionContext, side: str
                     ) -> tuple[Quote | None, list[float], tuple[str, str] | None]:
        """Choose which of the exchange's offered strikes this rule trades.

        Returns ``(quote, strikes_offered, blocking)``.  The exchange lists a whole ladder of
        "Over k.5" contracts per game, so "the total" is ambiguous until the rule says which
        rung it means.  The choice is made from :attr:`target_strike`, declared in the seed
        before any price is seen:

        * only strikes inside ``[min_strike, max_strike]`` are candidates -- an "Over 1.5"
          contract is a near-certainty priced at 0.99 and is not a totals bet;
        * among those, the one nearest the target wins, ties going to the lower strike so the
          choice is deterministic and reproducible;
        * a contract with no readable strike is never a candidate, and never guessed;
        * if nothing is tradable the caller gets a blocking reason that lists what *was*
          offered, so the ledger shows the book rather than a bare "no".
        """
        cands = self.find_quotes(ctx, "total", self.direction, side)
        q, offered, code = pick_strike(cands, target=self.target_strike, lo=self.min_strike,
                                       hi=self.max_strike)
        if code is None:
            return q, offered, None
        if code == "none_quoted":
            return None, offered, ("WATCHING", "no KXNHLTOTAL contract quoted for this game")
        if code == "no_readable_strike":
            return None, offered, ("WAITING FOR OTHER INFORMATION",
                                   f"{len(cands)} contract(s) quoted but none carries a readable "
                                   "floor_strike/strike_type; the line is not known, so it is "
                                   "not assumed")
        return None, offered, ("WATCHING",
                               f"strikes offered {[f'{k:g}' for k in offered]} are all "
                               f"outside the traded range "
                               f"[{self.min_strike:g}, {self.max_strike:g}]")

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

        q, offered, why = self._pick_strike(ctx, side)
        if q is None:
            return [Signal(status=why[0], quote=None, blocking_reason=why[1], **base)]
        base["quote"] = q
        strike = float(q.strike)
        base["supporting"] = {**base["supporting"], "strike": strike,
                              "contract": q.contract, "contract_label": q.label,
                              "price_basis": q.price_basis,
                              "strikes_offered": offered, "target_strike": self.target_strike,
                              "strike_choice": f"offered strike nearest {self.target_strike:g} "
                                               f"inside [{self.min_strike}, {self.max_strike}]"}

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


# ------------------------------------------------------------------ strike ladders
def pick_strike(cands: Sequence[Quote], *, target: float, lo: float, hi: float
                ) -> tuple[Quote | None, list[float], str | None]:
    """Choose which rung of a strike ladder a rule trades.

    Shared by the totals and puck-line rules because both markets are quoted as a ladder --
    one contract per strike -- and in both cases "the line" is ambiguous until the rule says
    which rung it means.  Returns ``(quote, strikes_offered, reason_code)``; ``reason_code``
    is None when a rung was picked, otherwise one of ``none_quoted``,
    ``no_readable_strike`` or ``out_of_range``, and the calling strategy turns it into the
    blocking reason it records in the ledger.

    The rung is chosen from a target declared in the seed, before any price is seen, and a
    contract with no readable strike is never a candidate: on 2026-09-21 the same puck-line
    event carried ``-VGK3`` at 2.5 in the live tier while the historical tier carried
    ``-VGK2`` at 2.5, so anything read out of a ticker suffix would have traded the wrong
    line on one of the two.
    """
    if not cands:
        return None, [], "none_quoted"
    known: list[tuple[float, Quote]] = []
    for c in cands:
        st = getattr(c, "strike", None)
        if st is None:
            continue
        try:
            known.append((float(st), c))
        except (TypeError, ValueError):
            continue
    offered = sorted({k for k, _ in known})
    if not known:
        return None, offered, "no_readable_strike"
    in_range = [(k, c) for k, c in known if lo <= k <= hi]
    if not in_range:
        return None, offered, "out_of_range"
    in_range.sort(key=lambda kc: (abs(kc[0] - target), kc[0]))
    return in_range[0][1], offered, None


# --------------------------------------------------------------------- puck line
class PuckLineStrategy(ThresholdStrategy):
    """Trade a Kalshi KXNHLSPREAD contract ("<team> wins by over k.5 goals").

    Why this market deserves its own rule instead of being folded into the moneyline: the
    payoff is a *margin*, not a winner, so the model has to answer a different question --
    P(final margin > k) -- and the answer is not the moneyline probability.  A -1.5 favourite
    is a much bigger claim than "wins", because a one-goal win (including every overtime and
    shootout win, whose official margin is exactly one) loses the contract.

    Verified contract shape (2026-09-21, live and historical tiers):
    ``strike_type='greater'``, ``floor_strike`` = k.5, one contract per team per rung, and
    the numeric suffix in the ticker is a rung index, **not** the line.

    Two sides are possible and, exactly as for totals, they are NOT symmetric in what can be
    verified:

    * ``YES`` buys "<team> covers -k".  Kalshi's candlestick history publishes ``yes_ask``,
      so a YES rule can be BACKTESTED at real, timestamped offers.
    * ``NO`` buys "<team> does not cover -k", which is the standard ``+k`` puck line on the
      opponent.  The candle feed publishes no NO-side offer and ``no_ask = 1 - yes_bid`` is
      contradicted by this ledger's own quotes, so a NO rule carries ``no_history_reason``
      and is FORWARD TEST only, entered at the live ``no_ask_dollars``.
    """

    strategy_id = "NHL_PUCK_LINE"
    category = "puck_line"
    market = "puck_line"
    markets = "puck_line"

    #: which team's contract the rule looks at
    CONTRACT_TEAMS = ("home", "away", "model_stronger", "model_weaker")

    def __init__(self, *, contract_team: str = "model_stronger", exchange_side: str = "YES",
                 target_strike: float = 1.5, min_strike: float = 1.5, max_strike: float = 2.5,
                 min_edge: float = 0.04, min_price: float = 0.05,
                 no_history_reason: str | None = None, **kw: Any):
        if contract_team not in self.CONTRACT_TEAMS:
            raise ValueError(f"contract_team must be one of {self.CONTRACT_TEAMS}")
        if exchange_side not in ("YES", "NO"):
            raise ValueError("exchange_side must be YES or NO")
        side_token = "home" if contract_team in ("home", "model_stronger", "model_weaker") else "away"
        kw.setdefault("feature", "home_n_prior")
        kw.setdefault("operator", ">=")
        kw.setdefault("threshold", 0)
        kw.setdefault("bet_side", side_token)
        kw.setdefault("use_model", "poisson")
        kw["min_edge"] = min_edge
        kw["min_price"] = min_price
        super().__init__(**kw)
        self.contract_team = contract_team
        self.exchange_side = exchange_side
        self.quote_side = exchange_side
        self.market = "puck_line"
        self.markets = "puck_line"
        self.target_strike = float(target_strike)
        self.min_strike = float(min_strike)
        self.max_strike = float(max_strike)
        self.no_history_reason = no_history_reason
        self.settlement_rule = (
            "Official NHL final score. A YES contract on team T at strike k pays when T's final "
            "goal differential (regulation, overtime and the shootout goal the official score "
            "already contains) is strictly greater than k; a NO contract pays otherwise. Kalshi's "
            "own settlement result is cross-checked against it and a disagreement is recorded "
            "with both values and left unsettled rather than resolved silently.")
        self.params = {**self.params, "kind": "puck_line", "contract_team": contract_team,
                       "exchange_side": exchange_side, "market": "puck_line",
                       "target_strike": self.target_strike, "min_strike": self.min_strike,
                       "max_strike": self.max_strike, "no_history_reason": no_history_reason,
                       "quote_side": self.quote_side}
        verb = "covers" if exchange_side == "YES" else "fails to cover"
        who = {"model_stronger": "the team the expected-goals model rates stronger",
               "model_weaker": "the team the expected-goals model rates weaker",
               "home": "the home team", "away": "the away team"}[contract_team]
        self.hypothesis = (
            f"The independent-Poisson margin distribution prices '{who} {verb} the "
            f"{self.target_strike:g}-goal line' more accurately than the exchange does, so "
            f"buying the {exchange_side} side when the offer is at least {min_edge:.3f} below "
            f"the model probability earns the difference. A one-goal win -- including every "
            f"overtime and shootout win -- loses a -1.5 contract, which is a different claim "
            f"from the moneyline and is priced separately here.")
        self.entry_rule = (
            f"At decision time compute expected goals from prior games only, then "
            f"P(margin > k) for the offered rung k nearest {self.target_strike:g} inside "
            f"[{self.min_strike:g}, {self.max_strike:g}]; buy the {exchange_side} side of "
            f"{who}'s contract when its offer is at or below that probability minus "
            f"{min_edge:.3f}. A game whose offered rungs are all outside that range is "
            f"skipped, and a contract with no readable floor_strike is never traded.")
        self.price_rule = (f"Require an executable {exchange_side}-side offer <= model "
                           f"P(cover) - {self.min_edge:.3f}; never bet into a stale or "
                           f"missing quote.")
        self.data_used = ("kalshi.trade_api KXNHLSPREAD contracts (floor_strike, strike_type, "
                          "yes/no offers, candlesticks) + api-web.nhle.com results for the "
                          "expected-goals margin distribution.")
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

    def _resolve_team(self, ctx: DecisionContext, lam: tuple[float, float]) -> tuple[str | None, str]:
        """Which team's contract this rule trades, decided from model output only."""
        from .models import p_margin_over
        if self.contract_team in ("home", "away"):
            return self.contract_team, f"contract_team={self.contract_team} (declared in the rule)"
        ph = p_margin_over(lam[0], lam[1], "home", self.target_strike)
        pa = p_margin_over(lam[0], lam[1], "away", self.target_strike)
        if ph != ph or pa != pa:      # NaN
            return None, "margin model returned no probability"
        stronger = "home" if ph >= pa else "away"
        pick = stronger if self.contract_team == "model_stronger" else (
            "away" if stronger == "home" else "home")
        return pick, (f"P(home covers {self.target_strike:g})={ph:.4f}, "
                      f"P(away covers {self.target_strike:g})={pa:.4f} -> "
                      f"{self.contract_team}={pick}")

    def evaluate(self, ctx: DecisionContext) -> list[Signal]:
        from .models import p_margin_over
        f = ctx.features
        base: dict[str, Any] = dict(
            strategy_id=self.strategy_id, version=self.version, username=self.username,
            game_id=int(f.get("game_id") or 0), game_date=f.get("game_date") or "",
            matchup=f"{f.get('away_id')}@{f.get('home_id')}", market="puck_line",
            selection="puck_line", side=self.exchange_side, model_prob=float("nan"),
            fair_price=float("nan"), required_price=float("nan"), stake=0.0,
            supporting={"market": "puck_line", "exchange_side": self.exchange_side,
                        "contract_team_rule": self.contract_team})

        lam = self._lambdas(ctx)
        if lam is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="no expected-goals model output for this game", **base)]
        team, why_team = self._resolve_team(ctx, lam)
        if team is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason=why_team, **base)]
        base["supporting"] = {**base["supporting"], "contract_team": team,
                              "contract_team_reason": why_team}

        cands = self.find_quotes(ctx, "puck_line", team, self.exchange_side)
        q, offered, code = pick_strike(cands, target=self.target_strike, lo=self.min_strike,
                                       hi=self.max_strike)
        if q is None:
            if code == "none_quoted":
                return [Signal(status="WATCHING",
                               blocking_reason=f"no KXNHLSPREAD contract quoted for the {team} "
                                               f"side of this game", **base)]
            if code == "no_readable_strike":
                return [Signal(status="WAITING FOR OTHER INFORMATION",
                               blocking_reason=f"{len(cands)} contract(s) quoted but none carries "
                                               "a readable floor_strike/strike_type; the line is "
                                               "not known, so it is not assumed", **base)]
            return [Signal(status="WATCHING",
                           blocking_reason=f"strikes offered {[f'{k:g}' for k in offered]} are all "
                                           f"outside the traded range [{self.min_strike:g}, "
                                           f"{self.max_strike:g}]", **base)]
        strike = float(q.strike)
        base["quote"] = q
        base["selection"] = f"{team}_cover" if self.exchange_side == "YES" else f"{team}_no_cover"
        p_cover = p_margin_over(lam[0], lam[1], team, strike)
        if p_cover != p_cover:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason=f"margin model returned no probability for strike "
                                           f"{strike:g}", **base)]
        p = p_cover if self.exchange_side == "YES" else round(1.0 - p_cover, 6)
        base["model_prob"] = p
        base["supporting"] = {**base["supporting"], "strike": strike, "contract": q.contract,
                              "contract_label": q.label, "price_basis": q.price_basis,
                              "strikes_offered": offered, "target_strike": self.target_strike,
                              "strike_choice": f"offered strike nearest {self.target_strike:g} "
                                               f"inside [{self.min_strike}, {self.max_strike}]",
                              "p_cover": p_cover, "lam_home": lam[0], "lam_away": lam[1],
                              "data_labels": {"price": f"SOURCE DATA (kalshi, {q.price_basis})",
                                              "features": "DERIVED (point-in-time)",
                                              "model_prob": "MODEL OUTPUT (independent-Poisson "
                                                            "margin distribution)"}}

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
                                           f"(model P={p:.4f})", **base)]
        if q.ask_size is not None and float(q.ask_size) <= 0:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="zero offer size at the quoted price", **base)]
        stake = self.size_stake(p, float(ask), ctx.bankroll, ctx.open_exposure)
        if stake <= 0:
            return [Signal(status="CANCELLED", blocking_reason="zero bankroll available", **base)]
        base.update(stake=stake)
        return [Signal(status="READY TO BET", **base)]


# --------------------------------------------------------------------- overtime
class OvertimeStrategy(ThresholdStrategy):
    """Trade a Kalshi KXNHLOVERTIME contract ("will this game go to overtime?").

    The contract is strike-less (verified 2026-09-21 on the historical tier: no
    ``floor_strike`` and no ``strike_type``, one ``-OT`` contract per game) and pays on a
    single question, so it needs neither a ladder nor a team -- it needs an honest
    probability that a game is tied after regulation.

    The independent-Poisson tie mass is *not* that probability by itself: real NHL games go
    past regulation at a rate the raw model tends to miss, because scoring is correlated and
    because a trailing team changes its behaviour.  So the rule prices the contract with a
    tie mass rescaled by :class:`nhlcomp.models.OtCalibration`, fitted on a chronologically
    earlier window of official results only, and it refuses to trade when that calibration
    has too little history behind it.  The calibration's own evidence (n, observed rate,
    model mean, window) is written onto every signal.

    **Definition risk, recorded rather than assumed.**  Kalshi's rules text says only "go to
    overtime".  Every settled contract recovered so far is a playoff game, where a shootout
    is impossible, so the recovered history cannot say whether a regular-season game decided
    in a shootout settles YES.  This rule therefore treats "past regulation" as
    ``lastPeriodType in (OT, SO)`` -- the reading the NHL's own outcome field supports -- and
    ``verify.py`` cross-tabulates every settled Kalshi OT contract against the NHL's
    ``lastPeriodType`` and records the result, so the definition is measured on real
    settlements instead of argued about.  A BACKTEST row settles on the **exchange's own**
    result, never on this project's reading of it.
    """

    strategy_id = "NHL_OVERTIME"
    category = "ot_shootout"
    market = "overtime"
    markets = "overtime"

    def __init__(self, *, direction: str = "yes", min_edge: float = 0.03, min_price: float = 0.05,
                 require_calibration: bool = True, no_history_reason: str | None = None, **kw: Any):
        if direction not in ("yes", "no"):
            raise ValueError("direction must be 'yes' (game goes to OT) or 'no'")
        kw.setdefault("feature", "home_n_prior")
        kw.setdefault("operator", ">=")
        kw.setdefault("threshold", 0)
        kw.setdefault("bet_side", "ot")
        kw.setdefault("use_model", "poisson")
        kw["min_edge"] = min_edge
        kw["min_price"] = min_price
        super().__init__(**kw)
        self.direction = direction
        self.quote_side = "YES" if direction == "yes" else "NO"
        self.market = "overtime"
        self.markets = "overtime"
        self.require_calibration = bool(require_calibration)
        self.no_history_reason = no_history_reason
        self.settlement_rule = (
            "The exchange's own settlement result for BACKTEST rows. For FORWARD TEST rows the "
            "official NHL lastPeriodType (OT or SO = the game went past regulation, REG = it did "
            "not), cross-checked against Kalshi's result when the contract settles; a "
            "disagreement is recorded with both values and the row is left OPEN.")
        self.params = {**self.params, "kind": "overtime", "direction": direction,
                       "market": "overtime", "require_calibration": self.require_calibration,
                       "no_history_reason": no_history_reason,
                       "quote_side": self.quote_side}
        buys = "game goes to overtime" if direction == "yes" else "game is decided in regulation"
        self.hypothesis = (
            f"The Poisson tie mass, rescaled by the observed rate of games that went past "
            f"regulation on an earlier chronological window, prices '{buys}' more accurately "
            f"than the exchange does; buying the {direction.upper()} side when the offer is at "
            f"least {min_edge:.3f} below that probability earns the difference.")
        self.entry_rule = (
            f"At decision time compute P(tied after regulation) from prior games only, rescale "
            f"it by the fitted OT calibration, and buy the {direction.upper()} side of the "
            f"KXNHLOVERTIME contract when the offer is at or below that probability minus "
            f"{min_edge:.3f}. If the calibration's sample is below its minimum the rule does "
            f"not trade.")
        self.price_rule = (f"Require an executable {direction.upper()}-side offer <= model P - "
                           f"{self.min_edge:.3f}; never bet into a stale or missing quote.")
        self.data_used = ("kalshi.trade_api KXNHLOVERTIME contracts (offers, candlesticks, own "
                          "settlement result) + api-web.nhle.com lastPeriodType for the observed "
                          "overtime rate.")
        if no_history_reason:
            self.data_used += f" BACKTEST NOTE: {no_history_reason}"

    def evaluate(self, ctx: DecisionContext) -> list[Signal]:
        f = ctx.features
        side = "YES" if self.direction == "yes" else "NO"
        base: dict[str, Any] = dict(
            strategy_id=self.strategy_id, version=self.version, username=self.username,
            game_id=int(f.get("game_id") or 0), game_date=f.get("game_date") or "",
            matchup=f"{f.get('away_id')}@{f.get('home_id')}", market="overtime",
            selection=f"ot_{self.direction}", side=side, model_prob=float("nan"),
            fair_price=float("nan"), required_price=float("nan"), stake=0.0,
            supporting={"market": "overtime", "direction": self.direction})

        cal = ctx.predictions.get("ot_calibration") or {}
        p_raw = ctx.predictions.get("p_overtime")
        p_cal = ctx.predictions.get("p_ot_cal")
        if p_raw is None and p_cal is None:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason="no tie-after-regulation probability for this game",
                           **base)]
        use_cal = p_cal is not None and cal.get("sufficient")
        if p_cal is not None and not cal.get("sufficient") and self.require_calibration:
            return [Signal(status="WAITING FOR OTHER INFORMATION",
                           blocking_reason=f"the overtime calibration is fitted on n="
                                           f"{cal.get('n')} games, below its minimum of "
                                           f"{cal.get('min_n', 100)}; the raw Poisson tie mass is "
                                           f"not checked against observed results, so this rule "
                                           f"does not trade on it", **base)]
        p_raw_v = float(p_raw) if p_raw is not None else float("nan")
        p = float(p_cal if use_cal else p_raw_v)
        if self.direction == "no":
            p = round(1.0 - p, 6)
        base["model_prob"] = p
        base["supporting"] = {**base["supporting"], "p_tie_reg_raw": p_raw_v,
                              "p_ot_used": p, "calibrated": bool(use_cal),
                              "ot_calibration": cal,
                              "data_labels": {"features": "DERIVED (point-in-time)",
                                              "model_prob": "MODEL OUTPUT (Poisson tie mass"
                                                            + (", calibrated)" if use_cal else ")")}}

        q = self.find_quote(ctx, "overtime", "ot", side)
        if q is None:
            return [Signal(status="QUALIFIED",
                           blocking_reason="condition priced, but no KXNHLOVERTIME contract "
                                           f"quoted for this game ({side} side)", **base)]
        base["quote"] = q
        base["supporting"] = {**base["supporting"], "contract": q.contract,
                              "contract_label": q.label, "price_basis": q.price_basis}

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
                                           f"(model P={p:.4f})", **base)]
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

    # -- totals (KXNHLTOTAL).  The exchange lists one contract per strike ("Over k.5") --
    # eight of them on the 2026-09-24 slate, 1.5 through 8.5 -- so each rule declares the
    # strike it means (target_strike) before any price is seen and trades the offered rung
    # nearest to it.  Buying YES is an Over bet and buying NO is an Under bet.  Only the Over
    # side has a verified historical price (the candlestick feed publishes the YES ask), so
    # the Under rule declares why it cannot be backtested instead of inventing a NO-side
    # history.  6.5 is the NHL's most common full-game line; 5.5 is the low rung of the
    # traded range and is paired with the tighter edge so the two variants differ in one
    # dimension at a time.
    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_OVER", username="NHL_TOTALS_OVER_030", category="totals",
        name="Expected-goals model over the exchange total", direction="over", min_edge=0.04,
        target_strike=6.5,
        data_used="kalshi.trade_api KXNHLTOTAL floor_strike + yes_ask candlesticks (verified "
                  "2026-09-21, historical tier back to the 2026 Stanley Cup Final); "
                  "api-web.nhle.com results for expected goals."))
    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_UNDER", username="NHL_TOTALS_UNDER_031", category="totals",
        name="Expected-goals model under the exchange total (forward only)", direction="under",
        min_edge=0.04, target_strike=6.5,
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
        direction="over", min_edge=0.02, target_strike=5.5,
        data_used="kalshi.trade_api KXNHLTOTAL floor_strike + yes_ask candlesticks (verified); "
                  "api-web.nhle.com results for expected goals."))

    # -- puck line (Kalshi KXNHLSPREAD, "<team> wins by over k.5 goals").  Verified shape
    # 2026-09-21 on both tiers: strike_type='greater', floor_strike=k.5, one contract per
    # team per rung, and a ticker suffix digit that is a rung index rather than the line
    # (live -VGK3 = 2.5 while historical -VGK2 = 2.5).  A margin is a different question
    # from a winner, so these rules price P(margin > k) and not the moneyline.
    out.append(PuckLineStrategy(
        strategy_id="NHL_PUCK_LINE_MODEL_COVER", username="NHL_PUCK_LINE_MODEL_COVER_033",
        category="puck_line", name="Model margin distribution covers the -1.5 puck line",
        contract_team="model_stronger", exchange_side="YES", target_strike=1.5,
        min_strike=1.5, max_strike=2.5, min_edge=0.04,
        data_used="kalshi.trade_api KXNHLSPREAD floor_strike/strike_type + yes_ask "
                  "candlesticks (verified 2026-09-21, live tier quoting the 2026-09-24 slate "
                  "and historical tier back to the 2026 Stanley Cup Final, 182,606.94 contracts "
                  "of volume on one rung); api-web.nhle.com results for expected goals."))
    # The mirror image: NO on the stronger team's -1.5 contract IS the weaker team's +1.5
    # puck line.  The candle feed publishes the YES offer only, and no_ask = 1 - yes_bid does
    # not hold on this ledger's quotes, so this rule declares that it has no history and is
    # forward-tested at the live no_ask_dollars instead of being backtested at an invented
    # price.
    out.append(PuckLineStrategy(
        strategy_id="NHL_PUCK_LINE_DOG_PLUS", username="NHL_PUCK_LINE_DOG_PLUS_034",
        category="puck_line", name="Model's weaker side covers +1.5 (forward only)",
        contract_team="model_stronger", exchange_side="NO", target_strike=1.5,
        min_strike=1.5, max_strike=2.5, min_edge=0.04,
        no_history_reason="Kalshi's candlestick feed publishes the YES bid/ask only, so no "
                          "historical NO-side offer exists to buy a +1.5 puck line at, and "
                          "no_ask = 1 - yes_bid does not hold on the quotes in this ledger. "
                          "FORWARD TEST ONLY, entered at the live no_ask_dollars."))
    # A second rung of the same ladder, so the strike itself is an experiment rather than a
    # tuned constant: -2.5 pays only on a multi-goal win, which the Poisson margin
    # distribution and the moneyline disagree about most sharply.
    out.append(PuckLineStrategy(
        strategy_id="NHL_PUCK_LINE_HOME_25", username="NHL_PUCK_LINE_HOME_25_035",
        category="puck_line", name="Home team covers the -2.5 rung", contract_team="home",
        exchange_side="YES", target_strike=2.5, min_strike=2.5, max_strike=3.5, min_edge=0.05,
        data_used="kalshi.trade_api KXNHLSPREAD floor_strike/strike_type + yes_ask "
                  "candlesticks (verified 2026-09-21); api-web.nhle.com results."))

    # -- overtime / shootout (Kalshi KXNHLOVERTIME, one strike-less -OT contract per game).
    # Priced with the Poisson tie mass rescaled by the observed rate of games that went past
    # regulation on an earlier chronological window; the rule refuses to trade when that
    # calibration has too little history.  Verified 2026-09-21: the historical tier holds
    # settled OT contracts with real volume (19,512.97 and 39,558.66 on two 2026 Final games)
    # and no floor_strike/strike_type at all.
    out.append(OvertimeStrategy(
        strategy_id="NHL_OT_MODEL_YES", username="NHL_OT_MODEL_YES_036", category="ot_shootout",
        name="Calibrated tie mass over the exchange's overtime price", direction="yes",
        min_edge=0.03,
        data_used="kalshi.trade_api KXNHLOVERTIME contracts (verified 2026-09-21, historical "
                  "tier settled contracts with volume; no open contracts were listed on that "
                  "date, so the live side is captured from the day the exchange lists one); "
                  "api-web.nhle.com lastPeriodType for the observed overtime rate."))
    out.append(OvertimeStrategy(
        strategy_id="NHL_OT_MODEL_NO", username="NHL_OT_MODEL_NO_037", category="ot_shootout",
        name="Calibrated tie mass under the exchange's overtime price (forward only)",
        direction="no", min_edge=0.03,
        no_history_reason="The NO side of KXNHLOVERTIME has no historical offer in Kalshi's "
                          "candlestick feed (YES bid/ask only), so this rule is FORWARD TEST "
                          "ONLY, entered at the live no_ask_dollars."))

    # -- missing market families required by brief, registered as blocked so the absence is
    # a decision, not an oversight. Each has no verified price history on 2026-09-21 and/or
    # no model, so nothing is bet even if a quote appears.

    # team totals: per-team over/under, distinct from full-game totals
    out.append(ThresholdStrategy(
        strategy_id="NHL_TEAM_TOTAL_OVER", username="NHL_TEAM_TOTAL_OVER_038", category="team_totals",
        name="Team totals over (blocked: no verified market shape yet)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="team_total",
        blocked_reason="KXNHLTEAMTOTAL series not observed in /series listing on 2026-09-21; no contract shape, no price history, no settlement rule verified. Blocked until shape appears.",
        data_used="NONE VERIFIED for team totals market shape."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_TEAM_TOTAL_UNDER", username="NHL_TEAM_TOTAL_UNDER_039", category="team_totals",
        name="Team totals under (blocked)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="team_total",
        blocked_reason="Team totals market not observed; no price history, no model for team-specific scoring vs opponent defense separately.",
        data_used="NONE VERIFIED."))

    # regulation: win in regulation vs OT/SO
    out.append(ThresholdStrategy(
        strategy_id="NHL_REGULATION", username="NHL_REGULATION_040", category="regulation",
        name="Regulation win (blocked: no verified contracts)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="regulation",
        blocked_reason="KXNHLREG / KXNHL60MIN series not observed on 2026-09-21; no contracts, no price history. Settlement would be lastPeriodType==REG and winner, already stored, but market absent.",
        data_used="NONE VERIFIED for regulation market prices."))

    # period moneyline: 1P/2P/3P
    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_ML_1P", username="NHL_PERIOD_ML_1P_041", category="period_betting",
        name="First-period moneyline (blocked: market empty)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="period",
        blocked_reason="KXNHL1P listed in /series but 0 open and 0 historical contracts on 2026-09-21 (verified negative). No price history, no period-level expected-goals model yet.",
        data_used="game_period_scores table ingests period goals for research, but no period model exists; kalshi_series_watch records verified negative."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_ML_2P", username="NHL_PERIOD_ML_2P_042", category="period_betting",
        name="Second-period moneyline (blocked)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="period",
        blocked_reason="KXNHL2P empty on 2026-09-21; no price history, no period model.",
        data_used="NONE VERIFIED for period prices."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_ML_3P", username="NHL_PERIOD_ML_3P_043", category="period_betting",
        name="Third-period moneyline (blocked)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="period",
        blocked_reason="KXNHL3P empty on 2026-09-21; no price history.",
        data_used="NONE VERIFIED."))

    # period totals
    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_TOTAL_1P_OVER", username="NHL_PERIOD_TOTAL_1P_OVER_044",
        category="period_totals", name="First-period total over (blocked)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="over", market="period_total",
        blocked_reason="KXNHL1PTOTAL empty on 2026-09-21; no price history; period totals typically 0.5/1.5/2.5 and need period-level scoring model.",
        data_used="NONE VERIFIED for period total prices."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_TOTAL_2P_OVER", username="NHL_PERIOD_TOTAL_2P_OVER_045",
        category="period_totals", name="Second-period total over (blocked)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="over", market="period_total",
        blocked_reason="KXNHL2PTOTAL empty on 2026-09-21; no price history.",
        data_used="NONE VERIFIED."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_TOTAL_3P_OVER", username="NHL_PERIOD_TOTAL_3P_OVER_046",
        category="period_totals", name="Third-period total over (blocked)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="over", market="period_total",
        blocked_reason="KXNHL3PTOTAL empty on 2026-09-21; no price history.",
        data_used="NONE VERIFIED."))

    # alternate lines: explicitly trade 5.5 and 7.5 vs 6.5 standard
    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_ALT_55_OVER", username="NHL_TOTALS_ALT_55_OVER_047",
        category="alternate_lines", name="Alternate total 5.5 over (tight line value)",
        direction="over", min_edge=0.03, target_strike=5.5, min_strike=5.5, max_strike=5.5,
        data_used="kalshi.trade_api KXNHLTOTAL alternate line 5.5 (verified ladder rung); same data as standard totals."))

    out.append(TotalsStrategy(
        strategy_id="NHL_TOTALS_ALT_75_UNDER", username="NHL_TOTALS_ALT_75_UNDER_048",
        category="alternate_lines", name="Alternate total 7.5 under (high line fade)",
        direction="under", min_edge=0.03, target_strike=7.5, min_strike=7.5, max_strike=7.5,
        no_history_reason="Under side has no historical NO-side offer; forward-only, same as other unders."))

    out.append(PuckLineStrategy(
        strategy_id="NHL_PUCK_ALT_25", username="NHL_PUCK_ALT_25_049",
        category="alternate_lines", name="Alternate puck line -2.5 cover (multi-goal)",
        contract_team="model_stronger", exchange_side="YES", target_strike=2.5,
        min_strike=2.5, max_strike=2.5, min_edge=0.05,
        data_used="kalshi KXNHLSPREAD alternate rung 2.5 (verified)."))

    # player props: goal, assist, points, first goal
    out.append(ThresholdStrategy(
        strategy_id="NHL_PLAYER_GOAL", username="NHL_PLAYER_GOAL_050", category="player_props",
        name="Player to score a goal (blocked: no model, no verified pre-game feed)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="home", market="player_goal",
        blocked_reason="KXNHLGOAL listed but 0 contracts on 2026-09-21; no player-level projection model exists; no verified mapping of player names to NHL player_ids for pre-game deployment; no price history.",
        data_used="NONE VERIFIED for player props pricing/model."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PLAYER_ANYGOAL", username="NHL_PLAYER_ANYGOAL_051", category="player_props",
        name="Player anytime goal (blocked)", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", market="player_anygoal",
        blocked_reason="KXNHLANYGOAL empty on 2026-09-21; no model, no price history.",
        data_used="NONE VERIFIED."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PLAYER_ASSIST", username="NHL_PLAYER_ASSIST_052", category="player_props",
        name="Player assist (blocked)", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", market="player_assist",
        blocked_reason="KXNHLAST empty on 2026-09-21; no model.",
        data_used="NONE VERIFIED."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PLAYER_POINTS", username="NHL_PLAYER_POINTS_053", category="player_props",
        name="Player points (goal+assist) (blocked)", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", market="player_points",
        blocked_reason="KXNHLPTS empty on 2026-09-21; no model.",
        data_used="NONE VERIFIED."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_PLAYER_FIRSTGOAL", username="NHL_PLAYER_FIRSTGOAL_054", category="player_props",
        name="First goal scorer (blocked)", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", market="player_firstgoal",
        blocked_reason="KXNHLFIRSTGOAL empty on 2026-09-21; first-goal is low-probability and needs lineup confirmation; no verified feed.",
        data_used="NONE VERIFIED."))

    # goalie props: saves
    out.append(ThresholdStrategy(
        strategy_id="NHL_GOALIE_SAVES_OVER", username="NHL_GOALIE_SAVES_OVER_055", category="goalie_props",
        name="Goalie saves over (blocked: no starter feed, no model)",
        feature="home_n_prior", operator=">=", threshold=10, bet_side="over", market="goalie_saves",
        blocked_reason="KXNHLSAVES listed but 0 contracts on 2026-09-21 (verified negative); no verified pre-game starter source, so even if listed forward-test only when starter known; no saves projection model.",
        data_used="api.nhle.com goalie/summary per-game saves for research; no saves model."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_GOALIE_SAVES_UNDER", username="NHL_GOALIE_SAVES_UNDER_056", category="goalie_props",
        name="Goalie saves under (blocked)", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="under", market="goalie_saves",
        blocked_reason="KXNHLSAVES empty; no model; no price history.",
        data_used="NONE VERIFIED for goalie saves market."))

    # futures
    out.append(ThresholdStrategy(
        strategy_id="NHL_FUTURES_CUP", username="NHL_FUTURES_CUP_057", category="futures",
        name="Stanley Cup futures (blocked: no season model)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="futures",
        blocked_reason="KXNHL futures (Stanley Cup) have been listed but require season simulation model, long-horizon bankroll lockup, and settlement months in future. No season model exists here; blocked.",
        data_used="kalshi.trade_api KXNHL futures market exists but not walked for history in this project."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_FUTURES_DIVISION", username="NHL_FUTURES_DIVISION_058", category="futures",
        name="Division winner futures (blocked)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="futures",
        blocked_reason="Futures require season model; not traded.",
        data_used="NONE VERIFIED for futures pricing model."))

    # live/in-game additional blocked variants
    out.append(ThresholdStrategy(
        strategy_id="NHL_LIVE_MONEYLINE", username="NHL_LIVE_MONEYLINE_059", category="live_in_game",
        name="Live moneyline reaction (blocked: batch pipeline)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="moneyline",
        blocked_reason="Kalshi candlesticks provide 60-min in-game points for research (ig60/ig120), but paper-trading loop runs batch and cannot act during a game; no live execution.",
        data_used="kalshi candlesticks intraday for research only."))

    out.append(ThresholdStrategy(
        strategy_id="NHL_LIVE_TOTALS", username="NHL_LIVE_TOTALS_060", category="live_in_game",
        name="Live totals reaction (blocked)", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="over", market="total",
        blocked_reason="Live totals would require in-game scoring model and live price feed execution; batch pipeline cannot execute intraday.",
        data_used="NONE VERIFIED for live execution."))

    # period spread
    out.append(ThresholdStrategy(
        strategy_id="NHL_PERIOD_SPREAD", username="NHL_PERIOD_SPREAD_061", category="period_spread",
        name="Period spread (blocked: no verified market)", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", market="period_spread",
        blocked_reason="No KXNHL period spread series observed; period spread would be margin in a single period, needs period model and market listing.",
        data_used="NONE VERIFIED."))

    return out
