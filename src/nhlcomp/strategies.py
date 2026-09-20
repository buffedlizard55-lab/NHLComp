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
        self.params = {"feature": feature, "side_feature": side_feature, "operator": operator,
                       "threshold": threshold, "bet_side": bet_side, "market": market,
                       "use_model": use_model, "min_edge": self.min_edge,
                       "requires_goalie": self.requires_goalie,
                       "requires_lineup": self.requires_lineup,
                       "injury_sensitive": self.injury_sensitive, "min_price": self.min_price,
                       "blocked_reason": self.blocked_reason,
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

    add(strategy_id="NHL_GOALIE_NEWS", username="NHL_GOALIE_NEWS_012", category="goalie_news",
        name="Market reaction to a goalie announcement", feature="home_n_prior",
        operator=">=", threshold=10, bet_side="home", requires_goalie=True,
        injury_sensitive=True, min_edge=0.06,
        data_used="espn.nhl_api injuries (verified, cross-check only); kalshi.trade_api quotes. "
                  "Goalie announcements: NO VERIFIED SOURCE.",
        hypothesis="A starter announcement moves the price, and the first mover is paid. "
                   "UNTESTED: neither the announcement feed nor intraday prices exist here.",
        entry_rule="Blocked: requires the confirmed starter and a pre-announcement price.")

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

    add(strategy_id="NHL_PERIOD_BET", username="NHL_PERIOD_BET_019", category="period_betting",
        name="Period-specific value", feature="home_n_prior", operator=">=", threshold=10,
        bet_side="home", min_edge=0.05,
        blocked_reason="Kalshi KXNHLGAME exposes game-level moneyline contracts only; no "
                       "verified first-period or period-specific market was found, and none is "
                       "assumed to exist",
        data_used="kalshi.trade_api KXNHLGAME (game-level only, verified).",
        hypothesis="Period markets are less efficient than game markets. UNTESTED.",
        entry_rule="Blocked: requires a verified period-level market.")

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

    return out
