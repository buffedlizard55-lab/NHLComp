"""Paper-trading engine: decision -> quote -> simulated execution -> settlement.

Execution model (explicit, because the brief forbids assuming unlimited liquidity):

* We buy at the **offer (ask)**, never the mid and never the bid.
* The number of contracts we can buy at that price is capped by ``yes_ask_size_fp``
  from the quote.  If our desired stake needs more contracts than are offered we take a
  **partial fill** and record the unfilled remainder -- we never invent a second price
  level that the quote did not show.
* Slippage is reported as 0 only because a single book level is all we have; the field is
  kept so a multi-level order book can populate it later.
* A missing or zero-size offer means **no bet**, recorded as a blocking status.
* Kalshi's general taker fee (``0.07 x C x P x (1-P)``, fee schedule effective 2026-07-07)
  is charged on every simulated fill and deducted from the settled P&L, because a paper
  result that ignores a real, published fee overstates what the strategy earns.

Nothing in this module can place a real order: there is no order-placement code at all.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .market import covers_margin, kalshi_taker_fee
from .store import Store, utcnow
from .strategies import DecisionContext, Quote, Signal, Strategy


#: Contracts this engine is willing to assume it could take when the venue publishes **no**
#: depth for the side being bought and no traded volume either.  Kalshi publishes no NO-side
#: offer size at all -- every one of the 302 NO-side rows in the 2026-09-21 ledger has
#: ``ask_size`` NULL -- so a NO-side rule (an Under, a +1.5 puck line, a "no overtime") would
#: otherwise fill any size it asked for, which is exactly the unlimited-liquidity assumption
#: the brief forbids.  This is a declared, bounded ASSUMPTION, not source data: every fill
#: that falls back to it is stamped ``depth_basis='declared_cap_no_published_size'`` and its
#: unfilled remainder is recorded, so a reader can see which wagers rest on it and how much
#: of the intended stake the assumption is carrying.
UNKNOWN_DEPTH_CAP_CONTRACTS = 100.0

#: Where a price comes from a candlestick rather than a live book there is no offer size, but
#: the candle does publish how many contracts actually traded in that hour.  A taker cannot
#: have been filled more than the hour traded, so that volume is used as the depth evidence
#: (SOURCE DATA) and a candle in which nothing traded yields no fill at all rather than an
#: imagined one.
DEPTH_BASIS_OFFER = "published_offer_size"
DEPTH_BASIS_VOLUME = "traded_volume_at_entry_candle"
DEPTH_BASIS_CAP = "declared_cap_no_published_size"
#: the fill was cut short not by the book itself but by depth an earlier wager had already
#: taken from the same book level (see :class:`PaperEngine` on shared depth)
DEPTH_BASIS_SHARED = "shared_with_earlier_wagers"

#: A two-sided book wider than this is not treated as a tradeable offer.  On a thin
#: preseason NHL market Kalshi does publish rows such as bid 0.11 / ask 0.74 -- a 0.63-wide
#: book with an 87-contract "offer" on the ask -- and filling against one produces a wager
#: whose price no participant would ever have met.  This is a declared ASSUMPTION (a chosen
#: threshold, not source data); every entry it blocks is recorded as an irregularity with the
#: bid, the ask and the spread, never dropped in silence.  The 2025-26 regular-season
#: book that this project *did* backtest against had a median spread of 0.02 and an
#: interquartile range well under 0.10, so the threshold does not touch the liquid market.
WIDE_BOOK_MAX_SPREAD = 0.50


@dataclass
class Fill:
    contracts_requested: float
    contracts_filled: float
    price: float
    stake: float
    unfilled_stake: float
    slippage: float
    liquidity: float | None
    #: which depth evidence capped this fill (see DEPTH_BASIS_*)
    depth_basis: str = DEPTH_BASIS_OFFER
    #: contracts still claimable at this book level after this fill, when a shared budget was
    #: applied (None when the caller did not ask for one)
    depth_remaining_after: float | None = None


def evidence_depth(ask_size: float | None, traded_volume: float | None, *,
                   unknown_cap: float = UNKNOWN_DEPTH_CAP_CONTRACTS
                   ) -> tuple[float, float | None, str]:
    """Best available evidence for how many contracts are on offer, and its basis.

    Order of evidence, strongest first:

    1. ``ask_size`` -- the offer size the venue published for the side being bought.
    2. ``traded_volume`` -- contracts that actually traded in the entry candle (a candle has
       no book size, but it does have volume).  Zero means no fill: nothing traded at that
       price in that hour, so claiming a fill would be inventing liquidity.
    3. ``unknown_cap`` -- a declared assumption of last resort, labelled as such on the row.

    Returns ``(capacity_in_contracts, liquidity_field, depth_basis)``.  ``capacity`` is the
    size of the book level itself; it is *not* yet net of anything already consumed.
    """
    if ask_size is not None:
        return float(ask_size), float(ask_size), DEPTH_BASIS_OFFER
    if traded_volume is not None:
        return float(traded_volume), float(traded_volume), DEPTH_BASIS_VOLUME
    return float(unknown_cap), None, DEPTH_BASIS_CAP


def simulate_fill(stake: float, ask: float, ask_size: float | None, *,
                  traded_volume: float | None = None,
                  unknown_cap: float = UNKNOWN_DEPTH_CAP_CONTRACTS,
                  remaining: float | None = None) -> Fill:
    """Single-level fill, capped by the best depth evidence available.

    ``remaining`` is the part of that book level nobody has claimed yet.  A published offer
    size describes the *book*, not any one strategy: when several rules decide against the
    same quote they are bidding for the same contracts, and the second one cannot also have
    the whole level.  When ``remaining`` is supplied the fill is capped by it and
    ``depth_remaining_after`` reports what is left; ``None`` means the caller is not
    modelling a shared book and the evidence cap is used as-is.
    """
    capacity, liquidity, basis = evidence_depth(ask_size, traded_volume,
                                                unknown_cap=unknown_cap)
    if remaining is not None:
        if remaining < capacity:
            basis = DEPTH_BASIS_SHARED
        capacity = min(capacity, max(0.0, float(remaining)))
    if stake <= 0 or ask is None or ask <= 0:
        return Fill(0.0, 0.0, ask or 0.0, 0.0, max(stake, 0.0), 0.0, liquidity,
                    DEPTH_BASIS_OFFER if ask_size is not None else DEPTH_BASIS_CAP,
                    None if remaining is None else round(float(remaining), 4))
    requested = stake / ask
    filled = min(requested, capacity) if capacity > 0 else 0.0
    # what is left is the budget minus what this fill actually took, not minus the whole
    # level: a wager smaller than the level must leave the rest of it for the next rule
    left_after = None if remaining is None else round(max(0.0, float(remaining) - filled), 4)
    filled_stake = filled * ask
    return Fill(contracts_requested=round(requested, 4), contracts_filled=round(filled, 4),
                price=ask, stake=round(filled_stake, 4),
                unfilled_stake=round(max(stake - filled_stake, 0.0), 4),
                slippage=0.0, liquidity=liquidity, depth_basis=basis,
                depth_remaining_after=left_after)


def binary_settlement(entry_price: float, contracts: float, won: bool, *, fee: float = 0.0) -> float:
    """A YES contract costs ``entry_price`` and pays 1.00 if it wins.  ``fee`` (dollars,
    already paid at the fill) is deducted from the result in both branches."""
    gross = contracts * (1.0 - entry_price) if won else -contracts * entry_price
    return round(gross - float(fee or 0.0), 4)


def _fee_of(bet: Any) -> float:
    """Fee recorded on a bet row (0 for rows written before fees were modelled)."""
    try:
        v = bet["fee"]
    except (KeyError, IndexError):
        return 0.0
    return float(v or 0.0)


class PaperEngine:
    """Execution, settlement and the shared-depth book.

    Every wager this engine writes consumes part of a published book level, and that level
    belongs to the *market*, not to the strategy that happened to look at it first.  The
    engine therefore keeps a per-``(contract, side, quote timestamp)`` budget, seeded from
    the wagers already in the ledger and decremented by each new fill in the same run.

    This matters because the failure it prevents is invisible in the output: on the
    2026-09-21 ledger, ``KXNHLGAME-26SEP21NYRNJ-NYR`` published an ``ask_size`` of **6**
    contracts at the close, and seven strategies independently booked fills summing to
    **1,443.7** contracts against it -- each one priced as though the whole level were
    available to it alone.  Seventeen of the twenty contracts more than one strategy traded
    on that ledger were over-claimed.  A wager whose recorded price and size could not both
    have existed is not a paper trade, it is an assumption, and it was inflating every
    strategy that shared a popular quote.
    """

    def __init__(self, store: Store, *, actor: str = "paper_engine"):
        self.store = store
        self.actor = actor
        self._depth_used: dict[tuple[str, str, str], float] | None = None
        #: book levels a rule asked for and could not get, keyed by ``(contract, entry ts)``:
        #: one contract can be blocked at two entry points, and collapsing them would
        #: under-report the blocked opportunity set.  The pipeline turns each into one
        #: irregularity naming which of the two causes it was, so a blocked wager is
        #: visible rather than merely absent.
        self.depth_blocked: dict[tuple[str, str], dict[str, Any]] = {}
        #: book levels refused because the two sides were quoted implausibly far apart
        self.wide_book_skipped: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ depth book
    @staticmethod
    def depth_key(contract: str | None, side: str | None, ts: str | None) -> tuple[str, str, str]:
        return (contract or "", (side or "YES").upper(), ts or "")

    def depth_used(self) -> dict[tuple[str, str, str], float]:
        """Contracts already claimed at each book level, from the ledger.

        Built once per engine and then incremented in memory, so the work a run does is one
        grouped scan rather than a query per fill.
        """
        if self._depth_used is None:
            used: dict[tuple[str, str, str], float] = {}
            for r in self.store.query(
                    """SELECT json_extract(notes, '$.contract') AS c,
                              COALESCE(exchange_side,
                                       json_extract(notes, '$.exchange_side'),
                                       'YES') AS s,
                              decision_ts AS t,
                              SUM(COALESCE(filled_size, 0)) AS n
                         FROM bets
                        WHERE json_extract(notes, '$.contract') IS NOT NULL
                          AND result <> 'VOID'
                        GROUP BY 1, 2, 3"""):
                if r["n"]:
                    used[self.depth_key(r["c"], r["s"], r["t"])] = float(r["n"])
            self._depth_used = used
        return self._depth_used

    def _claim_depth(self, key: tuple[str, str, str], contracts: float) -> None:
        used = self.depth_used()
        used[key] = used.get(key, 0.0) + float(contracts)

    def depth_remaining(self, contract: str | None, side: str | None, ts: str | None,
                        capacity: float) -> float:
        key = self.depth_key(contract, side, ts)
        return max(0.0, float(capacity) - self.depth_used().get(key, 0.0))

    # ------------------------------------------------------------------ record
    def record_upcoming(self, sig: Signal, *, provider: str, source_url: str = "") -> None:
        self.store.execute(
            """INSERT INTO upcoming_bets(strategy_id, strategy_version, username, game_id, game_date,
                                         matchup, market, selection, provider, current_price,
                                         required_price, model_prob, fair_price, edge, stake,
                                         decision_ts, status, blocking_reason, supporting_data,
                                         source_url)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(strategy_id, strategy_version, game_id, market, selection) DO UPDATE SET
                 current_price=excluded.current_price,
                 required_price=excluded.required_price,
                 model_prob=excluded.model_prob,
                 fair_price=excluded.fair_price,
                 edge=excluded.edge,
                 stake=excluded.stake,
                 decision_ts=excluded.decision_ts,
                 status=excluded.status,
                 blocking_reason=excluded.blocking_reason,
                 supporting_data=excluded.supporting_data""",
            (sig.strategy_id, sig.version, sig.username, sig.game_id, sig.game_date, sig.matchup,
             sig.market, sig.selection, provider,
             (sig.quote.ask if sig.quote else None), sig.required_price,
             (sig.model_prob if sig.model_prob == sig.model_prob else None),
             sig.fair_price, sig.edge, sig.stake, sig.supporting.get("decision_ts", ""),
             sig.status, sig.blocking_reason,
             json.dumps({k: v for k, v in sig.supporting.items()}, default=str), source_url),
        )
        self.store.commit()

    def place(self, sig: Signal, *, decision_ts: str, test_mode: str, provider: str = "kalshi",
              source_url: str = "", verification: str = "single_source",
              depth_volume: float | None = None) -> str | None:
        """Convert a READY TO BET signal into a ledger entry.  Returns the bet_id or None.

        ``depth_volume`` is the number of contracts that traded in the entry candle, and is
        passed only for candle-priced BACKTEST rows: a candlestick publishes no offer size,
        but it does publish volume, which is real evidence of how much could have been taken
        at that price.  For a live quote it must stay None -- a contract's cumulative volume
        is not depth available now, and using it as a cap would claim liquidity the book
        never showed.
        """
        if sig.status != "READY TO BET" or sig.quote is None:
            return None
        ask = sig.quote.ask
        # ---------------------------------------------------------------- wide book
        # A two-sided book quoted this far apart is not an offer anyone would meet; see
        # WIDE_BOOK_MAX_SPREAD.  Recorded with the numbers, never dropped silently.
        bid, ask_v = sig.quote.bid, ask
        if bid is not None and ask_v is not None and (float(ask_v) - float(bid)) >= WIDE_BOOK_MAX_SPREAD:
            contract = sig.quote.contract or sig.quote.market_key
            self.wide_book_skipped[contract] = {
                "contract": contract, "bid": float(bid), "ask": float(ask_v),
                "spread": round(float(ask_v) - float(bid), 4),
                "ask_size": sig.quote.ask_size, "side": sig.quote.side,
                "strategy_id": sig.strategy_id, "max_spread": WIDE_BOOK_MAX_SPREAD}
            return None
        # ---------------------------------------------------------------- shared depth
        capacity, _liquidity, _basis = evidence_depth(
            sig.quote.ask_size, depth_volume)
        remaining = self.depth_remaining(sig.quote.contract, sig.quote.side, decision_ts,
                                         capacity)
        fill = simulate_fill(sig.stake, ask, sig.quote.ask_size,
                            traded_volume=depth_volume, remaining=remaining)
        if fill.contracts_filled <= 0:
            # Nothing left at this book level.  This is not "no signal": it is a signal the
            # venue could not have filled, and it is recorded as such so a reader can see
            # how much of a strategy's opportunity set is blocked by real, finite depth.
            #
            # *Why* it could not fill is recorded too, because the two causes are not the
            # same finding and blaming the wrong one would be a lie about the data:
            #
            #   ``no_depth_published``  the entry point carries no depth evidence at all
            #                           (a candle whose volume is zero, say) -- a limitation
            #                           of what the venue published, not of the strategies;
            #   ``already_claimed``     the level did publish depth and earlier wagers in the
            #                           same run took it -- the shared-book effect.
            #
            # Keyed by (contract, timestamp): one contract can be blocked at two entry
            # points, and collapsing them would under-report the blocked opportunity set.
            contract = sig.quote.contract or sig.quote.market_key
            reason = "already_claimed" if capacity > 0 else "no_depth_published"
            entry = self.depth_blocked.setdefault((contract, decision_ts), {
                "contract": contract, "side": sig.quote.side,
                "ts": decision_ts, "capacity": capacity,
                "already_claimed": round(capacity - remaining, 4),
                "reason": reason, "ask": ask, "strategies": []})
            if sig.strategy_id not in entry["strategies"]:
                entry["strategies"].append(sig.strategy_id)
            return None
        fee = kalshi_taker_fee(fill.price, fill.contracts_filled) if provider == "kalshi" else 0.0
        strike = getattr(sig.quote, "strike", None)
        bet_id = f"{sig.strategy_id}-v{sig.version}-{sig.game_id}-{sig.market}-{sig.selection}"
        if strike is not None:
            # a totals contract is written at a line, so the same strategy can hold an
            # Over 5.5 and an Over 6.5 on one game; the line is part of the bet's identity
            bet_id += f"-{strike:g}"
        model_p = sig.model_prob if sig.model_prob == sig.model_prob else None
        row = {
            "bet_id": bet_id,
            "strategy_id": sig.strategy_id,
            "strategy_version": sig.version,
            "username": sig.username,
            "test_mode": test_mode,
            "season": None,
            "game_id": sig.game_id,
            "game_date": sig.game_date,
            "matchup": sig.matchup,
            "market": sig.market,
            "selection": sig.selection,
            "bet_type": "binary_contract" if provider == "kalshi" else "moneyline",
            "provider": provider,
            "odds_format": "binary" if provider == "kalshi" else "decimal",
            "price": fill.price,
            "implied_prob": fill.price,
            "model_prob": model_p,
            "edge": (round(model_p - fill.price, 4) if model_p is not None else None),
            "fair_price": (round(1.0 / model_p, 4) if model_p else None),
            "decision_ts": decision_ts,
            "bet_ts": utcnow(),
            "stake": fill.stake,
            "liquidity": fill.liquidity,
            "filled_size": fill.contracts_filled,
            "slippage": fill.slippage,
            "entry_price": fill.price,
            "fee": fee,
            "strike": (float(strike) if strike is not None else None),
            "price_basis": (getattr(sig.quote, "price_basis", "exchange") or "exchange"),
            # which side of the book was bought.  A NO-side entry (an Under, a +1.5 puck
            # line, a "no overtime") is NOT comparable to a YES closing price, so this is
            # stored on the row rather than left inside the notes blob.
            "exchange_side": sig.quote.side,
            "depth_basis": fill.depth_basis,
            "result": "OPEN",
            "pnl": None,
            "source_url": source_url or (sig.quote.source_url if hasattr(sig.quote, "source_url") else ""),
            "verification_status": verification,
            "notes": json.dumps({
                "contracts_requested": fill.contracts_requested,
                "unfilled_stake": fill.unfilled_stake,
                "taker_fee": fee,
                "fee_basis": "kalshi general taker fee 0.07*C*P*(1-P), schedule eff. 2026-07-07",
                "bid": sig.quote.bid, "ask": sig.quote.ask,
                "ask_size": sig.quote.ask_size, "volume": sig.quote.volume,
                "market_key": sig.quote.market_key, "contract": sig.quote.contract,
                # the exchange's own wording for the contract, kept verbatim so the ledger
                # shows exactly which instrument was bought
                "contract_label": getattr(sig.quote, "label", None),
                "strike": strike,
                # kept with the wager because settlement needs it: 'greater' is the only
                # strike_type this engine will enter (anything else is refused at ingest),
                # but the row must still say which payoff it holds.
                "strike_type": getattr(sig.quote, "strike_type", None),
                "price_basis": getattr(sig.quote, "price_basis", "exchange"),
                "exchange_side": sig.quote.side,
                "depth_basis": fill.depth_basis,
                "depth_remaining_after": fill.depth_remaining_after,
                "entry_bid": sig.quote.bid,
                "entry_spread": (round(float(sig.quote.ask) - float(sig.quote.bid), 4)
                                 if (sig.quote.bid is not None and sig.quote.ask is not None)
                                 else None),
                "depth_cap_note": (
                    "the venue published no offer size for this side; the fill is capped at the "
                    f"declared {UNKNOWN_DEPTH_CAP_CONTRACTS:g}-contract assumption"
                    if fill.depth_basis == DEPTH_BASIS_CAP else
                    "the best available depth evidence for this book level had already been "
                    "partly claimed by earlier wagers on the same contract and quote; this "
                    "fill takes only what was left"
                    if fill.depth_basis == DEPTH_BASIS_SHARED else None),
            }, default=str),
            "features_json": json.dumps(sig.supporting, default=str),
            "created_at": utcnow(),
        }
        g = self.store.one("SELECT season, game_type FROM games WHERE game_id=?", (sig.game_id,))
        row["season"] = int(g["season"]) if g else None
        # which phase of the season this wager belongs to, stored on the row rather than
        # derived at read time: the competition is reported per phase, and a bet must never
        # move between tables because somebody edited a games row.
        row["game_type"] = int(g["game_type"]) if g else None
        ok = self.store.record_bet(row)
        self.store.commit()
        if ok:
            # the book level is only spent once the wager is actually in the ledger, so a
            # rejected duplicate bet_id (a re-run) does not silently consume liquidity
            self._claim_depth(
                self.depth_key(sig.quote.contract, sig.quote.side, decision_ts),
                fill.contracts_filled)
            self.store.execute(
                """UPDATE upcoming_bets SET status='EXECUTED'
                   WHERE strategy_id=? AND strategy_version=? AND game_id=? AND market=?
                     AND selection=?""",
                (sig.strategy_id, sig.version, sig.game_id, sig.market, sig.selection))
            self.store.sync_bankroll(sig.strategy_id, sig.version)
            self.store.commit()
            self.store.audit(self.actor, "PLACE", bet_id, f"{test_mode} stake={fill.stake}")
            return bet_id
        return None

    # ------------------------------------------------------------------ settle
    #: How a KXNHLOVERTIME contract settles, keyed on the NHL's own ``lastPeriodType``.
    #:
    #: VERIFIED against the exchange's own settled results (fetched 2026-09-21 from Kalshi's
    #: historical tier): ``KXNHLOVERTIME-26JUN04VGKCAR-OT`` resolved **yes** for NHL game
    #: 2025030412 (2026-06-04, CAR 4-3 VGK, lastPeriodType OT), and
    #: ``KXNHLOVERTIME-26JUN11VGKCAR-OT`` / ``KXNHLOVERTIME-26JUN14CARVGK-OT`` resolved
    #: **no** for games 2025030415 / 2025030416 (both lastPeriodType REG).  The series'
    #: settled history is ~100 events, every one a 2026 playoff game between 2026-04-28 and
    #: 2026-06-14, and the playoffs have no shootout -- so **a shootout's treatment is
    #: UNVERIFIED**: the rules text says only "If <teams> go to overtime ... the market
    #: resolves to Yes" and does not say whether a game that reaches a shootout counts.  An
    #: SO game is therefore left OPEN with an irregularity recorded, and is settled from the
    #: exchange's own result (:meth:`settle_from_exchange`) if that contract is ingested.
    OT_VERIFIED_YES = ("OT",)
    OT_VERIFIED_NO = ("REG",)
    OT_UNVERIFIED = ("SO",)

    def settle_game(self, game_id: int, *, home_id: int, away_id: int, home_score: int | None,
                    away_score: int | None, state: str, last_period_type: str | None,
                    closing_quotes: Sequence[Quote] = ()) -> int:
        """Settle every open bet on this game from the official NHL result.

        Covers the moneyline, the total, the puck line and the overtime contract.  Never
        invents a result, and where the payoff of an instrument has not been verified
        against the exchange's own settlement the wager is left OPEN with the reason
        recorded rather than guessed.
        """
        if state not in ("FINAL", "OFF"):
            return 0
        if home_score is None or away_score is None:
            # The wording is deliberate and shared with Verifier.check_games: the same
            # condition must produce one queue row, not two rows saying it two ways.
            self.store.flag("missing_result", f"game {game_id} {state} without scores",
                            severity="error", entity_type="game", entity_id=str(game_id))
            return 0
        home_won = home_score > away_score
        away_won = away_score > home_score
        if not home_won and not away_won:
            self.store.flag("impossible_result",
                            f"game {game_id} final score tied {home_score}-{away_score}",
                            severity="critical", entity_type="game", entity_id=str(game_id))
            return 0

        closing = {q.selection: q for q in closing_quotes}
        n = 0
        for bet in self.store.query(
                "SELECT * FROM bets WHERE game_id=? AND result='OPEN' "
                "AND market IN ('moneyline','total','puck_line','overtime')",
                (game_id,)):
            market = bet["market"] or "moneyline"
            total_goals = home_score + away_score
            if market == "total":
                won, why = self._settle_total(bet, total_goals)
                if won is None:
                    self.store.flag(
                        "unsettleable_total",
                        f"bet {bet['bet_id']} cannot be settled: strike="
                        f"{bet['strike'] if 'strike' in bet.keys() else None} "
                        f"selection='{bet['selection']}'; left OPEN rather than guessed",
                        severity="error", entity_type="bet", entity_id=bet["bet_id"])
                    continue
            elif market == "puck_line":
                won, why = self._settle_puck_line(bet, home_score, away_score)
                if won is None:
                    self.store.flag(
                        "unsettleable_puck_line",
                        f"bet {bet['bet_id']} cannot be settled from the official margin: "
                        f"strike={bet['strike']} selection='{bet['selection']}'; left OPEN "
                        "rather than guessed",
                        severity="error", entity_type="bet", entity_id=bet["bet_id"])
                    continue
            elif market == "overtime":
                won, why = self._settle_overtime(bet, last_period_type)
                if won is None:
                    lpt = (last_period_type or "").upper()
                    kind = ("settlement_rule_unverified" if lpt in self.OT_UNVERIFIED
                            else "unsettleable_overtime")
                    detail = (
                        f"bet {bet['bet_id']} on a KXNHLOVERTIME contract cannot be settled "
                        f"from NHL data: lastPeriodType={lpt or 'missing'}. "
                        + ("The exchange's verified settled history is playoff-only, where no "
                           "shootout exists, so whether a shootout counts as 'going to "
                           "overtime' is UNVERIFIED; left OPEN, and it will be settled from "
                           "the exchange's own result if that contract is ingested."
                           if lpt in self.OT_UNVERIFIED else
                           "Left OPEN rather than guessed."))
                    self.store.flag(kind, detail, severity="error",
                                    entity_type="bet", entity_id=bet["bet_id"])
                    continue
            else:
                sel = (bet["selection"] or "").lower()
                if "home" in sel:
                    won = home_won
                elif "away" in sel:
                    won = away_won
                else:
                    self.store.flag("unmapped_selection",
                                    f"bet {bet['bet_id']} selection '{bet['selection']}' cannot be "
                                    f"mapped to a side", severity="error",
                                    entity_type="bet", entity_id=bet["bet_id"])
                    continue
                why = (f"official result {home_score}-{away_score} "
                       f"({'OT' if last_period_type == 'OT' else 'REG'})")
            price = float(bet["entry_price"] or bet["price"])
            contracts = float(bet["filled_size"] or (float(bet["stake"]) / price if price else 0))
            pnl = binary_settlement(price, contracts, won, fee=_fee_of(bet))
            close = closing.get(bet["selection"])
            close_price = close.ask if close else None
            clv = round(close_price - price, 4) if close_price is not None else None
            self.store.settle_bet(
                bet["bet_id"], result="WIN" if won else "LOSS", pnl=pnl,
                close_price=close_price, clv=clv, reason=why)
            self.store.sync_bankroll(bet["strategy_id"], int(bet["strategy_version"]))
            n += 1
        self.store.commit()
        return n

    @staticmethod
    def _settle_total(bet: Any, total_goals: int) -> tuple[bool | None, str]:
        """Settle a totals contract from the official final score.

        Kalshi's KXNHLTOTAL rules count regulation and overtime goals normally and count a
        shootout as one goal for the winner -- which is exactly what the NHL's official
        final score already contains -- so the settled total is ``home + away`` with no
        adjustment.  Returns (None, reason) when the wager cannot be settled honestly, in
        which case the row stays OPEN and an irregularity is recorded rather than guessed.
        """
        try:
            strike = bet["strike"]
        except (KeyError, IndexError):
            strike = None
        if strike is None:
            return None, ""
        sel = (bet["selection"] or "").lower()
        if sel not in ("over", "under"):
            return None, ""
        over = total_goals > float(strike)
        # an "under" wager is the NO side of an Over contract: it pays when the contract
        # resolves NO, i.e. when the total is at or below the strike
        won = over if sel == "over" else (not over)
        return won, (f"official total {total_goals} vs strike {strike:g} "
                     f"-> {'over' if over else 'under'}")

    @staticmethod
    def _settle_puck_line(bet: Any, home_score: int, away_score: int
                          ) -> tuple[bool | None, str]:
        """Settle a KXNHLSPREAD contract ("<team> wins by over k.5 goals") from the score.

        The margin is the official final differential, overtime and shootout goals included
        -- which is what Kalshi's own rules text describes ("wins by over X goals" on the
        final result).  ``selection`` carries both halves of the wager: ``home_cover`` is the
        YES side of the home team's contract and ``away_no_cover`` is the NO side of the away
        team's (i.e. the away team's +k.5 line).  Returns ``(None, "")`` when the row does
        not carry a readable strike and side, so it stays OPEN instead of being guessed.
        """
        try:
            strike = bet["strike"]
        except (KeyError, IndexError):
            strike = None
        if strike is None:
            return None, ""
        sel = (bet["selection"] or "").lower()
        if sel.endswith("_no_cover"):
            team, bought = sel[: -len("_no_cover")], "NO"
        elif sel.endswith("_cover"):
            team, bought = sel[: -len("_cover")], "YES"
        else:
            return None, ""
        if team not in ("home", "away"):
            return None, ""
        comparison = "greater"
        try:
            notes = json.loads(bet["notes"] or "{}")
            comparison = str(notes.get("strike_type") or "greater")
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        margin = home_score - away_score if team == "home" else away_score - home_score
        covered = covers_margin(margin, float(strike), comparison)
        if covered is None:
            return None, ""
        won = covered if bought == "YES" else (not covered)
        return won, (f"official {team} margin {margin:+d} vs strike {strike:g} "
                     f"({comparison}) -> {'covers' if covered else 'does not cover'}; "
                     f"wager held the {bought} side")

    @staticmethod
    def _settle_overtime(bet: Any, last_period_type: str | None) -> tuple[bool | None, str]:
        """Settle a KXNHLOVERTIME contract from the NHL's ``lastPeriodType``.

        See :attr:`OT_VERIFIED_YES` for what has been verified against the exchange's own
        results and what has not.  ``selection`` is ``ot_yes`` (game goes past regulation) or
        ``ot_no`` (decided in regulation).  Anything outside the verified mapping returns
        ``(None, reason)`` and the wager stays OPEN.
        """
        sel = (bet["selection"] or "").lower()
        if sel not in ("ot_yes", "ot_no"):
            return None, f"unmapped selection '{sel}'"
        lpt = (last_period_type or "").upper()
        if lpt in PaperEngine.OT_VERIFIED_YES:
            went = True
        elif lpt in PaperEngine.OT_VERIFIED_NO:
            went = False
        else:
            return None, f"lastPeriodType={lpt or 'missing'} is outside the verified mapping"
        won = went if sel == "ot_yes" else (not went)
        return won, (f"official lastPeriodType={lpt} -> the game "
                     f"{'did' if went else 'did not'} go past regulation; wager held the "
                     f"{sel[3:].upper()} side")

    def settle_from_exchange(self) -> int:
        """Settle OPEN wagers from the exchange's own settled contract result.

        For a contract the exchange has finalized, its ``result`` is the authoritative
        statement of what that instrument paid -- it is what a real account would have been
        settled on -- so it takes precedence over this project's reading of the official
        score, and it is the only honest way to settle an instrument whose payoff has not
        been independently verified (an overtime contract on a shootout game).  Only OPEN
        rows are touched; a wager already settled from the official result is left alone,
        and any disagreement between the two is surfaced by ``verify`` rather than by
        overwriting a settled row here.
        """
        n = 0
        for row in self.store.query(
                """SELECT b.bet_id, b.strategy_id, b.strategy_version, b.entry_price, b.price,
                          b.filled_size, b.stake, b.fee, b.exchange_side, b.close_price, b.clv,
                          ms.result, ms.settlement_ts, ms.contract
                     FROM bets b
                     JOIN market_settlements ms
                       ON ms.contract = json_extract(b.notes, '$.contract')
                    WHERE b.result='OPEN' AND lower(ms.result) IN ('yes','no')"""):
            side = (row["exchange_side"] or "YES").upper()
            result = str(row["result"]).lower()
            won = (result == "yes") if side == "YES" else (result == "no")
            price = float(row["entry_price"] or row["price"] or 0)
            if price <= 0:
                continue
            contracts = float(row["filled_size"] or (float(row["stake"] or 0) / price))
            pnl = binary_settlement(price, contracts, won, fee=_fee_of(row))
            self.store.settle_bet(
                row["bet_id"], result="WIN" if won else "LOSS", pnl=pnl,
                close_price=row["close_price"], clv=row["clv"],
                settle_ts=row["settlement_ts"],
                reason=f"kalshi settled {row['contract']} as {result.upper()}; the wager held "
                       f"the {side} side (exchange result is authoritative for the contract)")
            self.store.sync_bankroll(row["strategy_id"], int(row["strategy_version"]))
            n += 1
        self.store.commit()
        return n

    PENDING_STATUSES = ('WATCHING', 'QUALIFIED', 'READY TO BET', 'PRICE TOO HIGH', 'PRICE TOO LOW',
                        'WAITING FOR GOALIE', 'WAITING FOR LINEUP', 'WAITING FOR INJURY',
                        'WAITING FOR OTHER INFORMATION')

    def expire_started(self, now_iso: str) -> int:
        """A pending signal on a game that has already started can no longer be acted on;
        mark it EXPIRED (the row stays, with its last status reason, for the record)."""
        marks = ",".join("?" * len(self.PENDING_STATUSES))
        cur = self.store.execute(
            f"""UPDATE upcoming_bets SET status='EXPIRED'
                 WHERE status IN ({marks})
                   AND game_id IN (SELECT game_id FROM games WHERE start_time_utc < ?)""",
            (*self.PENDING_STATUSES, now_iso))
        self.store.commit()
        return cur.rowcount

    def expire_stale(self, cutoff_ts: str) -> int:
        cur = self.store.execute(
            """UPDATE upcoming_bets SET status='EXPIRED'
               WHERE status IN ('WATCHING','QUALIFIED','READY TO BET','PRICE TOO HIGH',
                                'PRICE TOO LOW','WAITING FOR GOALIE','WAITING FOR LINEUP',
                                'WAITING FOR INJURY','WAITING FOR OTHER INFORMATION')
                 AND decision_ts < ?""",
            (cutoff_ts,))
        self.store.commit()
        return cur.rowcount
