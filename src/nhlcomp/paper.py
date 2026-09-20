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

Nothing in this module can place a real order: there is no order-placement code at all.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .store import Store, utcnow
from .strategies import DecisionContext, Quote, Signal, Strategy


@dataclass
class Fill:
    contracts_requested: float
    contracts_filled: float
    price: float
    stake: float
    unfilled_stake: float
    slippage: float
    liquidity: float | None


def simulate_fill(stake: float, ask: float, ask_size: float | None) -> Fill:
    """Single-level fill.  ``ask_size`` is in contracts."""
    if stake <= 0 or ask is None or ask <= 0:
        return Fill(0.0, 0.0, ask or 0.0, 0.0, max(stake, 0.0), 0.0, ask_size)
    requested = stake / ask
    if ask_size is None:
        filled = requested          # unknown size: cannot claim depth, so do not cap
        liquidity = None
    else:
        filled = min(requested, float(ask_size))
        liquidity = float(ask_size)
    filled_stake = filled * ask
    return Fill(contracts_requested=round(requested, 4), contracts_filled=round(filled, 4),
                price=ask, stake=round(filled_stake, 4),
                unfilled_stake=round(max(stake - filled_stake, 0.0), 4),
                slippage=0.0, liquidity=liquidity)


def binary_settlement(entry_price: float, contracts: float, won: bool) -> float:
    """A YES contract costs ``entry_price`` and pays 1.00 if it wins."""
    return round(contracts * (1.0 - entry_price), 4) if won else round(-contracts * entry_price, 4)


class PaperEngine:
    def __init__(self, store: Store, *, actor: str = "paper_engine"):
        self.store = store
        self.actor = actor

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
              source_url: str = "", verification: str = "single_source") -> str | None:
        """Convert a READY TO BET signal into a ledger entry.  Returns the bet_id or None."""
        if sig.status != "READY TO BET" or sig.quote is None:
            return None
        fill = simulate_fill(sig.stake, sig.quote.ask, sig.quote.ask_size)
        if fill.contracts_filled <= 0:
            return None
        bet_id = f"{sig.strategy_id}-v{sig.version}-{sig.game_id}-{sig.market}-{sig.selection}"
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
            "result": "OPEN",
            "pnl": None,
            "source_url": source_url or (sig.quote.source_url if hasattr(sig.quote, "source_url") else ""),
            "verification_status": verification,
            "notes": json.dumps({
                "contracts_requested": fill.contracts_requested,
                "unfilled_stake": fill.unfilled_stake,
                "bid": sig.quote.bid, "ask": sig.quote.ask,
                "ask_size": sig.quote.ask_size, "volume": sig.quote.volume,
                "market_key": sig.quote.market_key, "contract": sig.quote.contract,
            }, default=str),
            "features_json": json.dumps(sig.supporting, default=str),
            "created_at": utcnow(),
        }
        row["season"] = self.store.one("SELECT season FROM games WHERE game_id=?",
                                       (sig.game_id,))
        row["season"] = int(row["season"]["season"]) if row["season"] else None
        ok = self.store.record_bet(row)
        self.store.commit()
        if ok:
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
    def settle_game(self, game_id: int, *, home_id: int, away_id: int, home_score: int | None,
                    away_score: int | None, state: str, last_period_type: str | None,
                    closing_quotes: Sequence[Quote] = ()) -> int:
        """Settle every open moneyline bet on this game.  Never invents a result."""
        if state not in ("FINAL", "OFF"):
            return 0
        if home_score is None or away_score is None:
            self.store.flag("missing_result", f"game {game_id} is {state} with no score",
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
                "SELECT * FROM bets WHERE game_id=? AND result='OPEN' AND market='moneyline'",
                (game_id,)):
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
            price = float(bet["entry_price"] or bet["price"])
            contracts = float(bet["filled_size"] or (float(bet["stake"]) / price if price else 0))
            pnl = binary_settlement(price, contracts, won)
            close = closing.get(bet["selection"])
            close_price = close.ask if close else None
            clv = round(close_price - price, 4) if close_price is not None else None
            self.store.settle_bet(
                bet["bet_id"], result="WIN" if won else "LOSS", pnl=pnl,
                close_price=close_price, clv=clv,
                reason=f"official result {home_score}-{away_score} "
                       f"({'OT' if last_period_type == 'OT' else 'REG'})")
            self.store.sync_bankroll(bet["strategy_id"], int(bet["strategy_version"]))
            n += 1
        self.store.commit()
        return n

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
