"""Verification, cross-validation and the irregularity queue.

The brief forbids silently choosing between conflicting sources and forbids silently
correcting suspicious data.  Every check here only *records*; resolution is a separate,
audited step.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from .store import Store, utcnow


class Verifier:
    def __init__(self, store: Store):
        self.store = store

    # ------------------------------------------------------------------ games
    def check_games(self) -> dict[str, int]:
        counts = {"missing_start": 0, "missing_result": 0, "tied_final": 0,
                  "impossible_score": 0, "duplicate": 0, "bad_season": 0}
        for g in self.store.query("SELECT * FROM games"):
            gid = str(g["game_id"])
            if not g["start_time_utc"]:
                counts["missing_start"] += 1
                self.store.flag("missing_timestamp", f"game {gid} has no startTimeUTC",
                                entity_type="game", entity_id=gid, severity="error")
            if g["state"] in ("FINAL", "OFF"):
                if g["home_score"] is None or g["away_score"] is None:
                    counts["missing_result"] += 1
                    self.store.flag("missing_result", f"game {gid} FINAL without scores",
                                    entity_type="game", entity_id=gid, severity="error")
                elif g["home_score"] == g["away_score"]:
                    counts["tied_final"] += 1
                    self.store.flag("impossible_result",
                                    f"game {gid} final score tied {g['home_score']}-"
                                    f"{g['away_score']}",
                                    entity_type="game", entity_id=gid, severity="critical")
                elif min(g["home_score"], g["away_score"]) < 0 or \
                        max(g["home_score"], g["away_score"]) > 30:
                    counts["impossible_score"] += 1
                    self.store.flag("impossible_result",
                                    f"game {gid} score {g['home_score']}-{g['away_score']} "
                                    f"outside plausible NHL range",
                                    entity_type="game", entity_id=gid, severity="critical")
            if g["season"] and not (20152016 <= int(g["season"]) <= 20302031):
                counts["bad_season"] += 1
                self.store.flag("impossible_season", f"game {gid} season {g['season']}",
                                entity_type="game", entity_id=gid, severity="warn")
        dup = self.store.query(
            """SELECT game_date, home_id, away_id, COUNT(*) c FROM games
               GROUP BY game_date, home_id, away_id HAVING c > 1""")
        for d in dup:
            counts["duplicate"] += 1
            self.store.flag("duplicate_game",
                            f"{d['game_date']} {d['away_id']}@{d['home_id']} appears {d['c']} times",
                            entity_type="game", entity_id=f"{d['game_date']}:{d['home_id']}:"
                                                           f"{d['away_id']}", severity="warn")
        return counts

    # ------------------------------------------------------------------ bets
    def check_bets(self) -> dict[str, int]:
        counts = {"impossible_odds": 0, "missing_timestamp": 0, "bad_pnl": 0,
                  "duplicate_bet": 0, "stake_exceeds_bankroll": 0, "unmapped": 0}
        for b in self.store.query("SELECT * FROM bets"):
            bid = b["bet_id"]
            p = b["price"]
            if b["odds_format"] == "binary":
                if p is None or not (0.0 < p < 1.0):
                    counts["impossible_odds"] += 1
                    self.store.flag("impossible_odds",
                                    f"bet {bid} binary price {p} outside (0,1)",
                                    entity_type="bet", entity_id=bid, severity="error")
            elif b["odds_format"] == "decimal":
                if p is None or p <= 1.0:
                    counts["impossible_odds"] += 1
                    self.store.flag("impossible_odds",
                                    f"bet {bid} decimal price {p} <= 1.0",
                                    entity_type="bet", entity_id=bid, severity="error")
            if not b["decision_ts"] or not b["bet_ts"]:
                counts["missing_timestamp"] += 1
                self.store.flag("missing_timestamp", f"bet {bid} missing a timestamp",
                                entity_type="bet", entity_id=bid, severity="error")
            if b["decision_ts"] and b["bet_ts"] and b["decision_ts"] > b["bet_ts"]:
                self.store.flag("timestamp_order",
                                f"bet {bid} decided after it was placed",
                                entity_type="bet", entity_id=bid, severity="error")
            if b["result"] in ("WIN", "LOSS"):
                stake = float(b["stake"] or 0)
                pnl = float(b["pnl"] or 0)
                if b["odds_format"] == "binary":
                    contracts = float(b["filled_size"] or 0)
                    exp_win = contracts * (1 - float(b["entry_price"] or 0))
                    exp_loss = -contracts * float(b["entry_price"] or 0)
                    if not (abs(pnl - exp_win) < 0.01 or abs(pnl - exp_loss) < 0.01):
                        counts["bad_pnl"] += 1
                        self.store.flag("incorrect_pnl",
                                        f"bet {bid} pnl {pnl} inconsistent with "
                                        f"{contracts} contracts at {b['entry_price']}",
                                        entity_type="bet", entity_id=bid, severity="critical")
                elif abs(pnl) > stake * 50:
                    counts["bad_pnl"] += 1
                    self.store.flag("incorrect_pnl",
                                    f"bet {bid} pnl {pnl} implausible vs stake {stake}",
                                    entity_type="bet", entity_id=bid, severity="critical")
            if b["market"] == "moneyline" and (b["selection"] or "").lower() not in ("home", "away"):
                counts["unmapped"] += 1
                self.store.flag("unmapped_selection",
                                f"bet {bid} selection '{b['selection']}' is not home/away",
                                entity_type="bet", entity_id=bid, severity="error")
        # duplicate wagers: same strategy/game/market/selection placed more than once
        for d in self.store.query(
                """SELECT strategy_id, strategy_version, game_id, market, selection, COUNT(*) c
                   FROM bets GROUP BY 1,2,3,4,5 HAVING c > 1"""):
            counts["duplicate_bet"] += 1
            self.store.flag("duplicate_bet",
                            f"{d['strategy_id']} v{d['strategy_version']} has {d['c']} bets on "
                            f"{d['market']}/{d['selection']} for game {d['game_id']}",
                            entity_type="bet",
                            entity_id=f"{d['strategy_id']}:{d['game_id']}:{d['selection']}",
                            severity="warn")
        # bankroll discipline
        for s in self.store.latest_versions():
            open_exposure = self.store.one(
                """SELECT COALESCE(SUM(stake),0) e FROM bets WHERE strategy_id=? AND
                   strategy_version=? AND result='OPEN'""",
                (s["strategy_id"], s["version"]))["e"] or 0.0
            if float(open_exposure) > float(s["bankroll"]) + 0.01:
                counts["stake_exceeds_bankroll"] += 1
                self.store.flag("exposure_exceeds_bankroll",
                                f"{s['strategy_id']} v{s['version']} open exposure "
                                f"{open_exposure} exceeds bankroll {s['bankroll']}",
                                entity_type="strategy", entity_id=s["strategy_id"],
                                severity="error")
        return counts

    # ------------------------------------------------------------------ quotes
    def check_quotes(self) -> dict[str, int]:
        counts = {"crossed_book": 0, "impossible_price": 0, "zero_liquidity": 0, "missing_ts": 0}
        for q in self.store.query("SELECT * FROM market_quotes"):
            qid = f"{q['provider']}:{q['contract']}:{q['side']}:{q['ts_utc']}"
            if q["bid"] is not None and q["ask"] is not None:
                if q["provider"] == "kalshi" and q["bid"] > q["ask"]:
                    counts["crossed_book"] += 1
                    self.store.flag("crossed_book", f"{qid} bid {q['bid']} > ask {q['ask']}",
                                    entity_type="quote", entity_id=qid, severity="error")
                if q["provider"] == "kalshi" and not (0 <= q["ask"] <= 1):
                    counts["impossible_price"] += 1
                    self.store.flag("impossible_odds", f"{qid} ask {q['ask']} outside [0,1]",
                                    entity_type="quote", entity_id=qid, severity="error")
            # only an explicit 0 is thin liquidity; NULL means the feed did not report size,
            # which is a different problem and must not be conflated with it
            if q["ask_size"] == 0 and q["bid_size"] == 0:
                counts["zero_liquidity"] += 1
                self.store.flag("insufficient_liquidity",
                                f"{qid} shows zero size on both sides",
                                entity_type="quote", entity_id=qid, severity="info")
            if not q["ts_utc"]:
                counts["missing_ts"] += 1
                self.store.flag("missing_timestamp", f"{qid} has no timestamp",
                                entity_type="quote", entity_id=qid, severity="error")
        return counts

    # ------------------------------------------------------------------ cross-checks
    def cross_validate_games(self, pairs: Sequence[tuple[str, dict[int, tuple[int, int]]]]) -> int:
        """Compare the same games as reported by two independent sources.

        ``pairs`` is a list of ``(source_name, {game_id: (home_score, away_score)})``.  Any
        disagreement is recorded with both values; nothing is overwritten.
        """
        if len(pairs) < 2:
            return 0
        conflicts = 0
        base_name, base = pairs[0]
        for other_name, other in pairs[1:]:
            for gid, (hs, as_) in other.items():
                if gid not in base:
                    continue
                bhs, bas = base[gid]
                if (hs, as_) != (bhs, bas):
                    conflicts += 1
                    self.store.flag(
                        "conflicting_source",
                        f"game {gid}: {base_name} says {bhs}-{bas}, {other_name} says {hs}-{as_}",
                        entity_type="game", entity_id=str(gid), severity="error",
                        sources=f"{base_name}|{other_name}")
        return conflicts

    def cross_validate_starters(self, claims: dict[int, list[tuple[str, str | None]]]) -> int:
        """``{game_id: [(source, goalie_name_or_None), ...]}`` -> record disagreements."""
        conflicts = 0
        for gid, entries in claims.items():
            named = {src: g for src, g in entries if g}
            if len(set(named.values())) > 1:
                conflicts += 1
                self.store.flag(
                    "conflicting_source",
                    f"game {gid}: starting goalie disagreement {json.dumps(named)}",
                    entity_type="game", entity_id=str(gid), severity="warn",
                    sources="|".join(named.keys()))
        return conflicts

    # ------------------------------------------------------------------ report
    def run_all(self) -> dict[str, Any]:
        summary = {}
        summary["games"] = self.check_games()
        summary["bets"] = self.check_bets()
        summary["quotes"] = self.check_quotes()
        summary["open_irregularities"] = self.store.one(
            "SELECT COUNT(*) c FROM irregularities WHERE status='open'")["c"]
        self.store.audit("verifier", "RUN_ALL", "", json.dumps(summary))
        self.store.commit()
        return summary
