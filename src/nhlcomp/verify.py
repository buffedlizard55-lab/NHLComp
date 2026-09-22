"""Verification, cross-validation and the irregularity queue.

The brief forbids silently choosing between conflicting sources and forbids silently
correcting suspicious data.  Every check here only *records*; resolution is a separate,
audited step.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from .http import parse_iso
from .store import Store, utcnow


#: Polymarket's NHL game markets come in exactly two shapes, verified on the 2026-09-21
#: ingest of ``gamma-api.polymarket.com/events?tag_slug=nhl``: a moneyline whose question is
#: the matchup ("Red Wings vs. Blue Jackets", outcomes = the two team nicknames) and a total
#: whose question appends the line ("Red Wings vs. Blue Jackets: O/U 5.5", outcomes =
#: ["Over", "Under"]).  It listed no puck line at all, so there is nothing to compare a
#: KXNHLSPREAD contract against -- recorded as an absence, never filled with a total or a
#: moneyline price pretending to be one.
TOTAL_QUESTION_RE = re.compile(r"\bO/U\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def classify_polymarket_question(question: str | None) -> tuple[str, float | None]:
    """``('total', line)`` for an over/under question, otherwise ``('moneyline', None)``."""
    m = TOTAL_QUESTION_RE.search(question or "")
    if m:
        return "total", float(m.group(1))
    return "moneyline", None


def _col(row: Any, name: str) -> Any:
    """Column value or None when an older ledger lacks the column."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


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
                    self.store.flag("missing_result", f"game {gid} {g['state']} without scores",
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
        # Preseason split-squad doubleheaders legitimately have same date and
        # matchup but different start times (e.g. 2024-09-22 18:00 vs 22:00, same
        # venue). Grouping only by date+teams flagged those as duplicates (2 rows
        # in the 2026-09-20 ledger). Group by start_time_utc as well so only
        # truly identical scheduled starts are flagged.
        dup = self.store.query(
            """SELECT game_date, home_id, away_id, start_time_utc, COUNT(*) c FROM games
               GROUP BY game_date, home_id, away_id, start_time_utc HAVING c > 1""")
        for d in dup:
            counts["duplicate"] += 1
            self.store.flag("duplicate_game",
                            f"{d['game_date']} {d['away_id']}@{d['home_id']} at {d['start_time_utc']} appears {d['c']} times",
                            entity_type="game", entity_id=f"{d['game_date']}:{d['home_id']}:"
                                                           f"{d['away_id']}:{d['start_time_utc']}", severity="warn")
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
                    # settled P&L is net of the exchange fee recorded on the row
                    fee = float(_col(b, "fee") or 0.0)
                    exp_win = contracts * (1 - float(b["entry_price"] or 0)) - fee
                    exp_loss = -contracts * float(b["entry_price"] or 0) - fee
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
        # duplicate wagers: same strategy/game/market/selection **and same line** placed
        # more than once.
        #
        # The line is part of the instrument's identity, not decoration: Kalshi lists one
        # "Over k.5" contract per strike, and ``PaperEngine.place`` puts the strike in the
        # ``bet_id`` precisely so that one rule may hold both an Over 5.5 and an Over 7.5 on
        # the same game.  Grouping without the strike made every such pair look like a
        # duplicate and wrote 64 false ``duplicate_bet`` irregularities into the 2026-09-21
        # ledger -- which is worse than noise, because a queue full of false positives is a
        # queue nobody reads.  With the strike included, the same query returns none on that
        # ledger, while a genuine double-entry at the *same* strike still trips it (the
        # append-only ``record_bet`` also refuses a repeated ``bet_id``).
        for d in self.store.query(
                """SELECT strategy_id, strategy_version, game_id, market, selection,
                          COALESCE(strike, -1) AS line, COALESCE(test_mode, '') AS mode,
                          COUNT(*) c
                     FROM bets GROUP BY 1,2,3,4,5,6,7 HAVING c > 1"""):
            counts["duplicate_bet"] += 1
            line = "" if d["line"] == -1 else f" at line {d['line']:g}"
            self.store.flag(
                "duplicate_bet",
                f"{d['strategy_id']} v{d['strategy_version']} has {d['c']} {d['mode']} bets on "
                f"{d['market']}/{d['selection']}{line} for game {d['game_id']}",
                entity_type="bet",
                entity_id=f"{d['strategy_id']}:{d['game_id']}:{d['selection']}:{d['line']:g}",
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
    def cross_validate_stats_vs_schedule(self) -> dict[str, int]:
        """Second-source check of every final score: the stats REST per-game team line
        (goalsFor / goalsAgainst, one row per team) against the schedule feed's score.
        Both are NHL first-party feeds but different systems; a disagreement is recorded with
        both values and left for review -- never resolved by picking one."""
        compared = conflicts = so_adjusted = 0
        for r in self.store.query(
                """SELECT g.game_id, g.home_score, g.away_score, g.last_period_type, t.team_id, t.gf,
                          t.ga, t.home_road
                     FROM team_game_stats t JOIN games g ON g.game_id = t.game_id
                    WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                      AND t.gf IS NOT NULL AND t.ga IS NOT NULL"""):
            compared += 1
            is_home = (r["home_road"] == "H")
            exp_gf, exp_ga = ((r["home_score"], r["away_score"]) if is_home
                              else (r["away_score"], r["home_score"]))
            got = (int(r["gf"]), int(r["ga"]))
            if got == (int(exp_gf), int(exp_ga)):
                continue
            if r["last_period_type"] == "SO":
                # Definitional difference, verified on the 2024-25 data: the NHL stats REST
                # team line excludes the shootout-deciding goal (a 2-1 SO win is GF 1 / GA 1),
                # while the schedule feed credits it.  Consistent once that goal is removed.
                hs, as_ = int(r["home_score"]), int(r["away_score"])
                if hs > as_:
                    hs -= 1
                else:
                    as_ -= 1
                adj = (hs, as_) if is_home else (as_, hs)
                if got == adj:
                    so_adjusted += 1
                    continue
            conflicts += 1
            self.store.flag(
                "conflicting_source",
                f"game {r['game_id']} team {r['team_id']}: schedule says GF {exp_gf} / GA {exp_ga}, "
                f"stats REST team/summary says GF {r['gf']} / GA {r['ga']}",
                entity_type="game", entity_id=str(r["game_id"]), severity="error",
                sources="nhl.api_web|nhl.stats_rest_game")
        # Settlement results vs the schedule winner.  This check is a MONEYLINE rule -- Kalshi
        # 'yes' on "<team> wins" must mean that team actually won -- and it is applied only to
        # moneyline contracts.  A puck-line contract also names a team but settles on the
        # margin, a totals contract on the goal count and an overtime contract on whether the
        # game reached overtime, so judging those by "did the team win" reports a perfectly
        # correct settlement as a critical conflict.  That assumption was live until the
        # 2026-09-21 run, when 5,186 settled KXNHLSPREAD contracts entered the ledger and it
        # produced 1,353 false CRITICAL conflicts; each strike market is checked by its own
        # study below, against the condition the contract actually states.
        settle_conf = settle_compared = settle_skipped = 0
        for r in self.store.query(
                """SELECT ms.contract, ms.result, ms.team_abbrev, ms.market_type,
                          g.game_id, g.home_id, g.away_id,
                          g.home_score, g.away_score
                     FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                    WHERE ms.provider='kalshi' AND ms.result IN ('yes','no') AND ms.team_abbrev IS NOT NULL
                      AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                      AND g.home_score <> g.away_score"""):
            if (_col(r, "market_type") or "moneyline") != "moneyline":
                settle_skipped += 1
                continue
            tid = self.store.team_id_for(r["team_abbrev"])
            if tid not in (r["home_id"], r["away_id"]):
                continue
            settle_compared += 1
            team_won = (r["home_score"] > r["away_score"]) == (tid == r["home_id"])
            if team_won != (r["result"] == "yes"):
                settle_conf += 1
                self.store.flag(
                    "conflicting_source",
                    f"contract {r['contract']} settled {r['result']} for {r['team_abbrev']} but the NHL "
                    f"score is {r['home_score']}-{r['away_score']} (game {r['game_id']})",
                    entity_type="quote", entity_id=r["contract"], severity="critical",
                    sources="kalshi.historical|nhl.api_web")
        return {"team_game_rows_compared": compared, "score_conflicts": conflicts,
                "shootout_goal_definition_adjusted": so_adjusted,
                "settlement_conflicts": settle_conf,
                "settlement_moneyline_compared": settle_compared,
                "settlement_skipped_strike_markets": settle_skipped}

    def cross_validate_totals_settlements(self) -> dict[str, int]:
        """Check every settled KXNHLTOTAL contract against the official final score.

        Kalshi resolves an "Over k.5" contract on regulation + overtime goals, counting a
        shootout as one goal for the winner -- which is exactly what the NHL's official
        final score already contains.  So the exchange's own ``result`` and this project's
        settlement rule are two independent statements about the same fact, and any
        disagreement means one of them is wrong.  Both values are recorded; neither is
        corrected here.
        """
        compared = conflicts = missing_strike = 0
        for r in self.store.query(
                """SELECT ms.contract, ms.game_id, ms.result, ms.floor_strike, ms.strike_type,
                          ms.selection, g.home_score, g.away_score
                     FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                    WHERE ms.provider='kalshi' AND ms.series_ticker='KXNHLTOTAL'
                      AND ms.result IN ('yes','no')
                      AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL"""):
            strike = _col(r, "floor_strike")
            if strike is None:
                missing_strike += 1
                self.store.flag("missing_strike",
                                f"totals contract {r['contract']} settled {r['result']} but has "
                                f"no floor_strike; it cannot be re-derived from the score",
                                entity_type="market", entity_id=r["contract"], severity="error")
                continue
            compared += 1
            total = int(r["home_score"]) + int(r["away_score"])
            implied = "yes" if total > float(strike) else "no"
            if implied != r["result"]:
                conflicts += 1
                self.store.flag(
                    "settlement_conflict",
                    f"totals contract {r['contract']} (game {r['game_id']}, strike {strike}, "
                    f"strike_type={_col(r, 'strike_type')!r}, selection={r['selection']!r}): "
                    f"kalshi settled '{r['result']}' but the official total is {total} "
                    f"({r['home_score']}-{r['away_score']}), which implies '{implied}'",
                    entity_type="market", entity_id=r["contract"], severity="critical",
                    sources="kalshi.trade_api,nhl.api_web")
        return {"totals_contracts_compared": compared, "totals_conflicts": conflicts,
                "totals_missing_strike": missing_strike}

    def check_totals_bets(self) -> dict[str, int]:
        """A totals wager without a line cannot be settled; catch it before settlement does."""
        counts = {"totals_bets": 0, "totals_missing_strike": 0}
        for b in self.store.query("SELECT * FROM bets WHERE market='total'"):
            counts["totals_bets"] += 1
            if _col(b, "strike") is None:
                counts["totals_missing_strike"] += 1
                self.store.flag("missing_strike",
                                f"totals bet {b['bet_id']} has no strike recorded",
                                entity_type="bet", entity_id=b["bet_id"], severity="error")
        return counts

    def cross_validate_puck_line_settlements(self) -> dict[str, Any]:
        """Check every settled KXNHLSPREAD contract against the official margin.

        The exchange's ``result`` and this project's reading of the payoff ("the named team's
        final margin, overtime and shootout goals included, strictly greater than
        ``floor_strike``") are two independent statements about the same fact, so any
        disagreement means one of them is wrong.  Both are recorded; neither is corrected
        here.  The count of settled contracts whose game went past regulation is reported
        separately, because those are the rows that prove the margin includes the OT goal.
        """
        from .market import covers_margin
        from .sources.kalshi import KALSHI_TEAM_ALIASES, suffix_team_code
        compared = conflicts = unreadable = past_reg = 0
        for r in self.store.query(
                """SELECT ms.contract, ms.game_id, ms.result, ms.floor_strike, ms.strike_type,
                          ms.rung, ms.team_abbrev, ms.selection, g.home_score, g.away_score,
                          g.home_id, g.away_id, g.last_period_type
                     FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                    WHERE ms.provider='kalshi' AND ms.series_ticker='KXNHLSPREAD'
                      AND ms.result IN ('yes','no')
                      AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL"""):
            strike = _col(r, "floor_strike")
            code = suffix_team_code(_col(r, "rung"))
            team_id = self.store.team_id_for(KALSHI_TEAM_ALIASES.get(code, code)) if code else None
            side = None
            if team_id is not None:
                side = ("home" if team_id == int(r["home_id"])
                        else "away" if team_id == int(r["away_id"]) else None)
            comparison = (_col(r, "strike_type") or "").lower()
            if strike is None or side is None or comparison not in ("greater", "less"):
                unreadable += 1
                self.store.flag(
                    "unreadable_puck_line",
                    f"settled puck-line contract {r['contract']} cannot be re-derived from the "
                    f"official score: floor_strike={strike!r}, strike_type={comparison!r}, "
                    f"rung={_col(r, 'rung')!r} -> team {side!r}; it is excluded from the "
                    "cross-check rather than assumed",
                    entity_type="market", entity_id=r["contract"], severity="error")
                continue
            compared += 1
            if str(_col(r, "last_period_type") or "").upper() in ("OT", "SO"):
                past_reg += 1
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            margin = hs - as_ if side == "home" else as_ - hs
            covered = covers_margin(margin, float(strike), comparison)
            implied = "yes" if covered else "no"
            if implied != r["result"]:
                conflicts += 1
                self.store.flag(
                    "settlement_conflict",
                    f"puck-line contract {r['contract']} (game {r['game_id']}, {side} side, "
                    f"strike {strike}, strike_type={comparison!r}, selection={r['selection']!r}):"
                    f" kalshi settled '{r['result']}' but the official margin is {margin:+d} "
                    f"({hs}-{as_}, lastPeriodType={_col(r, 'last_period_type')}), which implies "
                    f"'{implied}'",
                    entity_type="market", entity_id=r["contract"], severity="critical",
                    sources="kalshi.trade_api,nhl.api_web")
        return {"puck_line_contracts_compared": compared, "puck_line_conflicts": conflicts,
                "puck_line_unreadable": unreadable,
                "puck_line_compared_games_that_went_past_regulation": past_reg}

    def cross_validate_overtime_settlements(self) -> dict[str, Any]:
        """Cross-tabulate the exchange's overtime results against the NHL's own period type.

        This is the study that decides what a KXNHLOVERTIME contract pays on.  It is not
        assumed from the rules text ("if the teams go to overtime ... resolves to Yes"),
        because the text does not say whether a game decided in a **shootout** counts -- and a
        shootout is only reachable through overtime, so the two readings differ exactly on the
        games that matter.  Every settled contract with an official ``lastPeriodType`` becomes
        one cell of the table:

        * cells the evidence supports (REG -> no, OT -> yes) are counted as verified, and a
          contract landing in the *wrong* verified cell is a critical conflict;
        * an SO row is evidence about the shootout question, whichever way it fell, so it is
          recorded verbatim in ``shootout_evidence`` and the settlement rule is updated only
          by that evidence -- never by inference.

        The window the evidence covers is reported too, because a series whose settled history
        is playoff-only says nothing about shootouts: the playoffs have none.
        """
        table: dict[str, int] = {}
        conflicts = 0
        shootout_evidence: list[str] = []
        dates: list[str] = []
        game_types: dict[str, int] = {}
        for r in self.store.query(
                """SELECT ms.contract, ms.game_id, ms.result, ms.settlement_ts, g.last_period_type,
                          g.game_type, g.game_date, g.home_score, g.away_score
                     FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                    WHERE ms.provider='kalshi' AND ms.series_ticker='KXNHLOVERTIME'
                      AND ms.result IN ('yes','no')
                    ORDER BY g.game_date"""):
            lpt = str(_col(r, "last_period_type") or "UNKNOWN").upper()
            cell = f"{lpt}->{r['result']}"
            table[cell] = table.get(cell, 0) + 1
            dates.append(str(r["game_date"]))
            gt = {"1": "preseason", "2": "regular", "3": "playoff"}.get(str(r["game_type"]),
                                                                       str(r["game_type"]))
            game_types[gt] = game_types.get(gt, 0) + 1
            if lpt == "SO":
                shootout_evidence.append(
                    f"{r['contract']} (game {r['game_id']}, {r['game_date']}, "
                    f"{r['home_score']}-{r['away_score']}): NHL lastPeriodType=SO and kalshi "
                    f"settled '{r['result']}'")
            elif lpt in ("REG", "OT"):
                expected = "yes" if lpt == "OT" else "no"
                if r["result"] != expected:
                    conflicts += 1
                    self.store.flag(
                        "settlement_conflict",
                        f"overtime contract {r['contract']} (game {r['game_id']}, "
                        f"{r['game_date']}): kalshi settled '{r['result']}' but the NHL's own "
                        f"lastPeriodType is '{lpt}', which this project's verified mapping reads "
                        f"as '{expected}'",
                        entity_type="market", entity_id=r["contract"], severity="critical",
                        sources="kalshi.trade_api,nhl.api_web")
            else:
                self.store.flag(
                    "unsettleable_overtime",
                    f"overtime contract {r['contract']} (game {r['game_id']}) settled "
                    f"'{r['result']}' but the game has no readable lastPeriodType "
                    f"({_col(r, 'last_period_type')!r}), so the mapping cannot be checked",
                    entity_type="market", entity_id=r["contract"], severity="warn")
        out = {"overtime_contracts_compared": sum(table.values()),
               "overtime_conflicts": conflicts,
               "cross_tab": table,
               "window": {"first_game": min(dates) if dates else None,
                          "last_game": max(dates) if dates else None,
                          "game_types": game_types},
               "shootout_evidence": shootout_evidence}
        if not shootout_evidence:
            out["shootout_treatment"] = (
                "UNVERIFIED: no settled KXNHLOVERTIME contract in this ledger belongs to a game "
                "the NHL recorded as decided in a shootout, and the series' verified window is "
                f"{out['window']['first_game']}..{out['window']['last_game']} "
                f"(game types {game_types}). A shootout is only reachable through overtime, so "
                "the rules text does not settle the question by itself. Wagers on such a game "
                "are left OPEN and settled from the exchange's own result; nothing is inferred.")
            self.store.flag(
                "settlement_rule_unverified",
                "overtime market: " + out["shootout_treatment"],
                entity_type="market", entity_id="KXNHLOVERTIME", severity="warn",
                sources="kalshi.trade_api,nhl.api_web")
        else:
            out["shootout_treatment"] = (
                f"evidence exists on {len(shootout_evidence)} shootout game(s): "
                + "; ".join(shootout_evidence))
            # the rule is now verified by settled evidence, so an earlier
            # "settlement rule unverified" flag no longer describes reality
            self.store.resolve_irregularities_where(
                "settlement_rule_unverified", "overtime market: UNVERIFIED",
                f"the exchange's settled history now includes {len(shootout_evidence)} game(s) "
                f"the NHL recorded as decided in a shootout, so the overtime settlement rule "
                f"is verified against both sources; " + out["shootout_treatment"][:220],
                entity_id="KXNHLOVERTIME", actor="verifier")
        return out

    def check_strike_market_bets(self) -> dict[str, Any]:
        """Wagers on the strike-priced and strike-less markets, and what depth they claim.

        Three things a reader of the ledger must be able to see without asking:

        * a puck-line wager with no strike recorded cannot be re-settled, so it is an error;
        * an overtime wager whose selection is not ``ot_yes``/``ot_no`` cannot be mapped to a
          payoff, so it is an error;
        * how many wagers were filled against a **declared** depth cap because the venue
          published no offer size for the side bought.  That number is an assumption this
          project makes, and it is reported rather than buried in the notes blob.
        """
        counts: dict[str, Any] = {"puck_line_bets": 0, "puck_line_missing_strike": 0,
                                  "overtime_bets": 0, "overtime_unmapped_selection": 0,
                                  "bets_on_declared_depth_cap": 0, "bets_by_depth_basis": {},
                                  "no_side_bets": 0}
        for b in self.store.query(
                "SELECT * FROM bets WHERE market IN ('puck_line','overtime') "
                "OR depth_basis IS NOT NULL OR exchange_side='NO'"):
            market = b["market"]
            basis = _col(b, "depth_basis")
            if basis:
                counts["bets_by_depth_basis"][basis] = counts["bets_by_depth_basis"].get(basis, 0) + 1
                if basis == "declared_cap_no_published_size":
                    counts["bets_on_declared_depth_cap"] += 1
            if str(_col(b, "exchange_side") or "").upper() == "NO":
                counts["no_side_bets"] += 1
            if market == "puck_line":
                counts["puck_line_bets"] += 1
                if _col(b, "strike") is None:
                    counts["puck_line_missing_strike"] += 1
                    self.store.flag("missing_strike",
                                    f"puck-line bet {b['bet_id']} has no strike recorded, so its "
                                    "payoff cannot be re-derived",
                                    entity_type="bet", entity_id=b["bet_id"], severity="error")
            elif market == "overtime":
                counts["overtime_bets"] += 1
                if (b["selection"] or "") not in ("ot_yes", "ot_no"):
                    counts["overtime_unmapped_selection"] += 1
                    self.store.flag(
                        "unmapped_selection",
                        f"overtime bet {b['bet_id']} has selection {b['selection']!r}, which maps "
                        "to no payoff",
                        entity_type="bet", entity_id=b["bet_id"], severity="error")
        return counts

    def check_period_goal_reconciliation(self) -> dict[str, Any]:
        """Derived period scores vs the official final score, per game.

        A difference is not automatically an error: the deciding shootout attempt is not
        published as a goal row.  What matters is that every difference is *explained* on the
        row, so an unexplained one is flagged here.
        """
        out = {"games_with_period_scores": 0, "reconciled": 0, "unreconciled": 0,
               "unreconciled_without_a_note": 0, "goal_rows": 0}
        out["goal_rows"] = int(self.store.one(
            "SELECT COUNT(*) c FROM game_period_goals")["c"] or 0)
        for r in self.store.query(
                """SELECT game_id, SUM(home_goals) h, SUM(away_goals) a,
                          MIN(reconciled) reconciled,
                          SUM(CASE WHEN reconciled=0 AND (reconciliation_note IS NULL
                               OR reconciliation_note='') THEN 1 ELSE 0 END) silent
                     FROM game_period_scores GROUP BY game_id"""):
            out["games_with_period_scores"] += 1
            if int(r["reconciled"] or 0) == 1:
                out["reconciled"] += 1
            else:
                out["unreconciled"] += 1
                out["unreconciled_without_a_note"] += int(r["silent"] or 0)
                if int(r["silent"] or 0):
                    self.store.flag(
                        "unreconciled_period_goals",
                        f"game {r['game_id']}: derived period goals {r['h']}-{r['a']} differ from "
                        "the official final score and no explanation is recorded on the row",
                        entity_type="game", entity_id=str(r["game_id"]), severity="error",
                        sources="nhl.score,nhl.scoreboard")
        return out

    #: How far apart two prices may be in time and still be called a disagreement about the
    #: same question.  Kalshi's historical tier only serves settled contracts, so a stored
    #: Kalshi price can be many hours older than the Polymarket snapshot; recording both is
    #: still useful, but calling that a disagreement would be inventing one.
    COMPARISON_MAX_AGE_MINUTES = 30.0
    #: Price gap (in dollars per contract) at which two venues are recorded as disagreeing
    DISAGREEMENT_THRESHOLD = 0.05

    def _polymarket_team_abbrev(self, name: str | None) -> str | None:
        """Map a Polymarket outcome nickname ("Red Wings") to exactly one NHL abbrev.

        Two rules, each requiring a **unique** match among active franchises, and the second
        only consulted when the first finds nothing:

        1. **Suffix** on the full name ("Red Wings" -> Detroit Red Wings).
        2. **Word-boundary prefix** on the full name ("Utah" -> Utah Mammoth).  This is what
           resolves the one case the suffix rule could not: Polymarket names the Utah
           franchise simply "Utah", and its full name is neither "Utah Hockey Club" (the
           club's earlier name, stored inactive) nor anything ending in "utah".  A prefix is
           only ever used when it names exactly one active club -- "New" prefixes three of
           them and so stays unmapped, which is the point: a wrong team turns a corroboration
           into a fake conflict.
        """
        if not name:
            return None
        want = name.strip().lower()
        if not want:
            return None
        teams = [(r["abbrev"], (r["full_name"] or "").lower()) for r in self.store.query(
            "SELECT abbrev, full_name FROM teams WHERE active=1")]
        hits = {ab for ab, full in teams if full.endswith(want)}
        if len(hits) == 1:
            return hits.pop()
        if hits:
            return None      # genuinely ambiguous nickname: never guessed
        prefix = {ab for ab, full in teams if full.startswith(want + " ")}
        return prefix.pop() if len(prefix) == 1 else None

    def _kalshi_quote_for(self, game_id: Any, before_ts: str | None, *, market_type: str,
                          side: str, strike: float | None = None,
                          team_abbrev: str | None = None) -> Any:
        """The latest Kalshi quote for exactly that question at or before ``before_ts``.

        One price per contract, not one per quote row: comparing every candle a contract ever
        published against a single Polymarket snapshot turns one market into hundreds of
        "comparisons".
        """
        sql = ["""SELECT contract, side, selection, bid, ask, ts_utc, strike, market_type
                    FROM market_quotes
                   WHERE provider='kalshi' AND game_id=? AND market_type=? AND side=?
                     AND ask IS NOT NULL"""]
        params: list[Any] = [game_id, market_type, side]
        if strike is not None:
            sql.append(" AND strike IS NOT NULL AND ABS(strike - ?) < 0.0001")
            params.append(strike)
        if team_abbrev is not None:
            sql.append(" AND team_abbrev=?")
            params.append(team_abbrev)
        if before_ts:
            sql.append(" AND ts_utc <= ?")
            params.append(before_ts)
        sql.append(" ORDER BY ts_utc DESC LIMIT 1")
        return self.store.one("".join(sql), tuple(params))

    @staticmethod
    def _minutes_between(a: str | None, b: str | None) -> float | None:
        try:
            ta, tb = parse_iso(a), parse_iso(b)
        except Exception:
            return None
        if ta is None or tb is None:
            return None
        return abs((ta - tb).total_seconds()) / 60.0

    def cross_check_polymarket(self) -> dict[str, Any]:
        """Compare a second prediction market with Kalshi -- question for question.

        Reference only: this project executes on Kalshi, so a Polymarket price never funds or
        settles a wager.  What it can do is corroborate or contradict one, and both outcomes
        are recorded.

        Two rules keep that honest:

        * **the same question only.**  A Polymarket total is compared with the Kalshi totals
          contract on the same line; a Polymarket moneyline outcome with the Kalshi YES
          contract on the same team.  Joining on the game alone -- which is what an earlier
          version of this study did -- compared a Polymarket "O/U 7.5" with a Kalshi puck-line
          contract and reported 1,115 disagreements that were not disagreements about
          anything.  A market one venue does not list is recorded as an absence.
        * **one price per contract, close in time.**  The Kalshi side is the latest quote at or
          before the Polymarket snapshot, and a pair further apart than
          :attr:`COMPARISON_MAX_AGE_MINUTES` is recorded with ``stale: true`` and never
          flagged as a disagreement.

        The price basis is Polymarket's published outcome price against Kalshi's mid, because
        Polymarket's ``best_ask`` is published for the first outcome only; both venues' raw
        numbers travel with every comparison so nothing depends on this choice being right.
        """
        out: dict[str, Any] = {
            "polymarket_markets": 0, "families": {}, "matched_to_games": 0,
            "comparisons": [], "disagreements": 0, "stale_comparisons": 0,
            "unmatched_reasons": {}, "market_kinds_compared": {},
            "price_basis": ("polymarket published outcome price (outcome_prices aligned with "
                            "outcomes) vs the mid of the latest kalshi quote for the same "
                            "question at or before the polymarket snapshot"),
            "disagreement_threshold": self.DISAGREEMENT_THRESHOLD,
            "max_age_minutes": self.COMPARISON_MAX_AGE_MINUTES,
        }

        def reason(key: str) -> None:
            out["unmatched_reasons"][key] = out["unmatched_reasons"].get(key, 0) + 1

        for r in self.store.query(
                """SELECT market_family, COUNT(*) c, SUM(CASE WHEN game_id IS NOT NULL THEN 1
                          ELSE 0 END) matched FROM polymarket_markets GROUP BY market_family"""):
            out["polymarket_markets"] += int(r["c"])
            out["families"][r["market_family"]] = int(r["c"])
            out["matched_to_games"] += int(r["matched"] or 0)

        for pm in self.store.query(
                """SELECT id, game_id, question, group_item_title, outcomes, outcome_prices,
                          best_bid, best_ask, market_family, retrieved_at
                     FROM polymarket_markets WHERE game_id IS NOT NULL"""):
            if pm["market_family"] != "game":
                reason("not_a_game_market")
                continue
            try:
                outcomes = json.loads(pm["outcomes"] or "[]")
                prices = [float(x) for x in json.loads(pm["outcome_prices"] or "[]")]
            except (TypeError, ValueError):
                reason("unparseable_outcomes_or_prices")
                continue
            if not outcomes or len(outcomes) != len(prices):
                reason("outcomes_and_prices_not_aligned")
                continue
            kind, line = classify_polymarket_question(pm["question"])
            for name, price in zip(outcomes, prices):
                if kind == "total":
                    low = str(name).strip().lower()
                    if low.startswith("over"):
                        side = "YES"
                    elif low.startswith("under"):
                        side = "NO"
                    else:
                        reason(f"unrecognised_total_outcome:{name}")
                        continue
                    q = self._kalshi_quote_for(pm["game_id"], pm["retrieved_at"],
                                               market_type="total", side=side, strike=line)
                else:
                    abbrev = self._polymarket_team_abbrev(name)
                    if abbrev is None:
                        reason(f"team_nickname_not_mapped:{name}")
                        continue
                    q = self._kalshi_quote_for(pm["game_id"], pm["retrieved_at"],
                                               market_type="moneyline", side="YES",
                                               team_abbrev=abbrev)
                if q is None:
                    reason(f"no_kalshi_quote_for_{kind}")
                    continue
                bid, ask = q["bid"], q["ask"]
                mid = round((float(bid) + float(ask)) / 2.0, 4) if bid is not None else float(ask)
                age = self._minutes_between(q["ts_utc"], pm["retrieved_at"])
                stale = age is None or age > self.COMPARISON_MAX_AGE_MINUTES
                diff = round(price - mid, 4)
                out["comparisons"].append({
                    "polymarket_id": pm["id"], "game_id": pm["game_id"],
                    "question": pm["question"], "market_kind": kind, "line": line,
                    "polymarket_outcome": name, "polymarket_price": price,
                    "polymarket_best_bid": pm["best_bid"], "polymarket_best_ask": pm["best_ask"],
                    "kalshi_contract": q["contract"], "kalshi_side": q["side"],
                    "kalshi_selection": q["selection"], "kalshi_bid": bid, "kalshi_ask": ask,
                    "kalshi_mid": mid, "difference_vs_kalshi_mid": diff,
                    "kalshi_quote_ts": q["ts_utc"], "polymarket_snapshot_ts": pm["retrieved_at"],
                    "age_minutes": (round(age, 1) if age is not None else None),
                    "stale": bool(stale),
                })
                out["market_kinds_compared"][kind] = out["market_kinds_compared"].get(kind, 0) + 1
                if stale:
                    out["stale_comparisons"] += 1
                    continue
                if abs(diff) > self.DISAGREEMENT_THRESHOLD:
                    out["disagreements"] += 1
                    self.store.flag(
                        "source_disagreement",
                        f"game {pm['game_id']}: the same question is priced differently by two "
                        f"venues -- polymarket {name!r} at {price} vs kalshi {q['contract']} "
                        f"({q['side']}) mid {mid} (bid {bid}/ask {ask}), quoted "
                        f"{(round(age, 1) if age is not None else '?')} minutes before the "
                        f"polymarket snapshot; both prices are kept and neither is treated as "
                        "the truth",
                        entity_type="game", entity_id=str(pm["game_id"]), severity="info",
                        sources="polymarket.gamma,kalshi.trade_api")

        if not out["comparisons"]:
            out["finding"] = (
                f"no common market: {out['polymarket_markets']} Polymarket NHL market(s) are "
                f"stored (families {out['families']}), {out['matched_to_games']} matched to an "
                "NHL game, and none of them is the same question as a stored Kalshi quote "
                f"(reasons: {out['unmatched_reasons']}). There is therefore no second-price "
                "cross-check this run; the absence is recorded rather than filled with a "
                "comparison that does not exist.")
        else:
            fresh = len(out["comparisons"]) - out["stale_comparisons"]
            out["finding"] = (
                f"{len(out['comparisons'])} like-for-like price comparison(s) "
                f"({out['market_kinds_compared']}), {fresh} of them within "
                f"{self.COMPARISON_MAX_AGE_MINUTES:.0f} minutes and {out['stale_comparisons']} "
                f"recorded as stale; {out['disagreements']} disagreement(s) over "
                f"{self.DISAGREEMENT_THRESHOLD} dollars. Unmatched: {out['unmatched_reasons']}.")
        return out

    # ------------------------------------------------------------------ self-repair, audited
    #: Irregularities this project recorded and has since proved were false, with the reason.
    #: A correction is never a deletion: the row stays in the queue, its status changes to
    #: ``resolved``, and the note says what was wrong, what the correct check is, and where the
    #: evidence lives.  Each entry is matched by kind plus a WHERE clause over columns this
    #: project wrote, so a genuine irregularity can never be swept up by it.
    FALSE_POSITIVE_REPAIRS: tuple[dict[str, str], ...] = (
        {"kind": "unmapped_injury",
         "where": "entity_id='injuries'",
         "resolution": ("Self-inflicted, resolved not deleted. The NHL API and ESPN's injury feed "
                        "do not agree on abbreviation style: the NHL returns LAK/NJD/SJS/TBL and "
                        "ESPN returns LA/NJ/SJ/TB. Every ESPN injury row for those four clubs "
                        "failed to resolve, so four franchises had no injury context feeding the "
                        "injury-gated strategies at all while this queue recorded 11-13 "
                        "unmappable entries. Fixed on 2026-09-22 by the ABBREV_ALIASES table in "
                        "store.py, applied only when the abbreviation is not already a team in "
                        "its own right. Re-probe the flag by re-ingesting: if any entry still "
                        "cannot be mapped, _injury_context raises a fresh flag with the count.")},
        {"kind": "conflicting_source",
         "where": ("sources='kalshi.historical|nhl.api_web' AND (entity_id LIKE 'KXNHLSPREAD-%' "
                   "OR entity_id LIKE 'KXNHLTOTAL-%' OR entity_id LIKE 'KXNHLOVERTIME-%')"),
         "resolution": ("False positive from a verifier bug, resolved not deleted. The moneyline "
                        "settlement check judged every Kalshi contract that names a team by 'did "
                        "that team win', which is the wrong condition for a strike market: a "
                        "puck-line contract settles on the margin, a totals contract on the goal "
                        "count, an overtime contract on whether the game reached overtime. Fixed "
                        "in cross_validate_stats_vs_schedule (moneyline contracts only) on "
                        "2026-09-21; each strike market is checked by its own study -- "
                        "cross_validate_puck_line_settlements, cross_validate_totals_settlements, "
                        "cross_validate_overtime_settlements -- which found 0 conflicts on the "
                        "same ledger.")},
        {"kind": "depth_exhausted_by_earlier_wagers",
         "where": "detail LIKE '%published depth 0 contract(s)%'",
         "resolution": ("Self-inflicted mislabel, resolved not deleted.  The first version of the "
                        "shared-depth book reported every unfillable signal as 'an earlier wager "
                        "took this level', including entry points that published no depth at all "
                        "(a candle that traded nothing) -- so the row read 'published depth 0 "
                        "contract(s) ... was already fully claimed (0) by earlier wagers', which "
                        "contradicts itself and blames the strategies for a gap in the venue's "
                        "data.  Fixed on 2026-09-22: a zero-depth entry point is now recorded as "
                        "no_depth_evidence_at_entry and this kind is kept for levels that really "
                        "were shared.  The correct flag is raised afresh on the next run for every "
                        "entry point that still publishes nothing.")},
        {"kind": "source_disagreement",
         "where": ("sources='polymarket.gamma,kalshi.trade_api' AND "
                   "detail LIKE '%while kalshi asks%'"),
         "resolution": ("False positive from a verifier bug, resolved not deleted. The "
                        "Polymarket cross-check joined on game_id only, so it compared different "
                        "questions (a Polymarket 'O/U 7.5' total against a Kalshi puck-line "
                        "contract) and compared every stored quote row instead of one price per "
                        "contract. Fixed in cross_check_polymarket on 2026-09-21: comparisons are "
                        "now like-for-like (same market kind, same side/outcome, same line) "
                        "against the latest Kalshi quote before the Polymarket snapshot, and "
                        "pairs further apart than 30 minutes are recorded as stale rather than "
                        "flagged.")},
    )

    def audit_false_positive_flags(self) -> dict[str, int]:
        """Resolve the irregularities this project has proved were its own mistakes.

        Called at the start of :meth:`run_all` so a run's counts are not inflated by flags a
        previous version of the verifier raised wrongly.  Every resolution writes an
        ``audit_log`` row naming the bug and the fix; nothing is deleted, and a genuine
        irregularity is untouched because each rule matches on the columns that identified the
        false positive in the first place.
        """
        resolved: dict[str, int] = {}
        for rep in self.FALSE_POSITIVE_REPAIRS:
            rows = self.store.query(
                f"SELECT id FROM irregularities WHERE kind=? AND status='open' AND {rep['where']}",
                (rep["kind"],))
            if not rows:
                continue
            for r in rows:
                self.store.resolve_irregularity(int(r["id"]), rep["resolution"],
                                                status="resolved")
            self.store.audit("verifier", "RESOLVE_FALSE_POSITIVE", rep["kind"],
                             json.dumps({"resolved": len(rows), "matched_by": rep["where"],
                                         "resolution": rep["resolution"]}))
            resolved[rep["kind"]] = resolved.get(rep["kind"], 0) + len(rows)
        self.store.commit()
        return resolved

    #: Irregularity kinds that this verifier re-derives in full on **every** call to
    #: :meth:`run_all`.  Because each of these iterates every row of its table, a flag of
    #: this kind that is still open but was *not* re-raised this run can only mean the
    #: condition behind it has gone away -- a data repair, a corrected game row, a strategy
    #: version that no longer trades.  Anything not on this list (an ingest-time flag such as
    #: ``broken_api``, or a pipeline-time flag such as ``unreadable_puck_line``) is left
    #: alone: this verifier never re-derives it, so its silence proves nothing.
    RECONCILABLE_KINDS = (
        "crossed_book", "duplicate_bet", "duplicate_game", "exposure_exceeds_bankroll",
        "impossible_odds", "impossible_result", "impossible_season", "incorrect_pnl",
        "insufficient_liquidity", "missing_result", "missing_strike", "missing_timestamp",
        "timestamp_order", "unmapped_selection",
    )

    #: Resolutions carried out by a human-scale research step, recorded in code so they are
    #: re-applied idempotently on every run and survive a ledger reseed from a snapshot.
    #: Each is evidence-carrying: the queue row is marked ``resolved`` with the evidence in
    #: the resolution text, one ``audit_log`` entry names the researcher and verification
    #: date, and nothing is deleted.  If the identical condition were ever re-raised,
    #: ``Store.flag`` re-opens the row and the next run re-resolves it with the same
    #: evidence, so the audit trail never loses either side.
    MANUAL_RESEARCH_RESOLUTIONS = (
        {
            "kind": "unmatched_market",
            "entity_id": "KXNHLSPREAD-26JAN26LACBJ",
            "researcher": "arena session 2026-09-22",
            "verified_at": "2026-09-22",
            "resolution": (
                "Investigated 2026-09-22 and closed as an exchange listing anomaly. "
                "(1) The NHL's own schedule has no Los Angeles @ Columbus game on 2026-01-26: "
                "CBJ hosted DAL Jan 22, TBL Jan 24 and PHI Jan 28 and visited CHI Jan 30 "
                "(api-web.nhle.com/v1/club-schedule/CBJ/month/2026-01, cross-checked against "
                "this ledger's games table), while LAK was at DET on Jan 27 (game 2025020833). "
                "(2) The exchange no longer lists the market: GET "
                "https://api.elections.kalshi.com/trade-api/v2/markets/KXNHLSPREAD-26JAN26LACBJ "
                "returned {\"error\":{\"code\":\"not_found\"}} on 2026-09-22. "
                "The ingest behaved correctly: the contract was never attached to a guessed "
                "game. There is nothing to attach and no wager can be derived from it."),
        },
    )

    def apply_manual_resolutions(self) -> int:
        n = 0
        for res in self.MANUAL_RESEARCH_RESOLUTIONS:
            rows = self.store.query(
                "SELECT id FROM irregularities WHERE kind=? AND entity_id=? AND status='open'",
                (res["kind"], res["entity_id"]))
            for r in rows:
                self.store.resolve_irregularity(
                    int(r["id"]),
                    f"{res['resolution']} [manual research resolution by "
                    f"{res['researcher']}, verified {res['verified_at']}]")
                self.store.audit("verifier", "MANUAL_RESOLUTION", res["kind"],
                                 json.dumps({"id": r["id"], "entity_id": res["entity_id"],
                                             "researcher": res["researcher"],
                                             "verified_at": res["verified_at"]}))
                n += 1
        if n:
            self.store.commit()
        return n

    def reconcile_unmatched_markets(self) -> int:
        """Close ``unmatched_market`` flags whose event has since been matched by an ingest.

        ``unmatched_market`` is raised at ingest time, so it cannot go into
        ``RECONCILABLE_KINDS``: the verifier does not re-derive it, and its silence in a
        verifier-only process proves nothing.  But whether the market was *later matched*
        is observable here directly -- if the event ticker now has quotes in the ledger,
        some ingest did attach it, and the flag no longer describes reality.
        """
        rows = self.store.query(
            "SELECT id, entity_id FROM irregularities "
            "WHERE kind='unmatched_market' AND status='open'")
        closed = 0
        for r in rows:
            ev = r["entity_id"] or ""
            if not ev or not ev.startswith("KXNHL"):
                continue
            hit = self.store.one(
                "SELECT 1 FROM market_quotes WHERE market_key=? LIMIT 1", (ev,))
            if hit:
                self.store.resolve_irregularity(
                    int(r["id"]),
                    "a later ingest matched this event to an NHL game and stored quotes for "
                    "it, so the flag no longer describes reality (audited; nothing deleted)")
                self.store.audit("verifier", "RECONCILE_UNMATCHED_MARKET", "unmatched_market",
                                 json.dumps({"id": r["id"], "event_ticker": ev}))
                closed += 1
        self.store.commit()
        return closed

    def reconcile_stale_flags(self) -> dict[str, int]:
        """Close open flags of a re-derived kind that no longer reproduce.

        ``Store.flag`` uses ``INSERT OR IGNORE`` so a condition is never recorded twice --
        but that also means a flag raised once stayed open forever, even after the code that
        raised it was fixed.  The 2026-09-21 ledger still carried two ``duplicate_game``
        irregularities for 2024-09-22 and 2025-09-21 that the current check no longer
        produces: those are genuine NHL preseason *split-squad doubleheaders* (FLA v NSH at
        Amerant Bank Arena at 18:00Z **and** 22:00Z, two different game ids, two different
        final scores), which an earlier version matched on date/home/away alone.  Leaving
        them open overstated the verification queue by two and, worse, trained a reader to
        ignore it.

        Nothing is deleted.  Each row is marked ``resolved`` with the reason, and one
        ``audit_log`` entry per kind records how many rows were closed and why.
        """
        closed: dict[str, int] = {}
        for kind in self.RECONCILABLE_KINDS:
            rows = self.store.query(
                "SELECT id, entity_id FROM irregularities WHERE kind=? AND status='open'",
                (kind,))
            stale = [r for r in rows if (kind, r["entity_id"] or "") not in self.store.flag_seen]
            if not stale:
                continue
            detail = ("the check that raises this kind re-ran on every row of its table in "
                      "this pass and did not reproduce this entity, so the condition no "
                      "longer holds; closed rather than deleted, original detail retained")
            for r in stale:
                self.store.resolve_irregularity(int(r["id"]), detail, status="resolved")
            self.store.audit("verifier", "RECONCILE_STALE_FLAG", kind,
                             json.dumps({"closed": len(stale), "checked": len(rows),
                                         "entity_ids": [r["entity_id"] for r in stale][:20]}))
            closed[kind] = len(stale)
        self.store.commit()
        return closed

    def run_all(self) -> dict[str, Any]:
        summary = {}
        # first, so a run's counts are not inflated by flags a previous version of this
        # verifier raised wrongly; each resolution is audited, none is deleted
        summary["false_positives_resolved"] = self.audit_false_positive_flags()
        summary["games"] = self.check_games()
        summary["bets"] = self.check_bets()
        summary["totals_bets"] = self.check_totals_bets()
        summary["quotes"] = self.check_quotes()
        summary["cross_validation"] = self.cross_validate_stats_vs_schedule()
        summary["totals_settlements"] = self.cross_validate_totals_settlements()
        summary["puck_line_settlements"] = self.cross_validate_puck_line_settlements()
        summary["overtime_settlements"] = self.cross_validate_overtime_settlements()
        summary["strike_market_bets"] = self.check_strike_market_bets()
        summary["period_goals"] = self.check_period_goal_reconciliation()
        summary["second_market"] = self.cross_check_polymarket()
        # evidence-carrying research resolutions first, so a queue count reflects what a
        # reader should act on; then flags whose condition is observable as recovered
        summary["manual_resolutions"] = self.apply_manual_resolutions()
        summary["unmatched_markets_closed"] = self.reconcile_unmatched_markets()
        # last: every check above has now re-raised whatever it still finds, so anything left
        # open without having been re-raised describes a condition that no longer exists
        summary["stale_flags_closed"] = self.reconcile_stale_flags()
        summary["open_irregularities"] = self.store.one(
            "SELECT COUNT(*) c FROM irregularities WHERE status='open'")["c"]
        summary["irregularities_total"] = self.store.one(
            "SELECT COUNT(*) c FROM irregularities")["c"]
        self.store.audit("verifier", "RUN_ALL", "", json.dumps(summary))
        self.store.commit()
        return summary
