"""Ingestion of the sources verified in the 2026-09-20 research pass.

* Kalshi historical tier: settled NHL contracts + timestamped candlesticks  -> real
  closing/opening prices for BACKTESTS (``market_settlements`` + ``market_price_points``)
* NHL stats REST per-game team and goalie lines                              -> special
  teams, shot share and goalie workload/quality features
* NHL partner (DraftKings) odds feed                                          -> a second,
  independent price for cross-validation / CLV on FORWARD TESTS (no history exists)
* NHL EDGE team aggregates                                                    -> snapshots

Everything is written with provenance; nothing is inferred when a source is silent.
The mixin is composed into :class:`nhlcomp.ingest.Ingestor`.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from .http import NetworkUnavailable, parse_iso
from .market import derive_price_points
from .sources.kalshi import (BASE as KALSHI_BASE, HIST_BASE, KalshiApiError, market_team_code,
                             market_type_for, parse_event_ticker)
from .sources.nhl import (API_WEB, STATS_REST, parse_goalie_game_row, parse_partner_odds,
                          parse_team_game_row)
from .store import utcnow


def _epoch(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return int(parse_iso(ts).timestamp())
    except Exception:
        return None


class IngestExtensions:
    """Methods mixed into ``Ingestor`` (expects ``self.store``, ``self.kalshi``, ``self.nhl``,
    ``self.rest``, ``self.log`` and ``self._match_kalshi_game``)."""

    # ------------------------------------------------------------------ kalshi series
    def kalshi_series_discovery(self) -> int:
        """Record every hockey series Kalshi lists, so new market types are noticed
        automatically instead of being hard-coded."""
        series = self.kalshi.series_list(category="Sports", tags="Hockey")
        if not series:
            self.store.flag("no_markets", "kalshi /series?category=Sports&tags=Hockey returned "
                            "nothing", severity="warn", entity_type="source",
                            entity_id="kalshi.trade_api")
            return 0
        now = utcnow()
        self.store.record_raw(f"{KALSHI_BASE}/series?category=Sports&tags=Hockey",
                              json.dumps(series), source_id="kalshi.trade_api")
        for s in series:
            t = s.get("ticker")
            if not t:
                continue
            self.store.execute(
                """INSERT INTO kalshi_series(ticker, title, category, tags, fee_type, frequency,
                                             settlement_sources, contract_terms_url,
                                             first_seen, last_seen)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(ticker) DO UPDATE SET title=excluded.title,
                     fee_type=excluded.fee_type, settlement_sources=excluded.settlement_sources,
                     last_seen=excluded.last_seen""",
                (t, s.get("title"), s.get("category"), json.dumps(s.get("tags") or []),
                 s.get("fee_type"), s.get("frequency"),
                 json.dumps(s.get("settlement_sources") or []), s.get("contract_terms_url"),
                 now, now))
        self.store.commit()
        nhl = [s.get("ticker") for s in series if str(s.get("ticker", "")).startswith("KXNHL")]
        self.log(f"kalshi series: {len(series)} hockey series, NHL: {sorted(nhl)}")
        return len(series)

    # ------------------------------------------------------------------ kalshi history
    def _upsert_settlement(self, m: dict, *, tier: str, source_url: str, ts: str) -> int | None:
        """Insert/refresh one settled contract; returns the matched game_id (or None)."""
        game_id, _ = self._match_kalshi_game(m)
        ticker = m.get("ticker") or ""
        result = (m.get("result") or "").lower()
        code = market_team_code(m)
        parsed = parse_event_ticker(m.get("event_ticker") or "")
        team_abbrev = None
        if code and parsed and not parsed.get("ambiguous"):
            team_abbrev = (parsed["away_abbrev"] if code == parsed.get("away_code")
                           else parsed["home_abbrev"] if code == parsed.get("home_code") else None)
        self.store.execute(
            """INSERT INTO market_settlements(provider, event_ticker, contract, game_id,
                                              game_date, selection, side, result,
                                              settle_price, price_before, bid_before,
                                              ask_before, volume, open_interest, open_time,
                                              settlement_ts, retrieved_at, source_url,
                                              series_ticker, tier, floor_strike, close_time,
                                              title, occurrence_datetime, team_abbrev, market_type)
               VALUES('kalshi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(provider, contract, side) DO UPDATE SET
                 result=excluded.result, settle_price=excluded.settle_price,
                 price_before=excluded.price_before, bid_before=excluded.bid_before,
                 ask_before=excluded.ask_before, volume=excluded.volume,
                 open_interest=excluded.open_interest,
                 game_id=COALESCE(excluded.game_id, market_settlements.game_id),
                 series_ticker=excluded.series_ticker, tier=excluded.tier,
                 floor_strike=excluded.floor_strike, close_time=excluded.close_time,
                 title=excluded.title, occurrence_datetime=excluded.occurrence_datetime,
                 team_abbrev=COALESCE(excluded.team_abbrev, market_settlements.team_abbrev),
                 market_type=excluded.market_type,
                 settlement_ts=COALESCE(excluded.settlement_ts, market_settlements.settlement_ts)""",
            (m.get("event_ticker"), ticker, game_id,
             (m.get("occurrence_datetime") or "")[:10] or (parsed or {}).get("game_date"),
             m.get("yes_sub_title") or m.get("title") or ticker, "YES", result,
             _f(m.get("settlement_value_dollars")), _f(m.get("previous_price_dollars")),
             _f(m.get("previous_yes_bid_dollars")), _f(m.get("previous_yes_ask_dollars")),
             _f(m.get("volume_fp")), _f(m.get("open_interest_fp")),
             m.get("open_time"), m.get("settlement_ts"), ts, source_url,
             (m.get("event_ticker") or "").split("-", 1)[0] or None, tier,
             _f(m.get("floor_strike")), m.get("close_time"), m.get("title"),
             m.get("occurrence_datetime"), team_abbrev, market_type_for(m)))
        return game_id

    def kalshi_history(self, *, series: str = "KXNHLGAME", max_calls: int = 1500,
                       period_interval: int = 60, max_pages: int = 20) -> dict[str, Any]:
        """Load the settled history of one Kalshi series and its pre-game price history.

        Steps (each resumable, each bounded by ``max_calls`` API reads):

        1. ``/historical/cutoff`` -> which tier a contract's candles live in.
        2. Live-tier settled contracts (``/markets?status=settled``): the most recent games.
        3. Historical-tier contracts (``/historical/markets?series_ticker=``), newest first,
           stopping as soon as a whole page is already known.
        4. For every contract matched to an NHL game that has no price points yet, fetch
           hourly candlesticks over [open_time, close_time] and store the named points.

        The entry price a BACKTEST uses is the *closing* candle's offer (``close.ask``):
        the last executable price before puck drop.  ``open``/``t24h``/``t6h`` give line
        movement; ``ig60``/``ig120`` give in-game reference prices.
        """
        stats: dict[str, Any] = {"series": series, "calls_start": self.kalshi.calls}
        ts = utcnow()
        cutoff = self.kalshi.historical_cutoff()
        stats["cutoff"] = cutoff
        cutoff_ts = _epoch(cutoff)
        if cutoff is None:
            self.store.flag("broken_api", "kalshi /historical/cutoff unavailable; candle tier "
                            "routing falls back to trying historical then live",
                            severity="warn", entity_type="source", entity_id="kalshi.historical")
        # -- 2. live-tier settled
        n_live = 0
        try:
            live = self.kalshi.settled_markets(series, max_pages=5)
            url = f"{KALSHI_BASE}/markets?series_ticker={series}&status=settled"
            if live:
                self.store.record_raw(url, json.dumps(live), source_id="kalshi.settled_markets")
            for m in live:
                self._upsert_settlement(m, tier="live", source_url=url, ts=ts)
                n_live += 1
        except (KalshiApiError, NetworkUnavailable) as exc:
            self.store.flag("broken_api", f"kalshi live settled {series}: {exc}", severity="error",
                            entity_type="source", entity_id="kalshi.settled_markets")
        stats["live_settled"] = n_live
        self.store.commit()

        # -- 3. historical tier (resume-aware)
        known = {r["contract"] for r in self.store.query(
            "SELECT contract FROM market_settlements WHERE provider='kalshi' AND series_ticker=?",
            (series,))}
        n_hist, new_hist = 0, 0
        cursor = None
        pages = 0
        earliest = None
        try:
            while pages < max_pages and (self.kalshi.calls - stats["calls_start"]) < max_calls:
                page, cursor = self.kalshi.historical_markets(series, cursor=cursor, max_pages=1)
                pages += 1
                url = f"{HIST_BASE}/markets?series_ticker={series}&limit=1000"
                if page:
                    self.store.record_raw(url + (f"&cursor={cursor}" if cursor else ""),
                                          json.dumps(page), source_id="kalshi.historical")
                fresh = 0
                for m in page:
                    self._upsert_settlement(m, tier="historical", source_url=url, ts=ts)
                    n_hist += 1
                    if m.get("ticker") not in known:
                        fresh += 1
                        known.add(m.get("ticker"))
                    d = (m.get("occurrence_datetime") or m.get("close_time") or "")[:10]
                    if d and (earliest is None or d < earliest):
                        earliest = d
                new_hist += fresh
                self.store.commit()
                complete = self.store.one("SELECT history_complete FROM kalshi_series WHERE ticker=?",
                                          (series,))
                if not page or cursor is None:
                    self.store.execute(
                        """INSERT INTO kalshi_series(ticker, first_seen, last_seen, history_complete,
                                                     earliest_event)
                           VALUES(?,?,?,1,?)
                           ON CONFLICT(ticker) DO UPDATE SET history_complete=1, last_seen=excluded.last_seen,
                             earliest_event=COALESCE(excluded.earliest_event, kalshi_series.earliest_event)""",
                        (series, ts, ts, earliest))
                    break
                if fresh == 0 and complete and int(complete["history_complete"] or 0) == 1:
                    break   # everything below this page was loaded by an earlier run
        except (KalshiApiError, NetworkUnavailable) as exc:
            self.store.flag("broken_api", f"kalshi historical markets {series}: {exc}",
                            severity="error", entity_type="source", entity_id="kalshi.historical")
        stats["historical_rows"] = n_hist
        stats["historical_new"] = new_hist
        stats["historical_pages"] = pages
        stats["unmatched_contracts"] = int(self.store.one(
            """SELECT COUNT(*) c FROM market_settlements
                WHERE provider='kalshi' AND series_ticker=? AND game_id IS NULL""", (series,))["c"])
        self.store.commit()

        # -- 4. candlesticks -> price points
        stats.update(self._kalshi_price_points(series, cutoff_ts=cutoff_ts, budget=max_calls,
                                                calls_start=stats["calls_start"],
                                                period_interval=period_interval))
        stats["calls_used"] = self.kalshi.calls - stats["calls_start"]
        self.log(f"kalshi history {series}: cutoff={cutoff} live={n_live} hist={n_hist} "
                 f"(new {new_hist}, unmatched {stats['unmatched_contracts']}) candles "
                 f"ok={stats.get('candles_ok')} empty={stats.get('candles_empty')} "
                 f"err={stats.get('candles_error')} pending={stats.get('candles_pending')} "
                 f"calls={stats['calls_used']}")
        return stats

    def _kalshi_price_points(self, series: str, *, cutoff_ts: int | None, budget: int,
                             calls_start: int, period_interval: int = 60) -> dict[str, int]:
        rows = self.store.query(
            """SELECT ms.contract, ms.event_ticker, ms.game_id, ms.open_time, ms.close_time,
                      ms.settlement_ts, ms.tier, ms.team_abbrev, ms.market_type, g.start_time_utc
                 FROM market_settlements ms JOIN games g ON g.game_id = ms.game_id
                WHERE ms.provider='kalshi' AND ms.series_ticker=? AND ms.result IN ('yes','no')
                  AND (ms.candles_state IS NULL OR ms.candles_state='pending'
                       OR ms.candles_state LIKE 'error:%')
                ORDER BY g.start_time_utc DESC""", (series,))
        ok = empty = err = 0
        for r in rows:
            if (self.kalshi.calls - calls_start) >= budget:
                break
            start_ts = _epoch(r["start_time_utc"])
            open_ts = _epoch(r["open_time"]) or (start_ts - 4 * 86400 if start_ts else None)
            end_ts = _epoch(r["close_time"]) or _epoch(r["settlement_ts"]) or (
                start_ts + 6 * 3600 if start_ts else None)
            if start_ts is None or open_ts is None or end_ts is None:
                self.store.execute("UPDATE market_settlements SET candles_state=? WHERE contract=?",
                                   ("error:missing timestamps", r["contract"]))
                err += 1
                continue
            end_ts = max(end_ts, start_ts + 4 * 3600)
            settled_ts = _epoch(r["settlement_ts"])
            if cutoff_ts is not None and settled_ts is not None:
                tier = "historical" if settled_ts < cutoff_ts else "live"
            else:
                tier = r["tier"] or "historical"
            candles: list[dict] = []
            errors: list[str] = []
            for t in (tier, "live" if tier == "historical" else "historical"):
                try:
                    candles = self.kalshi.candlesticks(series, r["contract"], start_ts=open_ts,
                                                       end_ts=end_ts, period_interval=period_interval,
                                                       tier=t)
                except (KalshiApiError, NetworkUnavailable) as exc:
                    errors.append(f"{t}: {str(exc)[:120]}")
                    candles = []
                time.sleep(self.kalshi.pace_seconds)
                if candles:
                    tier = t
                    break
            if not candles:
                # two clean-but-empty responses -> 'empty'; any API failure -> retry next run
                state = "empty" if len(errors) < 2 else "error:" + " | ".join(errors)[:200]
                self.store.execute("UPDATE market_settlements SET candles_state=? WHERE contract=?",
                                   (state, r["contract"]))
                if state == "empty":
                    empty += 1
                else:
                    err += 1
                continue
            points = derive_price_points(candles, start_ts=start_ts)
            if tier == "historical":
                src = (f"{HIST_BASE}/markets/{r['contract']}/candlesticks?start_ts={open_ts}"
                       f"&end_ts={end_ts}&period_interval={period_interval}")
            else:
                src = (f"{KALSHI_BASE}/series/{series}/markets/{r['contract']}/candlesticks"
                       f"?start_ts={open_ts}&end_ts={end_ts}&period_interval={period_interval}")
            self._write_points(r["contract"], points, game_id=r["game_id"], team_abbrev=r["team_abbrev"],
                               series=series, market_type=r["market_type"], tier=tier,
                               period_interval=period_interval, src=src)
            self.store.execute("UPDATE market_settlements SET candles_state=?, tier=? WHERE contract=?",
                               ("ok" if "close" in points else "no_pregame_candle", tier, r["contract"]))
            ok += 1
            if ok % 50 == 0:
                self.store.commit()
            time.sleep(self.kalshi.pace_seconds)
        self.store.commit()
        pending = self.store.one(
            """SELECT COUNT(*) c FROM market_settlements ms JOIN games g ON g.game_id=ms.game_id
                WHERE ms.provider='kalshi' AND ms.series_ticker=? AND ms.result IN ('yes','no')
                  AND (ms.candles_state IS NULL OR ms.candles_state='pending')""", (series,))["c"]
        return {"candles_ok": ok, "candles_empty": empty, "candles_error": err,
                "candles_pending": int(pending)}

    def _write_points(self, contract: str, points: dict[str, dict], *, game_id: int | None,
                      team_abbrev: str | None, series: str, market_type: str | None, tier: str,
                      period_interval: int, src: str) -> None:
        now = utcnow()
        for name, p in points.items():
            self.store.execute(
                """INSERT INTO market_price_points(contract, point, end_period_ts, game_id, team_abbrev,
                                                   series_ticker, market_type, bid, ask, last, mean,
                                                   volume, open_interest, period_interval, tier,
                                                   retrieved_at, source_url)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(contract, point) DO UPDATE SET end_period_ts=excluded.end_period_ts,
                     game_id=COALESCE(excluded.game_id, market_price_points.game_id),
                     team_abbrev=COALESCE(excluded.team_abbrev, market_price_points.team_abbrev),
                     bid=excluded.bid, ask=excluded.ask, last=excluded.last, mean=excluded.mean,
                     volume=excluded.volume, open_interest=excluded.open_interest,
                     tier=excluded.tier, retrieved_at=excluded.retrieved_at,
                     source_url=excluded.source_url""",
                (contract, name, p["end_period_ts"], game_id, team_abbrev, series, market_type,
                 p["bid"], p["ask"], p["last"], p["mean"], p["volume"], p["open_interest"],
                 period_interval, tier, now, src))

    def kalshi_live_price_points(self, *, series: str = "KXNHLGAME", max_calls: int = 120,
                                 period_interval: int = 60) -> int:
        """Opening / T-24h / T-6h / T-1h / latest points for contracts that are still open.

        Uses the live-tier candlestick endpoint, so FORWARD-TEST line-movement features are
        computed from Kalshi's own timestamped history rather than from whenever this job
        happened to first observe a quote.
        """
        rows = self.store.query(
            """SELECT q.contract, q.market_key, q.game_id, g.start_time_utc, MAX(q.ts_utc) ts
                 FROM market_quotes q JOIN games g ON g.game_id = q.game_id
                WHERE q.provider='kalshi' AND q.side='YES' AND q.market_key LIKE ? AND q.game_id IS NOT NULL
                  AND g.home_score IS NULL
                GROUP BY q.contract ORDER BY g.start_time_utc""", (series + "-%",))
        n = 0
        now_ts = int(datetime.now(timezone.utc).timestamp())
        for r in rows[:max_calls]:
            start_ts = _epoch(r["start_time_utc"])
            if start_ts is None:
                continue
            open_ts = start_ts - 6 * 86400
            end_ts = min(now_ts, start_ts + 4 * 3600)
            try:
                candles = self.kalshi.candlesticks(series, r["contract"], start_ts=open_ts, end_ts=end_ts,
                                                   period_interval=period_interval, tier="live")
            except (KalshiApiError, NetworkUnavailable):
                continue
            time.sleep(self.kalshi.pace_seconds)
            if not candles:
                continue
            points = derive_price_points(candles, start_ts=start_ts)
            # before puck drop there is no 'close'; expose the newest candle as 'latest'
            if now_ts < start_ts:
                points.pop("close", None)
                points.pop("ig60", None)
                points.pop("ig120", None)
                latest = points.pop("final", None)
                if latest:
                    points["latest"] = latest
            parsed = parse_event_ticker(r["market_key"])
            code = r["contract"][len(r["market_key"]) + 1:] if r["contract"].startswith(r["market_key"] + "-") else None
            team_abbrev = None
            if parsed and not parsed.get("ambiguous") and code:
                team_abbrev = (parsed["away_abbrev"] if code == parsed.get("away_code")
                               else parsed["home_abbrev"] if code == parsed.get("home_code") else None)
            src = (f"{KALSHI_BASE}/series/{series}/markets/{r['contract']}/candlesticks"
                   f"?start_ts={open_ts}&end_ts={end_ts}&period_interval={period_interval}")
            self._write_points(r["contract"], points, game_id=int(r["game_id"]), team_abbrev=team_abbrev,
                               series=series, market_type="moneyline", tier="live",
                               period_interval=period_interval, src=src)
            n += 1
        self.store.commit()
        self.log(f"kalshi live price points: {n} open contracts")
        return n

    # ------------------------------------------------------------------ NHL per-game stats
    def nhl_team_game_stats(self, seasons: Sequence[int], game_types: Sequence[int] = (2,),
                            *, current_season: int | None = None) -> int:
        n = 0
        for season in seasons:
            for gt in game_types:
                fresh = (current_season is None) or (season >= current_season)
                try:
                    rows = self.rest.team_game_rows(season, gt, use_cache=not fresh)
                except (NetworkUnavailable, ValueError) as exc:
                    self.store.flag("broken_api", f"stats REST team/summary isGame {season}/{gt}: {exc}",
                                    severity="warn", entity_type="source", entity_id="nhl.stats_rest")
                    continue
                url = (f"{STATS_REST}/team/summary?isAggregate=false&isGame=true&cayenneExp="
                       f"seasonId={season} and gameTypeId={gt}")
                if rows:
                    self.store.record_raw(url, json.dumps(rows[:50]) if len(rows) > 50 else json.dumps(rows),
                                          source_id="nhl.stats_rest_game")
                ts = utcnow()
                for raw in rows:
                    r = parse_team_game_row(raw)
                    if r is None:
                        continue
                    self.store.execute(
                        """INSERT INTO team_game_stats(game_id, team_id, game_date, home_road,
                                 opponent_abbrev, team_name, gf, ga, sf, sa, pp_pct, pk_pct,
                                 pp_net_pct, pk_net_pct, fo_pct, wins, losses, ot_losses, points,
                                 reg_wins, so_wins, season, game_type, source_id, retrieved_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(game_id, team_id) DO UPDATE SET gf=excluded.gf, ga=excluded.ga,
                             sf=excluded.sf, sa=excluded.sa, pp_pct=excluded.pp_pct, pk_pct=excluded.pk_pct,
                             pp_net_pct=excluded.pp_net_pct, pk_net_pct=excluded.pk_net_pct,
                             fo_pct=excluded.fo_pct, wins=excluded.wins, losses=excluded.losses,
                             ot_losses=excluded.ot_losses, points=excluded.points,
                             reg_wins=excluded.reg_wins, so_wins=excluded.so_wins,
                             retrieved_at=excluded.retrieved_at""",
                        (r["game_id"], r["team_id"], r["game_date"], r["home_road"],
                         r["opponent_abbrev"], r["team_name"], r["gf"], r["ga"], r["sf"], r["sa"],
                         r["pp_pct"], r["pk_pct"], r["pp_net_pct"], r["pk_net_pct"], r["fo_pct"],
                         r["wins"], r["losses"], r["ot_losses"], r["points"], r["reg_wins"],
                         r["so_wins"], season, gt, "nhl.stats_rest", ts))
                    n += 1
                self.store.commit()
                self.log(f"stats REST team game rows {season}/{gt}: {len(rows)}")
        return n

    def nhl_goalie_game_stats(self, seasons: Sequence[int], game_types: Sequence[int] = (2,),
                              *, current_season: int | None = None) -> int:
        n = 0
        for season in seasons:
            for gt in game_types:
                fresh = (current_season is None) or (season >= current_season)
                try:
                    rows = self.rest.goalie_game_rows(season, gt, use_cache=not fresh)
                except (NetworkUnavailable, ValueError) as exc:
                    self.store.flag("broken_api", f"stats REST goalie/summary isGame {season}/{gt}: {exc}",
                                    severity="warn", entity_type="source", entity_id="nhl.stats_rest")
                    continue
                url = (f"{STATS_REST}/goalie/summary?isAggregate=false&isGame=true&cayenneExp="
                       f"seasonId={season} and gameTypeId={gt}")
                if rows:
                    self.store.record_raw(url, json.dumps(rows[:50]), source_id="nhl.stats_rest_game")
                ts = utcnow()
                for raw in rows:
                    r = parse_goalie_game_row(raw)
                    if r is None:
                        continue
                    self.store.execute(
                        """INSERT INTO goalie_game_stats(game_id, player_id, goalie_name, team_abbrev,
                                 opponent_abbrev, home_road, game_date, started, saves, shots_against,
                                 goals_against, save_pct, toi_seconds, decision, season, game_type,
                                 source_id, retrieved_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(game_id, player_id) DO UPDATE SET started=excluded.started,
                             saves=excluded.saves, shots_against=excluded.shots_against,
                             goals_against=excluded.goals_against, save_pct=excluded.save_pct,
                             toi_seconds=excluded.toi_seconds, decision=excluded.decision,
                             retrieved_at=excluded.retrieved_at""",
                        (r["game_id"], r["player_id"], r["goalie_name"], r["team_abbrev"],
                         r["opponent_abbrev"], r["home_road"], r["game_date"], r["started"],
                         r["saves"], r["shots_against"], r["goals_against"], r["save_pct"],
                         r["toi_seconds"], r["decision"], season, gt, "nhl.stats_rest", ts))
                    n += 1
                self.store.commit()
                self.log(f"stats REST goalie game rows {season}/{gt}: {len(rows)}")
        return n

    # ------------------------------------------------------------------ partner odds
    def nhl_partner_odds(self, country: str = "US") -> int:
        url = f"{API_WEB}/partner-game/{country}/now"
        try:
            payload = self.nhl.partner_odds(country)
        except (NetworkUnavailable, ValueError) as exc:
            self.store.flag("broken_api", f"NHL partner odds: {exc}", severity="warn",
                            entity_type="source", entity_id="nhl.partner_odds")
            return 0
        rows = parse_partner_odds(payload)
        self.store.record_raw(url, json.dumps(payload), source_id="nhl.partner_odds")
        ts = utcnow()
        n = 0
        for r in rows:
            self.store.execute(
                """INSERT OR IGNORE INTO odds_snapshots(game_id, partner, market_desc, home_price,
                         away_price, home_qualifier, away_qualifier, home_abbrev, away_abbrev,
                         start_time_utc, source_updated_utc, retrieved_at, source_url)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r["game_id"], r["partner"], r["market_desc"], r["home_price"], r["away_price"],
                 r["home_qualifier"], r["away_qualifier"], r["home_abbrev"], r["away_abbrev"],
                 r["start_time_utc"], r["source_updated_utc"], ts, url))
            n += 1
        self.store.commit()
        self.log(f"NHL partner odds ({(rows or [{}])[0].get('partner', '?')}): {n} rows for "
                 f"{len({r['game_id'] for r in rows})} games")
        return n

    # ------------------------------------------------------------------ EDGE
    def nhl_edge_snapshots(self, season: int, game_type: int = 2, *, max_teams: int = 40) -> int:
        teams = self.store.query("SELECT team_id, abbrev FROM teams WHERE active=1 ORDER BY team_id")
        n = 0
        ts = utcnow()
        for t in teams[:max_teams]:
            url = f"{API_WEB}/edge/team-comparison/{t['team_id']}/{season}/{game_type}"
            try:
                # one attempt: before a season starts this legitimately 404s for every team
                payload = self.http.get(url, use_cache=False, retries=1).json
            except (NetworkUnavailable, ValueError):
                continue
            if not isinstance(payload, dict) or "team" not in payload:
                continue
            ssd = payload.get("shotSpeedDetails") or {}
            sk = payload.get("skatingSpeedDetails") or {}
            last10 = payload.get("skatingDistanceLast10") or []
            dists = [((g.get("distanceSkated") or {}).get("imperial")) for g in last10]
            dists = [d for d in dists if isinstance(d, (int, float))]
            self.store.execute(
                """INSERT OR IGNORE INTO edge_team_snapshots(team_id, season, game_type, retrieved_at,
                         games_played, avg_shot_speed, shot_attempts_90_plus, bursts_over_22,
                         bursts_20_22, max_skating_speed, distance_last10_avg, payload_json, source_url)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (t["team_id"], season, game_type, ts,
                 (payload.get("team") or {}).get("gamesPlayed"),
                 (ssd.get("avgShotSpeed") or {}).get("imperial"),
                 (ssd.get("shotAttemptsOver100") or 0) + (ssd.get("shotAttempts90To100") or 0),
                 sk.get("burstsOver22"), sk.get("bursts20To22"),
                 (sk.get("maxSkatingSpeed") or {}).get("imperial"),
                 (sum(dists) / len(dists)) if dists else None,
                 json.dumps({k: v for k, v in payload.items() if k != "skatingDistanceLast10"})[:20000],
                 url))
            n += 1
            time.sleep(0.05)
        self.store.commit()
        self.log(f"NHL EDGE team snapshots {season}/{game_type}: {n}")
        return n


def _f(v: Any) -> float | None:
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
