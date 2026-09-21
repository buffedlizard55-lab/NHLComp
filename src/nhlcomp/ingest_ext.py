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
from .sources.kalshi import (BASE as KALSHI_BASE, HIST_BASE, SERIES_MARKET_TYPE, WATCH_SERIES,
                             KalshiApiError, contract_suffix, market_team_code, market_type_for,
                             parse_event_ticker)
from .sources.nhl import (API_WEB, STATS_REST, parse_goalie_game_row, parse_partner_odds,
                          parse_team_game_row)
from .sources.polymarket import GAMMA as POLYMARKET_BASE, numeric_disagreements, normalize_event
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
        """Insert/refresh one settled contract; returns the matched game_id (or None).

        The same contract can arrive twice -- from the live tier and later from the
        historical tier, or from a page where Kalshi omitted a field it had sent before -- and
        the two payloads are not equally complete.  Every descriptive column is therefore
        updated with ``COALESCE(excluded.x, existing.x)``: a newer value wins, but a missing
        one never erases a fact already on the record.  ``rung`` matters most here, because
        the ticker suffix is what tells the puck-line settlement code which line a contract
        was struck at; losing it would leave a settled bet unresolvable.  ``retrieved_at`` and
        ``source_url`` are deliberately overwritten, since they describe this row's most recent
        retrieval (each individual fetch is recorded separately in ``raw_response``).
        """
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
                                              title, occurrence_datetime, team_abbrev,
                                              market_type, strike_type, rung)
               VALUES('kalshi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(provider, contract, side) DO UPDATE SET
                 result=excluded.result, settle_price=excluded.settle_price,
                 price_before=excluded.price_before, bid_before=excluded.bid_before,
                 ask_before=excluded.ask_before, volume=excluded.volume,
                 open_interest=excluded.open_interest,
                 retrieved_at=excluded.retrieved_at, source_url=excluded.source_url,
                 game_id=COALESCE(excluded.game_id, market_settlements.game_id),
                 game_date=COALESCE(excluded.game_date, market_settlements.game_date),
                 -- selection falls back to the ticker when a payload carries no label, and
                 -- that fallback must not overwrite a real label recorded earlier
                 selection=CASE WHEN excluded.selection IS NOT NULL
                                 AND excluded.selection <> excluded.contract
                                THEN excluded.selection
                                ELSE market_settlements.selection END,
                 series_ticker=COALESCE(excluded.series_ticker, market_settlements.series_ticker),
                 tier=COALESCE(excluded.tier, market_settlements.tier),
                 floor_strike=COALESCE(excluded.floor_strike, market_settlements.floor_strike),
                 close_time=COALESCE(excluded.close_time, market_settlements.close_time),
                 title=COALESCE(excluded.title, market_settlements.title),
                 occurrence_datetime=COALESCE(excluded.occurrence_datetime,
                                              market_settlements.occurrence_datetime),
                 team_abbrev=COALESCE(excluded.team_abbrev, market_settlements.team_abbrev),
                 market_type=COALESCE(excluded.market_type, market_settlements.market_type),
                 strike_type=COALESCE(excluded.strike_type, market_settlements.strike_type),
                 rung=COALESCE(excluded.rung, market_settlements.rung),
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
             m.get("occurrence_datetime"), team_abbrev, market_type_for(m),
             m.get("strike_type") or None, contract_suffix(m)))
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


    # ------------------------------------------------------- kalshi series watch
    def kalshi_series_watch(self, series: Sequence[str] = WATCH_SERIES, *,
                            limit: int = 200) -> dict[str, Any]:
        """Poll the hockey series this project is ready to trade but that list nothing yet.

        An empty answer is a **verified negative**, not a missing value: the query ran, the
        exchange answered, and the answer was "no open contracts".  That is recorded per
        series with its timestamp (``kalshi_series.open_contracts`` /
        ``last_listed_check`` / ``listing_note``) so a later run can show the day a market
        appeared -- and so no code path ever treats an empty answer as licence to invent a
        price.  On 2026-09-21 all fourteen of these series were empty (preseason); the
        observation is captured in ``data/captured/kalshi_unlisted_series_20260921.json``.

        When a series *does* list contracts the event is flagged, because from that moment
        the market is tradable and its prices must start being captured.
        """
        out: dict[str, Any] = {}
        ts = utcnow()
        for ticker in series:
            url = f"{KALSHI_BASE}/markets?series_ticker={ticker}&limit={limit}&status=open"
            try:
                markets = self.kalshi.markets(ticker, limit=limit, status="open", max_pages=1)
            except (KalshiApiError, NetworkUnavailable) as exc:
                out[ticker] = {"error": f"{type(exc).__name__}: {exc}"}
                self.store.flag("broken_api", f"kalshi open markets {ticker}: {exc}",
                                severity="error", entity_type="source",
                                entity_id=f"kalshi.{ticker}")
                continue
            self.store.record_raw(url, json.dumps(markets), source_id="kalshi.trade_api")
            note = ("listed and open" if markets else
                    "queried and empty: the exchange lists no open contract for this series "
                    f"as of {ts}. Verified negative; nothing is inferred to fill it.")
            out[ticker] = {"open_contracts": len(markets), "checked_at": ts,
                           "market_type": SERIES_MARKET_TYPE.get(ticker), "note": note}
            self.store.execute(
                """INSERT INTO kalshi_series(ticker, first_seen, last_seen, open_contracts,
                                             last_listed_check, listing_note)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(ticker) DO UPDATE SET
                     open_contracts=excluded.open_contracts,
                     last_listed_check=excluded.last_listed_check,
                     listing_note=excluded.listing_note, last_seen=excluded.last_seen""",
                (ticker, ts, ts, len(markets), ts, note))
            if markets:
                self.store.flag(
                    "new_market_listed",
                    f"kalshi series {ticker} now lists {len(markets)} open contract(s) "
                    f"(market_type {SERIES_MARKET_TYPE.get(ticker)}); it was empty when last "
                    "polled, so its prices start being captured from this run",
                    severity="info", entity_type="market", entity_id=ticker)
        self.store.commit()
        listed = [t for t, v in out.items() if v.get("open_contracts")]
        self.log(f"kalshi series watch: {len(out)} series polled, "
                 f"{len(listed)} listing contracts{': ' + ', '.join(sorted(listed)) if listed else ''}")
        return out

    # ------------------------------------------------------------- nhl period goals
    def nhl_period_goals(self, days: Sequence[str]) -> dict[str, Any]:
        """Official per-goal rows from ``/v1/score/{date}`` -> period goals and period scores.

        Why a second NHL feed for the same games: the scoreboard endpoint this project
        already ingests gives the final score and ``lastPeriodType``, while ``/v1/score``
        gives one row per goal with the period it was scored in, the running score and the
        strength.  That is what a period market settles on, and it is an independent reading
        of the final result to cross-check the first feed against.

        Two things are recorded rather than assumed:

        * **reconciliation** -- the derived period goals are summed per team and compared to
          the official final score.  A game decided in a shootout publishes no goal row for
          the deciding shot, so the sums can legitimately differ; when they do, the row says
          so (``reconciled=0`` + note) instead of being quietly patched to match.
        * **disagreement** -- when ``gameOutcome.lastPeriodType`` here differs from the
          ``games.last_period_type`` already stored from the scoreboard feed, both values are
          kept and the conflict is flagged.  Conflicting sources are preserved, never
          silently resolved.
        """
        stats: dict[str, Any] = {"days": list(days), "games": 0, "goals": 0,
                                 "period_score_rows": 0, "unreconciled": 0, "conflicts": 0}
        for day in days:
            url = f"{API_WEB}/score/{day}"
            payload = self.nhl.score(day)
            self.store.record_raw(url, json.dumps(payload), source_id="nhl.api_web")
            games = payload.get("games") if isinstance(payload, dict) else None
            if not isinstance(games, list):
                self.store.flag("unexpected_shape",
                                f"nhl /v1/score/{day} published no 'games' list; nothing was "
                                "written and nothing was inferred",
                                severity="warn", entity_type="dataset", entity_id="nhl.score")
                continue
            for g in games:
                if not isinstance(g, dict) or g.get("id") is None:
                    continue
                gid = int(g["id"])
                stats["games"] += 1
                ts = utcnow()
                home = g.get("homeTeam") or {}
                away = g.get("awayTeam") or {}
                official_home, official_away = home.get("score"), away.get("score")
                lpt_score = ((g.get("gameOutcome") or {}).get("lastPeriodType")
                             if isinstance(g.get("gameOutcome"), dict) else None)
                per_period: dict[int, dict[str, Any]] = {}
                for gl in (g.get("goals") or []):
                    if not isinstance(gl, dict):
                        continue
                    pd = gl.get("periodDescriptor") or {}
                    period = gl.get("period") if gl.get("period") is not None else pd.get("number")
                    if period is None or not gl.get("teamAbbrev"):
                        self.store.flag(
                            "unexpected_shape",
                            f"nhl /v1/score/{day} game {gid} published a goal row without a "
                            f"period or a team ({json.dumps(gl)[:200]}); skipped, not guessed",
                            severity="warn", entity_type="game", entity_id=str(gid))
                        continue
                    self.store.execute(
                        """INSERT INTO game_period_goals(game_id, period, period_type,
                                                        team_abbrev, player_id, player_name,
                                                        time_in_period, strength, goal_modifier,
                                                        away_score, home_score, source_id,
                                                        source_url, retrieved_at, provenance)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'SOURCE')
                           ON CONFLICT(game_id, period, time_in_period, team_abbrev, player_id)
                             DO UPDATE SET period_type=excluded.period_type,
                               player_name=excluded.player_name, strength=excluded.strength,
                               goal_modifier=excluded.goal_modifier,
                               away_score=excluded.away_score, home_score=excluded.home_score,
                               retrieved_at=excluded.retrieved_at,
                               source_url=excluded.source_url""",
                        (gid, int(period), pd.get("periodType"), gl.get("teamAbbrev"),
                         int(gl.get("playerId") or 0),
                         ((gl.get("name") or {}).get("default")
                          if isinstance(gl.get("name"), dict) else gl.get("name")),
                         gl.get("timeInPeriod"), gl.get("strength"), gl.get("goalModifier"),
                         gl.get("awayScore"), gl.get("homeScore"), "nhl.api_web", url, ts))
                    stats["goals"] += 1
                    slot = per_period.setdefault(int(period), {"period_type": pd.get("periodType"),
                                                               "home": 0, "away": 0})
                    if pd.get("periodType") and not slot.get("period_type"):
                        slot["period_type"] = pd.get("periodType")
                    # which side scored is the feed's own teamAbbrev, matched against the
                    # game's own team ids; an abbreviation that matches neither is a conflict
                    # to report, not a side to pick
                    home_abbr, away_abbr = home.get("abbrev"), away.get("abbrev")
                    if gl.get("teamAbbrev") == home_abbr:
                        slot["home"] += 1
                    elif gl.get("teamAbbrev") == away_abbr:
                        slot["away"] += 1
                    else:
                        self.store.flag(
                            "unmapped_team",
                            f"nhl /v1/score/{day} game {gid} credits a goal to "
                            f"'{gl.get('teamAbbrev')}', which is neither the home team "
                            f"({home_abbr}) nor the away team ({away_abbr}) of that game",
                            severity="error", entity_type="game", entity_id=str(gid))
                # derived period scores + reconciliation against the official final score
                for period, slot in sorted(per_period.items()):
                    self.store.execute(
                        """INSERT INTO game_period_scores(game_id, period, period_type,
                                                          home_goals, away_goals, derived_from,
                                                          reconciled, reconciliation_note,
                                                          source_id, retrieved_at, provenance)
                           VALUES(?,?,?,?,?, 'nhl.score.goals', 1, NULL, 'nhl.api_web', ?,
                                  'DERIVED')
                           ON CONFLICT(game_id, period) DO UPDATE SET
                             period_type=excluded.period_type, home_goals=excluded.home_goals,
                             away_goals=excluded.away_goals,
                             reconciliation_note=excluded.reconciliation_note,
                             retrieved_at=excluded.retrieved_at""",
                        (gid, period, slot["period_type"], slot["home"], slot["away"], ts))
                    stats["period_score_rows"] += 1
                if official_home is not None and official_away is not None:
                    dh = sum(v["home"] for v in per_period.values())
                    da = sum(v["away"] for v in per_period.values())
                    if (dh, da) != (int(official_home), int(official_away)):
                        note = (f"derived period goals sum to {dh}-{da} (home-away) against an "
                                f"official final score of {official_home}-{official_away}; "
                                f"lastPeriodType={lpt_score}. ")
                        if str(lpt_score).upper() == "SO":
                            note += ("The deciding shootout attempt is not published as a goal "
                                     "row, so a one-goal difference for the winner is expected. "
                                     "Both readings are kept; neither is edited.")
                        else:
                            note += ("No shootout explains the difference, so the two readings "
                                     "of the same feed disagree and the discrepancy is kept "
                                     "open rather than resolved by preferring one.")
                        self.store.execute(
                            """UPDATE game_period_scores SET reconciled=0, reconciliation_note=?
                                WHERE game_id=?""", (note, gid))
                        self.store.flag("unreconciled_period_goals",
                                        f"game {gid} ({day}): {note}", severity="warn",
                                        entity_type="game", entity_id=str(gid),
                                        sources="nhl.score,nhl.scoreboard")
                        stats["unreconciled"] += 1
                # cross-check the second feed's lastPeriodType against the stored one
                stored = self.store.one(
                    "SELECT last_period_type, home_score, away_score FROM games WHERE game_id=?",
                    (gid,))
                if stored is not None and lpt_score:
                    if (stored["last_period_type"] or "").upper() != str(lpt_score).upper():
                        self.store.flag(
                            "source_disagreement",
                            f"game {gid} ({day}): scoreboard feed says lastPeriodType="
                            f"{stored['last_period_type']!r} while /v1/score says "
                            f"{lpt_score!r}; both kept, neither overwritten",
                            severity="warn", entity_type="game", entity_id=str(gid),
                            sources="nhl.scoreboard,nhl.score")
                        stats["conflicts"] += 1
                    for side, official, stored_score in (
                            ("home", official_home, stored["home_score"]),
                            ("away", official_away, stored["away_score"])):
                        if official is not None and stored_score is not None and \
                                int(official) != int(stored_score):
                            self.store.flag(
                                "source_disagreement",
                                f"game {gid} ({day}): {side} score is {stored_score} in the "
                                f"scoreboard feed and {official} in /v1/score; both kept",
                                severity="warn", entity_type="game", entity_id=str(gid),
                                sources="nhl.scoreboard,nhl.score")
                            stats["conflicts"] += 1
        self.store.commit()
        self.log(f"nhl period goals: {stats['games']} games, {stats['goals']} goal rows, "
                 f"{stats['period_score_rows']} derived period scores, "
                 f"{stats['unreconciled']} unreconciled, {stats['conflicts']} source conflicts")
        return stats

    # ------------------------------------------------------------------ polymarket
    def polymarket_nhl(self, *, limit: int = 100, closed: bool = False) -> dict[str, Any]:
        """Ingest Polymarket's NHL markets as a second, independent prediction-market price.

        Reference data only: this project executes on Kalshi, so nothing here funds a wager,
        and a Polymarket price is never used to settle one.  What it is good for is a
        cross-check -- two prediction markets pricing the same question is evidence about the
        price, and their published fee schedule is data rather than a guess.

        On 2026-09-21 the only open NHL events were season-level (the 2026-27 Stanley Cup
        Champion), so there was no per-game market to compare against a Kalshi game contract.
        That absence is recorded as a verified negative, and the poll keeps running so the
        day a game market is listed the prices are captured from that day forward.
        """
        url = (f"{POLYMARKET_BASE}/events?tag_slug=nhl"
               f"&closed={'true' if closed else 'false'}&limit={int(limit)}&offset=0")
        stats: dict[str, Any] = {"events": 0, "markets": 0, "families": {}, "matched_games": 0,
                                 "disagreements": [], "source_url": url}
        events = self.pm.events(tag_slug="nhl", closed=closed, limit=limit)
        self.store.record_raw(url, json.dumps(events), source_id="polymarket.gamma")
        stats["events"] = len(events)
        raw_markets: list[dict] = []
        ts = utcnow()
        for ev in events:
            for m in (ev.get("markets") or []):
                if isinstance(m, dict):
                    raw_markets.append(m)
            for row in normalize_event(ev, retrieved_at=ts, source_url=url):
                if not row["id"]:
                    continue
                # a game market is only attached to a game when the match is unambiguous
                gid, basis = self._match_polymarket_game(row)
                row["game_id"] = gid
                if basis:
                    row["match_basis"] = f"{row['match_basis']}; {basis}"
                if gid is not None:
                    stats["matched_games"] += 1
                cols = list(row.keys())
                self.store.execute(
                    f"""INSERT INTO polymarket_markets({','.join(cols)})
                        VALUES({','.join('?' * len(cols))})
                        ON CONFLICT(id) DO UPDATE SET
                          best_bid=excluded.best_bid, best_ask=excluded.best_ask,
                          spread=excluded.spread, last_trade_price=excluded.last_trade_price,
                          liquidity=excluded.liquidity, volume=excluded.volume,
                          volume_24hr=excluded.volume_24hr, active=excluded.active,
                          closed=excluded.closed, accepting_orders=excluded.accepting_orders,
                          enable_order_book=excluded.enable_order_book,
                          game_id=COALESCE(excluded.game_id, polymarket_markets.game_id),
                          match_basis=excluded.match_basis, market_family=excluded.market_family,
                          retrieved_at=excluded.retrieved_at""",
                    tuple(row[c] for c in cols))
                stats["markets"] += 1
                stats["families"][row["market_family"]] = \
                    stats["families"].get(row["market_family"], 0) + 1
        stats["disagreements"] = numeric_disagreements(
            [dict(r) for r in self.store.query(
                "SELECT id, liquidity, volume FROM polymarket_markets")], raw_markets)
        for d in stats["disagreements"]:
            self.store.flag("source_disagreement",
                            f"polymarket publishes two forms of the same figure that differ: {d}",
                            severity="warn", entity_type="market", entity_id="polymarket.gamma")
        if not stats.get("families", {}).get("game"):
            self.store.flag(
                "no_game_markets",
                f"polymarket listed {stats['markets']} NHL market(s) across {stats['events']} "
                f"event(s) but none is a per-game market (families: {stats['families']}); there "
                "is therefore nothing to cross-check against a Kalshi game contract this run. "
                "Verified negative, not missing data.",
                severity="info", entity_type="market", entity_id="polymarket.gamma")
        self.store.commit()
        self.log(f"polymarket: {stats['events']} events, {stats['markets']} markets, "
                 f"families {stats['families']}, matched to {stats['matched_games']} game(s)")
        return stats

    def _match_polymarket_game(self, row: dict[str, Any]) -> tuple[int | None, str]:
        """Match a Polymarket market to one NHL game, or say why it was not matched.

        Only a market that carries a game start time is even a candidate, and only an
        unambiguous team+date match attaches a game_id.  Everything else returns None with
        the reason recorded, because a plausible-looking match is how a futures price ends up
        attached to a game it has nothing to do with.
        """
        if row.get("market_family") != "game":
            return None, (f"not matched to a game: market_family={row.get('market_family')} "
                          "(no game start time published)")
        start = row.get("game_start_time") or row.get("start_date") or ""
        day = str(start)[:10]
        text = " ".join(str(row.get(k) or "") for k in
                        ("question", "group_item_title", "event_title", "slug"))
        home_id = away_id = None
        for t in self.store.query("SELECT team_id, abbrev, full_name FROM teams WHERE active=1"):
            for name in {t["abbrev"], t["full_name"]}:
                if name and len(str(name)) > 2 and str(name).lower() in text.lower():
                    if home_id is None:
                        home_id = int(t["team_id"])
                    elif away_id is None and int(t["team_id"]) != home_id:
                        away_id = int(t["team_id"])
        if not day or home_id is None or away_id is None:
            return None, (f"game market but no unambiguous match: date={day or 'unknown'}, "
                          f"teams resolved={home_id is not None and away_id is not None}")
        g = self.store.one(
            """SELECT game_id FROM games WHERE game_date=? AND
                ((home_id=? AND away_id=?) OR (home_id=? AND away_id=?))""",
            (day, home_id, away_id, away_id, home_id))
        if g is None:
            return None, f"no NHL game on {day} between team_ids {away_id} and {home_id}"
        return int(g["game_id"]), f"matched on published game start date {day} and both team names"

def _f(v: Any) -> float | None:
    if v in (None, "", "-"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
