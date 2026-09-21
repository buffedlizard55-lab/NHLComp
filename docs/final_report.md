# NHLComp Final Report — PASS3 FULL RECHECK
Generated 2026-09-21 UTC from data/nhlcomp.db (copy of data/ledger.db)

## Competition Summary
- **78 strategies** (61 seed + 17 generated variants, per build_seed_strategies)
- **BACKTEST**: 3091 settled wagers, PnL +1961.88, staked 101875.18, ROI 1.93%, win rate 48.75%
- **FORWARD TEST**: 81 open paper bets, 0 settled yet, staked 2172.32, PnL 0.0
- **Market baseline** (buy every side at close, net fees): home ROI -6.06% (n=1312, hit 52.21%), away ROI -1.8% (n=1312, hit 47.79%) — demonstrates vig.
- **Open irregularities**: 3 (down from 74 pre-fix)

## Verified Facts (SOURCE DATA)
- NHL schedule/results from api-web.nhle.com: 4410 games, 3968 regular-season feature rows, 0 leakage problems (assert_no_leakage)
- Team per-game stats from api.nhle.com/stats/rest team/summary isGame=true: PP%/PK%, shots for/against, used for shot_share, PDO, special_teams edge — verified via captured payload nhl_stats_rest_team.json
- Goalie per-game stats from same endpoint: starter save % last 10, b2b detection — verified
- Kalshi markets: 16244 settlements (KXNHLGAME 3098 with close candle, KXNHLTOTAL 7904 total, 3562 with close, 446 games, 352 backtestable; KXNHLSPREAD 5182 settlements, 390 with close across 98 games, 4 backtestable; KXNHLOVERTIME 49 settlements, 48 with close across 48 games, 0 backtestable — playoff-only window 2026-04-28..2026-06-14)
- Kalshi quotes: 916 latest quotes, 57038 price points (open/t24h/t6h/t1h/close candles)
- Fee model: Kalshi general taker fee round up(0.07*C*P*(1-P)), M=1 for KXNHLGAME, from official fee schedule effective 2026-07-07 — stored on every bet row
- Cross-validation: team_game_rows_compared 5248, score_conflicts 0, shootout_goal_definition_adjusted 392 (NHL stats REST excludes shootout goal), settlement_moneyline_compared 3098, settlement_conflicts 0, totals_conflicts 0, puck_line_conflicts 0, overtime cross_tab REG->no 36, OT->yes 13
- Polymarket: 707 markets (futures 223, game 484), matched_to_games 160, 100 like-for-like comparisons (moneyline 20, total 80), 0 disagreements >0.05 within 30min, 100 stale — reference only, never execution
- Source registry: 29 sources (was 21), including kalshi.period, period_total, team_total, regulation, player_props, goalie_props, futures, alternate, spread, overtime, etc., all probed via Ingestor.verify_sources()
- Strategy lifecycle: candidate→active→retired, 78 total, 690 triggers tested, 0 promoted via discovery significance (Holm), 0 significant on accuracy/price — honest reporting of search size

## Assumptions & Limitations (explicitly recorded, never hidden)
- **Starter identity for PAST games**: post-game goalie log identifies starter (ASSUMPTION: public at morning skate). For UPCOMING games, no verified pre-game starter source exists → goalie-gated strategies stay in WAITING FOR GOALIE forward, not assumed.
- **No historical injury feed**: ESPN injuries current list only, no history → injury-sensitive rules FORWARD TEST only, BACKTEST rows annotated backtest_without_injury_context via stage_reconcile (audit row exists)
- **No historical sportsbook odds**: DraftKings via NHL partner feed current slate only, de-vigged proportionally, reference only → book_vs_exchange FORWARD TEST only, no backtest claimed
- **Kalshi closing candle hourly**: close = last candle ending at or before scheduled start, may include final minutes before puck drop; thin books outside playoffs
- **Totals Under, puck-line +1.5, overtime NO**: NO-side has no historical offer in candle feed (YES bid/ask only), and no_ask = 1 - yes_bid contradicted by ledger (2 of 12 contracts) → FORWARD TEST only at live no_ask_dollars, no_history_reason recorded
- **Arena lat/lon unavailable**: statsapi.web.nhl.com unreachable, venues.lat/lon NULL by design, travel from UTC offsets, outdoor games not identified without guessing — blocked
- **EDGE**: api-web.nhle.com/v1/edge/team-comparison verified JSON, snapshots only, season-to-date aggregate, no per-game history → blocked until dated history accumulates
- **Period markets**: KXNHL1P listed but 0 contracts open/historical on 2026-09-21 verified negative, game_period_scores ingests period goals for research but no period model → blocked
- **Player props, goalie saves, team totals, regulation, futures**: KXNHLGOAL, KXNHLSAVES, KXNHLTEAMTOTAL etc. listed but empty on 2026-09-21 → blocked with explicit blocked_reason, no price history, no model
- **Live/in-game**: ig60/ig120 candle points stored for research, but batch pipeline cannot execute intraday → research only
- **Slippage**: stored as 0 because only one book level available — limitation recorded
- **Shootout definition for overtime**: Kalshi settled history playoff-only (no shootout), so whether SO counts as overtime is UNVERIFIED — wagers on such game left OPEN and settled from exchange's own result, not inferred

## Adversarial Audit PASS2 Fixes
- **Duplicate bets**: TotalsStrategy _backtest_totals previously looped per-contract, placing 2 bets per game (5.5 and 7.5) → 64 duplicate_bet irregularities. Fixed 2026-09-21: group by game_id, hand whole ladder to strategy once, _pick_strike chooses single rung (nearest target, lower on ties) — matches forward-test ladder logic. Result: duplicate_bet 0, totals_bets 96 (was 260)
- **Duplicate games**: verify.check_games grouped by date+home+away only, flagged preseason split-squad doubleheaders (same date, same matchup, 4h apart, e.g. 2024-09-22 18:00 vs 22:00, 2025-09-21 19:00 vs 23:00) as duplicates → 2 duplicate_game flags. Fixed: group by date+home+away+start_time_utc, so only truly identical scheduled starts flagged. Result: duplicate 0
- **Filtering UI**: site.py enhanced — leaderboard dual boards lb/lbb with controls lb_controls/lbb_controls, search setTextFilter, category/market/status selects, min ROI/PnL; _bets_table emits data-strategy/market/team/season/month/mode/pnl/roi/edge/price and sortable headers; upcoming has up_controls with market/status/team/season/month/min_edge; history has th_controls; strategies page has strat_list container, strat_controls with search, category, market selects, details data-category/market/strategy; JS applyFilters branches on TABLE vs details list, supports q, strategy/market/team/player/goalie/season/month/category/status/mode AND, min_roi/min_pnl/min_edge numeric
- **Source registry expanded**: 29 sources (was 21) including kalshi.period, period_total, team_total, regulation, player_props, goalie_props, futures, alternate, spread, overtime — all verified via probe
- **DB restore**: data/nhlcomp.db was wiped to 0 games after cli init; restored from data/ledger.db (4410 games, 16244 settlements) and re-ran full pipeline offline

## PASS3 Full Recheck
- Tests: PYTHONPATH=src python -m pytest tests -q = 231 passed, 72 subtests
- Verification: open_irregularities 3 (unmapped_injury 13 entries, overtime_price_coverage 48 contracts 0 backtestable, settlement_rule_unverified shootout) — all documented known limitations, not bugs
- No strategy_leakage, no impossible_odds, no missing_timestamp, no bad_pnl, no duplicate_bet, no stake_exceeds_bankroll, no unmapped, no crossed_book, no impossible_price
- Settlement cross-checks: 0 conflicts across moneyline, totals, puck_line, overtime (verified mapping)
- Site: wrote 23 files to docs, 11 HTML pages with controls, methodology updated to 78 strategies / 81 forward bets

## Remaining Open Irregularities (3) — why acceptable
1. unmapped_injury 13 entries: ESPN feed includes players with abbrev SJ, TB, LA, NJ that cannot be mapped to active team_id via Store.team_id_for (triCode shared by two franchises or non-NHL). Counted, not dropped, severity warn, source espn.nhl_api — expected, not a data invention
2. overtime_price_coverage: 48 contracts with pre-game offer, 0 backtestable because feature history (regular season) and price history (playoff) do not yet overlap — gap recorded as gap, not as empty result that looks like “rules found nothing”
3. settlement_rule_unverified KXNHLOVERTIME: shootout treatment unverified because settled history playoff-only, no SO evidence — wagers left OPEN, settled from exchange's own result, nothing inferred

## Deployment
- GitHub Pages site built from ledger: docs/ with dashboard, leaderboard, strategies, upcoming, positions, history, performance, research, sources, verification, methodology, final_report
- CI workflow .github/workflows/research.yml restores ledger from cache, seeds from data/ledger.db on cold start, ingests live data with budgets (kalshi 1500, totals 600, spread 400, overtime 250), runs full pipeline, verifies, builds site, commits snapshot back to branch
- Data files committed: data/ledger.db (74M), data/captured/*.json (real payloads), docs/data/*.json (leaderboard, strategies, bets, etc.)

## Conclusion
PASS1 BUILD done, PASS2 adversarial audit fixed duplicate bets and duplicate game false positives and enhanced filtering, PASS3 full recheck confirms 0 leakage, 0 invented odds, fee model verified, bankroll discipline enforced, all blocked categories stay blocked with explicit reason, site filtering AND logic works, tests pass, open irregularities reduced from 74 to 3 documented limitations.
