# NHLComp

An autonomous NHL betting-strategy **research, discovery, backtesting, forward-testing and
paper-trading** platform. It finds its own data sources, generates its own strategy
hypotheses, tests them against real timestamped exchange prices, and paper-trades the
survivors — logging every simulated wager with its price, its source and its verification
status.

> **Paper trading only.** No real money is wagered and this repository contains no
> order-placement code. There is no function anywhere in `src/` that can submit an order to
> an exchange or a sportsbook.

---

## Read this before reading any result

* **BACKTEST and FORWARD TEST are never merged.** `bets.test_mode` is one or the other; the
  dashboard, leaderboard, report and JSON mirrors show them side by side, never added.
* **BACKTEST prices are real, timestamped Kalshi candles.** Kalshi publishes hourly
  candlesticks for every contract — the live tier for recent contracts and the
  `/historical` tier for everything settled before the cutoff (`/historical/cutoff`, currently
  2026-07-22). The entry price is the `yes_ask` of the last hourly candle ending at or before
  the NHL scheduled start; such rows carry `verification_status=kalshi_candle_close`,
  `price_point=close` and the candle's own timestamp as `decision_ts`. A few rows written
  before the candle history was recovered were priced from Kalshi's undocumented
  `previous_yes_ask` and keep `single_source_timing_unverified` for good — labels are never
  upgraded after the fact.
* **FORWARD TEST rows are opened only against a live quote** before puck drop and settle from
  the official NHL result. Only forward P&L moves a strategy's virtual bankroll; backtest P&L is
  reported separately and never funds a forward stake.
* **A quoted contract is always matched to the rule that wants it.** Kalshi words a contract as
  a team name ("Anaheim wins"); a rule asks for a side ("home"). `Pipeline._live_quotes` maps one
  to the other through the games table and keeps the exchange's wording in `Quote.label`. Before
  that mapping existed the two were compared directly, never matched, and the 2026-09-20 ledger
  recorded 183 opportunities as "no quote published for this market yet" while holding **zero**
  forward wagers. Re-running that ledger after the fix placed 40 forward wagers; the regression
  test reproduces the production wording (`test_a_live_quote_worded_the_way_kalshi_words_it_is_traded`).
* **Totals are traded, and the two sides are not symmetric.** An **Over** is the YES side of an
  "Over k.5" contract, whose offer the candlestick history publishes, so it backtests at real
  prices. An **Under** is the NO side; the candle feed publishes no NO offer and
  `no_ask = 1 − yes_bid` does **not** hold on this ledger's quotes (2 of 12 contracts on
  2026-09-20), so an Under rule is forward-test only at the live `no_ask_dollars` and its
  backtest row reports accuracy only with `data_sufficient=0`. The line is read from Kalshi's
  `floor_strike`/`strike_type`, never parsed out of a title that changes shape between tiers.
* **Fees are charged.** Kalshi's general taker fee, `round up(0.07 × C × P × (1 − P))` (fee
  schedule effective 2026-07-07; KXNHLGAME is not on the non-standard list, so the multiplier
  is 1), is applied to every simulated fill and deducted from the settled P&L. The fee is
  stored on each bet row. Priced backtests also deduct it.
* **No sportsbook history exists here.** The DraftKings line published by the NHL partner feed
  is collected from 2026-09-20 onward and attached to forward signals as a *reference* (stored
  as published, de-vigged for comparison). Nothing is backtested against sportsbook odds and
  nothing claims to be.
* **Where no price exists the backtest reports predictive accuracy only** (hit rate, lift over
  base rate, log loss, Brier) with `data_sufficient=0`. It never invents a profit figure.
* **Every number is labelled**: SOURCE DATA (NHL, Kalshi, partner feed, EDGE), DERIVED
  (features, price points, P&L), MODEL OUTPUT (Poisson / Elo / logistic probabilities),
  ASSUMPTION (recorded where used — e.g. the post-game goalie log identifies the starter for a
  past game), UNVERIFIED (anything in the verification queue or behind a `blocked_reason`).

---

## Architecture

```
src/nhlcomp/
  http.py          HTTP client, content-addressed cache, provenance capture
  store.py         SQLite schema (v5) + append-only ledger + audit trail
  sources/
    registry.py    the permanent NHL DATA SOURCE REGISTRY (17 entries, probed every run)
    nhl.py         api-web.nhle.com, api.nhle.com/stats/rest (incl. isGame per-game reports),
                   EDGE JSON API, partner-game odds feed
    kalshi.py      Kalshi public market data: live tier, /historical tier, candlesticks,
                   series discovery, ticker parsing
  market.py        candle -> named price points (open/T-24h/T-6h/T-1h/close/in-game),
                   implied probability, vig, de-vig, Kalshi fee model
  ingest.py        acquisition; stores raw payloads with SHA-256 digests
  ingest_ext.py    Kalshi history walk + candles, live price points, stats REST per-game
                   team/goalie logs, partner odds, EDGE snapshots
  features.py      point-in-time schedule features (rest, travel, density, streaks, pace)
  features_ext.py  point-in-time special-teams / shot-share / PDO / goalie / market features
  models.py        Elo, Poisson, logistic, home-ice constant, calibration metrics
  discovery.py     autonomous hypothesis generation; chronological search; priced scoring
  strategies.py    strategy rules (39 seeds incl. versioned re-definitions and the totals
                   family), staking, gates
  backtest.py      accuracy backtests + priced backtests (flat stakes, fees, splits, baseline)
  paper.py         execution model (offer, depth cap, fee), settlement, open positions
  mastersite.py    the reviewed owner MasterSite directory: per-project verdicts and the one
                   reuse candidate that was tested and rejected
  analysis.py      P&L, ROI, drawdown, CLV, Wilson intervals, per-mode leaderboards
  verify.py        irregularity queue and cross-source validation
  site.py          static GitHub Pages generator
  pipeline.py      end-to-end orchestration (ingest -> features -> models -> discovery ->
                   backtest -> forward -> settle -> CLV -> analysis -> verify)
  cli.py           command-line interface
```

**Standard library only.** No third-party runtime dependency.

## Quick start

```bash
make test            # unit tests, no network needed (fixtures replay real payload shapes)
make probe           # HTTP-probe every registered source and record the verdict
make ingest          # pull live data (needs outbound network)
make run             # models -> discovery -> backtest -> forward -> settle -> verify
make site            # regenerate docs/ from the ledger
make report          # plain-text competition status, per test mode
```

Useful ingest flags: `--kalshi-budget N` (historical/candle calls per run; the walk resumes
next run), `--live-points-budget N` (live-tier candle calls for open contracts),
`--stats-seasons` (per-game stats REST seasons).

## What runs, and how often

`.github/workflows/research.yml`:

* every 3 hours — ingest (schedule, results, Kalshi quotes + settlements + candles, stats REST
  per-game logs, DraftKings reference odds, EDGE snapshots, ESPN injuries), then features,
  models, discovery, priced + accuracy backtests, forward decisions, settlement, CLV,
  verification, and a Pages deploy (on `main`);
* the working ledger is carried between runs in the Actions cache and seeded from the
  committed `data/ledger.db` on a cold start;
* once a day (and on code pushes) the ledger, logs and `docs/` are committed back as a
  snapshot.

## The competition

Every strategy gets a username, a strategy ID, a version, a virtual bankroll, an explicit
hypothesis / data / entry / price / settlement rule, and its own ledger. Strategies are never
overwritten — a change creates `v2`, the old version is set to `retired` with the reason, and
its bets stay in the ledger. Blocked categories (no verified input yet) are registered with an
explicit `blocked_reason` and stay in `WAITING FOR OTHER INFORMATION` rather than betting on
an assumption.

Strategy names are generated by the system (`NHL_GEN_DIFF_STARTER_SV_PCT_L10`,
`NHL_STEAM_FADE`, `NHL_B2B_FADE`, …), not supplied by a human. The discovery engine scans
schedule, form, special-teams, shot-quality, goalie and market-movement triggers on
chronological train → validation → test windows, scores them on hit-rate lift **and** on
flat-stake ROI at the Kalshi close (net of fees), records rejections as findings, and reports
how many hypotheses it tested so a lucky survivor is visible as such.

## Guarantees the code enforces

| Claim | Where it is enforced | Test |
|---|---|---|
| No look-ahead bias | `FeatureBuilder._prior` and `ExtendedFeatures` cut off strictly before the game; `Backtester.assert_no_leakage` | `test_no_future_information_leaks`, `test_02_stats_rest_rows_become_point_in_time_features` |
| Backtest prices are timestamped | BACKTEST rows use the candle ending at or before the scheduled start; `decision_ts` = candle end | `test_05_forward_stage_writes_candle_priced_backtest_bets` |
| Elo prior is not leakage | seeded from the **previous** season's points % | `stage_models` |
| Bets are append-only | `Store.record_bet` refuses a duplicate `bet_id` | `test_record_bet_is_append_only` |
| Corrections are audited | `settle_bet` / `amend_bet` write before/after rows | `test_resettlement_is_recorded_not_silent`, `test_06_forward_bet_gets_clv_from_the_close_candle` |
| No profit without real prices | accuracy backtests store `pnl=NULL`, `data_sufficient=0` | `test_backtest_without_price_data_reports_no_pnl` |
| Fees are real | `kalshi_taker_fee` reproduces the published fee table | `test_fee_matches_the_published_table` |
| Modes never merge | separate leaderboards / totals per `test_mode`; backtest P&L never funds a bankroll | `test_09_site_builds_every_section`, `test_05_*` |
| No unlimited liquidity | `simulate_fill` caps at `yes_ask_size_fp` | `test_partial_fill_when_stake_exceeds_liquidity` |
| A quoted contract is tradeable | `_live_quotes` maps the exchange's wording to home/away/over/under | `test_a_live_quote_worded_the_way_kalshi_words_it_is_traded` |
| No invented Under price | the Under side is forward-only; `no_history_reason` blocks its priced backtest | `test_an_under_rule_declares_that_it_has_no_history_and_is_not_priced` |
| Totals settle from the official score | `PaperEngine._settle_total`; exchange result cross-checked | `test_a_forward_totals_bet_settles_from_the_official_final_score`, `test_a_disagreement_is_recorded_with_both_values_and_left_open` |
| No line is guessed | a totals contract without `floor_strike`/`strike_type` is skipped | `test_a_contract_with_no_readable_line_is_never_traded` |
| Strategies never chase | bet only when `ask <= model_prob - min_edge` | `test_05b_strategy_refuses_a_price_that_is_not_value` |
| Conflicts are recorded, not resolved | `Verifier.cross_validate_*` writes both values | `test_conflicting_sources_are_recorded_not_resolved` |
| Small samples say so | Wilson interval beside every win rate | `test_wilson_interval_is_wide_for_small_n` |

## Data sources

See **Data Sources** on the site, or `source_registry` in the ledger. A source is only marked
`verified` after an automated probe is written to `source_verification`. "Free tier" is never
recorded as free.

### Verified in this build (2026-09-20)

| Source | What it gives | Label |
|---|---|---|
| `api-web.nhle.com/v1` schedule, scoreboard, standings, club schedules | games, results, `lastPeriodType`, venue UTC offsets | SOURCE |
| `api.nhle.com/stats/rest/en/team/summary?isGame=true` | one row per team-game: PP%, PK%, shots for/against, FO% (2,624 rows for 2025-26) | SOURCE |
| `api.nhle.com/stats/rest/en/goalie/summary?isGame=true` | one row per goalie-game: starts, saves, SA, SV% (2,768 rows for 2025-26) | SOURCE |
| Kalshi `/markets` (live tier) | quotes with bid/ask/size/volume; settled contracts after the cutoff | SOURCE |
| Kalshi `/historical/markets` + `/historical/markets/{t}/candlesticks` | the full KXNHLGAME record before the cutoff and hourly candles | SOURCE |
| Kalshi `/series/{s}/markets/{t}/candlesticks` | hourly candles for live-tier contracts | SOURCE |
| Kalshi `/series?category=Sports&tags=Hockey` | the NHL series family (KXNHLGAME, KXNHL1P, KXNHLOVERTIME, KXNHLTOTAL, KXNHLSPREAD, …) | SOURCE |
| `api-web.nhle.com/v1/partner-game/US/now` | DraftKings moneyline / puck line / total for the current slate | SOURCE (reference only, no history) |
| `api-web.nhle.com/v1/edge/*` | team tracking aggregates (skating distance, speed bursts, shot speed) | SOURCE (season-to-date snapshots, forward only) |
| `site.api.espn.com/.../nhl/injuries` | injury list | SOURCE (cross-check only) |
| `site.api.espn.com/.../nhl/scoreboard?dates=` | any past date: results, period linescores, three stars, winning/losing goalie | SOURCE (cross-check only; **no odds block on completed games**) |

### Corrected verdicts

* **Kalshi candles are available.** The 2026-09-20 probes that returned 404 used the path
  segment `candles`; the endpoint is `candlesticks`. The registry entry records the
  correction instead of keeping a wrong rejection.
* **NHL EDGE is reachable as JSON** (`api-web.nhle.com/v1/edge/...`); the earlier
  "unreachable" verdict probed the HTML page. EDGE is season-to-date with no per-game history,
  so it is snapshotted forward and not assumed predictive.
* **MoneyPuck's listed downloads are permitted** (free for non-commercial use with credit, per
  its data page); the registry keeps it as a candidate rather than rejected. Natural Stat Trick
  / Evolving-Hockey remain rejected for automated use (scraping conflicts with their terms).

Still unavailable: `statsapi.web.nhl.com` and `api.nhl.com/api/v1` (arena coordinates), The
Odds API (paid key; none invented), any pre-game starting-goalie feed, line combinations,
player props.

### MasterSite review (requirement 13)

The owner's directory at <https://buffedlizard55-lab.github.io/MasterSite/> (49 verified sites)
was reviewed on 2026-09-21 and is rendered on the site's **Research Lab** page
(`src/nhlcomp/mastersite.py`): the index was fetched, every named project was checked against the
GitHub API, candidate READMEs were read, and the one data claim that mattered was tested against
the live endpoint before anything was reused.

* **CEO and PinePilot do not exist** among the account's public repositories (the MasterSite
  itself flags PinePilot as a 404). Nothing was inferred about them.
* **SportsPred's claim that NHL prices come from "the ESPN odds block" was tested and rejected
  for historical use**: `scoreboard?dates=20251115&limit=1` returns the finished game with
  results, linescores, stars and goalies and a Draft Kings provider attribution — and no odds
  block. The feed was still registered (`espn.nhl_scoreboard`) for what it verifiably supplies:
  a second independent result and a post-game goalie identification to cross-check the NHL stats
  REST log.
* **KalshiPaperSim's methodology was reused, its code was not**: the rule that a design which
  never traded is left unranked rather than shown at 0%, and the rule that every candidate source
  ends as traded / blocked-with-a-named-unblocker / verified-negative.

### Kalshi status filters

`status=active` and `status=finalized` are **not** valid on the live tier — the API returns
`{"error":{"code":"bad_request","details":"invalid status filter"}}`. The live filter is
`open`; settled contracts are `settled`. `markets()` raises `KalshiApiError` on an
application-level error so a rejected request cannot be mistaken for "no markets".

## Known limitations

* **No arena coordinates.** Travel is derived from published venue `venueUTCOffset` values
  rather than invented lat/lon.
* **The closing candle is hourly.** "Close" is the last candle ending at or before the
  scheduled start, so it can include trades from the final minutes before puck drop. The
  candle's `yes_ask` is a price, not a guarantee of depth; Kalshi NHL books are thin outside
  the playoffs.
* **No historical sportsbook prices.** The ESPN scoreboard carries no odds block on completed
  games (verified 2026-09-21 on the 2025-11-15 slate), and the NHL partner feed covers the
  current slate only. Sportsbook lines are therefore a forward-only reference, never a backtest
  price.
* **Totals price history is still filling in.** KXNHLTOTAL settled contracts exist back to the
  2026 Stanley Cup Final (verified 2026-09-21) and the walk is budgeted separately
  (`--totals-budget`), so the totals backtest covers only the games whose candles have been
  recovered so far; the totals accuracy row says which strikes it used.
* **Starting goalies for upcoming games have no verified pre-game source.** Goalie-gated
  strategies forward-test only when a starter is known; their backtests rely on the post-game
  log (an explicit ASSUMPTION recorded on the strategy).
* **No period-level model yet.** KXNHL1P / KXNHLOVERTIME exist on Kalshi and are registered,
  but their prices are not ingested and nothing is bet on them.
* **In-game candles are stored for research only.** The pipeline runs on a batch schedule and
  cannot execute during a game.
* **Preseason Kalshi liquidity is thin.** Contracts with zero size on both sides are recorded as
  `insufficient_liquidity` and not traded.

## Site

`docs/` is generated by `nhlcomp.site` from the ledger and deployed by the
`research-and-site` workflow. Sections: Dashboard, Leaderboard (forward test and backtest
boards, sortable, filterable, drill-down to each strategy's wagers), Strategies, Upcoming
Bets, Live Paper Trading, Trade History, Performance, Research Lab, Data Sources,
Verification, Methodology. Machine-readable mirrors live in `docs/data/*.json`.
