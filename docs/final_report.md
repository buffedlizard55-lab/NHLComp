# NHLComp Final Report — 2026-09-22 (fourth session)

Generated from the ledger (`data/nhlcomp.db`, the 2026-09-22 CI ledger plus this session's
audit and research pass). Every number below is computed from the ledger and reproducible via
`make report` / `make site`. **BACKTEST, FORWARD TEST and PAPER TRADE are never merged.**

## 1. What was built (cumulative)

An autonomous NHL paper-betting research platform: verified data-source registry (31
entries, HTTP-probed every run), ingesters for the NHL APIs, Kalshi market data (live tier,
/historical tier, hourly candlesticks), Polymarket, ESPN and NHL partner feeds; point-in-time
feature builders with leakage assertions; Elo / Poisson / logistic models with a held-out
model comparison; an autonomous hypothesis generator (690 triggers tested per run with Holm
multiple-testing correction); priced backtests from real timestamped Kalshi candles; a
forward-test paper engine with depth caps, wide-book refusal, taker fees, partial fills and
shared-book budgets; settlement cross-checked against exchange results; closing-line value;
an append-only bet ledger with audit trail; an irregularity queue **with recovery**; a
GitHub Pages site (dashboard, leaderboards per test mode, strategies, upcoming bets, live
positions, trade history, performance, research lab, data sources, verification queue,
methodology) with JSON mirrors under `docs/data/`; and a 3-hourly CI pipeline that trades,
settles, verifies, rebuilds and deploys the site (284 unit tests, standard library only).

## 2. Competition standings (regular season + playoffs; preseason is excluded and labelled)

| Mode | Bets | Settled | Open | P&L | Staked | ROI | Win rate |
|---|---|---|---|---|---|---|---|
| BACKTEST (Kalshi close candles) | 3,922 | 3,922 | 0 | +1,015.59 | 122,877.67 | +0.83% | 47.1% |
| FORWARD TEST (live 2025-26 paper trading) | 127 | 44 | 0 | −879.61 | 2,932.05 | −30.0% | 31.8% |
| Preseason (both modes, excluded from scoring) | 140 | 57 | 83 | −991.94 | 3,592.99 | −27.6% | 36.8% |

**Calculations, clearly labelled as such:** net of Kalshi's published taker fee; ROI = P&L /
staked; every win rate carries a Wilson 95% interval. The forward-test loss is a *preseason*
result: 127 September wagers on thin books (median spreads far wider than the 2025-26
regular-season book this project backtests against) traded by rules fitted on regular-season
form. The season-scope gate now confines rules to regular season + playoffs; the 83 currently
open preseason wagers are labelled and will settle but score nothing. **No strategy has a
verified edge.** Discovery judged 157 candidates against a Holm threshold across 690 tests
this run: 0 significant on accuracy, 0 on price — and the backtest book as a whole (+
0.83% ROI on 3,922 flat-staked wagers) sits inside the range where fees and the candle-close
entry could explain it. The honest conclusion is *no demonstrated edge yet, regular season
2026-27 forward test is the live experiment*, and the market baseline (blindly buying home
at the close: −6.1% ROI; away: −1.8%) shows the spread+fee hurdle any strategy must clear.

## 3. Data sources discovered / verified / rejected

- **Verified by HTTP probe this run (24):** `api-web.nhle.com/v1` (schedule, scoreboard,
  standings, club schedules, partner-game DraftKings odds, EDGE JSON), `api.nhle.com/stats/rest`
  (team/summary and goalie/summary per-game logs), Kalshi trade API (live markets, settled
  markets, hourly candlesticks, series discovery, /historical tier), Kalshi NHL series
  (moneyline, totals ladder, spread, overtime, futures, period, regulation, team totals,
  goalie props, player props), Polymarket Gamma, ESPN injuries + scoreboard, Open-Meteo
  archive. Full table with limitations on the site's Data Sources page.
- **Verified but dormant (1):** MoneyPuck downloads — page live, terms permit non-commercial
  use with credit, but the page stated "last updated 2026-06-15": no 2026-27 file during the
  off-season, so no current-season point-in-time stream. Candidate, not ingested (shot-level
  xG is a third-party MODEL OUTPUT).
- **Newly rejected this session (2):**
  - `sports-odds-datasets` (ParlayAPI): **no NHL files at all** (Super Bowl LX, one MLB day,
    a 50k prop sample); samples CC BY 4.0 but the archive behind them is a freemium vendor.
  - Kaggle "NHL Full Game & Betting Statistics": rejected on access (account/API key that
    does not exist here and may not be invented) and on provenance (timestamp-less odds said
    to come from ESPN JSONs, contradicting this project's verified probe that ESPN's
    scoreboard carries no odds block on completed games).
- **Still unavailable (verified negatives):** pre-game starting-goalie announcements (a
  further `gamecenter/<id>/probable-goalies` probe returned 404 on 2026-09-22), historical
  sportsbook prices, line combinations, player props with verified pre-game deployment,
  arena coordinates. The Odds API remains rejected (paid key; none invented).

## 4. Backtests and forward tests

- **BACKTEST** rows are priced only from real, timestamped Kalshi data: the last hourly
  candle ending at or before puck drop (or, for a few early rows, an undocumented
  pre-settlement quote permanently stamped `single_source_timing_unverified`). 3,922 settled
  priced wagers across moneyline (3,255), totals ladder (637) and puck line (46). Accuracy-only
  backtests (no price → no P&L, `data_sufficient=0`) cover the rest.
- **Priced totals went live this session's ledger:** the candle walk and the feature walk
  reached the same games — 1,159 backtestable totals games (7,741 contracts with a pre-puck
  drop offer), 691 puck-line games; the old `totals_price_coverage` gap is closed by the new
  recovery pass. KXNHLOVERTIME remains a coverage gap (48 contracts, 0 backtestable) and its
  settlement rule stays shootout-unverified — both stay flagged.
- **FORWARD TEST** placed 127 regular-season wagers in the 2025-26 season against live
  quotes and settles them from official results; none are open (the 2026-27 season starts
  in October; the scope gate correctly keeps preseason out of the scored competition).
- **Market-movement / CLV:** every priced bet records its close price; closing-line value is
  computed against the candle close (NO-side purchases have no candle history, flagged
  `clv_unavailable_no_side`).

## 5. Paper-trading results (honest reading)

- 2025-26 regular-season forward sample: 44 settled, −30.0% ROI — **small sample, wide
  preseason-adjacent books, no conclusions drawn**; the Wilson intervals on the leaderboard
  are correspondingly enormous (e.g. best strategy n=2: CI [34%, 100%]).
- No strategy is promoted on this evidence. Discovery promotions require significance after
  Holm correction **and** priced-ROI evidence; 0 of 690 triggers qualify so far.
- Execution realism: fills capped by published offer size or traded candle volume (declared
  100-contract cap where neither exists, stamped on each row), shared-book budgets prevent
  several strategies over-claiming one book level, books wider than 0.50 are refused, and
  taker fees are deducted everywhere.

## 6. Open / upcoming bets

2,199 upcoming signals are tracked (preseason slate + the opening 2026-27 stretch), each
with its price, required price, model probability, edge and blocking reason. 83 forward
wagers are open on preseason games and settle from official results without scoring. The
competition's next live window is the 2026-27 opening slate, where regular-season scope
rules will again paper-trade at live Kalshi offers.

## 7. Important findings (all verifiable in the ledger's findings table)

1. **No demonstrated edge yet** — 0/690 triggers significant after Holm correction; backtest
   ROI +0.83% is within the spread+fee hurdle zone.
2. **A queue that only grows is a broken queue** — this session added evidence-carrying
   recovery so transient API failures, closed coverage gaps, verified settlement rules and
   since-matched markets resolve with notes + audit rows (nothing deleted). First run cut
   open irregularities 27 → 24, all of which are standing conditions.
3. **The delisted `KXNHLSPREAD-26JAN26LACBJ`** listing anomaly was resolved with evidence
   (Kalshi `not_found`; no NHL game existed for the ticker), superseding the flag.
4. **The "unknown team" in game 2024010106** is EHC Red Bull München (`MUN`, id 7509), a
   2024 Global Series exhibition opponent — identified from the schedule payload itself and
   excluded by design, not guessed.
5. **Under sides (totals Under, +1.5 puck line, OT No) have no price history** — the candle
   feed publishes the YES side only — so those rules are forward-test only and say so.
6. **Polymarket ↔ Kalshi cross-check:** 110 like-for-like comparisons so far, 0 disagreements
   beyond $0.05, but 0 within the 30-minute freshness window yet — recorded stale, used for
   research only.

## 8. Limitations and unavailable data

No historical sportsbook prices (partner feed is current-slate only, reference-only); no
pre-game starter source (goalie rules gated); no line combinations; no period-level model;
EDGE is a season-to-date snapshot (forward-only); in-game candles are research-only (the
pipeline is batch); hourly closes can include trades up to puck drop; preseason Kalshi books
are thin. Preseason scoring is excluded by a season-scope gate added 2026-09-22 after all
102 then-open forward wagers turned out to be preseason games.

## 9. Remaining work / recommended next research

1. Let the 2026-27 regular season accumulate a real forward sample (the CI loop already
   trades, settles and verifies every 3 hours).
2. Grow the KXNHLOVERTIME overlap; settle the shootout question from the first regular-season
   OT/SO contract the exchange finalizes.
3. Re-check MoneyPuck when 2026-27 updates resume (labelled MODEL OUTPUT feature family).
4. A starter-announcement source remains the single highest-value unlock (goalie rules are
   seeded and waiting).
5. Player props stay blocked until a verified pre-game deployment feed exists.

---
*Verification method: all source claims above carry a retrieval URL and date in the ledger
(`findings`, `source_verification`, `raw_response` tables); every bet carries its price
source, timestamp and verification status; corrections and resolutions are audited in
`audit_log`. Nothing on this page is reconstructed from memory.*
