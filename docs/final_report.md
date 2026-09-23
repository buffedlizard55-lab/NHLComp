# NHLComp Final Report — 2026-09-22 (fifth session)

Generated from the ledger. Every number below is computed from the SQLite ledger and
reproducible via `make report` / `make site`; CI (`research-and-site`) re-runs the whole
pipeline against the live public APIs every 3 hours and redeploys this site. **BACKTEST,
FORWARD TEST and PAPER TRADE are never merged.**

Sessions 1–4 built the platform (see the Methodology page for the cumulative design).
This session was an autonomous research pass in the spirit of the operating rule —
**research → discover → verify → model → test → paper trade → measure → analyze →
improve → repeat** — plus adversarial audit passes over both the new work and the
existing ledger.

---

## 1. What was built this session

* **A verified pre-game starting-goalie source.** The single largest blocked input in the
  project — a starter name before puck drop — was unblocked. ESPN's public scoreboard
  publishes a `probableStartingGoalie` per competitor of a *scheduled* game, with the
  feed's own status object. Verified live on the 2026-09-23 slate (Linus Ullmark / OTT,
  Anthony Stolarz / TOR, both status `expected`), registered as `espn.nhl_probables`,
  captured verbatim in `data/captured/espn_scoreboard_probable_goalies_20260923.json`,
  and wired end to end: parser → append-only `goalie_starts` book with point-in-time
  reads → `probable_starter_*` features → three new versioned strategies → post-game
  conversion measurement.
* **Club rosters.** The `players` table existed since schema v1 and was **empty in every
  committed ledger** — `NhlApi.roster()` was implemented and never called. The feed was
  re-verified live (`/v1/roster/{club}/{season}`, keyless; the `/current` alias works),
  captured, and the ingest now stores one snapshot row per player. It gives the injury
  feed and the probable-starter feed an identity join, with every name→id match recorded
  together with its basis (exact or normalised — DERIVED is labelled DERIVED).
* **Split-squad scope flags.** Discovered that `/v1/schedule/{date}` (unlike the
  scoreboard feed) publishes `homeSplitSquad` / `awaySplitSquad`. The 2026-09-23 OTT/TOR
  preseason double-header is split-squad; MIN@DAL and LAK@ANA are not. Stored on the game
  row: a first-party scope fact, not a heuristic.
* **Full discovery-scan auditability.** Previously only promoted candidates and loud
  rejections left ledger rows; inconclusive triggers vanished. Now every scanned trigger
  lands in `experiments` (`EXP_SCAN_<sha1>`), idempotently, with train/validation numbers,
  p-value, Holm threshold and verdict. On the working ledger the scan wrote **157 outcome
  rows** (5 edge candidates, 50 rejected after the price test, 65 no-edge, 37
  inconclusive) — the search footprint is now reconstructible from the database.
* **Out-of-scope signals are recorded, not skipped.** Rules scoped to regular-season +
  playoff form no longer silently skip preseason quotes: an `OUT OF SCOPE` row lands in
  the same upcoming table with the reason and the offer that was *not* taken (702 such
  rows on the last working run).
* **Post-settlement analysis on the site** (requirement 12): the Trade History page now
  renders the brief per-wager analysis (model vs outcome, price obtained, CLV, sample
  size with its CI, and whether the evidence supports changing the rule) for the 20 most
  recent settled wagers.
* **Schema v8** (additive only): goalie_starts status/snapshot/join columns, players
  roster columns, injuries→player_id resolution, games split-squad flags. No historical
  row rewritten; migration tested against the committed ledger.

## 2. Data sources — discovered / verified / rejected

* **VERIFIED**: `espn.nhl_probables` (probable starters, pre-game, status `expected`;
  keyless). Probe evidence + capture retained; the registry entry records the four
  limitations that matter (no `confirmed` observed yet, no announcement timestamp →
  forward-only, ESPN ids ≠ NHL ids so the join is date+teams+venue with unmatched events
  stored and flagged, names resolved through rosters as DERIVED).
* **VERIFIED**: `nhl.roster` (club rosters per season) — previously registered as an API
  surface but never ingested; now actually ingested every run.
* **VERIFIED (new field)**: `api-web.nhle.com/v1/schedule/{date}` split-squad flags.
* **UNVERIFIED, deliberately**: whether ESPN ever publishes `confirmed` for NHL games.
  Nothing upgrades `expected`; CI re-probes every run and the strict goalie gate says in
  its blocking reason exactly what the feed named and why the rule still waits.
* Previously rejected sources stand: The Odds API (no key may be invented), ParlayAPI
  sports-odds-datasets (no NHL files), Kaggle ESPN-derived odds (access + provenance),
  NHL gamecenter probable-goalies endpoint (404 — the block lifted via ESPN instead).

## 3. Strategies created

All version changes are new rows; nothing was edited in place.

* **NHL_GOALIE_EDGE v3** — probable-starter SV% matchup (`diff_probable_starter_sv_pct_l10`),
  gated on the verified feed, forward test only. v2 (post-game-log version) retired.
* **NHL_GOALIE_FATIGUE v2** — fade the probable starter on consecutive nights
  (`away_probable_starter_b2b`), forward test only. v1 retired.
* **NHL_GOALIE_NEWS v3** — the announcement-timing rule stays **blocked**, but the reason
  narrowed with evidence: the NAME is now published, the TIME is not, so a pre-announcement
  price cannot be identified. v2 retired.
* Competition count: 83 strategies at latest version (3 new versions registered, 3 older
  versions retired with reasons in `strategy_lifecycle`).

## 4. Backtests / forward tests

* **No new priced backtest was fabricated.** Probable-starter strategies are excluded from
  priced backtests by construction — the feed has no history and no announcement time, so
  any historical P&L would be invented. `stage_backtest` writes an explicit note instead
  of an empty row, and the accuracy backtest has no signal for them by construction
  (decided rows carry no probable features at all).
* Existing priced backtests are unchanged this session (3922 BACKTEST rows on the working
  ledger snapshot, +1,015.59 units on 122,877.67 staked, ROI +0.8%, net of fees — still
  reported beside, never merged with, the forward test). No edge claim is made: the
  discovery engine's Holm-corrected significance test still reports **0 significant
  survivors** across 690 triggers, and the honest conclusion is that no strategy has
  demonstrated an edge yet.
* Forward test: 0 open positions on the working snapshot (the last live quotes expired
  with their games; the next CI run re-quotes the current slate).

## 5. Paper-trading results & standings

Headline (FORWARD TEST, competition phases only): no settled forward wagers on the
snapshot ledger — the preseason phase produced 127 forward rows that are **excluded** from
the competition and labelled as such everywhere. The competition is waiting for the
regular season (first puck drop 2026-10-06); from then, every rule trades only in-scope
games and every out-of-scope want is recorded rather than traded.

## 6. Open / upcoming bets

0 open positions; upcoming signals regenerate on the next CI ingest against live quotes.
The Upcoming Bets page shows every signal with its status, including the new `OUT OF
SCOPE` rows and goalie gates that now name the probable they refuse to treat as confirmed.

## 7. Important findings (all in the findings ledger, with evidence)

1. `FIND_ESPN_PROBABLE_STARTERS` — the unblock (high confidence).
2. `FIND_SPLIT_SQUAD_FLAGS` — first-party scope field discovered (high).
3. `FIND_PLAYERS_TABLE_WAS_EMPTY` — a requirement-10 gap that existed since v1, closed
   and documented (high).
4. `FIND_DISCOVERY_FULL_SCAN_RECORDED` — the search is now fully auditable (high).
5. `FIND_NO_NHL_PROBABLE_GOALIE_ENDPOINT` — updated with the resolution path; the NHL-side
   negative stands.
6. `FIND_STARTER_CONVERSION` (written by the pipeline once probable rows meet decided
   games) — measures the feed against the NHL goalie log instead of asserting accuracy.

## 8. Limitations & unavailable data (unchanged or newly precise)

* No sportsbook price history (nothing was invented); Kalshi candles remain the only
  historical price.
* The probable-starter feed carries no timestamp → forward-only, by rule.
* `confirmed` status never yet observed → strict goalie gate still waits.
* Line combinations, announcement timing, player props: no verified feed.
* Overtime series: shootout settlement still unverified; wagers on such games stay OPEN
  for the exchange's own result.
* KXNHLOVERTIME price history / feature history still do not overlap (flagged).

## 9. Remaining work & recommended next research

1. **Measure, don't assume**: after ~2 weeks of regular-season probable rows,
   read `FIND_STARTER_CONVERSION`; if conversion is high and the ESPN name matches the
   NHL starter ≥95% of the time, consider promoting the probable features into a
   pre-game xG adjustment (a goalie-quality term in the Poisson rate).
2. **Announcement timing**: check whether ESPN's status ever flips to `confirmed`, and
   whether the Kalshi hourly candle around that flip shows a tradeable reaction
   (GOALIE_NEWS v3's block would then lift with real evidence).
3. **Split-squad-aware features**: once the regular season starts, the flags should be
   NULL/false everywhere — verify, and keep the preseason exclusion on the flag as a
   second layer of the scope gate.
4. **Player props** remain blocked pending a player-level model and an id-mapped feed;
   rosters are step one of that identity work.
5. Continue the standing loop: re-probe, ingest, discover, test, reject loudly, and keep
   the ledger honest.

---

**Verification key.** *Verified facts* are HTTP-probed sources and captured payloads
(`data/captured/`, `source_verification` table). *Calculations* are every number on this
site, generated from the ledger. *Assumptions* are labelled where used (post-game goalie
log for backtests; hourly candle "close"). *Unverified* anything in the verification
queue. Nothing in this report was hand-written into a number: change the data and
`make site` changes the report.
