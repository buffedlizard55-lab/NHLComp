# Captured API responses

Real payloads retrieved from live public APIs on **2026-09-20 (UTC)** while developing this
repository. They are kept so that parser tests run against genuine response shapes rather
than hand-written guesses, and so the pipeline can be exercised without network access.

| File | Source URL | Status |
|---|---|---|
| `nhl_stats_rest_team.json` | `https://api.nhle.com/stats/rest/en/team` | complete response, 32 current-team rows retained verbatim from the 62 returned |
| `nhl_scoreboard_20260919.json` | `https://api-web.nhle.com/v1/scoreboard/2026-09-19` | **partial** — the fetch tool returned the first chunk of a 12-chunk response; the 6 complete game records are retained verbatim and the truncated 7th is dropped |
| `kalshi_settled_kxnhlgame.json` | `https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNHLGAME&limit=3&status=settled` | complete response, 3 contracts |
| `kalshi_historical_markets_kxnhlgame_limit2.json` | `https://api.elections.kalshi.com/trade-api/v2/historical/markets?series_ticker=KXNHLGAME&limit=2` | complete response, 2 finalized contracts (2026 Stanley Cup Final game 6), retrieved 2026-09-20 |
| `kalshi_historical_markets_kxnhltotal_limit2.json` | `https://api.elections.kalshi.com/trade-api/v2/historical/markets?series_ticker=KXNHLTOTAL&limit=2` | complete response, 2 finalized totals contracts (2026 Final), retrieved **2026-09-21**. Proves totals history exists and that the line lives in `floor_strike`/`strike_type`, not in the historical tier's title |
| `kalshi_historical_candlesticks_26JUN14CARVGK_CAR.json` | `https://api.elections.kalshi.com/trade-api/v2/historical/markets/KXNHLGAME-26JUN14CARVGK-CAR/candlesticks?start_ts=1781481600&end_ts=1781503200&period_interval=60` | complete response, 4 hourly candles, retrieved 2026-09-20 |

Nothing in these files was authored, inferred or reconstructed. Where a payload is partial
that is stated above and in the file itself under `_capture`.

## Verified negatives (no payload file, because the answer is the absence of a field)

| Endpoint checked | Date | Result |
|---|---|---|
| `site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=20251115&limit=1` | 2026-09-21 | Keyless, serves arbitrary past dates, returns results, per-period `linescores`, three stars, and **winning/losing goalie with save stats** — and **no `odds` block** on a completed game (it does carry a Draft Kings `provider` attribution). Consequence: ESPN cannot supply historical sportsbook prices for this project, so the SportsPred claim that NHL prices come from an ESPN odds block was tested and **rejected for backtesting**; the feed is registered as a cross-check source only. |
| `users/buffedlizard55-lab/repos` (GitHub API) | 2026-09-21 | `CEO` and `PinePilot` are **not** among the account's public repositories, though the MasterSite links to them. Nothing was inferred about them. |

These are **not** the project's dataset. The full ingest runs against the live APIs
(`python -m nhlcomp run`); CI does this on every push.
