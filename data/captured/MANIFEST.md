# Captured API responses

Real payloads retrieved from live public APIs on **2026-09-20 (UTC)** while developing this
repository. They are kept so that parser tests run against genuine response shapes rather
than hand-written guesses, and so the pipeline can be exercised without network access.

| File | Source URL | Status |
|---|---|---|
| `nhl_stats_rest_team.json` | `https://api.nhle.com/stats/rest/en/team` | complete response, 32 current-team rows retained verbatim from the 62 returned |
| `nhl_scoreboard_20260919.json` | `https://api-web.nhle.com/v1/scoreboard/2026-09-19` | **partial** — the fetch tool returned the first chunk of a 12-chunk response; the 6 complete game records are retained verbatim and the truncated 7th is dropped |
| `kalshi_settled_kxnhlgame.json` | `https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNHLGAME&limit=3&status=settled` | complete response, 3 contracts |

Nothing in these files was authored, inferred or reconstructed. Where a payload is partial
that is stated above and in the file itself under `_capture`.

These are **not** the project's dataset. The full ingest runs against the live APIs
(`python -m nhlcomp run`); CI does this on every push.
