"""Review of the owner's MasterSite directory, with a verdict per project.

The brief asks for the master directory at
``https://buffedlizard55-lab.github.io/MasterSite/`` to be reviewed and for useful
infrastructure to be reused *only after* it has been verified as relevant and functional.
So every row below records what was actually fetched and what was concluded -- including the
projects that do not exist and the one reuse candidate that was tested and rejected.

Method (all on 2026-09-21, no inference from memory):

1. ``GET https://buffedlizard55-lab.github.io/MasterSite/`` -> the directory lists 49 verified
   sites across 10 categories (Sports Data & Scoreboards 17, Markets & Trading Research 9, ...).
2. ``GET https://api.github.com/users/buffedlizard55-lab/repos?per_page=100&sort=updated`` ->
   the account's public repositories, with push dates, used to confirm which named projects
   exist at all.
3. ``GET /repos/{owner}/{repo}/readme`` for the projects that could plausibly feed an NHL
   model, read before deciding anything.
4. The one concrete data claim that mattered -- "NHL prices from the ESPN odds block"
   (SportsPred, ice-hockey page) -- was tested directly against
   ``site.api.espn.com/.../nhl/scoreboard?dates=20251115&limit=1``.

Nothing here is a claim about a project's quality; it is a record of what was checked and
what was found, so a later run can re-check rather than re-guess.
"""

from __future__ import annotations

from typing import Any

#: Verified by fetching the GitHub API on 2026-09-21.
REVIEW_DATE = "2026-09-21"
MASTER_SITE_URL = "https://buffedlizard55-lab.github.io/MasterSite/"

#: name, exists (GitHub API), relevance to NHL prediction, what was done, evidence
REVIEW: list[dict[str, Any]] = [
    {
        "project": "KalshiPaperSim",
        "url": "https://github.com/buffedlizard55-lab/KalshiPaperSim",
        "exists": "yes (repo pushed 2026-09-21, 55 MB)",
        "relevance": "high — methodology, not data",
        "verdict": "methodology reused: a paper-trading competition scored on captured exchange "
                   "data, with an explicit rule that a design which never traded is left "
                   "UNRANKED rather than shown at 0%",
        "evidence": "README fetched 2026-09-21. Its own directory review (src/signal-sources.js) "
                    "is the pattern this project follows: every candidate source ends in exactly "
                    "one of three states — traded, blocked with a named unblocker, or a verified "
                    "negative. No code or data was copied.",
    },
    {
        "project": "Commodities",
        "url": "https://github.com/buffedlizard55-lab/Commodities",
        "exists": "yes (repo pushed 2026-09-21)",
        "relevance": "medium — execution modelling",
        "verdict": "not reused; its forward-desk design (immediate-or-cancel against displayed "
                   "depth, sized at a fraction of free cash) is already implemented here in "
                   "paper.simulate_fill",
        "evidence": "README fetched 2026-09-21: two-cycle-per-hour desk, append-only trades.jsonl, "
                    "order-book evidence. Its universe (econ, crypto, gold, weather, NFL/NBA/"
                    "NCAAF/MLB games) contains no NHL series, so no data was transferable.",
    },
    {
        "project": "SportsPred",
        "url": "https://github.com/buffedlizard55-lab/SportsPred",
        "exists": "yes (repo pushed 2026-09-05)",
        "relevance": "high claim, rejected on test",
        "verdict": "its ice-hockey page states that NHL prices come from 'the ESPN odds block'. "
                   "Tested directly and it does not hold for completed games, so it is not a "
                   "historical price source and was not reused",
        "evidence": "GET site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                    "?dates=20251115&limit=1 (2026-09-21) returns the finished TB@FLA game with "
                    "linescores, three stars, winning/losing goalies and a Draft Kings provider "
                    "attribution — and no odds block. Registered as source "
                    "espn.nhl_scoreboard with that negative recorded in known_limits.",
    },
    {
        "project": "NFLInjuryReport / NBAInjuryReport",
        "url": "https://github.com/buffedlizard55-lab/NFLInjuryReport",
        "exists": "yes (both repos; NFL pushed 2026-09-21)",
        "relevance": "medium — corroborates this project's injury choices",
        "verdict": "no NHL feed to reuse, but its probed source ledger independently confirms "
                   "the two negatives this project already records: Reddit's JSON API is blocked "
                   "(403) and X has no free read path",
        "evidence": "README fetched 2026-09-21. Its table of 13 probed sources records ESPN's "
                    "injuries JSON as the primary keyless feed — the same feed this project "
                    "uses for NHL injuries — and marks the NFL's own API 401 (OAuth).",
    },
    {
        "project": "PriceKalshiHistorical",
        "url": "https://github.com/buffedlizard55-lab/PriceKalshiHistorical",
        "exists": "yes (repo pushed 2026-08-19)",
        "relevance": "medium — Kalshi capture design",
        "verdict": "not reused; it collects the same endpoints this project already calls "
                   "(/markets, /orderbook, /trades, candlesticks) and adds a Node exchange "
                   "simulator this project deliberately does not have",
        "evidence": "README fetched 2026-09-21. Its 'capture the book, not just the mid' point is "
                    "recorded here as a known limitation instead: the candlestick feed publishes "
                    "one bid/ask level per period, so slippage is stored as 0 rather than "
                    "modelled.",
    },
    {
        "project": "DrugAnalysis (FDA / drug analysis)",
        "url": "https://github.com/buffedlizard55-lab/DrugAnalysis",
        "exists": "yes (repo pushed 2026-09-20, 47 MB)",
        "relevance": "none for NHL prediction",
        "verdict": "not reused",
        "evidence": "Listed in the account's repositories on 2026-09-21. No NHL, hockey or "
                    "exchange content.",
    },
    {
        "project": "Scoreboards (NFL-scoreboard, MLB-Live-PBP, Ncaa-football-alerts)",
        "url": "https://github.com/buffedlizard55-lab/NFL-scoreboard",
        "exists": "yes (all three)",
        "relevance": "low — same pattern, different sport",
        "verdict": "not reused; the NHL schedule/results pipeline here reads the NHL's own "
                   "api-web feed directly",
        "evidence": "Repositories confirmed present on 2026-09-21.",
    },
    {
        "project": "GOLD",
        "url": "https://github.com/buffedlizard55-lab/GOLD",
        "exists": "yes (repo pushed 2026-08-28)",
        "relevance": "none",
        "verdict": "not reused",
        "evidence": "Confirmed present on 2026-09-21; a directory site, not a market-data feed.",
    },
    {
        "project": "Insider-trades",
        "url": "https://github.com/buffedlizard55-lab/Insider-trades",
        "exists": "yes (repo pushed 2026-08-19)",
        "relevance": "none for NHL",
        "verdict": "not reused — SEC Form 4 filings have no NHL signal",
        "evidence": "Confirmed present on 2026-09-21.",
    },
    {
        "project": "TradingViewTheLeap (TheLeap)",
        "url": "https://github.com/buffedlizard55-lab/TradingViewTheLeap",
        "exists": "yes (repo pushed 2026-09-20)",
        "relevance": "none for NHL",
        "verdict": "not reused — equity chart patterns, no hockey application",
        "evidence": "Confirmed present on 2026-09-21.",
    },
    {
        "project": "NFLComp / NBAComp / MLBComp",
        "url": "https://github.com/buffedlizard55-lab/NFLComp",
        "exists": "yes (all three; NFL pushed 2026-09-21)",
        "relevance": "high — sibling competitions with the same brief",
        "verdict": "not reused as code, but the naming/ledger conventions match, which is why "
                   "this project keeps the same BACKTEST / FORWARD TEST / PAPER TRADE separation "
                   "and the same append-only ledger",
        "evidence": "Repositories confirmed present on 2026-09-21. Not read line by line in this "
                    "pass; recorded as a follow-up rather than claimed as reused.",
    },
    {
        "project": "CEO",
        "url": "https://github.com/buffedlizard55-lab/CEO",
        "exists": "NO — not among the account's public repositories",
        "relevance": "unknown; cannot be assessed",
        "verdict": "not reusable: nothing to fetch",
        "evidence": "The repository list fetched on 2026-09-21 contains no 'CEO' entry, which "
                    "independently confirms the same negative KalshiPaperSim recorded.",
    },
    {
        "project": "PinePilot",
        "url": "https://github.com/buffedlizard55-lab/PinePilot",
        "exists": "NO — not among the account's public repositories",
        "relevance": "unknown; cannot be assessed",
        "verdict": "not reusable: nothing to fetch. The MasterSite itself flags PinePilot as "
                   "unverifiable (HTTP 404)",
        "evidence": "Not present in the repository list fetched on 2026-09-21. A "
                    "'Tradingview-pinescript-editor' repository does exist and may be what the "
                    "name refers to; that is a guess and is recorded as one.",
    },
    {
        "project": "NFLPRED",
        "url": "https://github.com/buffedlizard55-lab/NFLPRED",
        "exists": "yes but empty (0 KB, pushed 2026-08-17)",
        "relevance": "none — no content",
        "verdict": "not reusable",
        "evidence": "Repository size 0 KB in the API listing fetched 2026-09-21.",
    },
]

#: The one reuse that was tested and rejected, stated plainly so it is not re-litigated.
REJECTED_REUSE = [
    {
        "claim": "NHL betting prices can be read from ESPN's scoreboard odds block "
                 "(as SportsPred's ice-hockey page describes).",
        "test": "GET https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
                "?dates=20251115&limit=1 on 2026-09-21.",
        "result": "Rejected for historical prices. The completed game's payload carries results, "
                  "period linescores, three stars, winning/losing goalies, venue and broadcast, "
                  "with the feed's provider attributed to Draft Kings — and no odds block. An "
                  "odds block may exist for upcoming games, which is a forward-only reference at "
                  "best; it is not history, so no backtest may be priced from it.",
        "kept": "The feed itself was kept as a registered, probed source (espn.nhl_scoreboard) "
                "for what it verifiably does supply: a second independent result and a "
                "post-game goalie identification to cross-check the NHL stats REST log.",
    },
]


def summary() -> dict[str, Any]:
    """Machine-readable summary for the findings ledger."""
    return {
        "reviewed_at": REVIEW_DATE,
        "directory": MASTER_SITE_URL,
        "sites_in_directory": 49,
        "projects_reviewed": len(REVIEW),
        "missing_repositories": [r["project"] for r in REVIEW if r["exists"].startswith("NO")],
        "reused": [r["project"] for r in REVIEW if r["verdict"].startswith("methodology reused")],
        "rejected_after_test": [c["claim"] for c in REJECTED_REUSE],
        "method": "MasterSite index fetched; repository existence checked against the GitHub "
                  "API; READMEs fetched for every candidate; the one data claim that mattered "
                  "was tested against the live endpoint.",
    }
