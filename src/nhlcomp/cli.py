"""Command-line interface.

    python -m nhlcomp init            create/refresh the database schema and source registry
    python -m nhlcomp probe           HTTP-probe every registered source and record the result
    python -m nhlcomp ingest ...      pull live data (requires outbound network)
    python -m nhlcomp run ...         full pipeline: ingest -> models -> strategies -> bets -> site
    python -m nhlcomp build-site      regenerate docs/ from the database
    python -m nhlcomp verify          run the irregularity checks
    python -m nhlcomp report          print a plain-text competition report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

from .analysis import Performance
from .http import HttpClient
from .pipeline import Pipeline
from .site import build_site
from .sources.registry import SOURCES, seed_registry
from .store import Store
from .verify import Verifier

DEFAULT_DB = os.environ.get("NHLCOMP_DB", "data/nhlcomp.db")
DEFAULT_CACHE = os.environ.get("NHLCOMP_CACHE", "data/cache")
DEFAULT_SITE = os.environ.get("NHLCOMP_SITE", "docs")


def _ctx(args: argparse.Namespace) -> tuple[Store, HttpClient]:
    store = Store(args.db)
    http = HttpClient(args.cache, offline=args.offline)
    return store, http


def _ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seasons", default="", help="comma list, e.g. 20242025,20252026")
    parser.add_argument("--scoreboard-days", default="")
    parser.add_argument("--clubs", default="", help="comma list of club abbreviations")
    parser.add_argument("--settled-pages", type=int, default=5)
    parser.add_argument("--cross-check-clubs", default="")
    parser.add_argument("--kalshi-budget", type=int, default=1500,
                        help="max Kalshi historical/candle calls per run (the walk resumes next run)")
    parser.add_argument("--live-points-budget", type=int, default=120,
                        help="max live-tier candlestick calls per run for open contracts")
    parser.add_argument("--totals-budget", type=int, default=400,
                        help="max Kalshi calls per run for the KXNHLTOTAL (totals) history walk; "
                             "kept separate so a long totals walk cannot starve the moneyline one")
    parser.add_argument("--spread-budget", type=int, default=400,
                        help="max Kalshi calls per run for the KXNHLSPREAD (puck line) history "
                             "walk; separate so it cannot starve the moneyline or totals walks")
    parser.add_argument("--overtime-budget", type=int, default=250,
                        help="max Kalshi calls per run for the KXNHLOVERTIME history walk")
    parser.add_argument("--polymarket-limit", type=int, default=100,
                        help="max Polymarket NHL events to read per run (reference cross-check "
                             "only; Kalshi is the execution venue)")
    parser.add_argument("--stats-seasons", default="",
                        help="comma list of seasons for the per-game stats REST reports "
                             "(defaults to --seasons)")


def _seasons(args: argparse.Namespace) -> list[int]:
    cur = date.today().year
    cur += 1 if date.today().month >= 7 else 0
    default = [cur * 10000 + cur + 1 - 20000, (cur - 1) * 10000 + cur - 20000]
    if not args.seasons:
        return default
    return [int(x) for x in args.seasons.split(",") if x.strip()]


def _days(args: argparse.Namespace) -> list[str]:
    if args.scoreboard_days:
        return [d for d in args.scoreboard_days.split(",") if d.strip()]
    today = date.today()
    out = []
    d = today - timedelta(days=400)
    while d <= today + timedelta(days=14):
        out.append(d.isoformat())
        d += timedelta(days=7)
    return out


def _clubs(args: argparse.Namespace) -> list[str]:
    if args.clubs:
        return [c for c in args.clubs.split(",") if c.strip()]
    return []


def cmd_init(args: argparse.Namespace) -> int:
    store, _ = _ctx(args)
    n = seed_registry(store)
    print(f"schema ready at {args.db}; {n} sources registered")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    store, http = _ctx(args)
    pipe = Pipeline(store, http, verbose=not args.quiet)
    res = pipe.ing.verify_sources()
    print(json.dumps(res, indent=1))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    store, http = _ctx(args)
    pipe = Pipeline(store, http, verbose=not args.quiet)
    out = pipe.stage_ingest(
        seasons=_seasons(args), scoreboard_days=_days(args), club_abbrevs=_clubs(args),
        ingest_settled_pages=args.settled_pages,
        cross_check_abbrevs=[c for c in args.cross_check_clubs.split(",") if c.strip()],
        kalshi_budget=args.kalshi_budget, live_points_budget=args.live_points_budget,
        totals_budget=args.totals_budget,
        spread_budget=args.spread_budget, overtime_budget=args.overtime_budget,
        polymarket_limit=args.polymarket_limit,
        stats_seasons=[int(x) for x in args.stats_seasons.split(",") if x.strip()] or None)
    print(json.dumps({k: v for k, v in out.items() if k != "probes"}, indent=1, default=str))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store, http = _ctx(args)
    pipe = Pipeline(store, http, verbose=not args.quiet)
    seasons = _seasons(args)
    prior = min(seasons) if seasons else None
    kw = {}
    if not args.no_ingest:
        kw["ingest"] = dict(seasons=seasons, scoreboard_days=_days(args),
                            club_abbrevs=_clubs(args),
                            ingest_settled_pages=args.settled_pages,
                            cross_check_abbrevs=[c for c in args.cross_check_clubs.split(",")
                                                 if c.strip()],
                            kalshi_budget=args.kalshi_budget,
                            live_points_budget=args.live_points_budget,
                            totals_budget=args.totals_budget,
                            spread_budget=args.spread_budget,
                            overtime_budget=args.overtime_budget,
                            polymarket_limit=args.polymarket_limit,
                            stats_seasons=[int(x) for x in args.stats_seasons.split(",")
                                           if x.strip()] or None)
    kw["features"] = {"game_types": tuple(int(x) for x in (args.game_types or "2").split(","))}
    kw["prior_season"] = prior if not args.no_prior else None
    report = pipe.run_all(**kw)
    if args.site:
        build_site(store, args.site)
        print(f"site written to {args.site}")
    print(json.dumps(report.get("competition", {}), indent=1))
    return 0


def cmd_build_site(args: argparse.Namespace) -> int:
    store, _ = _ctx(args)
    n = build_site(store, args.site)
    print(f"wrote {n} files to {args.site}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    store, _ = _ctx(args)
    print(json.dumps(Verifier(store).run_all(), indent=1))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store, _ = _ctx(args)
    perf = Performance(store)
    print("NHLComp competition status (BACKTEST and FORWARD TEST are reported separately)")
    for mode in ("FORWARD TEST", "BACKTEST"):
        tot = perf.competition_totals(test_mode=mode)
        print(f"[{mode}] strategies: {tot['strategies']}  bets: {tot['bets']}  "
              f"settled: {tot['settled']}  open: {tot['open']}")
        print(f"[{mode}] pnl: {tot['pnl']}  staked: {tot['staked']}  roi: {tot['roi']}  "
              f"win rate: {tot['win_rate']}")
        print(f"\n{mode.lower()} leaderboard (settled wagers only)")
        shown = 0
        for r in perf.leaderboard(test_mode=mode):
            if not r["n_settled"]:
                continue
            shown += 1
            print(f"  {shown:>3}. {r['username']:<28} n={r['n_settled']:<4} "
                  f"pnl={str(r['pnl']):>8} roi={r['roi']} dd={r['max_drawdown']} "
                  f"wr_ci={r['win_rate_ci']}")
            if shown >= 20:
                break
        if not shown:
            print("  (none settled yet)")
        print()
    # Season phase.  The totals above are the SCORED competition only (regular season +
    # playoffs).  Every wager carries the game_type of the game it was on, so wagers placed
    # on preseason games -- a phase no rule here is fitted for -- are shown on their own line
    # instead of being folded into a headline number they do not belong to.
    print("season phase split (the headline totals above are regular season + playoffs only)")
    for p in perf.phase_breakdown():
        phase = p["phase"]
        mark = "scored " if p["scored_in_competition"] else "excluded"
        print(f"  [{p['mode']:>12}] {mark}  {phase:<24} bets={p['bets']:<5} settled={p['settled']:<5} "
              f"open={p['open']:<5} pnl={p['pnl']:>9} staked={p['staked']:>10} roi={p['roi']}")
    print()
    base = store.one("SELECT evidence FROM findings WHERE finding_id='FIND_MARKET_BASELINE'")
    if base:
        print(f"market baseline (buy every side at the close, net of fees): {base['evidence']}")
    cov = store.one(
        """SELECT COUNT(*) c, SUM(CASE WHEN candles_state='ok' THEN 1 ELSE 0 END) ok
             FROM market_settlements WHERE provider='kalshi' AND result IN ('yes','no')""")
    print(f"kalshi settled contracts: {cov['c']}  with pre-game candle: {cov['ok'] or 0}")
    irr = store.one("SELECT COUNT(*) c FROM irregularities WHERE status='open'")["c"]
    print(f"\nopen irregularities: {irr}")
    return 0


def _common(*, suppress: bool = False) -> argparse.ArgumentParser:
    """Global options, accepted both before and after the subcommand.

    ``nhlcomp --db X run`` and ``nhlcomp run --db X`` must mean the same thing.  When the
    parser is reused as a subparser parent its defaults are SUPPRESSed, otherwise the
    subparser's own default would silently overwrite a value the top level already parsed.
    """
    d = argparse.SUPPRESS if suppress else None
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--db", default=d if suppress else DEFAULT_DB)
    c.add_argument("--cache", default=d if suppress else DEFAULT_CACHE)
    c.add_argument("--site", default=d if suppress else DEFAULT_SITE)
    c.add_argument("--offline", action="store_true", default=d if suppress else False,
                   help="never touch the network; use only the response cache")
    c.add_argument("--quiet", action="store_true", default=d if suppress else False)
    return c


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nhlcomp", description=__doc__, parents=[_common()])
    sub_common = _common(suppress=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", parents=[sub_common]).set_defaults(fn=cmd_init)
    sub.add_parser("probe", parents=[sub_common]).set_defaults(fn=cmd_probe)

    for name, fn in (("ingest", cmd_ingest), ("run", cmd_run)):
        sp = sub.add_parser(name, parents=[sub_common])
        _ingest_args(sp)
        sp.set_defaults(fn=fn)
    sub.choices["run"].add_argument("--game-types", default="2")
    sub.choices["run"].add_argument("--no-ingest", action="store_true")
    sub.choices["run"].add_argument("--no-prior", action="store_true")

    sub.add_parser("build-site", parents=[sub_common]).set_defaults(fn=cmd_build_site)
    sub.add_parser("verify", parents=[sub_common]).set_defaults(fn=cmd_verify)
    sub.add_parser("report", parents=[sub_common]).set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
