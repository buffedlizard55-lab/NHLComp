"""Static GitHub Pages site generator.

Pure stdlib.  Renders one page per required section plus a JSON data directory so the
leaderboard can be sorted and filtered in the browser without a server.

Every number on the site is read out of the database; nothing is computed in the template.
Where a figure does not exist (no verified historical odds, no verified arena coordinates)
the page says so explicitly rather than showing a blank or a plausible-looking zero.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Iterable, Sequence

from .analysis import Performance
from .backtest import CAVEAT_NO_PRICE
from .store import Store, utcnow

SECTIONS = [
    ("index.html", "Dashboard"),
    ("leaderboard.html", "Leaderboard"),
    ("strategies.html", "Strategies"),
    ("upcoming.html", "Upcoming Bets"),
    ("positions.html", "Live Paper Trading"),
    ("history.html", "Trade History"),
    ("performance.html", "Performance"),
    ("research.html", "Research Lab"),
    ("sources.html", "Data Sources"),
    ("verification.html", "Verification"),
    ("methodology.html", "Methodology"),
]

CSS = """
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;
--acc:#58a6ff;--good:#3fb950;--bad:#f85149;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,
"Segoe UI",Helvetica,Arial,sans-serif}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
header{background:var(--panel);border-bottom:1px solid var(--bd);padding:12px 16px;
display:flex;flex-wrap:wrap;gap:10px;align-items:center;position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0 18px 0 0}
nav a{color:var(--mut);padding:4px 9px;border-radius:6px;font-size:13px;white-space:nowrap}
nav a.on{background:var(--panel2);color:var(--fg)}
main{padding:16px;max-width:1500px;margin:0 auto}
h2{font-size:15px;margin:22px 0 8px;border-bottom:1px solid var(--bd);padding-bottom:5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:8px;padding:11px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.card .v{font-size:21px;font-weight:600;margin-top:3px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--bd);
border-radius:8px;overflow:hidden;font-size:12.5px}
th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--bd);vertical-align:top}
th{background:var(--panel2);color:var(--mut);font-weight:600;cursor:pointer;user-select:none;
position:sticky;top:52px}
tr:hover td{background:#1a1f26}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--good)}.neg{color:var(--bad)}.mut{color:var(--mut)}.warn{color:var(--warn)}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;border:1px solid var(--bd);
background:var(--panel2)}
.pill.good{color:var(--good);border-color:#1f6f33}.pill.bad{color:var(--bad);border-color:#7d2220}
.pill.warn{color:var(--warn);border-color:#6d5314}.pill.info{color:var(--acc);border-color:#1f4d7a}
.note{background:#1a1f26;border-left:3px solid var(--acc);padding:9px 12px;border-radius:0 6px 6px 0;
margin:10px 0;color:#c9d1d9}
.caveat{border-left-color:var(--warn)}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0;align-items:center}
input,select{background:var(--panel2);border:1px solid var(--bd);color:var(--fg);padding:5px 8px;
border-radius:6px;font-size:13px}
.wrap{overflow-x:auto}
.small{font-size:11.5px;color:var(--mut)}
details{background:var(--panel);border:1px solid var(--bd);border-radius:8px;padding:8px 11px;
margin:7px 0}
summary{cursor:pointer;font-weight:600}
code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:12px}
footer{color:var(--mut);font-size:11.5px;padding:18px 16px;border-top:1px solid var(--bd);
margin-top:26px}
@media(max-width:700px){th{position:static}.card .v{font-size:17px}main{padding:10px}}
"""

JS = """
function sortTable(id, col, type){
  const t=document.getElementById(id); if(!t) return;
  const tb=t.tBodies[0]; const rows=[...tb.rows];
  const dir=t.dataset.sortDir==='asc'?'desc':'asc';
  t.dataset.sortDir=dir;
  rows.sort((a,b)=>{
    let x=a.cells[col].dataset.v ?? a.cells[col].textContent.trim();
    let y=b.cells[col].dataset.v ?? b.cells[col].textContent.trim();
    if(type==='n'){x=parseFloat(x)||0;y=parseFloat(y)||0;return dir==='asc'?x-y:y-x;}
    return dir==='asc'?String(x).localeCompare(y):String(y).localeCompare(x);
  });
  rows.forEach(r=>tb.appendChild(r));
}
function filterTable(id, q, cols){
  const t=document.getElementById(id); if(!t) return;
  q=(q||'').toLowerCase();
  for(const r of t.tBodies[0].rows){
    let hay='';
    (cols||[...Array(r.cells.length).keys()]).forEach(i=>{hay+=' '+(r.cells[i].textContent||'');});
    r.style.display = hay.toLowerCase().includes(q)?'':'none';
  }
}
"""


def e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def n(v: Any, digits: int = 2, dash: str = "—") -> str:
    if v is None or v == "":
        return dash
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return e(v)


def signed(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return e(v)
    cls = "pos" if f > 0 else ("neg" if f < 0 else "mut")
    return f'<span class="{cls}">{f:+,.{digits}f}</span>'


def pct(v: Any, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return e(v)


def _page(title: str, active: str, body: str, generated: str) -> str:
    nav = "".join(
        f'<a class="{"on" if href == active else ""}" href="{href}">{e(label)}</a>'
        for href, label in SECTIONS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)} · NHLComp</title>
<link rel="stylesheet" href="assets/style.css">
<script src="assets/app.js" defer></script></head>
<body><header><h1>NHLComp</h1><nav>{nav}</nav></header>
<main>{body}</main>
<footer>NHLComp · paper-trading research platform · no real money is ever wagered ·
generated {e(generated)} from the SQLite ledger. Every figure is read from the database;
values that could not be verified from a source are shown as — with a caveat.</footer>
</body></html>"""


def _table(id_: str, headers: Sequence[str], rows: Sequence[Sequence[str]],
           numeric: Sequence[int] = ()) -> str:
    ths = "".join(
        f'<th class="{"num" if i in numeric else ""}" '
        f'onclick="sortTable(\'{id_}\',{i},\'{"n" if i in numeric else "s"}\')">{e(h)}</th>'
        for i, h in enumerate(headers))
    trs = "".join("<tr>" + "".join(
        f'<td class="{"num" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r))
        + "</tr>" for r in rows)
    empty = f"<tr><td colspan='{len(headers)}'>No rows yet.</td></tr>"
    return (f'<div class="wrap"><table id="{id_}"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{trs or empty}</tbody></table></div>')


# --------------------------------------------------------------------- pages
def page_dashboard(store: Store, perf: Performance, gen: str) -> str:
    tot = perf.competition_totals()
    irr = store.one("SELECT COUNT(*) c FROM irregularities WHERE status='open'")["c"]
    src_ok = store.one("SELECT COUNT(*) c FROM source_registry WHERE status='verified'")["c"]
    src_all = store.one("SELECT COUNT(*) c FROM source_registry")["c"]
    upcoming = store.one("SELECT COUNT(*) c FROM upcoming_bets")["c"]
    cards = [
        ("Strategies", tot["strategies"]), ("Simulated bets", tot["bets"]),
        ("Settled", tot["settled"]), ("Open", tot["open"]),
        ("Total P&L", f'<span class="{"pos" if (tot["pnl"] or 0) > 0 else "neg"}">'
                      f'{n(tot["pnl"])}</span>'),
        ("Staked", n(tot["staked"])), ("ROI", pct(tot["roi"])), ("Win rate", pct(tot["win_rate"])),
        ("Upcoming signals", upcoming), ("Verified sources", f"{src_ok}/{src_all}"),
        ("Open irregularities", f'<span class="{"warn" if irr else ""}">{irr}</span>'),
    ]
    body = ["<h2>Competition dashboard</h2>",
            '<div class="cards">' + "".join(
                f'<div class="card"><div class="k">{e(k)}</div>'
                f'<div class="v">{v}</div></div>' for k, v in cards) + "</div>"]

    body.append('<div class="note">This is a <b>paper-trading research competition</b>. No real '
                'money is wagered and no order-placement code exists in this repository. Prices '
                'come from the Kalshi public market-data API and results from the NHL API.</div>')

    top = perf.leaderboard()[:10]
    rows = [[str(r["rank"]), e(r["username"]), e(r["category"]), str(r["n_settled"]),
             signed(r["pnl"]), pct(r["roi"]), pct(r["win_rate"]),
             f'{pct(r["win_rate_ci"][0], 1)}–{pct(r["win_rate_ci"][1], 1)}',
             n(r["max_drawdown"]), str(r["open"])] for r in top]
    body.append("<h2>Top strategies</h2>")
    body.append(_table("lb", ["#", "Username", "Category", "Settled", "P&L", "ROI", "Win rate",
                              "Win-rate 95% CI", "Max DD", "Open"], rows, numeric=(0, 3, 4, 5, 6, 8, 9)))

    mc = store.one("SELECT body, evidence FROM findings WHERE finding_id='FIND_MODEL_COMPARE'")
    if mc:
        body.append("<h2>Out-of-sample model comparison</h2>")
        body.append(f'<p class="small">{e(mc["body"])}</p>')
        try:
            comp = json.loads(mc["evidence"])
            rows = [[e(k), str(v.get("n")), n(v.get("log_loss"), 5), n(v.get("brier"), 5)]
                    for k, v in comp.items()]
            body.append(_table("mc", ["Model", "Held-out games", "Log loss", "Brier"], rows,
                               numeric=(1, 2, 3)))
            body.append('<p class="small">Lower is better. A model is only kept if it beats the '
                        'home-ice constant baseline.</p>')
        except (ValueError, TypeError):
            pass

    body.append("<h2>Verification status of the price feeds</h2>")
    rows = []
    src_pill = {"verified": "good", "rejected": "bad", "flagged": "warn"}
    for s in store.query("SELECT * FROM source_registry WHERE source_id LIKE 'kalshi%' OR "
                         "source_id LIKE 'odds%' ORDER BY source_id"):
        rows.append([e(s["source_id"]), e(s["name"]),
                     f'<span class="pill {src_pill.get(s["status"], "info")}">'
                     f'{e(s["status"])}</span>',
                     e(s["known_limits"])[:400]])
    body.append(_table("srcs", ["Source", "Name", "Status", "Known limits"], rows))
    body.append('<div class="note caveat"><b>Why some sections show no profit history:</b> '
                'Kalshi does not document the timestamp behind its <code>previous_*</code> price '
                'fields and its candle endpoint returned 404, so no precisely-timestamped '
                'historical price series exists. Bets built from settled-contract prices are '
                'labelled BACKTEST with '
                '<code>verification_status=single_source_timing_unverified</code>; everything '
                'else is FORWARD TEST and settles as games finish.</div>')
    return _page("Dashboard", "index.html", "".join(body), gen)


def page_leaderboard(store: Store, perf: Performance, gen: str) -> str:
    rows_data = perf.leaderboard()
    cats = sorted({r["category"] for r in rows_data})
    body = ["<h2>Leaderboard</h2>",
            '<div class="controls"><input id="q" placeholder="filter by username or category…" '
            'oninput="filterTable(\'lb\', this.value)">',
            '<select onchange="filterCategory(this.value)"><option value="">all categories</option>'
            + "".join(f'<option value="{e(c)}">{e(c)}</option>' for c in cats) + "</select></div>",
            '<p class="small">Win rate is always paired with its 95% Wilson interval — a 3-bet '
            'sample proves nothing. Click any header to sort; click a username for its wagers.</p>']
    rows = []
    for r in rows_data:
        rows.append([str(r["rank"]),
                     f'<a href="strategies.html#{e(r["strategy_id"])}">{e(r["username"])}</a>',
                     e(r["category"]), e(r["status"]), str(r["n_bets"]), str(r["n_settled"]),
                     str(r["wins"]), str(r["losses"]), str(r["pushes"]), str(r["open"]),
                     signed(r["pnl"]), n(r["staked"]), pct(r["roi"]), pct(r["win_rate"]),
                     f'{pct(r["win_rate_ci"][0], 1)}–{pct(r["win_rate_ci"][1], 1)}',
                     n(r["avg_price"], 3), n(r["avg_edge"], 4), n(r["max_drawdown"]),
                     n(r["volatility"]), str(r["longest_losing_streak"]),
                     pct(r["clv_beat_close"]), n(r["clv_avg"], 4), str(r["n_clv"]),
                     n(r["starting_bankroll"], 0), n(r["bankroll"])])
    body.append(_table("lb", ["#", "Username", "Category", "Status", "Bets", "Settled", "W", "L",
                              "Push", "Open", "P&L", "Staked", "ROI", "Win rate", "95% CI",
                              "Avg price", "Avg edge", "Max DD", "Volatility", "Worst streak",
                              "Beat close", "Avg CLV", "n CLV", "Start bank", "Bankroll"],
                       rows, numeric=tuple(i for i in range(25) if i not in (1, 2, 3))))
    body.append("""<script>
function filterCategory(v){
  const t=document.getElementById('lb');
  for(const r of t.tBodies[0].rows)
    r.style.display = (!v || r.cells[2].textContent===v)?'':'none';
}
</script>""")
    return _page("Leaderboard", "leaderboard.html", "".join(body), gen)


def page_strategies(store: Store, perf: Performance, gen: str) -> str:
    parts = ["<h2>Strategies</h2>",
             '<p class="small">Strategies are never overwritten — every change creates a new '
             'version so an edit can be judged on whether it actually helped.</p>']
    for s in store.query("SELECT * FROM strategies ORDER BY strategy_id, version"):
        key = f"{s['strategy_id']}_v{s['version']}"
        summ = perf.summarize(s["strategy_id"], int(s["version"]))
        m = summ.get("ALL", {})
        f = summ.get("FORWARD TEST", {})
        b = summ.get("BACKTEST", {})
        pill = {"active": "good", "candidate": "info", "paused": "warn",
                "retired": "bad", "rejected": "bad"}.get(s["status"], "info")
        parts.append(f"""<details id="{e(s['strategy_id'])}"><summary>
<span class="pill {pill}">{e(s['status'])}</span> <b>{e(s['username'])}</b>
<span class="mut">{e(s['strategy_id'])} v{s['version']} · {e(s['category'])} ·
origin {e(s['origin'])}</span></summary>
<table><tbody>
<tr><th>Hypothesis</th><td>{e(s['hypothesis'])}</td></tr>
<tr><th>Data</th><td>{e(s['data_used'])}</td></tr>
<tr><th>Entry rule</th><td>{e(s['entry_rule'])}</td></tr>
<tr><th>Price rule</th><td>{e(s['price_rule'])}</td></tr>
<tr><th>Settlement</th><td>{e(s['settlement_rule'])}</td></tr>
<tr><th>Markets</th><td>{e(s['markets'])}</td></tr>
<tr><th>Parameters</th><td><code>{e(s['params_json'])}</code></td></tr>
<tr><th>Forward test</th><td>{f.get('n_settled', 0)} settled · P&L {signed(f.get('pnl'))} ·
ROI {pct(f.get('roi'))} · win rate {pct(f.get('win_rate'))}
({pct((f.get('win_rate_ci') or [0, 0])[0], 1)}–{pct((f.get('win_rate_ci') or [0, 0])[1], 1)})</td></tr>
<tr><th>Backtest</th><td>{b.get('n_settled', 0)} settled · P&L {signed(b.get('pnl'))} ·
ROI {pct(b.get('roi'))}</td></tr>
<tr><th>Bankroll</th><td>{n(s['starting_bankroll'], 0)} → {n(m.get('bankroll'))}
(open exposure {n(m.get('open_exposure'))})</td></tr>
</tbody></table>""")
        bt = store.query("SELECT * FROM backtests WHERE strategy_id=? AND version=?",
                         (s["strategy_id"], s["version"]))
        if bt:
            rows = [[e(x["label"]), str(x["n_bets"]), str(x["n_wins"]), str(x["n_losses"]),
                     pct(_ratio(x["n_wins"], x["n_bets"])), e(x["test_from"]), e(x["test_to"]),
                     e("yes" if x["data_sufficient"] else "no"), e(x["caveat"])[:300]]
                    for x in bt]
            parts.append("<h2 style='font-size:13px'>Accuracy backtest (no price data)</h2>")
            parts.append(_table(f"bt_{key}", ["Window", "N", "W", "L", "Hit rate", "From", "To",
                                              "Price data OK?", "Caveat"], rows,
                                numeric=(1, 2, 3, 4)))
        parts.append("<h2 style='font-size:13px'>Wagers</h2>")
        total_bets = store.one(
            "SELECT COUNT(*) c FROM bets WHERE strategy_id=? AND strategy_version=?",
            (s["strategy_id"], s["version"]))["c"]
        bets = store.query("SELECT * FROM bets WHERE strategy_id=? AND strategy_version=? "
                           "ORDER BY bet_ts DESC LIMIT 50", (s["strategy_id"], s["version"]))
        if total_bets > len(bets):
            parts.append(f'<p class="small">showing the {len(bets)} most recent of '
                         f'{total_bets} wagers — the full ledger is in '
                         f'<code>data/bets.json</code> and the SQLite database.</p>')
        parts.append(_bets_table(f"bets_{key}", bets))
        parts.append("</details>")
    return _page("Strategies", "strategies.html", "".join(parts), gen)


def _ratio(a: Any, b: Any) -> float | None:
    try:
        return float(a) / float(b) if b else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _bets_table(id_: str, bets: Sequence[Any]) -> str:
    rows = []
    for b in bets:
        rows.append([e(b["bet_id"])[:52], e(b["test_mode"]), e(b["game_date"]), e(b["matchup"]),
                     e(b["market"]), e(b["selection"]), e(b["provider"]), n(b["price"], 3),
                     n(b["model_prob"], 4), n(b["edge"], 4), n(b["stake"]), n(b["filled_size"], 2),
                     n(b["liquidity"], 0), e(b["result"]), signed(b["pnl"]), pct(b["roi"]),
                     n(b["clv"], 4), e(b["verification_status"]), e(b["bet_ts"])])
    return _table(id_, ["Bet ID", "Mode", "Game", "Matchup", "Market", "Selection", "Provider",
                        "Price", "Model p", "Edge", "Stake", "Contracts", "Liquidity", "Result",
                        "P&L", "ROI", "CLV", "Verification", "Placed"], rows,
                  numeric=(7, 8, 9, 10, 11, 12, 14, 15, 16))


def page_upcoming(store: Store, gen: str) -> str:
    body = ["<h2>Upcoming strategy bets</h2>",
            '<p class="small">Every signal a strategy currently wants to act on, including the '
            'ones it is blocked on. A status of PRICE TOO HIGH means the model likes the side '
            'but the offer is not good enough — the strategy waits instead of chasing.</p>',
            '<div class="controls"><input placeholder="filter…" '
            'oninput="filterTable(\'up\', this.value)"></div>']
    rows = []
    for u in store.query("SELECT * FROM upcoming_bets ORDER BY decision_ts DESC LIMIT 1000"):
        pill = {"READY TO BET": "good", "EXECUTED": "good", "PRICE TOO HIGH": "warn",
                "PRICE TOO LOW": "warn", "WATCHING": "info", "CANCELLED": "bad",
                "EXPIRED": "bad"}.get(u["status"], "info")
        rows.append([e(u["username"]), e(u["game_date"]), e(u["matchup"]), e(u["market"]),
                     e(u["selection"]), e(u["provider"]), n(u["current_price"], 3),
                     n(u["required_price"], 3), n(u["model_prob"], 4), n(u["edge"], 4),
                     n(u["stake"]), f'<span class="pill {pill}">{e(u["status"])}</span>',
                     e(u["blocking_reason"]), e(u["supporting_data"])[:300], e(u["decision_ts"])])
    body.append(_table("up", ["Username", "Game", "Matchup", "Market", "Selection", "Provider",
                              "Current price", "Required price", "Model p", "Edge", "Stake",
                              "Status", "Blocking reason", "Supporting data", "Decision ts"],
                       rows, numeric=(6, 7, 8, 9, 10)))
    return _page("Upcoming Bets", "upcoming.html", "".join(body), gen)


def page_positions(store: Store, gen: str) -> str:
    body = ["<h2>Live paper trading — open positions</h2>"]
    rows = []
    for b in store.query("SELECT * FROM bets WHERE result='OPEN' ORDER BY bet_ts DESC"):
        rows.append([e(b["username"]), e(b["test_mode"]), e(b["game_date"]), e(b["matchup"]),
                     e(b["selection"]), n(b["entry_price"], 3), n(b["model_prob"], 4),
                     n(b["edge"], 4), n(b["stake"]), n(b["filled_size"], 2), n(b["liquidity"], 0),
                     n(b["slippage"], 4), e(b["bet_ts"]), e(b["verification_status"])])
    body.append(_table("op", ["Username", "Mode", "Game", "Matchup", "Selection", "Entry",
                              "Model p", "Edge", "Stake", "Contracts", "Liquidity", "Slippage",
                              "Opened", "Verification"], rows,
                       numeric=(5, 6, 7, 8, 9, 10, 11)))
    exp = store.one("SELECT COALESCE(SUM(stake),0) e, COUNT(*) c FROM bets WHERE result='OPEN'")
    body.append(f'<div class="note">{exp["c"]} open positions, {n(exp["e"])} of virtual stake '
                f'reserved. A position settles only from the official NHL result; nothing is '
                f'marked to model.</div>')
    return _page("Live Paper Trading", "positions.html", "".join(body), gen)


def page_history(store: Store, gen: str) -> str:
    body = ["<h2>Trade history</h2>",
            '<p class="small">Append-only. Corrections are never made in place — they are '
            'written as amendments with a before/after audit row.</p>',
            '<div class="controls"><input placeholder="filter by id, team, mode…" '
            'oninput="filterTable(\'th\', this.value)"></div>']
    bets = store.query("SELECT * FROM bets ORDER BY bet_ts DESC LIMIT 1500")
    total = store.one("SELECT COUNT(*) c FROM bets")["c"]
    if total > len(bets):
        body.append(f'<p class="small">showing the {len(bets)} most recent of {total} wagers; '
                    f'the complete ledger is in <code>data/bets.json</code>.</p>')
    body.append(_bets_table("th", bets))
    aud = store.one("SELECT COUNT(*) c FROM bet_audit")["c"]
    body.append(f'<p class="small">{aud} audit rows in <code>bet_audit</code>.</p>')
    return _page("Trade History", "history.html", "".join(body), gen)


def page_performance(store: Store, perf: Performance, gen: str) -> str:
    parts = ["<h2>Performance</h2>"]
    for s in store.query("SELECT strategy_id, version, username FROM strategies "
                         "ORDER BY strategy_id, version"):
        summ = perf.summarize(s["strategy_id"], int(s["version"]))
        if not summ.get("ALL", {}).get("n_settled"):
            continue
        parts.append(f"<details><summary><b>{e(s['username'])}</b> "
                     f"<span class='mut'>v{s['version']}</span></summary>")
        for mode in ("FORWARD TEST", "BACKTEST"):
            m = summ.get(mode, {})
            if not m.get("n_settled"):
                continue
            curve = perf.equity_curve(perf.strategy_bets(s["strategy_id"], int(s["version"]), mode),
                                      float(summ["strategy"]["starting_bankroll"]))
            parts.append(f"<h2 style='font-size:13px'>{e(mode)} — {m['n_settled']} settled</h2>")
            parts.append('<div class="cards">' + "".join(
                f'<div class="card"><div class="k">{e(k)}</div><div class="v">{v}</div></div>'
                for k, v in [("P&L", signed(m["pnl"])), ("ROI", pct(m["roi"])),
                             ("Win rate", pct(m["win_rate"])),
                             ("95% CI", f'{pct(m["win_rate_ci"][0], 1)}–'
                                        f'{pct(m["win_rate_ci"][1], 1)}'),
                             ("Max DD", n(m["max_drawdown"])),
                             ("Volatility", n(m["volatility"])),
                             ("Worst streak", m["longest_losing_streak"]),
                             ("Beat close", pct(m["clv_beat_close"])),
                             ("Avg CLV", n(m["clv_avg"], 4))]) + "</div>")
            parts.append(_sparkline(curve))
            for gname, buckets in (summ.get("breakdowns") or {}).items():
                if not buckets or gname in ("timing",):
                    continue
                rows = [[e(k), str(v["n"]), signed(v["pnl"]), n(v["staked"]), pct(v["roi"]),
                         pct(v["win_rate"])] for k, v in sorted(buckets.items())]
                parts.append(f"<h2 style='font-size:12px'>by {e(gname)}</h2>")
                parts.append(_table(f"bd_{s['strategy_id']}_{s['version']}_{mode[:2]}_{gname}",
                                    [gname, "N", "P&L", "Staked", "ROI", "Win rate"], rows,
                                    numeric=(1, 2, 3, 4, 5)))
        parts.append("</details>")
    return _page("Performance", "performance.html", "".join(parts), gen)


def _sparkline(curve: Sequence[float], w: int = 720, h: int = 90) -> str:
    if len(curve) < 2:
        return '<p class="small">Not enough settled bets to draw an equity curve.</p>'
    lo, hi = min(curve), max(curve)
    rng = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(curve):
        x = i * w / (len(curve) - 1)
        y = h - (v - lo) / rng * (h - 8) - 4
        pts.append(f"{x:.1f},{y:.1f}")
    base_y = h - (curve[0] - lo) / rng * (h - 8) - 4
    return (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;background:var(--panel);'
            f'border:1px solid var(--bd);border-radius:8px">'
            f'<line x1="0" y1="{base_y:.1f}" x2="{w}" y2="{base_y:.1f}" stroke="#30363d"/>'
            f'<polyline fill="none" stroke="#58a6ff" stroke-width="1.5" points="{" ".join(pts)}"/>'
            f'</svg><p class="small">equity curve: {n(curve[0])} → {n(curve[-1])} '
            f'(min {n(lo)}, max {n(hi)})</p>')


def page_research(store: Store, gen: str) -> str:
    parts = ["<h2>Research lab</h2>",
             '<div class="note">The discovery engine asks one question mechanically: '
             '<i>what was knowable before the market moved, and would it have predicted the '
             'outcome?</i> It enumerates triggers, splits the data chronologically, and only '
             'promotes what survives the validation window. Rejections are recorded too.</div>']
    parts.append("<h2>Hypotheses</h2>")
    rows = [[e(h["hyp_id"]), e(h["question"]), e(h["status"]),
             "yes" if h["testable"] else "no",
             "yes" if h["data_available"] else "no", e(h["origin"])]
            for h in store.query("SELECT * FROM hypotheses ORDER BY hyp_id LIMIT 300")]
    parts.append(_table("hyp", ["ID", "Question", "Status", "Testable", "Data available", "Origin"],
                       rows))
    parts.append("<h2>Experiments</h2>")
    rows = [[e(x["exp_id"]), e(x["kind"]), e(x["strategy_id"]), str(x["version"]),
             e(x["verdict"]), e(x["conclusion"]), e(x["result_json"])[:200]]
            for x in store.query("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 200")]
    parts.append(_table("exp", ["ID", "Kind", "Strategy", "Ver", "Verdict", "Conclusion", "Result"],
                       rows))
    parts.append("<h2>Findings</h2>")
    for f in store.query("SELECT * FROM findings ORDER BY created_at DESC LIMIT 100"):
        pill = {"high": "good", "medium": "info", "low": "warn"}.get(f["confidence"], "info")
        parts.append(f"""<details><summary><span class="pill {pill}">{e(f['confidence'])}</span>
<span class="pill">{e(f['kind'])}</span> <b>{e(f['title'])}</b></summary>
<p>{e(f['body'])}</p><pre class="small">{e(f['evidence'])[:900]}</pre></details>""")
    return _page("Research Lab", "research.html", "".join(parts), gen)


def page_sources(store: Store, gen: str) -> str:
    parts = ["<h2>NHL data source registry</h2>",
             '<div class="note">A source is only marked <b>verified</b> after an automated HTTP '
             'probe is recorded in <code>source_verification</code>. "Free tier" is never treated '
             'as free: anything needing a key, a trial or paid credits is marked freemium or '
             'paid.</div>']
    rows = []
    for s in store.query("SELECT * FROM source_registry ORDER BY status, source_id"):
        pill = {"verified": "good", "rejected": "bad", "flagged": "warn"}.get(s["status"], "info")
        rows.append([
            f'<b>{e(s["source_id"])}</b><br><span class="small">{e(s["name"])}</span>',
            f'<a href="{e(s["url"])}">{e(s["url"])[:70]}</a>',
            f'<span class="pill {pill}">{e(s["status"])}</span>',
            e(s["data_type"]), e(s["nhl_relevance"]), e(s["historical_depth"]),
            "yes" if s["live_available"] else "no", e(s["update_frequency"]),
            "yes" if s["api_available"] else "no", e(s["auth_required"]), e(s["cost"]),
            e(s["genuinely_free"]), e(s["usage_limits"]), e(s["licensing"]), e(s["provenance"]),
            e(s["reliability"]), e(s["accuracy"]), e(s["granularity"]), e(s["automated_access"]),
            e(s["last_verified_at"]), e(s["known_limits"]), e(s["notes"])])
    parts.append(_table("reg", ["Source", "URL", "Status", "Data type", "NHL relevance",
                                "Historical depth", "Live", "Update freq", "API", "Auth", "Cost",
                                "Genuinely free", "Usage limits", "Licensing", "Provenance",
                                "Reliability", "Accuracy", "Granularity", "Automated access",
                                "Last verified", "Known limits", "Notes"], rows))
    parts.append("<h2>Verification log</h2>")
    rows = [[e(v["checked_at"]), e(v["source_id"]), e(v["method"]),
             f'<a href="{e(v["url_probed"])}">{e(v["url_probed"])[:80]}</a>',
             str(v["http_status"]),
             f'<span class="pill {"good" if v["ok"] else "bad"}">{e(v["verdict"])}</span>',
             e(v["evidence"])]
            for v in store.query("SELECT * FROM source_verification ORDER BY checked_at DESC")]
    parts.append(_table("vlog", ["Checked", "Source", "Method", "URL probed", "HTTP", "Verdict",
                                 "Evidence"], rows))
    return _page("Data Sources", "sources.html", "".join(parts), gen)


def page_verification(store: Store, gen: str) -> str:
    parts = ["<h2>Verification &amp; irregularities</h2>",
             '<div class="note">Nothing here is auto-corrected. Every row is a recorded problem '
             'plus its resolution. Conflicting sources are shown side by side rather than one '
             'being silently preferred.</div>']
    counts = {}
    for r in store.query("SELECT kind, severity, status, COUNT(*) c FROM irregularities "
                         "GROUP BY kind, severity, status"):
        counts[(r["kind"], r["severity"])] = counts.get((r["kind"], r["severity"]), 0) + r["c"]
    rows = [[e(k[0]), f'<span class="pill {"bad" if k[1] in ("error", "critical") else "warn"}">'
                     f'{e(k[1])}</span>', str(v)] for k, v in sorted(counts.items())]
    parts.append("<h2>Summary</h2>")
    parts.append(_table("isum", ["Kind", "Severity", "Count"], rows, numeric=(2,)))
    parts.append("<h2>Queue</h2>")
    parts.append('<div class="controls"><input placeholder="filter…" '
                 'oninput="filterTable(\'iq\', this.value)"></div>')
    rows = []
    for i in store.query("SELECT * FROM irregularities ORDER BY "
                         "CASE severity WHEN 'critical' THEN 0 WHEN 'error' THEN 1 "
                         "WHEN 'warn' THEN 2 ELSE 3 END, ts_utc DESC LIMIT 1000"):
        rows.append([str(i["id"]), e(i["ts_utc"]), e(i["kind"]),
                     f'<span class="pill {"bad" if i["severity"] in ("error", "critical") else "warn"}">'
                     f'{e(i["severity"])}</span>', e(i["entity_type"]), e(i["entity_id"]),
                     e(i["detail"]), e(i["sources"]), e(i["status"]), e(i["resolution"]),
                     "yes" if i["auto_corrected"] else "no"])
    parts.append(_table("iq", ["ID", "Raised", "Kind", "Severity", "Entity", "Entity ID",
                               "Detail", "Sources", "Status", "Resolution", "Auto-corrected"],
                       rows))
    return _page("Verification", "verification.html", "".join(parts), gen)


def page_methodology(store: Store, gen: str) -> str:
    tot = Performance(store).competition_totals()
    body = f"""
<h2>Methodology</h2>
<div class="note">This is a paper-trading research competition. <b>No real money is wagered and
this repository contains no order-placement code.</b></div>

<h2>1. What the system does</h2>
<p>It discovers, tests, forward-tests and paper-trades NHL betting strategies. Strategies are
generated by the system itself from a search over point-in-time features, not supplied as a fixed
list. Every wager is logged permanently with its price, its source and its verification status.</p>

<h2>2. Point-in-time discipline</h2>
<p>For a game at time <code>T</code>, features are built only from games that started strictly
before <code>T</code>. This is enforced in <code>features.FeatureBuilder._prior</code> and
re-checked by <code>Backtester.assert_no_leakage</code>, which flags any violation as a
<code>strategy_leakage</code> irregularity. Elo ratings are seeded from the <b>previous</b>
season's points percentage — never the current season's final standings.</p>

<h2>3. Backtest versus forward test</h2>
<p>The distinction is never blurred. A run is labelled <b>BACKTEST</b> only when the price it used
was genuinely quoted in the past. Because Kalshi's candle endpoint returned HTTP 404 and its
<code>previous_*</code> fields have no documented timestamp, price-based backtests carry
<code>verification_status=single_source_timing_unverified</code>. Everything priced from a live
quote is <b>FORWARD TEST</b> and settles later.</p>
<p>Where no price feed exists at all, the backtest reports <i>predictive accuracy only</i> —
hit rate, lift over base rate, log loss and Brier score — and stores
<code>data_sufficient=0</code> with this caveat:</p>
<div class="note caveat">{e(CAVEAT_NO_PRICE)}</div>

<h2>4. Execution model</h2>
<p>Buys happen at the <b>offer</b>, never the mid. Size is capped by the quoted
<code>yes_ask_size_fp</code>; if the desired stake needs more contracts than are offered the order
is partially filled and the remainder is recorded as unfilled. A missing or zero-size offer means
no bet. Slippage is stored as 0 because only one book level is available — that is a limitation of
the data, recorded rather than papered over.</p>

<h2>5. Staking</h2>
<p>Fractional Kelly, capped, with open exposure reserved from the bankroll so a strategy cannot
over-commit. Aggressive staking is permitted by design — the drawdown, volatility and
losing-streak columns exist so the risk is visible, not so it is suppressed.</p>

<h2>6. Statistics</h2>
<p>Win rate is always shown with a 95% Wilson interval and a sample size. The discovery engine
reports how many triggers it evaluated, so a survivor of a large search is not mistaken for a
discovery. Models are compared against a home-ice constant baseline on a held-out window; a model
that does not beat the baseline is not kept.</p>

<h2>7. Auditability</h2>
<p>Every fetched payload is stored with its URL, retrieval time and SHA-256 digest in
<code>raw_response</code>. Every row carries a provenance label: SOURCE, DERIVED, MODEL,
ASSUMPTION or UNVERIFIED. Bets are append-only; corrections go through
<code>amend_bet</code>, which writes a before/after audit row.</p>

<h2>8. Current state</h2>
<p>{tot['strategies']} strategies · {tot['bets']} simulated bets · {tot['settled']} settled ·
{tot['open']} open · P&amp;L {n(tot['pnl'])} · ROI {pct(tot['roi'])}.</p>

<h2>9. Known limitations</h2>
<ul>
<li>No verified historical sportsbook odds. Closing-line value is measured against Kalshi only.</li>
<li>Arena latitude/longitude is unavailable: the legacy NHL venue host could not be reached, so
travel is derived from published venue UTC offsets rather than invented coordinates.</li>
<li>Goalie announcements, line combinations and EDGE tracking are not ingested at scale, so
goaltending, line-matchup and tracking strategies are hypotheses awaiting forward tests rather
than tested results.</li>
<li>Preseason Kalshi markets are thin; several contracts show zero size on both sides and are
recorded as <code>insufficient_liquidity</code> instead of being traded.</li>
<li>No intraday price history, so no live/in-game or line-movement backtest is attempted.</li>
</ul>
"""
    return _page("Methodology", "methodology.html", body, gen)


# --------------------------------------------------------------------- build
def build_site(store: Store, outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "assets"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    gen = utcnow()
    perf = Performance(store)

    pages = {
        "index.html": lambda: page_dashboard(store, perf, gen),
        "leaderboard.html": lambda: page_leaderboard(store, perf, gen),
        "strategies.html": lambda: page_strategies(store, perf, gen),
        "upcoming.html": lambda: page_upcoming(store, gen),
        "positions.html": lambda: page_positions(store, gen),
        "history.html": lambda: page_history(store, gen),
        "performance.html": lambda: page_performance(store, perf, gen),
        "research.html": lambda: page_research(store, gen),
        "sources.html": lambda: page_sources(store, gen),
        "verification.html": lambda: page_verification(store, gen),
        "methodology.html": lambda: page_methodology(store, gen),
    }
    n = 0
    for name, fn in pages.items():
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as fh:
            fh.write(fn())
        n += 1

    with open(os.path.join(outdir, "assets", "style.css"), "w", encoding="utf-8") as fh:
        fh.write(CSS)
    with open(os.path.join(outdir, "assets", "app.js"), "w", encoding="utf-8") as fh:
        fh.write(JS)
    n += 2

    # JSON mirrors so the data is machine-readable too
    dumps = {
        "leaderboard.json": perf.leaderboard(),
        "competition.json": perf.competition_totals(),
        "sources.json": [dict(r) for r in store.query("SELECT * FROM source_registry")],
        "strategies.json": [dict(r) for r in store.query("SELECT * FROM strategies")],
        "bets.json": [dict(r) for r in store.query(
            "SELECT * FROM bets ORDER BY bet_ts DESC LIMIT 5000")],
        "upcoming.json": [dict(r) for r in store.query("SELECT * FROM upcoming_bets")],
        "irregularities.json": [dict(r) for r in store.query(
            "SELECT * FROM irregularities ORDER BY ts_utc DESC LIMIT 2000")],
        "findings.json": [dict(r) for r in store.query("SELECT * FROM findings")],
        "backtests.json": [dict(r) for r in store.query("SELECT * FROM backtests")],
    }
    for name, payload in dumps.items():
        with open(os.path.join(outdir, "data", name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
        n += 1
    with open(os.path.join(outdir, ".nojekyll"), "w", encoding="utf-8") as fh:
        fh.write("")
    return n + 1
