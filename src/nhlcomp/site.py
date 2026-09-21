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

from . import mastersite
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
// Multi-field filtering: strategy, market, team, player, goalie, season, month, metrics
const activeFilters = {};
function applyFilters(id){
  const el=document.getElementById(id);
  if(!el) return;
  const f = activeFilters[id] || {};
  if(el.tagName==='TABLE'){
    for(const r of el.tBodies[0].rows){
      let show = true;
      if(f.q){
        let hay='';
        for(let i=0;i<r.cells.length;i++) hay+=' '+(r.cells[i].textContent||'');
        if(!hay.toLowerCase().includes(f.q.toLowerCase())) show=false;
      }
      for(const k of ['strategy','market','team','player','goalie','season','month','category','status','mode']){
        if(f[k] && f[k]!==''){
          const v = (r.dataset[k]||'').toLowerCase();
          if(!v.includes(f[k].toLowerCase())) show=false;
        }
      }
      if(f.min_roi!==undefined && f.min_roi!==''){
        const roi = parseFloat(r.dataset.roi||'');
        if(!isNaN(roi) && roi < parseFloat(f.min_roi)) show=false;
      }
      if(f.min_pnl!==undefined && f.min_pnl!==''){
        const pnl = parseFloat(r.dataset.pnl||'');
        if(!isNaN(pnl) && pnl < parseFloat(f.min_pnl)) show=false;
      }
      if(f.min_edge!==undefined && f.min_edge!==''){
        const ed = parseFloat(r.dataset.edge||'');
        if(!isNaN(ed) && ed < parseFloat(f.min_edge)) show=false;
      }
      r.style.display = show?'':'none';
    }
    return;
  }
  for(const d of el.querySelectorAll('details')){
    let show = true;
    if(f.q){
      const hay = (d.textContent||'').toLowerCase();
      if(!hay.includes(f.q.toLowerCase())) show=false;
    }
    for(const k of ['strategy','market','team','player','goalie','season','month','category','status','mode']){
      if(f[k] && f[k]!==''){
        const v = (d.dataset[k]||'').toLowerCase();
        const hay2 = (d.textContent||'').toLowerCase();
        if(!v.includes(f[k].toLowerCase()) && !hay2.includes(f[k].toLowerCase())) show=false;
      }
    }
    d.style.display = show?'':'none';
  }
}
function setFilter(id, key, val){
  if(!activeFilters[id]) activeFilters[id]={};
  activeFilters[id][key]=val;
  applyFilters(id);
}
function setTextFilter(id, val){
  if(!activeFilters[id]) activeFilters[id]={};
  activeFilters[id].q=val;
  applyFilters(id);
}
function clearFilters(id){
  activeFilters[id]={};
  const c=document.getElementById(id+'_controls');
  if(c){
    for(const el of c.querySelectorAll('input,select')) el.value='';
  }
  applyFilters(id);
}
function filterCategory(id, v){
  setFilter(id,'category',v);
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
    tot = perf.competition_totals(test_mode="FORWARD TEST")
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
            '<p class="small">The headline numbers are <b>FORWARD TEST</b> only: paper bets opened '
            'against a live Kalshi quote before puck drop. Backtests are shown separately below and '
            'never added in.</p>',
            '<div class="cards">' + "".join(
                f'<div class="card"><div class="k">{e(k)}</div>'
                f'<div class="v">{v}</div></div>' for k, v in cards) + "</div>"]

    # BACKTEST and FORWARD TEST are never merged: show them side by side
    modes = []
    for mode in ("BACKTEST", "FORWARD TEST"):
        r = store.one(
            """SELECT COUNT(*) c, SUM(CASE WHEN result IN ('WIN','LOSS','PUSH') THEN 1 ELSE 0 END) settled,
                      SUM(CASE WHEN result='OPEN' THEN 1 ELSE 0 END) open_n,
                      COALESCE(SUM(CASE WHEN result IN ('WIN','LOSS','PUSH') THEN pnl END),0) pnl,
                      COALESCE(SUM(CASE WHEN result IN ('WIN','LOSS','PUSH') THEN stake END),0) staked,
                      COALESCE(SUM(fee),0) fees,
                      SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) wins
                 FROM bets WHERE test_mode=?""", (mode,))
        settled = int(r["settled"] or 0)
        modes.append([e(mode), str(r["c"]), str(settled), str(r["open_n"] or 0),
                      signed(r["pnl"]), n(r["staked"]), pct((r["pnl"] / r["staked"]) if r["staked"] else None),
                      pct((r["wins"] / settled) if settled else None), n(r["fees"])])
    body.append("<h2>Backtest vs forward test (never merged)</h2>")
    body.append(_table("modes", ["Mode", "Bets", "Settled", "Open", "P&L", "Staked", "ROI",
                                 "Win rate", "Fees paid"], modes, numeric=(1, 2, 3, 4, 5, 6, 7, 8)))
    body.append('<p class="small">BACKTEST rows are priced from the recovered Kalshi candle at '
                'or before puck drop (or, for a few early rows, an undocumented pre-settlement '
                'quote — see the verification column). FORWARD TEST rows were opened against a '
                'live quote before the game and settle from the official result. Kalshi\'s '
                'published taker fee is deducted from both.</p>')

    # per market, per mode: a totals result and a moneyline result are different questions
    # and are never netted against each other
    mrows = []
    for mode in ("BACKTEST", "FORWARD TEST"):
        for r in store.query(
                """SELECT market,
                          COUNT(*) c,
                          SUM(CASE WHEN result IN ('WIN','LOSS','PUSH') THEN 1 ELSE 0 END) settled,
                          SUM(CASE WHEN result='OPEN' THEN 1 ELSE 0 END) open_n,
                          COALESCE(SUM(CASE WHEN result IN ('WIN','LOSS','PUSH') THEN pnl END),0) pnl,
                          COALESCE(SUM(CASE WHEN result IN ('WIN','LOSS','PUSH') THEN stake END),0) staked,
                          SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) wins
                     FROM bets WHERE test_mode=? GROUP BY market ORDER BY c DESC""", (mode,)):
            settled = int(r["settled"] or 0)
            mrows.append([e(mode), e(r["market"] or "?"), str(r["c"]), str(settled),
                          str(r["open_n"] or 0), signed(r["pnl"]), n(r["staked"]),
                          pct((r["pnl"] / r["staked"]) if r["staked"] else None),
                          pct((r["wins"] / settled) if settled else None)])
    if mrows:
        body.append("<h2>By market (moneyline and totals are reported apart)</h2>")
        body.append(_table("markets", ["Mode", "Market", "Bets", "Settled", "Open", "P&L",
                                       "Staked", "ROI", "Win rate"], mrows,
                           numeric=(2, 3, 4, 5, 6, 7, 8)))
        body.append('<p class="small">A totals wager settles on the official total '
                    '(home + away, which already includes the shootout goal exactly as Kalshi\'s '
                    'KXNHLTOTAL rules define it). An Over is the YES side of an "Over k.5" '
                    'contract; an Under is the NO side of the same contract.</p>')

    # how much real price history backs the backtests
    cov = store.one(
        """SELECT COUNT(*) contracts,
                  SUM(CASE WHEN candles_state='ok' THEN 1 ELSE 0 END) with_close,
                  SUM(CASE WHEN game_id IS NULL THEN 1 ELSE 0 END) unmatched,
                  MIN(game_date) first_game, MAX(game_date) last_game
             FROM market_settlements WHERE provider='kalshi' AND result IN ('yes','no')""")
    games_priced = store.one(
        "SELECT COUNT(DISTINCT game_id) c FROM market_price_points WHERE point='close' "
        "AND game_id IS NOT NULL")["c"]
    pts = store.one("SELECT COUNT(*) c FROM market_price_points")["c"]
    stats = store.one("SELECT COUNT(*) c, COUNT(DISTINCT game_id) g FROM team_game_stats")
    goalies = store.one("SELECT COUNT(*) c, COUNT(DISTINCT game_id) g FROM goalie_game_stats")
    odds = store.one("SELECT COUNT(*) c, COUNT(DISTINCT game_id) g FROM odds_snapshots")
    edge = store.one("SELECT COUNT(*) c, COUNT(DISTINCT substr(retrieved_at,1,10)) d FROM edge_team_snapshots")
    cards2 = [
        ("Settled Kalshi contracts", cov["contracts"] or 0),
        ("… with a pre-game candle", cov["with_close"] or 0),
        ("Games with a closing price", games_priced),
        ("Price points stored", pts),
        ("Contract window", f'{e(cov["first_game"] or "—")} → {e(cov["last_game"] or "—")}'),
        ("Team-game stat rows", f'{stats["c"]} ({stats["g"]} games)'),
        ("Goalie-game rows", f'{goalies["c"]} ({goalies["g"]} games)'),
        ("Sportsbook odds snapshots", f'{odds["c"]} ({odds["g"]} games)'),
        ("EDGE team snapshots", f'{edge["c"]} ({edge["d"]} days)'),
    ]
    body.append("<h2>Data behind the tests</h2>")
    body.append('<div class="cards">' + "".join(
        f'<div class="card"><div class="k">{e(k)}</div><div class="v" style="font-size:18px">{v}</div></div>'
        for k, v in cards2) + "</div>")
    mb = store.one("SELECT evidence FROM findings WHERE finding_id='FIND_MARKET_BASELINE'")
    if mb:
        try:
            base = json.loads(mb["evidence"])
            rows = [[e(side), str(v.get("n")), pct(v.get("roi")), n(v.get("avg_price"), 3),
                     pct(v.get("hit"))] for side, v in base.items()]
            body.append("<h3>Market baseline (buy every contract at the close, net of fees)</h3>")
            body.append(_table("mb", ["Side", "Games", "ROI", "Avg price", "Hit rate"], rows,
                               numeric=(1, 2, 3, 4)))
            body.append('<p class="small">A priced strategy has to beat this number, not zero: it '
                        'is what the spread and the taker fee cost a blind buyer.</p>')
        except (ValueError, TypeError):
            pass

    body.append('<div class="note">This is a <b>paper-trading research competition</b>. No real '
                'money is wagered and no order-placement code exists in this repository. Prices '
                'come from the Kalshi public market-data API and results from the NHL API.</div>')

    for mode, title in (("FORWARD TEST", "Top strategies — forward test (live paper trading)"),
                        ("BACKTEST", "Top strategies — backtest (Kalshi closing candles)")):
        top = [r for r in perf.leaderboard(test_mode=mode) if r["n_settled"]][:10]
        rows = [[str(i), e(r["username"]), e(r["category"]), str(r["n_settled"]),
                 signed(r["pnl"]), pct(r["roi"]), pct(r["win_rate"]),
                 f'{pct(r["win_rate_ci"][0], 1)}–{pct(r["win_rate_ci"][1], 1)}',
                 n(r["max_drawdown"]), str(r["open"])] for i, r in enumerate(top, 1)]
        body.append(f"<h2>{e(title)}</h2>")
        if rows:
            body.append(_table(f"lb_{mode[:4].lower()}", ["#", "Username", "Category", "Settled", "P&L",
                                                            "ROI", "Win rate", "Win-rate 95% CI",
                                                            "Max DD", "Open"], rows,
                               numeric=(0, 3, 4, 5, 6, 8, 9)))
        else:
            body.append('<p class="small">No settled wagers in this mode yet.</p>')

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
    body.append('<div class="note caveat"><b>How to read the P&amp;L:</b> BACKTEST rows use the '
                'Kalshi candlestick history (<code>verification_status=kalshi_candle_close</code>: '
                'entry at the last hourly candle ending at or before the scheduled start, with '
                'its timestamp). A handful of early rows were priced from Kalshi\'s undocumented '
                '<code>previous_yes_ask</code> and remain labelled '
                '<code>single_source_timing_unverified</code>; they are never relabelled. '
                'FORWARD TEST rows are opened only against a live quote and settle from the '
                'official NHL result. No sportsbook history exists in this system, so nothing '
                'is backtested against sportsbook odds; the DraftKings line from the NHL partner '
                'feed is recorded as a reference for forward signals only.</div>')
    return _page("Dashboard", "index.html", "".join(body), gen)


def page_leaderboard(store: Store, perf: Performance, gen: str) -> str:
    body = ["<h2>Leaderboard</h2>",
            '<p class="small">Two boards, never merged. <b>Forward test</b> is the competition: '
            'paper bets opened against a live Kalshi quote before puck drop, settled from the '
            'official result. <b>Backtest</b> is the same rules replayed against the recovered '
            'Kalshi closing candles (flat historical evidence, not live performance). Win rate is '
            'always paired with its 95% Wilson interval — a 3-bet sample proves nothing. Click any '
            'header to sort; click a username for its wagers. Filters combine with AND: '
            'strategy, market, team, player, goalie, season, month, metrics.</p>']
    for mode, tid, title in (("FORWARD TEST", "lb", "Forward test (live paper trading)"),
                             ("BACKTEST", "lbb", "Backtest (Kalshi closing candles, net of fees)")):
        rows_data = perf.leaderboard(test_mode=mode)
        cats = sorted({r["category"] for r in rows_data})
        markets = sorted({(r.get("markets") or r.get("market") or "") for r in rows_data if (r.get("markets") or r.get("market"))})
        body.append(f"<h2>{e(title)}</h2>")
        body.append(
            f"<div class=\"controls\" id=\"{tid}_controls\">"
            f"<input placeholder=\"search username...\" oninput=\"setTextFilter('{tid}', this.value)\">"
            f"<select onchange=\"setFilter('{tid}','category',this.value)\"><option value=\"\">all categories</option>"
            + "".join(f'<option value="{e(c)}">{e(c)}</option>' for c in cats) + "</select>"
            f"<select onchange=\"setFilter('{tid}','market',this.value)\"><option value=\"\">all markets</option>"
            + "".join(f'<option value="{e(m)}">{e(m)}</option>' for m in markets[:20]) + "</select>"
            f"<select onchange=\"setFilter('{tid}','status',this.value)\"><option value=\"\">all statuses</option>"
            f"<option value=\"active\">active</option><option value=\"retired\">retired</option></select>"
            f"<input placeholder=\"min ROI (e.g. -0.1)\" style=\"width:130px\" oninput=\"setFilter('{tid}','min_roi',this.value)\">"
            f"<input placeholder=\"min P&L\" style=\"width:100px\" oninput=\"setFilter('{tid}','min_pnl',this.value)\">"
            f"<button onclick=\"clearFilters('{tid}')\">clear</button>"
            f"</div>")
        rows_html = []
        for r in rows_data:
            cat = (r["category"] or "").lower()
            mkt = (r.get("markets") or r.get("market") or "").lower()
            strat = (r["strategy_id"] or "").lower()
            status = (r["status"] or "").lower()
            pnl = r["pnl"] or 0
            roi = r["roi"] or 0
            rows_html.append(
                f'<tr data-category="{e(cat)}" data-market="{e(mkt)}" data-strategy="{e(strat)}" '
                f'data-status="{e(status)}" data-pnl="{pnl}" data-roi="{roi}">'
                f'<td class="num">{r["rank"]}</td>'
                f'<td><a href="strategies.html#{e(r["strategy_id"])}">{e(r["username"])}</a></td>'
                f'<td>{e(r["category"])}</td><td>{e(r["status"])}</td>'
                f'<td class="num">{r["n_bets"]}</td><td class="num">{r["n_settled"]}</td>'
                f'<td class="num">{r["wins"]}</td><td class="num">{r["losses"]}</td><td class="num">{r["pushes"]}</td><td class="num">{r["open"]}</td>'
                f'<td class="num">{signed(r["pnl"])}</td><td class="num">{n(r["staked"])}</td><td class="num">{pct(r["roi"])}</td><td class="num">{pct(r["win_rate"])}</td>'
                f'<td class="num">{pct(r["win_rate_ci"][0],1)}–{pct(r["win_rate_ci"][1],1)}</td>'
                f'<td class="num">{n(r["avg_price"],3)}</td><td class="num">{n(r["avg_edge"],4)}</td><td class="num">{n(r["max_drawdown"])}</td>'
                f'<td class="num">{n(r["volatility"])}</td><td class="num">{r["longest_losing_streak"]}</td>'
                f'<td class="num">{pct(r["clv_beat_close"])}</td><td class="num">{n(r["clv_avg"],4)}</td><td class="num">{r["n_clv"]}</td>'
                f'<td class="num">{n(r["starting_bankroll"],0)}</td><td class="num">{n(r["bankroll"])}</td>'
                f'</tr>')
        ths = "".join(
            f'<th class="{"num" if i not in (1,2,3) else ""}" onclick="sortTable(\'{tid}\',{i},\'{"n" if i not in (1,2,3) else "s"}\')">{e(h)}</th>'
            for i, h in enumerate(["#", "Username", "Category", "Status", "Bets", "Settled", "W", "L", "Push", "Open", "P&L", "Staked", "ROI", "Win rate", "95% CI", "Avg price", "Avg edge", "Max DD", "Volatility", "Worst streak", "Beat close", "Avg CLV", "n CLV", "Start bank", "Bankroll"]))
        body.append(f'<div class="wrap"><table id="{tid}"><thead><tr>{ths}</tr></thead><tbody>{"".join(rows_html) or "<tr><td colspan=25>No rows</td></tr>"}</tbody></table></div>')
        if mode == "BACKTEST":
            body.append('<p class="small">Backtest bankroll columns are informational: BACKTEST P&amp;L never funds a forward stake.</p>')
    return _page("Leaderboard", "leaderboard.html", "".join(body), gen)



def page_strategies(store: Store, perf: Performance, gen: str) -> str:
    parts = ["<h2>Strategies</h2>",
             '<p class="small">Strategies are never overwritten — every change creates a new '
             'version so an edit can be judged on whether it actually helped. Filters: strategy, market, team, player, goalie, season, month, metrics.</p>',
             '<div class="controls" id="strat_controls">'
             '<input placeholder="search strategies..." oninput="setTextFilter(\'strat_list\', this.value)" style="width:260px">'
             '<select onchange="setFilter(\'strat_list\',\'category\',this.value)"><option value="">all categories</option>'
             '<option value="fatigue">fatigue</option><option value="goaltending">goaltending</option><option value="special_teams">special_teams</option>'
             '<option value="totals">totals</option><option value="puck_line">puck_line</option><option value="ot_shootout">ot_shootout</option>'
             '<option value="period">period</option><option value="team_totals">team_totals</option><option value="regulation">regulation</option>'
             '<option value="player_props">player_props</option><option value="goalie_props">goalie_props</option><option value="futures">futures</option>'
             '<option value="alternate_lines">alternate_lines</option><option value="live_in_game">live</option><option value="market">market</option>'
             '</select>'
             '<select onchange="setFilter(\'strat_list\',\'market\',this.value)"><option value="">all markets</option>'
             '<option value="moneyline">moneyline</option><option value="total">total</option><option value="puck_line">puck_line</option>'
             '<option value="overtime">overtime</option><option value="period">period</option><option value="team_total">team_total</option>'
             '<option value="regulation">regulation</option><option value="player">player</option><option value="goalie">goalie</option>'
             '<option value="futures">futures</option><option value="alternate">alternate</option></select>'
             '<button onclick="clearFilters(\'strat_list\')">clear</button>'
             '</div>',
             '<div id="strat_list">']
    for s in store.query("SELECT * FROM strategies ORDER BY strategy_id, version"):
        key = f"{s['strategy_id']}_v{s['version']}"
        summ = perf.summarize(s["strategy_id"], int(s["version"]))
        m = summ.get("ALL", {})
        f = summ.get("FORWARD TEST", {})
        b = summ.get("BACKTEST", {})
        pill = {"active": "good", "candidate": "info", "paused": "warn",
                "retired": "bad", "rejected": "bad"}.get(s["status"], "info")
        parts.append(f"""<details id="{e(s['strategy_id'])}" data-category="{e(s['category'])}" data-market="{e(s['markets'])}" data-strategy="{e(s['strategy_id'])}"><summary>
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
        acc = [x for x in bt if not (x["label"] or "").startswith("priced")]
        priced = [x for x in bt if (x["label"] or "").startswith("priced")]
        if acc:
            rows = [[e(x["label"]), str(x["n_bets"]), str(x["n_wins"]), str(x["n_losses"]),
                     pct(_ratio(x["n_wins"], x["n_bets"])), e(x["test_from"]), e(x["test_to"]),
                     e("yes" if x["data_sufficient"] else "no"), e(x["caveat"])[:300]]
                    for x in acc]
            parts.append("<h2 style='font-size:13px'>Accuracy backtest (all decided games, no prices)</h2>")
            parts.append(_table(f"bt_{key}", ["Window", "N", "W", "L", "Hit rate", "From", "To",
                                              "Price data OK?", "Caveat"], rows,
                                numeric=(1, 2, 3, 4)))
        if priced:
            order = {"priced_all": 0, "priced_train": 1, "priced_valid": 2, "priced_test": 3}
            priced.sort(key=lambda x: order.get(x["label"], 9))
            rows = [[e(x["label"]), str(x["n_games"] or 0), str(x["n_bets"]), pct(x["hit_rate"]),
                     pct(x["base_rate"]), n(x["avg_price"], 3), n(x["avg_edge"], 4),
                     signed(x["pnl"]), pct(x["roi"]), n(x["max_drawdown"]), n(x["sharpe"], 2),
                     e(x["test_from"]), e(x["test_to"]), e(x["price_basis"])] for x in priced]
            parts.append("<h2 style='font-size:13px'>Priced backtest (Kalshi closing candle, "
                         "flat 1-unit stakes, net of taker fee)</h2>")
            parts.append(_table(f"pbt_{key}", ["Window", "Priced games", "Bets", "Hit rate",
                                               "Base rate", "Avg price", "Avg edge", "P&L (units)",
                                               "ROI", "Max DD", "Sharpe", "From", "To", "Price basis"],
                                rows, numeric=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))
            parts.append('<p class="small">train / valid / test are consecutive chronological '
                         'windows. A rule is only interesting if the sign of the ROI survives '
                         'out of sample; the Research Lab lists the market baseline it must beat.</p>')
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
    parts.append("</div>")
    return _page("Strategies", "strategies.html", "".join(parts), gen)


def _ratio(a: Any, b: Any) -> float | None:
    try:
        return float(a) / float(b) if b else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _bets_table(id_: str, bets: Sequence[Any]) -> str:
    rows_html = []
    for b in bets:
        try:
            game_date = str(b["game_date"] or "")
            season = game_date[:4] if len(game_date)>=4 else ""
            month = game_date[5:7] if len(game_date)>=7 else ""
        except Exception:
            game_date = ""
            season = ""
            month = ""
        market = (b["market"] or "").lower()
        matchup = (b["matchup"] or "").lower()
        team = matchup
        strategy = (b["strategy_id"] or "").lower()
        mode = (b["test_mode"] or "").lower()
        player = ""
        goalie = ""
        pnl = b["pnl"] or 0
        roi = b["roi"] or 0
        edge = b["edge"] or 0
        price = b["price"] or 0
        rows_html.append(
            f'<tr data-strategy="{e(strategy)}" data-market="{e(market)}" data-team="{e(team)}" '
            f'data-player="{e(player)}" data-goalie="{e(goalie)}" data-season="{e(season)}" data-month="{e(month)}" '
            f'data-mode="{e(mode)}" data-pnl="{pnl}" data-roi="{roi}" data-edge="{edge}" data-price="{price}">'
            f'<td>{e(b["bet_id"])[:52]}</td>'
            f'<td>{e(b["test_mode"])}</td><td>{e(b["game_date"])}</td><td>{e(b["matchup"])}</td>'
            f'<td>{e(b["market"])}</td><td>{e(b["selection"])}</td><td>{e(b["provider"])}</td>'
            f'<td class="num">{n(b["price"],3)}</td>'
            f'<td>{e(_col(b, "price_point") or "")}</td>'
            f'<td class="num">{n(b["model_prob"],4)}</td><td class="num">{n(b["edge"],4)}</td>'
            f'<td class="num">{n(b["stake"])}</td><td class="num">{n(b["filled_size"],2)}</td><td class="num">{n(b["liquidity"],0)}</td>'
            f'<td class="num">{n(_col(b, "fee"),2)}</td><td>{e(b["result"])}</td><td class="num">{signed(b["pnl"])}</td><td class="num">{pct(b["roi"])}</td>'
            f'<td class="num">{n(b["close_price"],3)}</td><td class="num">{n(b["clv"],4)}</td><td>{e(b["verification_status"])}</td>'
            f'<td>{e(b["decision_ts"])[:19]}</td><td>{e(b["bet_ts"])}</td>'
            f'</tr>')
    headers = ["Bet ID", "Mode", "Game", "Matchup", "Market", "Selection", "Provider",
               "Price", "Price point", "Model p", "Edge", "Stake", "Contracts", "Liquidity",
               "Fee", "Result", "P&L", "ROI", "Close", "CLV", "Verification",
               "Decision ts", "Placed"]
    ths = "".join(f'<th class="{"num" if i in (7,9,10,11,12,13,14,16,17,18,19) else ""}" onclick="sortTable(\'{id_}\',{i},\'{"n" if i in (7,9,10,11,12,13,14,16,17,18,19) else "s"}\')">{e(h)}</th>' for i,h in enumerate(headers))
    return f'<div class="wrap"><table id="{id_}"><thead><tr>{ths}</tr></thead><tbody>{"".join(rows_html) or f"<tr><td colspan={len(headers)}>No rows yet.</td></tr>"}</tbody></table></div>' 


def _col(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def page_upcoming(store: Store, gen: str) -> str:
    body = ["<h2>Upcoming strategy bets</h2>",
            '<p class="small">Every signal a strategy currently wants to act on, including the '
            'ones it is blocked on. A status of PRICE TOO HIGH means the model likes the side '
            'but the offer is not good enough — the strategy waits instead of chasing. '
            'Filters: strategy, market, team, player, goalie, season, month, metrics (combine with AND).</p>']
    body.append(
        '<div class="controls" id="up_controls">'
        '<input placeholder="search..." oninput="setTextFilter(\'up\', this.value)">'
        '<select onchange="setFilter(\'up\',\'market\',this.value)"><option value="">all markets</option>'
        '<option value="moneyline">moneyline</option><option value="total">total</option><option value="puck_line">puck_line</option>'
        '<option value="overtime">overtime</option><option value="period">period</option><option value="team_total">team_total</option>'
        '<option value="regulation">regulation</option><option value="player">player</option><option value="goalie">goalie</option>'
        '<option value="futures">futures</option><option value="alternate">alternate</option></select>'
        '<select onchange="setFilter(\'up\',\'status\',this.value)"><option value="">all statuses</option>'
        '<option value="READY TO BET">READY TO BET</option><option value="PRICE TOO HIGH">PRICE TOO HIGH</option>'
        '<option value="WATCHING">WATCHING</option><option value="WAITING FOR">WAITING</option></select>'
        '<input placeholder="team id or matchup" style="width:140px" oninput="setFilter(\'up\',\'team\',this.value)">'
        '<input placeholder="season YYYY" style="width:110px" oninput="setFilter(\'up\',\'season\',this.value)">'
        '<input placeholder="month MM" style="width:90px" oninput="setFilter(\'up\',\'month\',this.value)">'
        '<input placeholder="min edge" style="width:90px" oninput="setFilter(\'up\',\'min_edge\',this.value)">'
        '<button onclick="clearFilters(\'up\')">clear</button>'
        '</div>')
    latest = {int(r["version"]): r["strategy_id"] for r in store.query(
        "SELECT strategy_id, MAX(version) version FROM strategies GROUP BY strategy_id")}
    status_of = {(r["strategy_id"], int(r["version"])): r["status"] for r in store.query(
        "SELECT strategy_id, version, status FROM strategies")}
    rows_html = []
    for u in store.query("SELECT * FROM upcoming_bets ORDER BY decision_ts DESC LIMIT 1000"):
        pill = {"READY TO BET": "good", "EXECUTED": "good", "PRICE TOO HIGH": "warn",
                "PRICE TOO LOW": "warn", "WATCHING": "info", "CANCELLED": "bad",
                "EXPIRED": "bad"}.get(u["status"], "info")
        sid, ver = u["strategy_id"], int(u["strategy_version"] or 1)
        superseded = latest.get(sid) not in (None, ver)
        vstat = status_of.get((sid, ver), "?")
        version_cell = (f'v{ver} <span class="pill {"bad" if superseded else "info"}">'
                        f'{"superseded" if superseded else e(vstat)}</span>')
        gd = str(u["game_date"] or "")
        season = gd[:4] if len(gd)>=4 else ""
        month = gd[5:7] if len(gd)>=7 else ""
        market = (u["market"] or "").lower()
        matchup = (u["matchup"] or "").lower()
        strategy = (u["strategy_id"] or "").lower()
        status = (u["status"] or "").lower()
        edge = u["edge"] or 0
        rows_html.append(
            f'<tr data-strategy="{e(strategy)}" data-market="{e(market)}" data-team="{e(matchup)}" '
            f'data-season="{e(season)}" data-month="{e(month)}" data-status="{e(status)}" data-edge="{edge}">'
            f'<td>{e(u["username"])}</td><td>{version_cell}</td><td>{e(u["game_date"])}</td><td>{e(u["matchup"])}</td>'
            f'<td>{e(u["market"])}</td><td>{e(u["selection"])}</td><td>{e(u["provider"])}</td>'
            f'<td class="num">{n(u["current_price"],3)}</td>'
            f'<td class="num">{n(u["required_price"],3)}</td><td class="num">{n(u["model_prob"],4)}</td><td class="num">{n(u["edge"],4)}</td>'
            f'<td class="num">{n(u["stake"])}</td><td><span class="pill {pill}">{e(u["status"])}</span></td>'
            f'<td>{e(u["blocking_reason"])}</td><td>{e(u["supporting_data"])[:300]}</td><td>{e(u["decision_ts"])}</td>'
            f'</tr>')
    headers = ["Username", "Version", "Game", "Matchup", "Market", "Selection",
               "Provider", "Current price", "Required price", "Model p", "Edge",
               "Stake", "Status", "Blocking reason", "Supporting data", "Decision ts"]
    ths = "".join(f'<th class="{"num" if i in (7,8,9,10,11) else ""}" onclick="sortTable(\'up\',{i},\'{"n" if i in (7,8,9,10,11) else "s"}\')">{e(h)}</th>' for i,h in enumerate(headers))
    body.append(f'<div class="wrap"><table id="up"><thead><tr>{ths}</tr></thead><tbody>{"".join(rows_html) or "<tr><td colspan=16>No signals</td></tr>"}</tbody></table></div>')
    body.append('<p class="small">Signals are never deleted. A row marked <b>superseded</b> was '
                'written by an older version of that strategy; it stays in the record with the '
                'reason it carried at the time, and only the latest version trades.</p>')
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
            'written as amendments with a before/after audit row. Filters combine with AND: '
            'strategy, market, team, player, goalie, season, month, metrics.</p>',
            '<div class="controls" id="th_controls">'
            '<input placeholder="search..." oninput="setTextFilter(\'th\', this.value)">'
            '<select onchange="setFilter(\'th\',\'market\',this.value)"><option value="">all markets</option>'
            '<option value="moneyline">moneyline</option><option value="total">total</option><option value="puck_line">puck_line</option>'
            '<option value="overtime">overtime</option><option value="period">period</option><option value="team_total">team_total</option>'
            '<option value="regulation">regulation</option><option value="player">player</option><option value="goalie">goalie</option>'
            '<option value="futures">futures</option><option value="alternate">alternate</option></select>'
            '<select onchange="setFilter(\'th\',\'mode\',this.value)"><option value="">all modes</option><option value="BACKTEST">BACKTEST</option><option value="FORWARD">FORWARD</option></select>'
            '<input placeholder="team/matchup" style="width:130px" oninput="setFilter(\'th\',\'team\',this.value)">'
            '<input placeholder="strategy id" style="width:130px" oninput="setFilter(\'th\',\'strategy\',this.value)">'
            '<input placeholder="season YYYY" style="width:100px" oninput="setFilter(\'th\',\'season\',this.value)">'
            '<input placeholder="month MM" style="width:80px" oninput="setFilter(\'th\',\'month\',this.value)">'
            '<input placeholder="min ROI" style="width:90px" oninput="setFilter(\'th\',\'min_roi\',this.value)">'
            '<button onclick="clearFilters(\'th\')">clear</button>'
            '</div>']
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
    parts.append("<h2>Master site review</h2>")
    parts.append(
        f'<p class="small">The owner\'s directory of verified GitHub Pages sites '
        f'(<a href="{e(mastersite.MASTER_SITE_URL)}">MasterSite</a>) was reviewed on '
        f'{e(mastersite.REVIEW_DATE)}: the index was fetched, every named project was checked '
        f'against the GitHub API, READMEs were read, and the one data claim that mattered was '
        f'tested against the live endpoint before anything was reused.</p>')
    rows = [[e(r["project"]), e(r["exists"]), e(r["relevance"]), e(r["verdict"]), e(r["evidence"])]
            for r in mastersite.REVIEW]
    parts.append(_table("msite", ["Project", "Exists?", "Relevance to NHL", "Verdict", "Evidence"],
                        rows))
    for c in mastersite.REJECTED_REUSE:
        parts.append(f'<div class="note caveat"><b>Tested and rejected:</b> {e(c["claim"])}<br>'
                     f'<span class="small">Test: {e(c["test"])}<br>Result: {e(c["result"])}<br>'
                     f'Kept: {e(c["kept"])}</span></div>')

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
    tot = Performance(store).competition_totals(test_mode="FORWARD TEST")
    tot_b = Performance(store).competition_totals(test_mode="BACKTEST")
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
was genuinely quoted in the past <i>with a timestamp</i>. Kalshi publishes hourly candlesticks for
every contract (live tier for recent contracts, <code>/historical</code> tier for contracts settled
before the cutoff); the <b>closing price</b> used here is the last hourly candle ending at or before
the NHL scheduled start, and the <code>open</code>, <code>T-24h</code>, <code>T-6h</code> and
<code>T-1h</code> candles give line movement. Such rows carry
<code>verification_status=kalshi_candle_close</code>. A few early rows were priced from Kalshi's
undocumented <code>previous_yes_ask</code> before the candle history was recovered; they keep the
label <code>single_source_timing_unverified</code> permanently. Everything priced from a live quote
is <b>FORWARD TEST</b> and settles later from the official NHL result. No sportsbook history exists
in this system, so no backtest is ever claimed against sportsbook odds.</p>
<p>Two backtests are reported per strategy and never combined: an <i>accuracy</i> backtest on every
decided game (hit rate, lift over base rate, log loss, Brier; <code>data_sufficient=0</code>, no
P&amp;L) and a <i>priced</i> backtest on the games whose Kalshi closing candle was recovered (flat
1-unit stakes at the offer, net of the exchange fee, on the whole window and on chronological
train / validation / test splits). The accuracy backtest carries this caveat:</p>
<div class="note caveat">{e(CAVEAT_NO_PRICE)}</div>

<h2>3b. Markets traded</h2>
<p><b>Moneyline</b> (KXNHLGAME) is traded on both the backtest and the forward test. <b>Totals</b>
(KXNHLTOTAL, "Over k.5 goals") is traded as follows, and the asymmetry is deliberate:</p>
<ul>
<li>The line is read from the exchange's own <code>floor_strike</code> and
<code>strike_type</code> fields — never parsed out of a title, because Kalshi words the same
contract differently on the live tier ("Full Game: Over 8.5 goals scored") and the historical
tier ("Carolina vs Vegas: Total Goals"). A contract with no readable line is skipped, not
guessed.</li>
<li>An <b>Over</b> is the YES side, and the candlestick history publishes the YES offer, so an
Over rule is backtested at real timestamped prices.</li>
<li>An <b>Under</b> is the NO side. The candlestick feed publishes only the YES bid/ask, so there
is no historical NO offer to buy at, and <code>no_ask = 1 − yes_bid</code> is <i>not</i> assumed:
on the quotes in this ledger that identity fails on 2 of 12 contracts
(KXNHLGAME-26SEP20SJANA: yes_bid 0.19 against no_ask 0.99). An Under rule therefore carries a
<code>no_history_reason</code> and is <b>forward-test only</b>, entered at the live
<code>no_ask_dollars</code> Kalshi actually quotes. Its backtest row reports accuracy only, with
<code>data_sufficient=0</code> and no P&amp;L.</li>
<li>Settlement uses the official final score. Kalshi counts regulation and overtime goals
normally and counts a shootout as one goal for the winner, which is what the NHL's official score
already contains — so the settled total is <code>home + away</code> with no adjustment. Every
settled totals contract is also cross-checked against the exchange's own result, and a
disagreement is recorded as a critical <code>settlement_conflict</code> rather than resolved.</li>
<li>Puck line (KXNHLSPREAD), first period (KXNHL1P), overtime (KXNHLOVERTIME) and player props
are registered as series and researched, but nothing is bet on them: there is no period-level
model and no verified prop feed. They stay in "waiting for other information" instead of being
traded on an assumption.</li>
</ul>

<h2>3a. Data labels</h2>
<ul>
<li><b>SOURCE DATA</b> — NHL schedule/results, stats REST per-game team and goalie lines, Kalshi
quotes, settlements and candlesticks, the NHL partner-feed (DraftKings) odds, EDGE snapshots.</li>
<li><b>DERIVED</b> — every feature (rest days, rolling PP%/PK%, shot share, PDO, starter save
percentage, line movement, vig), every price point picked from a candle, every P&amp;L figure.</li>
<li><b>MODEL OUTPUT</b> — Poisson / Elo / logistic probabilities and any edge computed from them;
also any third-party model number (e.g. xG) should one ever be ingested.</li>
<li><b>ASSUMPTION</b> — recorded where used: the post-game goalie log identifies the starter for a
<i>past</i> game (starters are announced pre-game, but this system did not observe the announcement);
the fee model is Kalshi's published general schedule (M = 1 for KXNHLGAME); no slippage beyond the
single quoted level.</li>
<li><b>UNVERIFIED</b> — anything flagged in the Verification queue, and every strategy still
carrying a <code>blocked_reason</code>.</li>
</ul>

<h2>4. Execution model</h2>
<p>Buys happen at the <b>offer</b>, never the mid. Size is capped by the quoted
<code>yes_ask_size_fp</code>; if the desired stake needs more contracts than are offered the order
is partially filled and the remainder is recorded as unfilled. A missing or zero-size offer means
no bet. Slippage is stored as 0 because only one book level is available — that is a limitation of
the data, recorded rather than papered over. <b>Fees:</b> Kalshi's general taker fee,
<code>round up(0.07 × C × P × (1 − P))</code> per the fee schedule effective 2026-07-07
(KXNHLGAME is not on the non-standard list, so the multiplier is 1), is charged on every simulated
fill and deducted from the settled P&amp;L; the fee is stored on each bet row.</p>

<h2>5. Staking and bankrolls</h2>
<p>Model-based strategies use fractional Kelly, capped, with open exposure reserved from the
bankroll so a strategy cannot over-commit. Market-structure rules (no model of their own) stake a
flat fraction. Each strategy version has its own virtual bankroll; <b>only FORWARD TEST results move
it</b> — BACKTEST P&amp;L is reported separately and never funds a forward stake. Aggressive staking is
permitted by design — the drawdown, volatility and losing-streak columns exist so the risk is
visible, not so it is suppressed.</p>

<h2>5a. Strategy versions and lifecycle</h2>
<p>A strategy is identified by <code>(strategy_id, version)</code>. Changing a rule creates a new
version; the old one is set to <code>retired</code> with the reason, and its bets stay in the
ledger. Lifecycle states: candidate → active → paused / retired / rejected, each transition logged
in <code>strategy_lifecycle</code> with evidence. Blocked categories (no verified input yet) are
registered with an explicit <code>blocked_reason</code> and stay in
<code>WAITING FOR OTHER INFORMATION</code> rather than betting on an assumption.</p>

<h2>5b. Sportsbook reference</h2>
<p>The NHL partner feed publishes DraftKings moneylines for the current slate. They are stored as
published (American odds), de-vigged proportionally, and attached to forward signals as a
<i>reference</i> — Kalshi remains the execution venue. Because the feed has no history, the
book-vs-exchange strategy is forward-test only and says so.</p>

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
<p>{tot['strategies']} strategies. FORWARD TEST: {tot['bets']} simulated bets · {tot['settled']}
settled · {tot['open']} open · P&amp;L {n(tot['pnl'])} · ROI {pct(tot['roi'])}. BACKTEST (reported
separately, never merged): {tot_b['bets']} priced wagers · P&amp;L {n(tot_b['pnl'])} · ROI
{pct(tot_b['roi'])}.</p>

<h2>9. Known limitations</h2>
<ul>
<li>No verified historical sportsbook odds. Closing-line value is measured against the Kalshi
closing candle only; the DraftKings reference exists only from the day collection started.</li>
<li>The Kalshi closing candle is hourly: "close" means the last candle ending at or before the
scheduled start, so it may include trades from the final minutes before puck drop. Kalshi's NHL
books are thin outside the playoffs; the candle's <code>yes_ask</code> is a price, not a
guarantee of size, and the fee model assumes the general schedule.</li>
<li>Starting goalies for <i>upcoming</i> games have no verified pre-game source, so goalie-gated
strategies forward-test only when a starter becomes known; their backtests rely on the post-game
log (an explicit ASSUMPTION).</li>
<li>Arena latitude/longitude is unavailable: the legacy NHL venue host could not be reached, so
travel is derived from published venue UTC offsets rather than invented coordinates.</li>
<li>Line combinations, goalie announcements and player props have no verified feed. EDGE tracking
is snapshotted forward only (season-to-date aggregates, no per-game history).</li>
<li>Preseason Kalshi markets are thin; contracts with zero size on both sides are recorded as
<code>insufficient_liquidity</code> instead of being traded.</li>
<li>In-game candles (60/120 minutes after the start) are stored for research; the pipeline runs on
a batch schedule and cannot execute during a game, so no in-game bet is simulated.</li>
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
        "leaderboard.json": {"FORWARD TEST": perf.leaderboard(test_mode="FORWARD TEST"),
                             "BACKTEST": perf.leaderboard(test_mode="BACKTEST")},
        "competition.json": {"FORWARD TEST": perf.competition_totals(test_mode="FORWARD TEST"),
                             "BACKTEST": perf.competition_totals(test_mode="BACKTEST")},
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
