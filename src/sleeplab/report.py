"""Static HTML report: one page per night plus an index with the multi-night summary.

Open ``reports/index.html`` in a browser. plotly.js is written once next to the
pages so the report works offline and carries no tracking.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .analysis import desaturations, event_triggered_average, lagged_xcorr, minute_grid, snore_bouts, summarise
from .night import Night

# reference palette (dataviz skill), light / dark steps of the same hues
SPO2 = ("#2a78d6", "#3987e5")
SNORE = ("#eb6834", "#d95926")
HR = ("#1baf7a", "#199e70")
STAGE = ("#6250d6", "#9085e9")
STAGE_Y = {"DEEP": 0, "LIGHT": 1, "ASLEEP": 1, "RESTLESS": 1.5, "REM": 2, "AWAKE": 3}

CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --rule:#e4e3df;
  --spo2:#2a78d6; --snore:#eb6834; --hr:#1baf7a; --stage:#6250d6; --warn:#a15c00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#141413; --card:#1a1a19; --ink:#f0efec;
  --ink2:#c3c2b7; --muted:#8f8e86; --rule:#2c2b29; --spo2:#3987e5; --snore:#d95926; --hr:#199e70; --stage:#9085e9; --warn:#e0a040; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#141413; --card:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86; --rule:#2c2b29;
  --spo2:#3987e5; --snore:#d95926; --hr:#199e70; --stage:#9085e9; --warn:#e0a040; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font: 15px/1.5 ui-sans-serif, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 32px 0 8px; }
.eyebrow { font: 600 12px/1 ui-monospace, Menlo, monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.verdict { font-size: 20px; margin: 12px 0 20px; }
.note { color: var(--ink2); font-size: 13px; }
.warn { color: var(--warn); }
.tiles { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 8px; }
.tile { background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:10px 12px; }
.tile .v { font-size: 22px; font-variant-numeric: tabular-nums; }
.tile .k { font-size: 12px; color: var(--ink2); }
.card { background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:8px; margin-top:8px; }
.row { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 760px) { .row { grid-template-columns: 1fr; } }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
.scroll { overflow-x: auto; }
a { color: var(--spo2); }
.key { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--ink2); margin: 4px 0; }
.key i { display:inline-block; width:12px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; }
"""

# Re-colour plotly figures for the current theme (plotly can't read CSS variables).
THEME_JS = """
function sleeplabTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim();
  const dark = cs.colorScheme === 'dark';
  document.querySelectorAll('.js-plotly-plot').forEach(el => {
    const upd = {'font.color': v('--ink2'), 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
                 'hoverlabel.bgcolor': v('--card'), 'hoverlabel.font.color': v('--ink')};
    for (const k of Object.keys(el.layout)) if (/^[xy]axis/.test(k)) {
      upd[k + '.gridcolor'] = v('--rule'); upd[k + '.linecolor'] = v('--rule'); upd[k + '.zerolinecolor'] = v('--rule');
    }
    Plotly.relayout(el, upd);
    const traces = [], colors = [];
    el.data.forEach((t, i) => { if (t.meta && t.meta.role) { traces.push(i); colors.push(v('--' + t.meta.role)); } });
    if (traces.length) Plotly.restyle(el, {'line.color': colors, 'marker.color': colors}, traces);
  });
}
window.addEventListener('load', sleeplabTheme);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', sleeplabTheme);
"""

LAYOUT = dict(
    margin=dict(l=56, r=16, t=16, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="ui-sans-serif, -apple-system, Segoe UI, sans-serif", size=12, color="#52514e"),
    hovermode="x unified",
    showlegend=False,
)
CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def _local(t: pd.Series | pd.DatetimeIndex, tz: str):
    return pd.DatetimeIndex(t).tz_convert(tz).tz_localize(None)


def timeline_figure(night: Night, grid: pd.DataFrame | None = None) -> go.Figure:
    tz = night.meta.get("timezone", "UTC")
    g = grid if grid is not None else minute_grid(night)
    x = _local(g.index, tz)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.22, 0.38, 0.22, 0.18],
        subplot_titles=("Sleep stage", "SpO2 (%)", "Snoring (seconds per minute)", "Heart rate (bpm)"),
    )
    # hypnogram as a step line
    if not night.stages.empty:
        xs, ys, txt = [], [], []
        for s, e, st in night.stages.itertuples(index=False):
            y = STAGE_Y.get(st, 1)
            xs += [s, e]
            ys += [y, y]
            txt += [st, st]
        fig.add_trace(
            go.Scatter(x=_local(pd.Series(xs), tz), y=ys, mode="lines", line=dict(color=STAGE[0], width=2, shape="hv"),
                       text=txt, hovertemplate="%{text}<extra></extra>", meta={"role": "stage"}, name="Stage"),
            row=1, col=1,
        )
    fig.update_yaxes(row=1, col=1, tickvals=[0, 1, 2, 3], ticktext=["Deep", "Light", "REM", "Awake"], range=[-0.4, 3.4])

    # SpO2 line (gaps stay gaps) + desaturation markers
    fig.add_trace(
        go.Scatter(x=x, y=g["spo2"], mode="lines", line=dict(color=SPO2[0], width=2), connectgaps=False,
                   hovertemplate="SpO2 %{y:.1f}%<extra></extra>", meta={"role": "spo2"}, name="SpO2"),
        row=2, col=1,
    )
    d = desaturations(g)
    if len(d):
        fig.add_trace(
            go.Scatter(x=_local(d, tz), y=g.loc[d, "spo2"], mode="markers",
                       marker=dict(color=SPO2[0], size=9, symbol="triangle-down", line=dict(width=2, color="rgba(255,255,255,0.9)")),
                       hovertemplate="≥3-point dip<extra></extra>", meta={"role": "spo2"}, name="Dip"),
            row=2, col=1,
        )
    lo = np.nanmin(g["spo2"]) if g["spo2"].notna().any() else 90
    fig.update_yaxes(row=2, col=1, range=[min(lo - 1, 90), 100])

    fig.add_trace(
        go.Bar(x=x, y=g["snore_s"], marker=dict(color=SNORE[0], line=dict(width=0)),
               hovertemplate="%{y:.0f} s of snoring<extra></extra>", meta={"role": "snore"}, name="Snoring"),
        row=3, col=1,
    )
    fig.update_yaxes(row=3, col=1, range=[0, 60])

    fig.add_trace(
        go.Scatter(x=x, y=g["bpm"], mode="lines", line=dict(color=HR[0], width=1.5), connectgaps=False,
                   hovertemplate="%{y:.0f} bpm<extra></extra>", meta={"role": "hr"}, name="Heart rate"),
        row=4, col=1,
    )

    # minutes without audio: shaded, so "no snoring" is not confused with "not listening"
    no_audio = ~g["audio"].to_numpy()
    if no_audio.any():
        edges = np.flatnonzero(np.diff(np.r_[0, no_audio.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            fig.add_vrect(x0=x[a], x1=x[min(b, len(x) - 1)], fillcolor="#8a8984", opacity=0.12, line_width=0, row=3, col=1)

    fig.update_layout(**LAYOUT, height=640, bargap=0.1)
    fig.update_xaxes(showgrid=False, tickformat="%H:%M")
    fig.update_yaxes(gridcolor="#e4e3df", zeroline=False)
    fig.update_annotations(font=dict(size=12), x=0, xanchor="left")
    return fig


def xcorr_figure(xc: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=xc["lag_min"], y=xc["r"], marker=dict(color=SPO2[0], cornerradius=4),
               hovertemplate="lag %{x} min: r = %{y:.2f}<extra></extra>", meta={"role": "spo2"})
    )
    fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=280,
                      xaxis_title="SpO2 lag after snoring (minutes)", yaxis_title="correlation r")
    fig.update_yaxes(range=[-1, 1], gridcolor="#e4e3df", zeroline=True)
    return fig


def eta_figure(lags, mean, lo, hi, n: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(lags) + list(lags)[::-1], y=list(hi) + list(lo)[::-1], fill="toself",
                             fillcolor="rgba(138,137,132,0.18)", line=dict(width=0), hoverinfo="skip", name="random times (95%)"))
    fig.add_trace(go.Scatter(x=lags, y=mean, mode="lines+markers", line=dict(color=SPO2[0], width=2),
                             marker=dict(size=8, color=SPO2[0]), meta={"role": "spo2"},
                             hovertemplate="%{x:+d} min: %{y:+.2f} points<extra></extra>", name=f"after snore bouts (n={n})"))
    fig.add_vline(x=0, line=dict(color="#8a8984", width=1, dash="dot"))
    fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=280,
                      xaxis_title="minutes from snore-bout onset", yaxis_title="SpO2 change (points)")
    fig.update_yaxes(gridcolor="#e4e3df", zeroline=True)
    return fig


def _div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style><script src="plotly.min.js"></script></head>
<body><main>{body}
<footer>Sleep Lab · generated locally · not a medical device: this is a curiosity tool, not a sleep study.
If you're worried about your breathing at night, talk to a doctor.</footer></main>
<script>{THEME_JS}</script></body></html>"""


def _fmt(v, suffix=""):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v}{suffix}"


def night_page(night: Night) -> tuple[str, dict]:
    g = minute_grid(night)
    s = summarise(night)
    bouts = snore_bouts(night.snores)
    eta = event_triggered_average(g, bouts)
    xc = lagged_xcorr(g)
    synth = night.meta.get("source") == "synthetic"
    tiles = [
        ("Sleep", _fmt(s["sleep_hours"], " h")),
        ("Snore bouts / hour", _fmt(s["bouts_per_hour"])),
        ("Time snoring", _fmt(s["snore_pct_of_sleep"], "%")),
        ("Mean SpO2", _fmt(s["spo2_mean"], "%")),
        ("Lowest SpO2", _fmt(s["spo2_min"], "%")),
        ("≥3-pt SpO2 dips / hour", _fmt(s["desats_per_hour"])),
        ("Resting HR", _fmt(s["resting_hr"], " bpm")),
        ("SpO2 coverage", _fmt(s["spo2_coverage_pct"], "%")),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in tiles)
    stage_rows = "".join(f"<tr><td>{k.title()}</td><td>{_fmt(v, '%')}</td></tr>" for k, v in s["snore_pct_by_stage"].items())
    lag, depth = eta.nadir
    body = f"""
<div class="eyebrow">Field report · night ending {night.date}{' · SYNTHETIC DATA' if synth else ''}</div>
<h1>Did I snore?</h1>
<p class="verdict">{html.escape(s['verdict'])}</p>
{'<p class="note warn">This night is generated by <code>sleeplab demo</code> (profile: “' + html.escape(str(s['profile'])) + '”). Real nights come from <code>sleeplab fetch</code> + <code>sleeplab detect</code>.</p>' if synth else ''}
<div class="tiles">{tiles_html}</div>
<h2>The night, on one timeline</h2>
<div class="key"><span><i style="background:var(--stage)"></i>sleep stage</span><span><i style="background:var(--spo2)"></i>SpO2 (▼ = 3-point dip)</span>
<span><i style="background:var(--snore)"></i>snoring</span><span><i style="background:var(--hr)"></i>heart rate</span>
<span><i style="background:var(--muted);opacity:.4;height:10px"></i>no audio recorded</span></div>
<div class="card">{_div(timeline_figure(night, g))}</div>
<h2>Does SpO2 drop after snoring?</h2>
<div class="row">
 <div><div class="card">{_div(eta_figure(eta.lags, eta.mean, eta.null_lo, eta.null_hi, eta.n_bouts))}</div>
 <p class="note">Average SpO2 change around the start of each snore bout (≥3 snores, ≤30 s apart), against each bout's own 5-minute baseline.
 Grey band: the same average at random sleeping minutes, 300 shuffles. Deepest point: {_fmt(None if np.isnan(depth) else round(depth, 2))} points at
 {_fmt(None if np.isnan(lag) else int(lag), ' min')}; {'outside' if eta.beats_null else 'inside'} the random band.</p></div>
 <div><div class="card">{_div(xcorr_figure(xc))}</div>
 <p class="note">Correlation between seconds of snoring in a minute and SpO2 that many minutes later.
 Negative bars at positive lags mean SpO2 tends to fall after snoring. Strongest: r = {_fmt(s['xcorr_best_r'])} at {_fmt(s['xcorr_best_lag_min'], ' min')}.</p></div>
</div>
<h2>When do I snore?</h2>
<div class="scroll"><table><tr><th>Stage</th><th>Share of minutes with snoring</th></tr>{stage_rows}</table></div>
<p class="note">{s['snore_events']} snore events grouped into {s['snore_bouts']} bouts. Detector: {html.escape(str((night.meta.get('audio') or {}).get('detector', 'none')))}.
SpO2 from the wrist is 1-minute resolution with gaps; a single low reading is often motion, not physiology.</p>
<p><a href="index.html">← all nights</a></p>"""
    return _page(f"Sleep Lab · {night.date}", body), s


def index_page(summaries: list[dict]) -> str:
    dates = [s["date"] for s in summaries]
    fig1 = go.Figure(go.Bar(x=dates, y=[s["bouts_per_hour"] for s in summaries], marker=dict(color=SNORE[0], cornerradius=4),
                            meta={"role": "snore"}, hovertemplate="%{x}: %{y:.1f} bouts/h<extra></extra>"))
    fig1.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=240, yaxis_title="snore bouts / hour")
    fig2 = go.Figure(go.Bar(x=dates, y=[s["desats_per_hour"] for s in summaries], marker=dict(color=SPO2[0], cornerradius=4),
                            meta={"role": "spo2"}, hovertemplate="%{x}: %{y:.1f} dips/h<extra></extra>"))
    fig2.update_layout(**{**LAYOUT, "hovermode": "closest"}, height=240, yaxis_title="≥3-pt SpO2 dips / hour")
    for f in (fig1, fig2):
        f.update_xaxes(type="category")
        f.update_yaxes(gridcolor="#e4e3df", rangemode="tozero")
    rows = "".join(
        f"<tr><td><a href='night-{s['date']}.html'>{s['date']}</a></td><td>{_fmt(s['sleep_hours'])}</td><td>{_fmt(s['bouts_per_hour'])}</td>"
        f"<td>{_fmt(s['snore_pct_of_sleep'], '%')}</td><td>{_fmt(s['spo2_mean'])}</td><td>{_fmt(s['spo2_min'])}</td><td>{_fmt(s['desats_per_hour'])}</td>"
        f"<td>{_fmt(s['eta_nadir_drop'])}</td><td>{'yes' if s['eta_beats_shuffle'] else 'no'}</td><td>{_fmt(s['resting_hr'])}</td><td style='text-align:left'>{html.escape(s['verdict'].split('.')[0])}</td></tr>"
        for s in summaries
    )
    synth = any(s["source"] == "synthetic" for s in summaries)
    total_bph = np.mean([s["bouts_per_hour"] for s in summaries]) if summaries else 0
    body = f"""
<div class="eyebrow">Field report · {len(summaries)} night{'s' if len(summaries) != 1 else ''}{' · SYNTHETIC DATA' if synth else ''}</div>
<h1>Do I snore?</h1>
<p class="verdict">Across {len(summaries)} nights: {total_bph:.1f} snore bouts per hour on average.</p>
<div class="row"><div class="card">{_div(fig1)}</div><div class="card">{_div(fig2)}</div></div>
<h2>Night by night</h2>
<div class="scroll"><table>
<tr><th>Night</th><th>Sleep h</th><th>Bouts/h</th><th>Snoring</th><th>Mean SpO2</th><th>Min SpO2</th><th>Dips/h</th><th>Dip after bout</th><th>Beats chance</th><th>Resting HR</th><th style='text-align:left'>Verdict</th></tr>
{rows}</table></div>"""
    return _page("Sleep Lab", body)


def build_report(nights: list[Night], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    js = out / "plotly.min.js"
    if not js.exists():
        js.write_text(get_plotlyjs())
    summaries = []
    for n in sorted(nights, key=lambda n: n.date):
        page, s = night_page(n)
        (out / f"night-{n.date}.html").write_text(page)
        summaries.append(s)
    (out / "index.html").write_text(index_page(summaries))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    return out / "index.html"
