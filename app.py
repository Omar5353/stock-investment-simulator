#!/usr/bin/env python3
"""
Stock Investment Simulator — Streamlit web app
Deploy: streamlit run app.py  |  share via Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

from simulator_core import (
    fetch_data, simulate_base_dca, simulate_drawdown_dca, simulate_ath_buy_sell,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Investment Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours (match desktop palette) ────────────────────────────────────
ACCENT = '#89b4fa'
GREEN  = '#a6e3a1'
RED    = '#f38ba8'
YELLOW = '#f9e2af'
BG     = '#1e1e2e'
PANEL  = '#24273a'
CARD   = '#2a2d3e'

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #1e1e2e; }
section[data-testid="stSidebar"] { background-color: #24273a; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #2a2d3e;
    border: 1px solid #363a4f;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"]  { color: #cdd6f4 !important; font-size: 0.82rem; }
[data-testid="stMetricValue"]  { color: #ffffff !important; font-size: 1.15rem; }
[data-testid="stMetricDelta"]  { font-size: 0.80rem; }

/* Dataframe */
[data-testid="stDataFrame"] { background-color: #2a2d3e; border-radius: 6px; }

/* Expander */
details { background-color: #24273a !important; border-radius: 8px; }
summary { color: #89b4fa !important; font-weight: 600; }

/* Divider */
hr { border-color: #363a4f; }

/* Buttons */
.stButton > button {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: 700;
    border: none;
    border-radius: 6px;
}
.stButton > button:hover { background-color: #74c7ec; color: #1e1e2e; }

/* Section headers */
h3 { color: #89b4fa !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached data fetch (1h TTL) ────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(ticker: str) -> pd.DataFrame:
    return fetch_data(ticker)


# ── Formatters ────────────────────────────────────────────────────────────────
def _usd(v):   return f"${float(v):,.2f}"
def _spct(v):
    fv = float(v)
    return ("N/A" if np.isnan(fv) else f"{fv:+.2f}%")
def _sign(v):  return f"{float(v):+,.2f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{ACCENT};margin-bottom:2px'>📈 Stock Simulator</h2>",
                unsafe_allow_html=True)
    st.caption("Yahoo Finance · DCA & ATH strategies")
    st.divider()

    ticker = st.text_input("Ticker Symbol", "MSFT",
                           help="US: MSFT, AAPL  |  International: BMW.DE, 9984.T").strip().upper()
    amount = st.number_input("Investment per Buy ($)", min_value=1.0,
                             value=100.0, step=10.0)
    freq   = st.selectbox("Buy Frequency",
                          ["Monthly", "Biweekly", "Weekly", "Daily"])
    strat  = st.selectbox("Strategy",
                          ["Base DCA", "Drawdown-Triggered DCA", "ATH Buy & Sell"])

    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date", value=date(2015, 1, 1))
    end_date   = c2.date_input("End Date",   value=date(2025, 1, 1))

    # Strategy-specific params
    dd_buy = dd_stop = ath_buy = ath_stop = ath_sell = None

    if strat == "Drawdown-Triggered DCA":
        st.markdown(f"<p style='color:{ACCENT};font-weight:600;margin:10px 0 4px'>⚙️ Drawdown Settings</p>",
                    unsafe_allow_html=True)
        dd_buy  = st.number_input("Buy trigger  (% below ATH)",
                                  value=15.0, min_value=0.1, max_value=99.0, step=0.5)
        dd_stop = st.number_input("Stop trigger  (% below ATH)",
                                  value=10.0, min_value=0.1, max_value=99.0, step=0.5)

    elif strat == "ATH Buy & Sell":
        st.markdown(f"<p style='color:{ACCENT};font-weight:600;margin:10px 0 4px'>⚙️ ATH Settings</p>",
                    unsafe_allow_html=True)
        ath_buy  = st.number_input("Buy when  (% below ATH)",
                                   value=10.0, min_value=0.1, max_value=99.0, step=0.5)
        ath_stop = st.number_input("Stop buying  (% below ATH, 0 = off)",
                                   value=0.0, min_value=0.0, max_value=99.0, step=0.5)
        ath_sell = st.number_input("Sell when  (% above ATH at buy)",
                                   value=5.0,  min_value=0.1, max_value=99.0, step=0.5)

    st.divider()
    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.markdown(
        "<p style='color:#6c7086;font-size:0.72rem;margin-top:16px'>"
        "Brought to you by<br>Mohammad Omar Faruk Murad</p>",
        unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='color:{ACCENT}'>📈 Stock Investment Simulator</h1>",
            unsafe_allow_html=True)
st.caption("Simulate DCA & ATH-based strategies on historical data · Yahoo Finance")

if not run:
    st.info("👈 Configure your strategy in the sidebar, then click **▶ Run Simulation**.")
    st.stop()

# ── Validate ──────────────────────────────────────────────────────────────────
errs = []
if not ticker:
    errs.append("Ticker symbol is required.")
if start_date >= end_date:
    errs.append("Start date must be before end date.")
if strat == "Drawdown-Triggered DCA" and dd_buy is not None and dd_stop is not None:
    if dd_stop >= dd_buy:
        errs.append("Stop trigger must be less than Buy trigger.")
if strat == "ATH Buy & Sell" and ath_stop is not None and ath_stop > 0 and ath_buy is not None:
    if ath_stop >= ath_buy:
        errs.append("Stop buying % must be less than Buy %.")
if errs:
    for e in errs:
        st.error(e)
    st.stop()

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
    try:
        df = _fetch(ticker)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

# ── Run simulation ────────────────────────────────────────────────────────────
with st.spinner("Running simulation…"):
    try:
        if strat == "Base DCA":
            daily, buy_df, summary = simulate_base_dca(
                df, start_date, end_date, amount, freq)
            sell_df = None

        elif strat == "Drawdown-Triggered DCA":
            daily, buy_df, summary = simulate_drawdown_dca(
                df, start_date, end_date, amount, freq,
                buy_dd=dd_buy / 100, stop_dd=dd_stop / 100)
            sell_df = None

        else:  # ATH Buy & Sell
            daily, buy_df, sell_df, summary = simulate_ath_buy_sell(
                df, start_date, end_date, amount, freq,
                buy_pct=ath_buy / 100,
                stop_buy_pct=ath_stop / 100,
                sell_pct=ath_sell / 100)

    except Exception as exc:
        st.error(str(exc))
        st.stop()

# ── Results metrics ───────────────────────────────────────────────────────────
st.markdown("### Results")

row1 = st.columns(5)
row1[0].metric("Total Invested",  _usd(summary['total_invested']))
row1[1].metric("Final Value",     _usd(summary['final_value']))
row1[2].metric("Capital Gain",    _usd(summary['profit']),
               delta=_spct(summary['pct_gain']))
row1[3].metric("CAGR",            _spct(summary['cagr']))
row1[4].metric("ARR",             _spct(summary['arr']))

row2 = st.columns(5)
row2[0].metric("Total Return",    _spct(summary['pct_gain']))
row2[1].metric("# of Buys",       str(summary['buy_count']))
row2[2].metric("# of Sells",      str(summary.get('sell_count', 0)))
row2[3].metric("Est. Tax",        _usd(summary.get('total_tax', 0.0)),
               delta=f"20% ST / 10% LT", delta_color="off")
row2[4].metric("Net Gain (After Tax)",
               _usd(summary.get('net_profit', summary['profit'])),
               delta=_spct(summary.get('net_pct', summary['pct_gain'])))

st.divider()

# ── Plotly charts ─────────────────────────────────────────────────────────────
profit  = float(summary['profit'])
pcolor  = GREEN if profit >= 0 else RED
pfill   = 'rgba(166,227,161,0.12)' if profit >= 0 else 'rgba(243,139,168,0.12)'

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.10,
    subplot_titles=[
        f"{ticker}  ·  {strat}  ·  {freq}  ·  "
        f"${summary['total_invested']:,.0f} invested",
        f"Portfolio Value  ·  Final: {_usd(summary['final_value'])}  ·  "
        f"Gain: {_sign(summary['profit'])} ({float(summary['pct_gain']):.1f}%)"
    ],
)

dates = daily['Date']

# Price line
fig.add_trace(go.Scatter(
    x=dates, y=daily['Close'],
    name='Close Price',
    line=dict(color=ACCENT, width=1.6),
    hovertemplate='%{x|%b %d %Y}  $%{y:,.2f}<extra>Close</extra>',
), row=1, col=1)

# Buy markers
if not buy_df.empty:
    pcol = 'Price' if 'Price' in buy_df.columns else 'Close'
    fig.add_trace(go.Scatter(
        x=buy_df['Date'], y=buy_df[pcol],
        mode='markers',
        name=f'Buy ({len(buy_df)})',
        marker=dict(symbol='triangle-up', color=RED, size=9, opacity=0.88,
                    line=dict(width=0)),
        hovertemplate='%{x|%b %d %Y}  $%{y:,.2f}<extra>Buy</extra>',
    ), row=1, col=1)

# Sell markers
if sell_df is not None and not sell_df.empty:
    fig.add_trace(go.Scatter(
        x=sell_df['Date'], y=sell_df['Price'],
        mode='markers',
        name=f'Sell ({len(sell_df)})',
        marker=dict(symbol='triangle-down', color=GREEN, size=11, opacity=0.92,
                    line=dict(width=0)),
        hovertemplate='%{x|%b %d %Y}  $%{y:,.2f}<extra>Sell</extra>',
    ), row=1, col=1)

# Portfolio value (filled area)
fig.add_trace(go.Scatter(
    x=dates, y=daily['port_value'],
    name='Portfolio Value',
    line=dict(color=pcolor, width=1.6),
    fill='tozeroy',
    fillcolor=pfill,
    hovertemplate='%{x|%b %d %Y}  $%{y:,.2f}<extra>Portfolio</extra>',
), row=2, col=1)

# Invested baseline
if summary['total_invested'] > 0:
    fig.add_hline(
        y=float(summary['total_invested']),
        line=dict(color=YELLOW, width=1.2, dash='dash'),
        annotation_text=f"  Invested ${float(summary['total_invested']):,.0f}",
        annotation_font_color=YELLOW,
        annotation_font_size=11,
        row=2, col=1,
    )

# Layout
fig.update_layout(
    height=680,
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color='#cdd6f4', size=11, family='Segoe UI, sans-serif'),
    legend=dict(bgcolor=CARD, bordercolor='#363a4f', borderwidth=1,
                font=dict(color='#cdd6f4')),
    margin=dict(l=60, r=30, t=55, b=40),
    hovermode='x unified',
    hoverlabel=dict(bgcolor=CARD, font_color='#cdd6f4', bordercolor='#363a4f'),
)
fig.update_xaxes(
    gridcolor='#363a4f', zerolinecolor='#363a4f',
    tickformat='%Y', tickfont=dict(color='#cdd6f4'),
    linecolor='#363a4f',
)
fig.update_yaxes(
    gridcolor='#363a4f', zerolinecolor='#363a4f',
    tickprefix='$', tickfont=dict(color='#cdd6f4'),
    linecolor='#363a4f',
)
fig.update_annotations(font_color='#cdd6f4')

st.plotly_chart(fig, use_container_width=True)

# ── Download buttons ──────────────────────────────────────────────────────────
_strat_slug = strat.replace(' ', '_').replace('&', 'and')
_gain_slug  = f"{float(summary['pct_gain']):+.0f}pct"
_base_name  = (f"{ticker}_{_strat_slug}_{freq}_"
               f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_"
               f"{_gain_slug}")

_dl1, _dl2, _spacer = st.columns([1, 1, 4])

# PNG — requires kaleido
with _dl1:
    try:
        _png_bytes = fig.to_image(format='png', width=1600, height=800, scale=2)
        st.download_button(
            label="⬇️ Download PNG",
            data=_png_bytes,
            file_name=f"{_base_name}.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception:
        st.caption("PNG unavailable (kaleido not installed)")

# Interactive HTML — no extra deps
with _dl2:
    _html_bytes = fig.to_html(include_plotlyjs='cdn', full_html=True).encode('utf-8')
    st.download_button(
        label="⬇️ Download HTML",
        data=_html_bytes,
        file_name=f"{_base_name}.html",
        mime="text/html",
        use_container_width=True,
    )

# ── Tax breakdown (ATH B&S only) ──────────────────────────────────────────────
if strat == "ATH Buy & Sell" and summary.get('tax_breakdown'):
    breakdown = summary['tax_breakdown']

    with st.expander(
        "📋 Tax Breakdown Detail  ·  FIFO  ·  Short-term ≤ 1 yr: 20%  ·  Long-term > 1 yr: 10%",
        expanded=False
    ):
        total_all_tax = 0.0
        sell_idx      = 0

        for event in breakdown:
            is_sell = event['type'] == 'sell'
            total_all_tax += event['total_tax']

            if is_sell:
                sell_idx += 1
                total_shares = sum(l['shares'] for l in event['lots'])
                st.markdown(
                    f"**SELL #{sell_idx}** — "
                    f"`{event['date'].strftime('%Y-%m-%d')}`  "
                    f"@ **${event['sell_price']:,.2f}**  "
                    f"({total_shares:.4f} shares)"
                )
            else:
                st.markdown(
                    f"**End of Period (Unrealized)** — "
                    f"`{event['date'].strftime('%Y-%m-%d')}`  "
                    f"@ **${event['sell_price']:,.2f}**"
                )

            # Build lot rows
            rows = []
            for lot in event['lots']:
                rows.append({
                    'Buy Date':  lot['buy_date'].strftime('%Y-%m-%d'),
                    'Buy Price': f"${lot['buy_price']:,.2f}",
                    'Shares':    round(lot['shares'], 4),
                    'Gain ($)':  round(lot['gain'], 2),
                    'Days Held': lot['days_held'],
                    'Term':      lot['term'],
                    'Rate':      f"{lot['rate']*100:.0f}%",
                    'Tax ($)':   round(lot['tax'], 2),
                })

            # Subtotal row
            rows.append({
                'Buy Date':  '— Subtotal —',
                'Buy Price': '',
                'Shares':    round(sum(l['shares'] for l in event['lots']), 4),
                'Gain ($)':  round(event['total_gain'], 2),
                'Days Held': '',
                'Term':      '',
                'Rate':      '',
                'Tax ($)':   round(event['total_tax'], 2),
            })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("")

        st.markdown(
            f"<p style='font-size:1.05rem;font-weight:700;color:{ACCENT}'>"
            f"Total Estimated Tax: ${total_all_tax:,.2f}</p>",
            unsafe_allow_html=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='color:#6c7086;font-size:0.75rem;margin:0'>"
    "Brought to you by Mohammad Omar Faruk Murad</p>",
    unsafe_allow_html=True
)

