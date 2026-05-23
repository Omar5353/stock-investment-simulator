#!/usr/bin/env python3
"""
simulator_core.py — pure simulation logic, no GUI dependencies.
Shared by stock_simulator.py (desktop) and app.py (Streamlit web).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV via yfinance. Supports all major markets."""
    symbol = ticker.strip().upper()
    raw = yf.download(symbol, period='max', auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(
            f"No data for '{symbol}'.\n"
            "US stocks: MSFT, AAPL, TSLA\n"
            "International: BMW.DE, 9984.T, 0700.HK"
        )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index.name = 'Date'
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.dropna(subset=['Open', 'Close']).sort_values('Date').reset_index(drop=True)
    return df


# ── ARR ───────────────────────────────────────────────────────────────────────

def calc_arr(total_profit: float, total_invested: float, years: float) -> float:
    """
    Accounting Rate of Return.
    ARR = (Average Annual Net Profit / Total Invested) * 100
    """
    if total_invested <= 0 or years <= 0:
        return float('nan')
    return (total_profit / years) / total_invested * 100


# ── Capital gains tax ─────────────────────────────────────────────────────────
# Short-term: held ≤ 365 days → 20%   Long-term: held > 365 days → 10%

def calc_tax_breakdown(buy_df: pd.DataFrame, end_date, final_close: float,
                       sell_df=None,
                       short_rate: float = 0.20, long_rate: float = 0.10):
    """
    FIFO capital gains tax with full per-lot detail.
    Returns (total_tax: float, breakdown: list[dict]).

    Each breakdown entry:
      type        : 'sell' | 'end_of_period'
      date        : pd.Timestamp
      sell_price  : float
      lots        : list of lot dicts
      total_gain  : float
      total_tax   : float

    Long-term threshold: held > 365 days.  Only positive gains taxed.
    """
    if buy_df.empty:
        return 0.0, []

    end_ts    = pd.Timestamp(end_date)
    price_col = 'Price' if 'Price' in buy_df.columns else 'Close'

    lots: list = []
    for _, row in buy_df.iterrows():
        lots.append([row['Date'], float(row['shares_bought']), float(row[price_col])])

    total_tax = 0.0
    breakdown = []

    def _lot_entry(lot_date, shares, lot_cost, exit_price, exit_date):
        gain      = (exit_price - lot_cost) * shares
        days_held = (exit_date - lot_date).days
        is_long   = days_held > 365
        rate      = long_rate if is_long else short_rate
        tax       = gain * rate if gain > 0 else 0.0
        return dict(buy_date=lot_date, buy_price=lot_cost, shares=shares,
                    gain=gain, days_held=days_held,
                    term='Long' if is_long else 'Short',
                    rate=rate, tax=tax)

    if sell_df is not None and not sell_df.empty:
        for _, sell_row in sell_df.iterrows():
            to_sell    = float(sell_row['Shares'])
            sell_price = float(sell_row['Price'])
            sell_date  = sell_row['Date']
            event      = dict(type='sell', date=sell_date, sell_price=sell_price,
                              lots=[], total_gain=0.0, total_tax=0.0)

            while to_sell > 1e-9 and lots:
                lot_date, lot_shares, lot_cost = lots[0]
                taken = min(to_sell, lot_shares)
                e     = _lot_entry(lot_date, taken, lot_cost, sell_price, sell_date)
                event['lots'].append(e)
                event['total_gain'] += e['gain']
                event['total_tax']  += e['tax']
                total_tax           += e['tax']
                lots[0][1] -= taken
                to_sell    -= taken
                if lots[0][1] < 1e-9:
                    lots.pop(0)

            breakdown.append(event)

    remaining = [(d, s, c) for d, s, c in lots if s > 1e-9]
    if remaining:
        event = dict(type='end_of_period', date=end_ts, sell_price=final_close,
                     lots=[], total_gain=0.0, total_tax=0.0)
        for lot_date, lot_shares, lot_cost in remaining:
            e = _lot_entry(lot_date, lot_shares, lot_cost, final_close, end_ts)
            event['lots'].append(e)
            event['total_gain'] += e['gain']
            event['total_tax']  += e['tax']
            total_tax           += e['tax']
        breakdown.append(event)

    return total_tax, breakdown


def calc_tax_liability(buy_df: pd.DataFrame, end_date, final_close: float,
                       sell_df=None,
                       short_rate: float = 0.20, long_rate: float = 0.10) -> float:
    """Convenience wrapper — returns only total tax."""
    total, _ = calc_tax_breakdown(buy_df, end_date, final_close,
                                  sell_df, short_rate, long_rate)
    return total


# ── Shared helpers ────────────────────────────────────────────────────────────

FREQ_STEPS = {'Daily': 1, 'Weekly': 5, 'Biweekly': 10, 'Monthly': 21}


def _portfolio_series(df: pd.DataFrame, buy_df: pd.DataFrame) -> pd.DataFrame:
    cum    = buy_df[['Date', 'shares_cum']].copy()
    merged = df.merge(cum, on='Date', how='left')
    merged['shares_cum'] = merged['shares_cum'].ffill().fillna(0)
    merged['port_value'] = merged['shares_cum'] * merged['Close']
    peak   = merged['port_value'].cummax()
    merged['drawdown'] = np.where(peak > 0, merged['port_value'] / peak - 1, 0)
    return merged


def _base_stats(daily, total_invested, buy_count, buy_dates, final_value=None, **extra):
    fv       = final_value if final_value is not None else float(daily['port_value'].iloc[-1])
    profit   = fv - total_invested
    pct_gain = profit / total_invested * 100 if total_invested else 0.0

    days  = (daily['Date'].iloc[-1] - daily['Date'].iloc[0]).days
    years = days / 365.25
    pv    = daily['port_value']
    first = float(pv[pv > 0].iloc[0]) if (pv > 0).any() else None
    cagr  = ((fv / first) ** (1 / years) - 1) * 100 if (first and years > 0) else float('nan')
    arr   = calc_arr(profit, total_invested, years)

    return dict(
        total_invested=total_invested,
        final_value=fv,
        profit=profit,
        pct_gain=pct_gain,
        cagr=cagr,
        arr=arr,
        total_shares=float(daily['shares_cum'].iloc[-1]),
        buy_count=buy_count,
        sell_count=0,
        **extra,
    )


# ── Base DCA ──────────────────────────────────────────────────────────────────

def simulate_base_dca(df, start_date, end_date, amount, frequency):
    """Buy fixed $ every N trading days at Close."""
    mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
    df   = df[mask].copy().reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Not enough data in date range.")

    step   = FREQ_STEPS.get(frequency, 1)
    buy_df = df.iloc[list(range(0, len(df), step))].copy()
    buy_df['shares_bought'] = amount / buy_df['Close']
    buy_df['dollars']       = amount
    buy_df['shares_cum']    = buy_df['shares_bought'].cumsum()
    buy_df = buy_df.reset_index(drop=True)

    daily          = _portfolio_series(df, buy_df)
    total_invested = amount * len(buy_df)
    summary        = _base_stats(daily, total_invested, len(buy_df), list(buy_df['Date']))

    final_close = float(daily['Close'].iloc[-1])
    total_tax   = calc_tax_liability(buy_df, end_date, final_close)
    net_profit  = summary['profit'] - total_tax
    net_pct     = net_profit / total_invested * 100 if total_invested else 0.0
    summary.update(total_tax=total_tax, net_profit=net_profit, net_pct=net_pct)

    return daily, buy_df, summary


# ── Drawdown-triggered DCA ────────────────────────────────────────────────────

def simulate_drawdown_dca(df, start_date, end_date, amount, frequency,
                           buy_dd=0.15, stop_dd=0.10):
    """
    Buy at `frequency` intervals while price is buy_dd% below prior ATH.
    Stop buying once recovered within stop_dd% of ATH.
    """
    mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
    df   = df[mask].copy().reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Not enough data in date range.")

    step = FREQ_STEPS.get(frequency, 1)
    df['prior_ath'] = df['Close'].expanding().max().shift(1)

    n          = len(df)
    active     = False
    cycle_id   = 0
    shares     = 0.0
    invested   = 0.0
    buy_ctr    = 0
    trades     = []
    shares_arr = np.zeros(n)

    for i in range(n):
        prior_ath = df['prior_ath'].iloc[i]
        close     = df['Close'].iloc[i]
        exec_p    = df['Open'].iloc[i]

        if pd.notna(prior_ath):
            if not active and close <= (1 - buy_dd) * prior_ath:
                active   = True
                cycle_id += 1
                buy_ctr  = 0
            if active and (buy_ctr % step == 0):
                s = amount / exec_p
                shares   += s
                invested += amount
                trades.append({'Date': df['Date'].iloc[i], 'Price': exec_p,
                               'shares_bought': s, 'dollars': amount,
                               'shares_cum': shares, 'Cycle': cycle_id})
            if active:
                buy_ctr += 1
            if active and close >= (1 - stop_dd) * prior_ath:
                active = False

        shares_arr[i] = shares

    df = df.copy()
    df['shares_cum'] = shares_arr
    df['port_value'] = df['shares_cum'] * df['Close']
    peak = df['port_value'].cummax()
    df['drawdown'] = np.where(peak > 0, df['port_value'] / peak - 1, 0)

    buy_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['Date', 'Price', 'shares_bought', 'dollars', 'shares_cum', 'Cycle'])

    buy_dates = list(buy_df['Date']) if not buy_df.empty else []
    summary   = _base_stats(df, invested, len(buy_df), buy_dates, cycles=cycle_id)

    final_close = float(df['Close'].iloc[-1])
    total_tax   = calc_tax_liability(buy_df, end_date, final_close)
    net_profit  = summary['profit'] - total_tax
    net_pct     = net_profit / invested * 100 if invested else 0.0
    summary.update(total_tax=total_tax, net_profit=net_profit, net_pct=net_pct)

    return df, buy_df, summary


# ── ATH Buy & Sell ────────────────────────────────────────────────────────────

def simulate_ath_buy_sell(df, start_date, end_date, amount, frequency,
                           buy_pct=0.10, sell_pct=0.05, stop_buy_pct=0.0):
    """
    Buy phase: enter when close drops buy_pct% below rolling prior ATH.
               Continue buying at `frequency` while price stays below
               (1 - stop_buy_pct) * prior_ath; 0 = buy until sell signal.
    Sell:      exit (sell ALL shares) when close rises sell_pct% above the
               prior ATH recorded at the start of the buy phase.
    Executes at next day's Open.
    """
    mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
    df   = df[mask].copy().reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Not enough data in date range.")

    step = FREQ_STEPS.get(frequency, 1)
    df['prior_ath'] = df['Close'].expanding().max().shift(1)

    n           = len(df)
    buying      = False
    buy_active  = False
    cycle_id    = 0
    shares      = 0.0
    invested    = 0.0
    realized    = 0.0
    buy_ctr     = 0
    ref_ath     = None
    buy_trades  = []
    sell_trades = []
    shares_arr  = np.zeros(n)
    port_arr    = np.zeros(n)

    pending_buy  = False
    pending_sell = False

    for i in range(n):
        prior_ath = df['prior_ath'].iloc[i]
        close     = df['Close'].iloc[i]
        exec_p    = df['Open'].iloc[i]
        date_i    = df['Date'].iloc[i]

        if pending_sell and shares > 0:
            proceeds  = shares * exec_p
            realized += proceeds
            sell_trades.append({'Date': date_i, 'Price': exec_p,
                                'Shares': shares, 'Proceeds': proceeds,
                                'Cycle': cycle_id})
            shares       = 0.0
            pending_sell = False
            buying       = False
            buy_active   = False

        if pending_buy:
            s = amount / exec_p
            shares   += s
            invested += amount
            buy_trades.append({'Date': date_i, 'Price': exec_p,
                               'shares_bought': s, 'dollars': amount,
                               'shares_cum': shares, 'Cycle': cycle_id})
            pending_buy = False

        if pd.notna(prior_ath):
            if not buying and close <= (1 - buy_pct) * prior_ath:
                buying     = True
                buy_active = True
                cycle_id  += 1
                buy_ctr    = 0
                ref_ath    = prior_ath

            if buying and buy_active and stop_buy_pct > 0 and \
                    close >= (1 - stop_buy_pct) * prior_ath:
                buy_active  = False
                pending_buy = False

            if buying and buy_active and (buy_ctr % step == 0) and i < n - 1:
                pending_buy = True
            if buying and buy_active:
                buy_ctr += 1

            if buying and ref_ath is not None and close >= (1 + sell_pct) * ref_ath:
                if i < n - 1:
                    pending_sell = True
                    pending_buy  = False

        shares_arr[i] = shares
        port_arr[i]   = realized + shares * close

    df = df.copy()
    df['shares_cum'] = shares_arr
    df['port_value'] = port_arr
    peak = df['port_value'].cummax()
    df['drawdown'] = np.where(peak > 0, df['port_value'] / peak - 1, 0)

    buy_df  = pd.DataFrame(buy_trades)  if buy_trades  else pd.DataFrame(
        columns=['Date', 'Price', 'shares_bought', 'dollars', 'shares_cum', 'Cycle'])
    sell_df = pd.DataFrame(sell_trades) if sell_trades else pd.DataFrame(
        columns=['Date', 'Price', 'Shares', 'Proceeds', 'Cycle'])

    final_value    = float(port_arr[-1])
    total_invested = invested
    profit         = final_value - total_invested
    pct_gain       = profit / total_invested * 100 if total_invested else 0.0
    days           = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    years          = days / 365.25

    pv    = df['port_value']
    first = float(pv[pv > 0].iloc[0]) if (pv > 0).any() else None
    cagr  = ((final_value / first) ** (1 / years) - 1) * 100 \
            if (first and years > 0 and final_value > 0) else float('nan')
    arr   = calc_arr(profit, total_invested, years)

    final_close = float(df['Close'].iloc[-1])
    total_tax, tax_breakdown = calc_tax_breakdown(
        buy_df, end_date, final_close,
        sell_df=sell_df if sell_trades else None)
    net_profit_val = profit - total_tax
    net_pct_val    = net_profit_val / total_invested * 100 if total_invested else 0.0

    summary = dict(
        total_invested=total_invested,
        final_value=final_value,
        profit=profit,
        pct_gain=pct_gain,
        cagr=cagr,
        arr=arr,
        total_shares=float(shares_arr[-1]),
        buy_count=len(buy_df),
        sell_count=len(sell_df),
        cycles=cycle_id,
        realized=realized,
        total_tax=total_tax,
        net_profit=net_profit_val,
        net_pct=net_pct_val,
        tax_breakdown=tax_breakdown,
    )
    return df, buy_df, sell_df, summary
