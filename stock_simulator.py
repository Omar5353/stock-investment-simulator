#!/usr/bin/env python3
"""
Stock Investment Simulator
GUI for DCA and ATH-based simulations using Yahoo Finance historical data.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from simulator_core import (
    fetch_data, calc_arr, calc_tax_breakdown, calc_tax_liability,
    FREQ_STEPS, _portfolio_series, _base_stats,
    simulate_base_dca, simulate_drawdown_dca, simulate_ath_buy_sell,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = '#1e1e2e'
PANEL   = '#24273a'
CARD    = '#2a2d3e'
ACCENT  = '#89b4fa'
GREEN   = '#a6e3a1'
RED     = '#f38ba8'
YELLOW  = '#f9e2af'
TEXT    = '#ffffff'        # white — main label text
SUBTEXT = '#cdd6f4'        # light lavender — secondary text / values
MUTED   = '#6c7086'        # grey — kept only for decorative separators / borders
BORDER  = '#363a4f'

FONT       = ('Segoe UI', 10)
FONT_BOLD  = ('Segoe UI', 10, 'bold')
FONT_SMALL = ('Segoe UI', 9)
FONT_LG    = ('Segoe UI', 13, 'bold')


# ── GUI ───────────────────────────────────────────────────────────────────────

class StockSimulator(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Stock Investment Simulator")
        self.geometry("1380x900")
        self.minsize(1000, 720)
        self.configure(bg=BG)
        self._tax_breakdown = None   # populated after ATH B&S run
        self._setup_style()
        self._build_ui()
        self.after(100, self.focus_force)   # ensure window gets focus on start

    # ── ttk style ─────────────────────────────────────────────────────────────

    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('TFrame',       background=BG)
        s.configure('TEntry',       fieldbackground='#313244', foreground=TEXT,
                    insertcolor=TEXT, font=FONT, relief='flat')
        s.configure('TCombobox',    fieldbackground='#313244', foreground=TEXT,
                    selectbackground='#313244', font=FONT)
        s.map('TCombobox',          fieldbackground=[('readonly', '#313244')],
                                    foreground=[('readonly', TEXT)])
        s.configure('Run.TButton',  background=ACCENT, foreground='#1e1e2e',
                    font=FONT_BOLD, padding=(8, 7), relief='flat', borderwidth=0)
        s.map('Run.TButton',        background=[('active', '#74c7ec'), ('disabled', MUTED)],
                                    foreground=[('disabled', BG)])
        s.configure('TSeparator',   background=BORDER)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Scrollable sidebar ────────────────────────────────────────────────
        sidebar_outer = tk.Frame(self, bg=PANEL, width=285)
        sidebar_outer.pack(side='left', fill='y', padx=(6, 0), pady=6)
        sidebar_outer.pack_propagate(False)

        canvas_sb = tk.Canvas(sidebar_outer, bg=PANEL, highlightthickness=0, width=283)
        scrollbar = ttk.Scrollbar(sidebar_outer, orient='vertical', command=canvas_sb.yview)
        canvas_sb.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas_sb.pack(side='left', fill='both', expand=True)

        sidebar = tk.Frame(canvas_sb, bg=PANEL)
        win_id  = canvas_sb.create_window((0, 0), window=sidebar, anchor='nw', width=265)

        def _on_resize(e):
            canvas_sb.itemconfig(win_id, width=e.width)
        canvas_sb.bind('<Configure>', _on_resize)

        def _on_frame_cfg(e):
            canvas_sb.configure(scrollregion=canvas_sb.bbox('all'))
        sidebar.bind('<Configure>', _on_frame_cfg)

        # mouse-wheel scroll
        def _on_wheel(e):
            canvas_sb.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        canvas_sb.bind_all('<MouseWheel>', _on_wheel)

        pad = {'padx': 14}

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(sidebar, text="Stock Simulator", bg=PANEL, fg=ACCENT,
                 font=FONT_LG).pack(anchor='w', pady=(14, 1), **pad)
        tk.Label(sidebar, text="Yahoo Finance  •  DCA & ATH strategies",
                 bg=PANEL, fg=TEXT, font=FONT_SMALL).pack(anchor='w', pady=(0, 10), **pad)
        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=10)

        # ── Input helpers ─────────────────────────────────────────────────────
        def lbl(parent, text, top=8, bottom=1):
            tk.Label(parent, text=text, bg=PANEL, fg=TEXT,
                     font=FONT_SMALL).pack(anchor='w', pady=(top, bottom), **pad)

        def entry(parent, default='', var=None):
            if var is None:
                var = tk.StringVar(value=default)
            e = ttk.Entry(parent, textvariable=var)
            e.pack(fill='x', padx=14, ipady=3)
            e.bind('<Return>', lambda _: self._run_async())
            return var

        def combo(parent, choices, default=None):
            var = tk.StringVar(value=default or choices[0])
            cb  = ttk.Combobox(parent, textvariable=var, values=choices,
                                state='readonly', font=FONT)
            cb.pack(fill='x', padx=14, ipady=3)
            return var

        lbl(sidebar, "Ticker Symbol", top=14)
        self.v_ticker = entry(sidebar, "MSFT")

        lbl(sidebar, "Investment per Buy ($)")
        self.v_amount = entry(sidebar, "100")

        lbl(sidebar, "Buy Frequency")
        self.v_freq = combo(sidebar, ["Daily", "Weekly", "Biweekly", "Monthly"], "Monthly")

        lbl(sidebar, "Strategy")
        self.v_strat = combo(sidebar,
                              ["Base DCA",
                               "Drawdown-Triggered DCA",
                               "ATH Buy & Sell"],
                              "Base DCA")

        lbl(sidebar, "Start Date  (YYYY-MM-DD)")
        self.v_start = entry(sidebar, "2015-01-01")

        lbl(sidebar, "End Date  (YYYY-MM-DD)")
        self.v_end = entry(sidebar, "2025-01-01")

        # ── Drawdown settings ─────────────────────────────────────────────────
        self.dd_frame = tk.Frame(sidebar, bg=PANEL)
        tk.Label(self.dd_frame, text="─── Drawdown Settings ───",
                 bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(anchor='w', padx=14, pady=(10, 4))
        lbl(self.dd_frame, "Buy trigger  (% below ATH)", top=2)
        self.v_dd_buy  = entry(self.dd_frame, "15")
        lbl(self.dd_frame, "Stop trigger  (% below ATH)", top=6)
        self.v_dd_stop = entry(self.dd_frame, "10")

        # ── ATH Buy & Sell settings ───────────────────────────────────────────
        self.ath_frame = tk.Frame(sidebar, bg=PANEL)
        tk.Label(self.ath_frame, text="─── ATH Buy & Sell Settings ───",
                 bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(anchor='w', padx=14, pady=(10, 4))
        lbl(self.ath_frame, "Buy when  (% below ATH)", top=2)
        self.v_ath_buy  = entry(self.ath_frame, "10")
        lbl(self.ath_frame, "Stop buying  (% below ATH, 0 = off)", top=6)
        self.v_ath_stop = entry(self.ath_frame, "0")
        lbl(self.ath_frame, "Sell when  (% above ATH at buy)", top=6)
        self.v_ath_sell = entry(self.ath_frame, "5")

        def on_strat(*_):
            self.dd_frame.pack_forget()
            self.ath_frame.pack_forget()
            s = self.v_strat.get()
            if s == "Drawdown-Triggered DCA":
                self.dd_frame.pack(fill='x')
            elif s == "ATH Buy & Sell":
                self.ath_frame.pack(fill='x')
        self.v_strat.trace_add('write', on_strat)
        on_strat()

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=10, pady=14)

        # ── Run button ────────────────────────────────────────────────────────
        # tk.Button used instead of ttk — more reliable click registration on macOS
        # inside Canvas-hosted frames. Dual binding: command= + ButtonRelease-1.
        self.run_btn = tk.Button(
            sidebar, text="▶  Run Simulation",
            bg=ACCENT, fg='#1e1e2e', font=FONT_BOLD,
            relief='flat', bd=0, padx=8, pady=8,
            activebackground='#74c7ec', activeforeground='#1e1e2e',
            cursor='hand2',
            command=self._run_async)
        self.run_btn.pack(fill='x', padx=14)
        self.run_btn.bind('<ButtonRelease-1>', lambda _: self._run_async())
        self.bind('<Return>', lambda _: self._run_async())

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(sidebar, textvariable=self.status_var,
                 bg=PANEL, fg=TEXT, font=FONT_SMALL).pack(anchor='w', padx=14, pady=(4, 0))

        ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=10, pady=14)

        # ── Results cards ─────────────────────────────────────────────────────
        tk.Label(sidebar, text="Results", bg=PANEL, fg=ACCENT,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', **pad, pady=(0, 8))

        self._result_vars = {}
        result_rows = [
            ('total_invested', 'Total Invested',       None),
            ('final_value',    'Final Value',           None),
            ('profit',         'Capital Gain',          'signed'),
            ('pct_gain',       'Total Return',          'signed_pct'),
            ('cagr',           'CAGR',                  'signed_pct'),
            ('arr',            'ARR',                   'signed_pct'),
            ('buy_count',      '# of Buys',             None),
            ('sell_count',     '# of Sells',            None),
            ('total_tax',      'Est. Tax (20%/10%)',    'signed'),
            ('net_profit',     'Net Gain (After Tax)',  'signed'),
            ('net_pct',        'Net Return (After Tax)','signed_pct'),
        ]

        results_frame = tk.Frame(sidebar, bg=PANEL)
        results_frame.pack(fill='x', padx=12, pady=(0, 14))

        for key, label, kind in result_rows:
            row = tk.Frame(results_frame, bg=CARD, pady=5)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg=CARD, fg=TEXT,
                     font=FONT_SMALL).pack(side='left', padx=8)
            var   = tk.StringVar(value="—")
            lbl_w = tk.Label(row, textvariable=var, bg=CARD, fg=GREEN,
                              font=FONT_BOLD, anchor='e')
            lbl_w.pack(side='right', padx=8)
            self._result_vars[key] = (var, lbl_w, kind)

        # ── Tax breakdown button (ATH B&S only) ───────────────────────────────
        self.tax_btn = tk.Button(
            sidebar, text="📋  Tax Breakdown Detail",
            bg=CARD, fg=MUTED, font=FONT_SMALL,
            relief='flat', bd=0, padx=8, pady=6,
            cursor='hand2', state='disabled',
            command=self._show_tax_breakdown)
        self.tax_btn.pack(fill='x', padx=12, pady=(0, 14))
        self.tax_btn.bind('<ButtonRelease-1>', lambda _: self._show_tax_breakdown())

        # ── Credit ────────────────────────────────────────────────────────────
        tk.Label(sidebar, text="Brought to you by Mohammad Omar Faruk Murad",
                 bg=PANEL, fg=MUTED, font=('Segoe UI', 7),
                 wraplength=240, justify='left').pack(
                     anchor='w', padx=14, pady=(4, 12))

        # ── Plot area ─────────────────────────────────────────────────────────
        plot_frame = tk.Frame(self, bg=BG)
        plot_frame.pack(side='right', fill='both', expand=True, padx=6, pady=6)

        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor=BG)
        self.fig.subplots_adjust(hspace=0.38, left=0.08, right=0.97, top=0.93, bottom=0.09)

        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar_frame = tk.Frame(plot_frame, bg=BG)
        toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.config(background=BG)
        self.toolbar.update()

        self._placeholder()

    # ── Axes styling ──────────────────────────────────────────────────────────

    def _style_axes(self):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.8)

    def _placeholder(self):
        for ax, txt in [(self.ax1, "Price + trade markers will appear here"),
                         (self.ax2, "Portfolio value will appear here")]:
            ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                    ha='center', va='center', color=MUTED, fontsize=11)
        self.canvas.draw()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        ticker = self.v_ticker.get().strip()
        if not ticker:
            raise ValueError("Ticker symbol is required.")

        try:
            amount = float(self.v_amount.get())
            if amount <= 0:
                raise ValueError
        except ValueError:
            raise ValueError("Investment amount must be a positive number.")

        try:
            start = datetime.strptime(self.v_start.get().strip(), '%Y-%m-%d').date()
            end   = datetime.strptime(self.v_end.get().strip(), '%Y-%m-%d').date()
        except ValueError:
            raise ValueError("Dates must be YYYY-MM-DD (e.g. 2020-01-01).")
        if start >= end:
            raise ValueError("Start date must be before end date.")

        strat = self.v_strat.get()
        extra = {}

        if strat == "Drawdown-Triggered DCA":
            try:
                dd_buy  = float(self.v_dd_buy.get())
                dd_stop = float(self.v_dd_stop.get())
            except ValueError:
                raise ValueError("Drawdown values must be numbers.")
            if not (0 < dd_stop < dd_buy < 100):
                raise ValueError("Need: 0 < Stop trigger < Buy trigger < 100")
            extra = {'dd_buy': dd_buy / 100, 'dd_stop': dd_stop / 100}

        elif strat == "ATH Buy & Sell":
            try:
                ath_buy  = float(self.v_ath_buy.get())
                ath_stop = float(self.v_ath_stop.get())
                ath_sell = float(self.v_ath_sell.get())
            except ValueError:
                raise ValueError("ATH values must be numbers.")
            if not (0 < ath_buy < 100):
                raise ValueError("Buy threshold must be between 0 and 100.")
            if not (0 <= ath_stop < 100):
                raise ValueError("Stop buying must be between 0 (off) and 100.")
            if ath_stop > 0 and ath_stop >= ath_buy:
                raise ValueError("Stop buying % must be less than Buy % (or 0 to disable).")
            if not (0 < ath_sell < 100):
                raise ValueError("Sell threshold must be between 0 and 100.")
            extra = {'ath_buy': ath_buy / 100, 'ath_stop': ath_stop / 100,
                     'ath_sell': ath_sell / 100}

        return ticker, amount, start, end, strat, extra

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run_async(self):
        if self.run_btn['state'] == 'disabled':
            return
        try:
            args = self._validate()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e), parent=self)
            return
        self.run_btn.config(state='disabled', bg=MUTED)
        self.status_var.set("Fetching data…")
        threading.Thread(target=self._run_thread, args=(args,), daemon=True).start()

    def _run_thread(self, args):
        ticker, amount, start, end, strat, extra = args
        try:
            df = fetch_data(ticker)
            self.after(0, lambda: self.status_var.set("Running simulation…"))
            freq = self.v_freq.get()

            if strat == "Base DCA":
                daily, buy_df, summary = simulate_base_dca(df, start, end, amount, freq)
                sell_df = None
            elif strat == "Drawdown-Triggered DCA":
                daily, buy_df, summary = simulate_drawdown_dca(
                    df, start, end, amount, freq,
                    buy_dd=extra['dd_buy'], stop_dd=extra['dd_stop'])
                sell_df = None
            else:  # ATH Buy & Sell
                daily, buy_df, sell_df, summary = simulate_ath_buy_sell(
                    df, start, end, amount, freq,
                    buy_pct=extra['ath_buy'], stop_buy_pct=extra['ath_stop'],
                    sell_pct=extra['ath_sell'])

            self.after(0, self._update_ui, ticker, daily, buy_df, sell_df, summary)
        except Exception as exc:
            self.after(0, self._on_error, str(exc))

    def _on_error(self, msg):
        self.run_btn.config(state='normal', bg=ACCENT)
        self.status_var.set("Error — check inputs")
        messagebox.showerror("Simulation Error", msg, parent=self)

    # ── Update UI ─────────────────────────────────────────────────────────────

    def _update_ui(self, ticker, daily, buy_df, sell_df, summary):
        self.run_btn.config(state='normal', bg=ACCENT)
        strat = self.v_strat.get()

        def usd(v): return f"${float(v):,.2f}"
        def pct(v): return f"{float(v):.2f}%"
        def sfmt(v, is_pct=False):
            fv = float(v)
            s  = pct(fv) if is_pct else usd(fv)
            return f"+{s}" if fv >= 0 else s

        def safe_pct(v):
            fv = float(v)
            if np.isnan(fv):
                return "N/A"
            return sfmt(fv, is_pct=True)

        vals = {
            'total_invested': usd(summary['total_invested']),
            'final_value':    usd(summary['final_value']),
            'profit':         sfmt(summary['profit']),
            'pct_gain':       sfmt(summary['pct_gain'], is_pct=True),
            'cagr':           safe_pct(summary['cagr']),
            'arr':            safe_pct(summary['arr']),
            'buy_count':      str(summary['buy_count']),
            'sell_count':     str(summary.get('sell_count', 0)),
            'total_tax':      sfmt(summary.get('total_tax', 0.0)),
            'net_profit':     sfmt(summary.get('net_profit', summary['profit'])),
            'net_pct':        sfmt(summary.get('net_pct', summary['pct_gain']), is_pct=True),
        }

        def _signed_color(text_val: str) -> str:
            """Return GREEN or RED based on numeric sign of a formatted value string."""
            clean = text_val.replace('$', '').replace('%', '').replace('+', '').replace(',', '')
            try:
                return GREEN if float(clean) >= 0 else RED
            except Exception:
                return SUBTEXT

        for key, (var, lbl_w, kind) in self._result_vars.items():
            var.set(vals[key])
            if kind in ('signed', 'signed_pct'):
                fg = _signed_color(vals[key])
                lbl_w.config(fg=fg)
            else:
                lbl_w.config(fg=GREEN)

        # Tax breakdown button — only active for ATH B&S
        if strat == "ATH Buy & Sell" and summary.get('tax_breakdown'):
            self._tax_breakdown = summary['tax_breakdown']
            self.tax_btn.config(state='normal', fg=ACCENT, bg=CARD,
                                activebackground=BORDER, activeforeground=TEXT)
        else:
            self._tax_breakdown = None
            self.tax_btn.config(state='disabled', fg=MUTED, bg=CARD)

        self._draw_plots(ticker, daily, buy_df, sell_df, summary)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _draw_plots(self, ticker, daily, buy_df, sell_df, summary):
        self.ax1.clear()
        self.ax2.clear()
        self._style_axes()

        dates = daily['Date']
        strat = self.v_strat.get()
        freq  = self.v_freq.get()

        # ── Subplot 1: Price + trade markers ──────────────────────────────────
        self.ax1.plot(dates, daily['Close'],
                      color=ACCENT, linewidth=1.3, label='Close Price', zorder=2)

        if not buy_df.empty:
            price_col = 'Price' if 'Price' in buy_df.columns else 'Close'
            self.ax1.scatter(buy_df['Date'], buy_df[price_col],
                             color=RED, s=24, zorder=5, alpha=0.88,
                             label=f'Buy ({len(buy_df)})', edgecolors='none', marker='^')

        if sell_df is not None and not sell_df.empty:
            self.ax1.scatter(sell_df['Date'], sell_df['Price'],
                             color=GREEN, s=30, zorder=5, alpha=0.9,
                             label=f'Sell ({len(sell_df)})', edgecolors='none', marker='v')

        self.ax1.set_title(
            f"{ticker.upper()}  —  {strat}  •  {freq}  •  "
            f"${summary['total_invested']:,.0f} invested",
            color=TEXT, fontsize=10, pad=8, loc='left')
        self.ax1.set_ylabel("Price ($)", color=TEXT, fontsize=8)
        self.ax1.legend(facecolor=CARD, edgecolor=BORDER,
                         labelcolor=TEXT, fontsize=8, framealpha=0.9)
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        self.ax1.xaxis.set_major_locator(mdates.YearLocator())

        # ── Subplot 2: Portfolio value ────────────────────────────────────────
        port     = daily['port_value']
        invested = summary['total_invested']
        final    = summary['final_value']
        profit   = summary['profit']
        pct_gain = summary['pct_gain']
        pcolor   = GREEN if float(profit) >= 0 else RED

        self.ax2.fill_between(dates, port, alpha=0.12, color=pcolor)
        self.ax2.plot(dates, port, color=pcolor, linewidth=1.4,
                      label='Portfolio Value', zorder=3)

        if invested > 0:
            self.ax2.axhline(float(invested), color=YELLOW, linewidth=0.9,
                              linestyle='--', alpha=0.7,
                              label=f'Invested  ${float(invested):,.0f}')

        self.ax2.annotate(
            f"  ${float(final):,.0f}",
            xy=(daily['Date'].iloc[-1], float(final)),
            color=pcolor, fontsize=8, va='center',
            xytext=(5, 0), textcoords='offset points')

        sign = '+' if float(pct_gain) >= 0 else ''
        self.ax2.set_title(
            f"Portfolio Value  |  Final: ${float(final):,.2f}  |  "
            f"Gain: ${float(profit):+,.2f}  ({sign}{float(pct_gain):.1f}%)",
            color=pcolor, fontsize=10, pad=8, loc='left')
        self.ax2.set_ylabel("Value ($)", color=TEXT, fontsize=8)
        self.ax2.set_xlabel("Date", color=TEXT, fontsize=8)
        self.ax2.legend(facecolor=CARD, edgecolor=BORDER,
                         labelcolor=TEXT, fontsize=8, framealpha=0.9)
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        self.ax2.xaxis.set_major_locator(mdates.YearLocator())

        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()

        # auto-save after every simulation run
        self._save_plots(ticker, summary)

    # ── Tax breakdown popup ───────────────────────────────────────────────────

    def _show_tax_breakdown(self):
        if not self._tax_breakdown or self.tax_btn['state'] == 'disabled':
            return

        win = tk.Toplevel(self)
        win.title("Tax Breakdown — ATH Buy & Sell")
        win.configure(bg=BG)
        win.geometry("880x620")
        win.minsize(700, 400)

        # header label
        tk.Label(win, text="Capital Gains Tax Breakdown  (FIFO)  •  Short-term ≤ 1 yr: 20%   Long-term > 1 yr: 10%",
                 bg=BG, fg=ACCENT, font=FONT_BOLD).pack(anchor='w', padx=10, pady=(8, 2))
        tk.Label(win, text="Red = short-term   Green = long-term   Yellow = unrealized (end of period)",
                 bg=BG, fg=MUTED, font=FONT_SMALL).pack(anchor='w', padx=10, pady=(0, 6))

        # scrollable text area
        frame = tk.Frame(win, bg=BG)
        frame.pack(fill='both', expand=True, padx=8, pady=(0, 4))

        sb_y = ttk.Scrollbar(frame, orient='vertical')
        sb_y.pack(side='right', fill='y')
        sb_x = ttk.Scrollbar(frame, orient='horizontal')
        sb_x.pack(side='bottom', fill='x')

        txt = tk.Text(frame, bg=CARD, fg=TEXT,
                      font=('Courier New', 9),
                      yscrollcommand=sb_y.set, xscrollcommand=sb_x.set,
                      wrap='none', padx=10, pady=8, relief='flat', bd=0,
                      selectbackground=BORDER)
        txt.pack(fill='both', expand=True)
        sb_y.config(command=txt.yview)
        sb_x.config(command=txt.xview)

        txt.tag_config('hdr',     foreground=ACCENT,  font=('Courier New', 9, 'bold'))
        txt.tag_config('col',     foreground=SUBTEXT, font=('Courier New', 9, 'bold'))
        txt.tag_config('short',   foreground=RED)
        txt.tag_config('long',    foreground=GREEN)
        txt.tag_config('unreal',  foreground=YELLOW)
        txt.tag_config('total',   foreground=ACCENT,  font=('Courier New', 9, 'bold'))
        txt.tag_config('muted',   foreground=MUTED)
        txt.tag_config('sep',     foreground=BORDER)

        W   = 96
        SEP  = '─' * W + '\n'
        SEP2 = '═' * W + '\n'

        COL = (f"  {'Buy Date':<12}  {'Buy $':>8}  {'Shares':>10}  "
               f"{'Gain ($)':>13}  {'Days':>5}  {'Term':<6}  {'Rate':>5}  {'Tax ($)':>13}\n")

        sell_idx   = 0
        total_all  = 0.0

        for event in self._tax_breakdown:
            is_sell = event['type'] == 'sell'

            if is_sell:
                sell_idx += 1
                total_shares = sum(l['shares'] for l in event['lots'])
                hdr = (f"  SELL #{sell_idx}  —  {event['date'].strftime('%Y-%m-%d')}  "
                       f"@  ${event['sell_price']:,.2f}  "
                       f"({total_shares:.4f} shares sold)\n")
                hdr_tag = 'hdr'
            else:
                hdr = (f"  END OF PERIOD (Unrealized)  —  {event['date'].strftime('%Y-%m-%d')}  "
                       f"@  ${event['sell_price']:,.2f}\n")
                hdr_tag = 'unreal'

            txt.insert('end', SEP, 'sep')
            txt.insert('end', hdr, hdr_tag)
            txt.insert('end', SEP, 'sep')
            txt.insert('end', COL, 'col')
            txt.insert('end', '  ' + '·' * (W - 2) + '\n', 'muted')

            for lot in event['lots']:
                tag  = 'long' if lot['term'] == 'Long' else ('unreal' if not is_sell else 'short')
                line = (f"  {lot['buy_date'].strftime('%Y-%m-%d'):<12}  "
                        f"${lot['buy_price']:>7,.2f}  "
                        f"{lot['shares']:>10.4f}  "
                        f"${lot['gain']:>12,.2f}  "
                        f"{lot['days_held']:>5}  "
                        f"{lot['term']:<6}  "
                        f"{lot['rate']*100:>4.0f}%  "
                        f"${lot['tax']:>12,.2f}\n")
                txt.insert('end', line, tag)

            # event subtotal
            txt.insert('end', '  ' + '─' * (W - 2) + '\n', 'muted')
            sub = (f"  {'Subtotal':<12}  {'':>8}  "
                   f"{sum(l['shares'] for l in event['lots']):>10.4f}  "
                   f"${event['total_gain']:>12,.2f}  "
                   f"{'':>5}  {'':>6}  {'':>5}  "
                   f"${event['total_tax']:>12,.2f}\n\n")
            txt.insert('end', sub, 'total')
            total_all += event['total_tax']

        txt.insert('end', SEP2, 'sep')
        txt.insert('end', f"  TOTAL ESTIMATED TAX:  ${total_all:,.2f}\n", 'total')
        txt.insert('end', SEP2, 'sep')
        txt.config(state='disabled')

    # ── Save plots ────────────────────────────────────────────────────────────

    def _save_plots(self, ticker: str, summary: dict):
        """Save current figure as PNG inside a datetime-stamped subfolder."""
        ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
        base   = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base, ts)
        os.makedirs(folder, exist_ok=True)

        strat  = self.v_strat.get().replace(' ', '_').replace('&', 'and')
        freq   = self.v_freq.get()
        start  = self.v_start.get().replace('-', '')
        end    = self.v_end.get().replace('-', '')
        gain   = f"{float(summary['pct_gain']):+.0f}pct"
        fname  = f"{ticker.upper()}_{strat}_{freq}_{start}_{end}_{gain}.png"
        out    = os.path.join(folder, fname)

        sells    = summary.get('sell_count', 0)
        sell_str = f"  {sells} sells  •" if sells else ""
        try:
            self.fig.savefig(out, dpi=150, bbox_inches='tight',
                             facecolor=BG, edgecolor='none')
            self.status_var.set(
                f"Done  •  {summary['buy_count']} buys  •{sell_str}  {ts}/{fname}")
        except Exception as e:
            self.status_var.set(f"Done  •  Save failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = StockSimulator()
    app.mainloop()
