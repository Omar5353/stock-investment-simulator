# Stock Investment Simulator

Python GUI for simulating DCA and ATH-based investment strategies using Yahoo Finance historical data.

## Features

- **3 strategies**
  - **Base DCA** — buy fixed $ at regular intervals
  - **Drawdown-Triggered DCA** — buy only when price drops X% below all-time high
  - **ATH Buy & Sell** — buy on dips below ATH, sell when price surpasses ATH by Y%
- **Metrics** — Total Return, CAGR, ARR (Accounting Rate of Return)
- **Plots** — price chart with buy/sell markers + portfolio value curve
- **Auto-save** — each simulation exports a PNG to the project folder

## Requirements

```
pip install yfinance matplotlib pandas numpy scipy
```

## Run

```bash
python3 stock_simulator.py
```

## Usage

1. Enter ticker symbol (e.g. `MSFT`, `AAPL`, `BMW.DE`)
2. Set investment amount per buy, frequency, strategy, date range
3. Press **Run Simulation** or hit `Return`
4. Plot saves automatically as `{TICKER}_{Strategy}_{dates}_{gain}.png`

## Data source

Yahoo Finance via `yfinance` — supports US stocks and international markets.
