# ETF Portfolio Dashboard

> Interactive dashboard for analyzing ETF portfolios: historical performance, volatility, and risk metrics.

## Overview

A web-based tool that allows users to build custom ETF portfolios by specifying tickers and weights, then analyze how those portfolios would have performed historically. The dashboard computes key financial metrics and visualizes results through interactive charts.

## Planned Features

- Portfolio construction with up to 10 ETFs and custom weights
- Historical performance simulation (buy & hold, with optional rebalancing)
- Key risk metrics: CAGR, annualized volatility, Sharpe ratio, maximum drawdown
- Correlation matrix across assets
- Comparison with benchmark indices
- Interactive charts (cumulative returns, drawdown, rolling volatility)

## Tech Stack

- **Python 3.14**
- **Streamlit** — web app framework
- **pandas / NumPy** — data manipulation and numerical computing
- **yfinance** — historical market data from Yahoo Finance
- **Plotly** — interactive charts

## Project Status

🚧 **Under active development.** Setup and project scaffolding complete. Core modules coming soon.

## Getting Started (for developers)

```bash
# Clone the repository
git clone https://github.com/giacomomignuzzi/etf-portfolio-dashboard.git
cd etf-portfolio-dashboard

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Author

**Giacomo Mignuzzi**  
[GitHub](https://github.com/giacomomignuzzi)

---

*This project is part of a personal portfolio to demonstrate skills in Python, financial analysis, and data visualization.*