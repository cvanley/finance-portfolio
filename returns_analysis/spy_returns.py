"""
spy_returns.py
-----------------
Annualized return and volatility for daily return series.

Data Source : OpenBB / yfinance
Ticker      : configurable (default: SPY)
Period      : configurable (default: 2020-01-01 to 2024-12-31)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Constants ---
TRADING_DAYS = 252


def fetch_prices(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Pull adjusted closing prices from OpenBB via yfinance.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (e.g., 'SPY').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.Series
        Daily adjusted closing prices, indexed by date.
    """
    result = obb.equity.price.historical(
        symbol=ticker,
        start_date=start_date,
        end_date=end_date,
        provider="yfinance",
    )
    prices = result.to_df()["close"]

    assert prices.isna().sum() == 0, f"Missing prices in {ticker} — check date range"
    assert (prices > 0).all(), f"Non-positive prices detected in {ticker}"

    return prices


def compute_annual_return(daily_returns: pd.Series) -> float:
    """
    Compound daily returns to an annualized figure.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily sample returns as decimals (e.g., 0.01 for 1%)

    Returns
    -------
    float
        Annualized return
    """
    assert daily_returns.isna().sum() == 0, "Missing values detected in returns"
    assert len(daily_returns) > 0, "Returns series is empty"

    total_return = (1 + daily_returns).prod()
    annualized = total_return ** (TRADING_DAYS / len(daily_returns)) - 1
    return annualized


def compute_annual_volatility(daily_returns: pd.Series) -> float:
    """
    Annualize daily return volatility using square-root-of-time scaling.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily simple returns as decimals.

    Returns
    -------
    float
        Annualized volatility.
    """
    assert (
        daily_returns.isna().sum() == 0
    ), "NaN values in returns — cannot compute volatility"
    assert len(daily_returns) > 1, "Need at least 2 observations"

    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def plot_cumulative_return(
    daily_returns: pd.Series, ticker: str, output_path: str
) -> None:
    """
    Plot and save cumulative return series to disk.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily simple returns.
    ticker : str
        Label used in chart title.
    output_path : str
        File path to save the figure (e.g., 'spy_cumulative_return.png').
    """
    cumulative = (1 + daily_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative.index, cumulative, linewidth=1.5, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{ticker} Cumulative Return", fontsize=14)
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    from openbb import obb

    TICKER = "SPY"
    START = "2020-01-01"
    END = "2024-12-31"

    prices = fetch_prices(TICKER, START, END)
    returns = prices.pct_change().dropna()

    annual_return = compute_annual_return(returns)
    annual_vol = compute_annual_volatility(returns)

    print(f"Ticker:                 {TICKER}")
    print(f"Period:                 {START} to {END}")
    print(f"Annualized Return:      {annual_return:.2%}")
    print(f"Annualized Volatility:  {annual_vol:.2%}")

    plot_cumulative_return(
        returns, TICKER, "returns_analysis/spy_cumulative_return.png"
    )
