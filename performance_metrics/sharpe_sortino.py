"""
sharpe_sortino.py
-----------------
Annualized Sharpe and Sortino ratios for daily return series.

Data Source : OpenBB / yfinance
Ticker      : configurable (default: SPY)
Period      : configurable (default: 2020-01-01 to 2024-12-31)
"""

import numpy as np
import pandas as pd
from openbb import obb

# --- Constants ---
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0525  # 3-month T-bill — update quarterly


def compute_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE
) -> float:
    """
    Compute the annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns (not prices).
    risk_free_rate : float
        Annualized risk-free rate. Defaults to module constant.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    assert (
        returns.isna().sum() == 0
    ), "Missing returns detected — clean data before computing Sharpe"
    assert len(returns) > 1, "Need at least 2 observations"

    daily_rf = risk_free_rate / TRADING_DAYS

    excess_returns = returns - daily_rf
    annualized_ex_ret = excess_returns.mean() * TRADING_DAYS
    annualized_vol = excess_returns.std() * np.sqrt(TRADING_DAYS)

    sharpe = annualized_ex_ret / annualized_vol

    return sharpe


if __name__ == "__main__":

    prices = obb.equity.price.historical(
        "SPY", start_date="2020-01-01", end_date="2024-12-31", provider="yfinance"
    ).to_df()["close"]
    returns = prices.pct_change().dropna()

    sharpe = compute_sharpe_ratio(returns)
    print(f"SPY Sharpe Ratio (2020-2024): {sharpe:.4f}")
