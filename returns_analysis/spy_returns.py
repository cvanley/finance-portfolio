# spy_returns.py
# Goal: Compute annualized return and volatility for SPY
# Data: Will use OpenBB in next exercise — synthetic data today

import pandas as pd

TRADING_DAYS = 252  # US equity market convention


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


if __name__ == "__main__":
    # Synthetic test — 1 year of flat 0.05% daily returns
    test_returns = pd.Series([0.0005] * TRADING_DAYS)
    result = compute_annual_return(test_returns)
    print(f"Annualized Return: {result:.2%}")
