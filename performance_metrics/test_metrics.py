"""
test_metrics.py
---------------
PyTest suite for sharpe_sortino.py.
"""

import numpy as np
import pandas as pd
import pytest
from sharpe_sortino import (
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)


def test_sharpe_raises_on_nan():
    """
    compute_sharpe_ratio raises AssertionError when returns contain NaN.
    """
    returns = pd.Series([0.01, np.nan, -0.01])

    with pytest.raises(AssertionError):
        compute_sharpe_ratio(returns)


def test_sharpe_known_output():
    """
    Sharpe ratio matches hand-calculated value on alternating return series.
    """
    returns = pd.Series([0.01, -0.01] * 126)
    daily_rf = 0.0525 / 252
    excess = returns - daily_rf
    expected = (excess.mean() * 252) / (excess.std() * np.sqrt(252))

    sharpe = compute_sharpe_ratio(returns)

    assert sharpe == pytest.approx(expected, abs=1e-4)


def test_sortino_known_output():
    """
    Sortino ratio matches hand-calculated value on alternating return series.
    """
    returns = pd.Series([0.01, -0.01] * 126)
    daily_rf = 0.0525 / 252
    excess = returns - daily_rf
    downside = excess.clip(upper=0)
    downside_deviation = np.sqrt((downside**2).mean() * 252)
    expected = (excess.mean() * 252) / downside_deviation

    sortino = compute_sortino_ratio(returns)

    assert sortino == pytest.approx(expected, abs=1e-4)


def test_max_drawdown_basic():
    """
    Max drawdown matches hand-calculated value on basic return series.
    """
    returns = pd.Series([0.10, -0.20])
    expected = -0.20

    max_drawdown = compute_max_drawdown(returns)

    assert max_drawdown == pytest.approx(expected, abs=1e-4)


def test_max_drawdown_no_drawdown():
    """
    Max drawdown matches hand-calculated value on monotonically rising series.
    """
    returns = pd.Series([0.05, 0.05, 0.05])
    expected = 0.0

    max_drawdown = compute_max_drawdown(returns)

    assert max_drawdown == pytest.approx(expected, abs=1e-4)


def test_calmar_ratio_known_output():
    """
    Calmar ratio matches hand-calculated value on a two-element series.
    """
    returns = pd.Series([0.001] * 251 + [-0.10])
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    max_drawdown = -0.10
    expected = annualized_return / abs(max_drawdown)

    calmar_ratio = compute_calmar_ratio(returns)

    assert calmar_ratio == pytest.approx(expected, abs=1e-4)
