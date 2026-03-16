"""
Backtesting functions for portfolio strategies

Contains:
- backtest_rebalance: Backtest a rebalancing strategy
- summary_table: Generate summary statistics table
- BacktestResult: Data class for backtest results
"""

from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class BacktestResult:
    """Result of a backtest"""
    strategy: str
    wealth: pd.Series
    returns: pd.Series
    annual_return: float
    std_dev: float
    sharpe: float
    max_drawdown: float
    turnover: float
    tx_cost_cum: float


def backtest_rebalance(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy_name: str = "Strategy",
    rebalance_every: int = 126,
    transaction_cost_rate: float = 0.003,
    weight_function: Optional[Callable] = None,
) -> BacktestResult:
    """
    Backtest a rebalancing strategy
    
    Args:
        prices: DataFrame với prices (index = dates, columns = tickers)
        returns: DataFrame với returns (index = dates, columns = tickers)
        strategy_name: Tên strategy
        rebalance_every: Số ngày giữa các lần rebalance (10_000 = không rebalance = Buy&Hold)
        transaction_cost_rate: Phí giao dịch (mặc định 0.3%)
        weight_function: Function để tính weights (prices, returns) -> weights array
                         Nếu None, dùng equal weights
    
    Returns:
        BacktestResult object
    """
    # Align prices and returns
    common_idx = prices.index.intersection(returns.index)
    prices = prices.loc[common_idx].sort_index()
    returns = returns.loc[common_idx].sort_index()
    
    if len(prices) == 0:
        raise ValueError("No common dates between prices and returns")
    
    n_stocks = len(prices.columns)
    n_days = len(prices)
    
    # Initialize wealth và weights
    wealth = pd.Series(index=prices.index, dtype=float)
    wealth.iloc[0] = 1.0  # Start with 1.0
    
    # Current weights (normalized)
    if weight_function is None:
        # Equal weights
        current_weights = np.ones(n_stocks) / n_stocks
    else:
        # Use weight function
        current_weights = weight_function(prices.iloc[:1], returns.iloc[:1])
        if len(current_weights) != n_stocks:
            current_weights = np.ones(n_stocks) / n_stocks
        current_weights = current_weights / (current_weights.sum() + 1e-8)
    
    # Track turnover và transaction costs
    turnover_series = pd.Series(index=prices.index, dtype=float)
    tx_cost_series = pd.Series(index=prices.index, dtype=float)
    tx_cost_cum = 0.0
    
    rebalance_counter = 0
    
    # Backtest loop
    for i in range(1, n_days):
        # Check if should rebalance
        should_rebalance = (rebalance_counter % rebalance_every == 0) or (i == 1)
        
        if should_rebalance and weight_function is not None:
            # Calculate new weights using weight function
            new_weights = weight_function(prices.iloc[:i+1], returns.iloc[:i+1])
            if len(new_weights) != n_stocks:
                new_weights = np.ones(n_stocks) / n_stocks
            new_weights = new_weights / (new_weights.sum() + 1e-8)
            
            # Calculate turnover
            turnover = np.abs(new_weights - current_weights).sum()
            turnover_series.iloc[i] = turnover
            
            # Transaction cost
            tx_cost = turnover * transaction_cost_rate * wealth.iloc[i-1]
            tx_cost_series.iloc[i] = tx_cost
            tx_cost_cum += tx_cost
            
            # Update weights
            current_weights = new_weights.copy()
            rebalance_counter = 0
        elif should_rebalance:
            # Equal weight rebalancing
            new_weights = np.ones(n_stocks) / n_stocks
            
            # Calculate turnover (only if weights actually changed)
            turnover = np.abs(new_weights - current_weights).sum()
            turnover_series.iloc[i] = turnover
            
            # Transaction cost (only charge if there's actual turnover)
            if turnover > 1e-6:  # Only charge if meaningful change
                tx_cost = turnover * transaction_cost_rate * wealth.iloc[i-1]
            else:
                tx_cost = 0.0
            tx_cost_series.iloc[i] = tx_cost
            tx_cost_cum += tx_cost
            
            # Update weights
            current_weights = new_weights.copy()
            rebalance_counter = 0
        else:
            # No rebalance, weights drift with returns
            turnover_series.iloc[i] = 0.0
            tx_cost_series.iloc[i] = 0.0
            rebalance_counter += 1
        
        # Calculate portfolio return for this day
        portfolio_return = np.dot(current_weights, returns.iloc[i].values)
        
        # Update wealth (after transaction costs)
        wealth.iloc[i] = wealth.iloc[i-1] * (1 + portfolio_return) - tx_cost_series.iloc[i]
        
        # Normalize weights after price changes (drift)
        if not should_rebalance:
            # Weights drift with returns
            price_changes = 1 + returns.iloc[i].values
            current_weights = current_weights * price_changes
            current_weights = current_weights / (current_weights.sum() + 1e-8)
    
    # Calculate statistics
    wealth_returns = wealth.pct_change().dropna()
    
    # Validation: Check wealth is positive
    if (wealth <= 0).any():
        raise ValueError(f"Wealth cannot be negative or zero. Min wealth: {wealth.min()}")
    
    # Annual return (assuming 252 trading days per year)
    if len(wealth_returns) > 0:
        total_return = (wealth.iloc[-1] / wealth.iloc[0]) - 1
        n_days = len(wealth_returns)
        years = n_days / 252.0
        
        # Compound annual return: (1 + total_return)^(1/years) - 1
        if years > 0 and total_return > -1:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0.0
        
        # Validation: Warn if annual return seems unrealistic (>500%)
        if annual_return > 5.0:
            print(f"⚠️  Warning: Annual return {annual_return*100:.2f}% seems very high for strategy '{strategy_name}'")
            print(f"   Total return: {total_return*100:.2f}%, Period: {n_days} days ({years:.2f} years)")
            print(f"   Final wealth: {wealth.iloc[-1]:.4f}, Initial wealth: {wealth.iloc[0]:.4f}")
    else:
        annual_return = 0.0
    
    # Standard deviation (annualized)
    if len(wealth_returns) > 0:
        std_dev = wealth_returns.std() * np.sqrt(252)
    else:
        std_dev = 0.0
    
    # Sharpe ratio (assuming risk-free rate = 0.045 = 4.5%)
    risk_free_rate = 0.045
    if std_dev > 0:
        sharpe = (annual_return - risk_free_rate) / std_dev
    else:
        sharpe = 0.0
    
    # Max drawdown
    cumulative = wealth / wealth.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Average turnover
    turnover_avg = turnover_series.mean()
    
    return BacktestResult(
        strategy=strategy_name,
        wealth=wealth,
        returns=wealth_returns,
        annual_return=annual_return,
        std_dev=std_dev,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        turnover=turnover_avg,
        tx_cost_cum=tx_cost_cum,
    )


def summary_table(results: list[BacktestResult]) -> pd.DataFrame:
    """
    Generate summary statistics table from backtest results
    
    Args:
        results: List of BacktestResult objects
    
    Returns:
        DataFrame với summary statistics
    """
    rows = []
    for result in results:
        rows.append({
            'Strategy': result.strategy,
            'Annual Return (%)': result.annual_return * 100,
            'Std Dev (%)': result.std_dev * 100,
            'Sharpe Ratio': result.sharpe,
            'Max Drawdown (%)': result.max_drawdown * 100,
            'Turnover': result.turnover,
            'Tx Cost (cum)': result.tx_cost_cum,
        })
    
    return pd.DataFrame(rows)

