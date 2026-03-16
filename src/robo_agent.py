"""
Robo-Advisor Implementation

Contains:
- IPOAgent: Inverse Portfolio Optimization Agent
- DDPGAgent: Deep Deterministic Policy Gradient Agent for portfolio optimization
- Training functions and utilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Constants
TRADING_DAYS_PER_YEAR = 252  # 252 trading days per year
ROLLING_WINDOW_DAYS = 126  # Rolling window 126 ngày (6 tháng)
RISK_FREE_RATE_ANNUAL = 0.045  # Risk-free rate: 4.5% annual (trái phiếu chính phủ VN)
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR  # Daily risk-free rate
# Transaction cost: ~0.3% theo tài liệu
TRANSACTION_COST_RATE = 0.003  # Transaction cost: 0.3% (phí giao dịch VN theo tài liệu)

# Try to import scipy for optimization
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class IPOAgent:
    """
    Inverse Portfolio Optimization Agent
    Learns risk preference from current portfolio allocation
    """
    
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
        self.risk_tolerance = 1.0  # Default balanced risk
        self.target_return = 0.1  # Default 10% annual return
        self.W = None  # Maximum acceptable risk (rủi ro tối đa có thể chịu được)
        self.W_vol = None  # Maximum acceptable volatility (độ biến động tối đa)
        self.W_drawdown = None  # Maximum acceptable drawdown (mức sụt giảm tối đa)
    
    def calculate_optimal_weights(self, mean_returns, cov_matrix, lambda_param=None, W_vol=None):
        """
        Tính optimal weights từ historical data dựa trên Mean-Variance Optimization
        với constraint W (maximum acceptable risk)
        
        Công thức: 
        max_w w^T * μ_t - λ * w^T * Σ_t * w
        s.t. Σ_i w_i = 1, w_i ≥ 0
             sqrt(w^T * Σ_t * w) ≤ W_vol  (NEW: Constraint rủi ro tối đa)
        
        Trong đó:
        - w: vector weights (n_stocks x 1)
        - μ_t: mean returns vector (n_stocks x 1)
        - Σ_t: covariance matrix (n_stocks x n_stocks)
        - λ: risk aversion parameter (λ > 0)
        - W_vol: Maximum acceptable volatility (rủi ro tối đa có thể chịu được)
        
        Args:
            mean_returns: Mean returns array (μ_t)
            cov_matrix: Covariance matrix (Σ_t)
            lambda_param: Risk aversion parameter λ (nếu None thì tự động tính)
            W_vol: Maximum acceptable volatility (optional, nếu có sẽ thêm constraint)
            
        Returns:
            optimal_weights: Optimal portfolio weights từ data (w*)
        """
        if not HAS_SCIPY:
            # Fallback to equal weights nếu không có scipy
            return np.ones(len(mean_returns)) / len(mean_returns)
        
        n_stocks = len(mean_returns)
        
        # Nếu không có lambda, tự động tính dựa trên Sharpe ratio
        if lambda_param is None:
            # Estimate lambda từ data: lambda ≈ mean(returns) / mean(variance)
            mean_var = np.mean(np.diag(cov_matrix))
            mean_ret = np.mean(np.abs(mean_returns))
            if mean_var > 0:
                lambda_param = mean_ret / (mean_var + 1e-8) * 0.5  # Scale factor
            else:
                lambda_param = 1.0  # Default
        
        # Objective function: maximize w^T * μ - λ * w^T * Σ * w
        # Chuyển thành minimize: minimize -(w^T * μ - λ * w^T * Σ * w)
        def objective(w):
            # w^T * μ
            portfolio_return = np.dot(w, mean_returns)
            
            # w^T * Σ * w
            portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
            
            # Objective: -(w^T * μ - λ * w^T * Σ * w)
            objective_value = -(portfolio_return - lambda_param * portfolio_variance)
            
            return objective_value
        
        # Constraints: Σ_i w_i = 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # NEW: Thêm constraint W (maximum acceptable risk) nếu có
        if W_vol is not None and W_vol > 0:
            def volatility_constraint(w):
                """Constraint: portfolio volatility ≤ W_vol"""
                portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                return W_vol - portfolio_vol  # ≤ 0 means portfolio_vol ≤ W_vol
            
            constraints.append({'type': 'ineq', 'fun': volatility_constraint})
        
        # Bounds: w_i ≥ 0 (weights between 0 and 1)
        bounds = [(0, 1) for _ in range(n_stocks)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_stocks) / n_stocks
        
        # Optimize
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                # Normalize to ensure sum = 1 (numerical precision)
                optimal_weights = optimal_weights / (optimal_weights.sum() + 1e-8)
                # Ensure non-negative
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / (optimal_weights.sum() + 1e-8)
                return optimal_weights
            else:
                # Fallback to equal weights
                return np.ones(n_stocks) / n_stocks
        except Exception:
            # Fallback to equal weights
            return np.ones(n_stocks) / n_stocks
    
    def learn_risk_preference(self, current_weights, returns, cov_matrix, prices=None):
        """
        Learn risk tolerance and W (maximum acceptable risk) from current portfolio weights
        
        Args:
            current_weights: Current portfolio weights (array)
            returns: Returns DataFrame
            cov_matrix: Covariance matrix
            prices: Prices DataFrame (optional, for drawdown calculation)
            
        Returns:
            target_return: Estimated target return based on risk preference
        """
        # Calculate current portfolio metrics
        mean_returns = returns.mean().values
        current_return = np.dot(current_weights, mean_returns)
        current_vol = np.sqrt(np.dot(current_weights, np.dot(cov_matrix, current_weights)))
        
        # === TÍNH W: Maximum Acceptable Risk (Rủi ro tối đa có thể chịu được) ===
        # W1: Maximum Acceptable Volatility (Độ biến động tối đa)
        # Suy ngược từ current volatility: nếu nhà đầu tư chọn portfolio này,
        # họ có thể chịu được ít nhất current_vol, thêm buffer 15% để an toàn
        risk_buffer_vol = 1.15  # 15% buffer
        self.W_vol = current_vol * risk_buffer_vol
        
        # W2: Maximum Acceptable Drawdown (Mức sụt giảm tối đa)
        # Suy ngược từ historical max drawdown nếu có prices
        self.W_drawdown = None
        if prices is not None and len(prices) > 20:
            try:
                portfolio_prices = (prices * current_weights).sum(axis=1)
                portfolio_returns = portfolio_prices.pct_change().fillna(0)
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / (running_max + 1e-8)
                max_drawdown_historical = abs(drawdown.min())
                
                # Thêm buffer 20% để an toàn
                risk_buffer_drawdown = 1.2
                self.W_drawdown = max_drawdown_historical * risk_buffer_drawdown
            except Exception:
                self.W_drawdown = None
        
        # Lưu W (dictionary chứa cả W_vol và W_drawdown)
        self.W = {
            'volatility': self.W_vol,
            'drawdown': self.W_drawdown,
            'volatility_annualized': self.W_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if self.W_vol is not None else None
        }
        
        # TÍNH OPTIMAL WEIGHTS TỪ HISTORICAL DATA
        # Đây là phần quan trọng: tính W* (optimal) từ data
        # Sử dụng công thức: max_w w^T * μ_t - λ * w^T * Σ_t * w
        # s.t. Σ_i w_i = 1, w_i ≥ 0
        #      sqrt(w^T * Σ_t * w) ≤ W_vol (NEW: constraint rủi ro tối đa)
        # Sử dụng W_vol để constraint optimization
        optimal_weights = self.calculate_optimal_weights(mean_returns, cov_matrix, W_vol=self.W_vol)
        
        # Tính optimal portfolio metrics
        optimal_return = np.dot(optimal_weights, mean_returns)
        optimal_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        
        # Tính DELTA (weight difference) giữa current và optimal
        weight_delta = np.abs(current_weights - optimal_weights)
        total_delta = np.sum(weight_delta)  # Tổng delta (0-2, vì mỗi weight có thể khác tối đa 1)
        max_delta = np.max(weight_delta)    # Delta lớn nhất của một mã
        
        # So sánh current vs optimal để học risk preference
        # Nếu current weights gần optimal → risk tolerance phù hợp với data
        # Nếu current weights khác optimal → có thể do risk preference khác
        
        # Estimate risk tolerance based on current allocation và so sánh với optimal
        if current_vol > 0:
            # Risk tolerance dựa trên volatility ratio
            vol_ratio = current_vol / (np.abs(current_return) + 1e-8)
            
            # Điều chỉnh dựa trên so sánh với optimal
            # Nếu current vol cao hơn optimal vol nhiều → risk tolerance cao
            if optimal_vol > 0:
                vol_ratio_vs_optimal = current_vol / (optimal_vol + 1e-8)
                # Nếu current vol > optimal vol → risk tolerance cao hơn
                vol_ratio_adjustment = (vol_ratio_vs_optimal - 1.0) * 0.3
            else:
                vol_ratio_adjustment = 0.0
            
            # ĐIỀU CHỈNH RISK TOLERANCE DỰA TRÊN DELTA
            # Logic: Nếu delta lớn → có thể muốn giảm rủi ro → risk tolerance thấp hơn
            #        Nếu delta nhỏ → cân bằng → risk tolerance balanced
            
            # Tính risk adjustment từ delta
            # total_delta: 0 (giống hệt) → 2 (hoàn toàn khác)
            # Nếu delta lớn (> 0.5) → có thể portfolio hiện tại rủi ro cao → giảm risk tolerance
            # Nếu delta nhỏ (< 0.3) → cân bằng → risk tolerance balanced
            
            delta_risk_adjustment = 0.0
            
            if total_delta > 0.5:
                # Delta lớn: Portfolio hiện tại khác optimal nhiều
                # Có thể do rủi ro cao → GIẢM risk tolerance (conservative hơn)
                # Adjustment âm để giảm risk tolerance
                delta_risk_adjustment = -(total_delta - 0.5) * 0.4  # Giảm risk tolerance
            elif total_delta < 0.3:
                # Delta nhỏ: Portfolio gần optimal → CÂN BẰNG
                # Điều chỉnh về balanced (risk_tolerance ≈ 1.0)
                delta_risk_adjustment = (0.3 - total_delta) * 0.2  # Điều chỉnh về balanced
            # Nếu 0.3 <= total_delta <= 0.5: giữ nguyên
            
            # Nếu max_delta lớn (một mã chiếm quá nhiều) → giảm risk tolerance
            if max_delta > 0.3:
                # Một mã có weight khác optimal quá nhiều → tập trung rủi ro
                delta_risk_adjustment -= (max_delta - 0.3) * 0.3  # Giảm risk tolerance
            
            # Risk tolerance: 0.5 (conservative) to 2.0 (aggressive)
            base_risk_tolerance = 0.5 + vol_ratio * 1.5 + vol_ratio_adjustment
            self.risk_tolerance = np.clip(base_risk_tolerance + delta_risk_adjustment, 0.5, 2.0)
        else:
            self.risk_tolerance = 1.0
        
        # Lưu delta để sử dụng sau
        self.weight_delta = total_delta
        self.max_delta = max_delta
        
        # Calculate target return based on risk tolerance
        # Higher risk tolerance -> higher target return
        max_return = np.max(mean_returns)
        min_return = np.min(mean_returns)
        self.target_return = min_return + (max_return - min_return) * (self.risk_tolerance - 0.5) / 1.5
        
        # Lưu optimal weights để sử dụng sau
        self.optimal_weights = optimal_weights
        self.optimal_return = optimal_return
        self.optimal_vol = optimal_vol
        
        return self.target_return


def extract_state_features(returns, prices=None, lookback=20, ohlcv=None, news_features=None):
    """
    Extract rich state features from market data (ENHANCED with OHLCV and News)
    
    Args:
        returns: Returns DataFrame
        prices: Prices DataFrame (optional, for drawdown calculation)
        lookback: Number of periods to look back
        ohlcv: Dictionary with Open, High, Low, Close, Volume DataFrames (optional)
        news_features: numpy array với news features [avg_sentiment, avg_impact, positive_ratio, news_activity] (optional)
        
    Returns:
        state_features: Array of state features
    """
    if len(returns) < lookback:
        lookback = len(returns)
    
    recent_returns = returns.iloc[-lookback:] if len(returns) >= lookback else returns
    
    # Check and fix NaN in recent_returns
    if recent_returns.isna().any().any():
        recent_returns = recent_returns.fillna(0.0)
    
    features = []
    
    # 1. Market momentum (average return over lookback period)
    try:
        market_momentum = recent_returns.mean().mean()
        if np.isnan(market_momentum) or np.isinf(market_momentum):
            market_momentum = 0.0
    except Exception:
        market_momentum = 0.0
    features.append(market_momentum)
    
    # 2. Market volatility (average volatility over lookback period)
    try:
        market_volatility = recent_returns.std().mean()
        if np.isnan(market_volatility) or np.isinf(market_volatility):
            market_volatility = 0.0
    except Exception:
        market_volatility = 0.0
    features.append(market_volatility)
    
    # 3. Average correlation between stocks
    if len(recent_returns.columns) > 1:
        try:
            # Check if returns have enough variation for correlation
            # If all returns are the same, correlation will be NaN
            returns_std = recent_returns.std()
            if returns_std.sum() == 0 or (returns_std == 0).all():
                # All returns are constant, correlation is undefined
                avg_correlation = 0.0
            else:
                corr_matrix = recent_returns.corr().values
                # Check for NaN in correlation matrix
                if np.any(np.isnan(corr_matrix)):
                    avg_correlation = 0.0
                else:
                    # Get upper triangle (excluding diagonal)
                    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                    if len(upper_triangle) > 0:
                        avg_correlation = np.mean(upper_triangle)
                        if np.isnan(avg_correlation) or np.isinf(avg_correlation):
                            avg_correlation = 0.0
                    else:
                        avg_correlation = 0.0
        except Exception:
            avg_correlation = 0.0
    else:
        avg_correlation = 0.0
    features.append(avg_correlation)
    
    # 4. Sharpe ratio (market-wide)
    try:
        if market_volatility > 0:
            market_sharpe = market_momentum / (market_volatility + 1e-8)
        else:
            market_sharpe = 0.0
        if np.isnan(market_sharpe) or np.isinf(market_sharpe):
            market_sharpe = 0.0
    except Exception:
        market_sharpe = 0.0
    features.append(market_sharpe)
    
    # 5. Trend indicator (positive vs negative returns ratio)
    try:
        total_returns = len(recent_returns) * len(recent_returns.columns)
        if total_returns > 0:
            positive_ratio = (recent_returns > 0).sum().sum() / (total_returns + 1e-8)
        else:
            positive_ratio = 0.5
        if np.isnan(positive_ratio) or np.isinf(positive_ratio):
            positive_ratio = 0.5  # Default to neutral
    except Exception:
        positive_ratio = 0.5
    features.append(positive_ratio)
    
    # 6. Max drawdown (if prices available)
    if prices is not None and len(prices) >= lookback:
        try:
            recent_prices = prices.iloc[-lookback:]
            # Check for NaN in prices
            if recent_prices.isna().any().any():
                features.append(0.0)
            else:
                portfolio_prices = recent_prices.mean(axis=1)  # Equal-weighted portfolio
                if np.any(np.isnan(portfolio_prices)) or np.any(np.isinf(portfolio_prices)):
                    features.append(0.0)
                else:
                    cumulative = (1 + portfolio_prices).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / (running_max + 1e-8)
                    max_drawdown = drawdown.min()
                    if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                        features.append(0.0)
                    else:
                        features.append(max_drawdown)
        except Exception:
            features.append(0.0)
    else:
        features.append(0.0)
    
    # NEW: OHLCV-based features
    if ohlcv is not None:
        try:
            recent_ohlcv = {}
            for key in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if key in ohlcv and len(ohlcv[key]) >= lookback:
                    recent_ohlcv[key] = ohlcv[key].iloc[-lookback:]
                elif key in ohlcv:
                    recent_ohlcv[key] = ohlcv[key]
                else:
                    recent_ohlcv[key] = None
            
            # 7. Average intraday volatility (High - Low) / Open
            if recent_ohlcv['High'] is not None and recent_ohlcv['Low'] is not None and recent_ohlcv['Open'] is not None:
                try:
                    intraday_range = ((recent_ohlcv['High'] - recent_ohlcv['Low']) / 
                                    (recent_ohlcv['Open'] + 1e-8)).mean().mean()
                    if np.isnan(intraday_range) or np.isinf(intraday_range):
                        intraday_range = 0.0
                except Exception:
                    intraday_range = 0.0
                features.append(intraday_range)
            else:
                features.append(0.0)
            
            # 8. Volume momentum (current volume vs average)
            if recent_ohlcv['Volume'] is not None:
                try:
                    current_volume = recent_ohlcv['Volume'].iloc[-1].mean()
                    avg_volume = recent_ohlcv['Volume'].mean().mean()
                    if np.isnan(current_volume) or np.isnan(avg_volume) or avg_volume <= 0:
                        volume_momentum = 0.0
                    else:
                        volume_ratio = current_volume / (avg_volume + 1e-8)
                        volume_momentum = np.tanh(volume_ratio - 1)  # Normalize around 0
                        if np.isnan(volume_momentum) or np.isinf(volume_momentum):
                            volume_momentum = 0.0
                except Exception:
                    volume_momentum = 0.0
                features.append(volume_momentum)
            else:
                features.append(0.0)
            
            # 9. Price range (volatility proxy from High-Low)
            if recent_ohlcv['High'] is not None and recent_ohlcv['Low'] is not None and recent_ohlcv['Close'] is not None:
                try:
                    price_range = ((recent_ohlcv['High'] - recent_ohlcv['Low']) / 
                                  (recent_ohlcv['Close'] + 1e-8)).mean().mean()
                    if np.isnan(price_range) or np.isinf(price_range):
                        price_range = 0.0
                except Exception:
                    price_range = 0.0
                features.append(price_range)
            else:
                features.append(0.0)
            
            # 10. Gap analysis (Open vs previous Close)
            if recent_ohlcv['Open'] is not None and recent_ohlcv['Close'] is not None:
                try:
                    gaps = (recent_ohlcv['Open'] - recent_ohlcv['Close'].shift(1)) / (recent_ohlcv['Close'].shift(1) + 1e-8)
                    avg_gap = gaps.mean().mean()
                    if np.isnan(avg_gap) or np.isinf(avg_gap):
                        avg_gap = 0.0
                except Exception:
                    avg_gap = 0.0
                features.append(avg_gap)
            else:
                features.append(0.0)
            
            # 11. Volume-weighted momentum
            if recent_ohlcv['Volume'] is not None and recent_ohlcv['Close'] is not None:
                try:
                    vwap = ((recent_ohlcv['Close'] * recent_ohlcv['Volume']).sum() / 
                           (recent_ohlcv['Volume'].sum() + 1e-8)).mean()
                    if np.isnan(vwap) or np.isinf(vwap) or vwap <= 0:
                        price_to_vwap = 0.0
                    else:
                        current_price = recent_ohlcv['Close'].iloc[-1].mean()
                        if np.isnan(current_price) or current_price <= 0:
                            price_to_vwap = 0.0
                        else:
                            price_to_vwap = current_price / (vwap + 1e-8) - 1
                            if np.isnan(price_to_vwap) or np.isinf(price_to_vwap):
                                price_to_vwap = 0.0
                except Exception:
                    price_to_vwap = 0.0
                features.append(price_to_vwap)
            else:
                features.append(0.0)
        except Exception:
            # If OHLCV processing fails, add zeros
            features.extend([0.0] * 5)
    else:
        # No OHLCV data, add zeros for OHLCV features
        features.extend([0.0] * 5)
    
    # Convert to numpy array and ensure no NaN/Inf values
    # NEW: Add news features if available
    if news_features is not None:
        # news_features should be array with 4 values: [avg_sentiment, avg_impact, positive_ratio, news_activity]
        if isinstance(news_features, np.ndarray) and len(news_features) == 4:
            # Normalize news features to [-1, 1] range
            news_normalized = np.clip(news_features, -1.0, 1.0)
            features.extend(news_normalized.tolist())
        else:
            # Default news features (neutral)
            features.extend([0.0, 0.0, 0.5, 0.0])
    else:
        # No news features, add zeros
        features.extend([0.0, 0.0, 0.5, 0.0])
    
    features_array = np.array(features, dtype=np.float32)
    
    # Final check and fix any remaining NaN/Inf BEFORE normalization
    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features (simple normalization to [-1, 1] range)
    # Use tanh for bounded normalization
    try:
        features_normalized = np.tanh(features_array * 10)  # Scale and tanh for normalization
        # Check for NaN after normalization
        if np.any(np.isnan(features_normalized)) or np.any(np.isinf(features_normalized)):
            features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        # If normalization fails, return zeros
        features_normalized = np.zeros_like(features_array)
    
    return features_normalized


class ActorNetwork(nn.Module):
    """
    Improved Actor Network for DDPG with BatchNorm and Dropout
    Outputs portfolio weights (action)
    """
    
    def __init__(self, n_stocks, state_dim=6, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.n_stocks = n_stocks
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, n_stocks)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state):
        # Handle both 1D and 2D inputs
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        x = torch.relu(self.bn1(self.fc1(state)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        # Output portfolio weights (softmax to ensure they sum to 1)
        weights = torch.softmax(self.fc4(x), dim=-1)
        return weights


class CriticNetwork(nn.Module):
    """
    Improved Critic Network for DDPG with BatchNorm
    Estimates Q-value of state-action pairs
    """
    
    def __init__(self, n_stocks, state_dim=6, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        # State + Action as input
        self.fc1 = nn.Linear(state_dim + n_stocks, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state, action):
        # Handle both 1D and 2D inputs
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
            
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        q_value = self.fc4(x)
        return q_value


class DDPGAgent:
    """
    Improved DDPG Agent for Portfolio Optimization
    """
    
    def __init__(self, n_stocks, state_dim=6, lr_actor=3e-5, lr_critic=1e-4, 
                 gamma=0.99, tau=0.001, device='cpu'):
        self.n_stocks = n_stocks
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(n_stocks, state_dim).to(device)
        self.critic = CriticNetwork(n_stocks, state_dim).to(device)
        self.target_actor = ActorNetwork(n_stocks, state_dim).to(device)
        self.target_critic = CriticNetwork(n_stocks, state_dim).to(device)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers with learning rate scheduling
        # Adjusted learning rates for better convergence with new reward formula
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)
        
        # Learning rate schedulers (slower decay for more training)
        # StepLR với step_size lớn hơn để giữ learning rate cao lâu hơn và decay chậm hơn
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.95)
        
        # Replay buffer (larger for better stability and more diverse experiences)
        self.replay_buffer = deque(maxlen=50000)  # Increased from 20000 to 50000 for more diverse training
        
        # Parameters for portfolio optimization
        self.omega = 1.0  # Risk aversion parameter
        self.target_return = 0.1  # Target annual return
    
    def select_action(self, state, explore=True, noise_scale=0.1):
        """
        Select action (portfolio weights) given state
        
        Args:
            state: State array
            explore: Whether to add exploration noise
            noise_scale: Scale of exploration noise
            
        Returns:
            action: Portfolio weights (numpy array)
        """
        self.actor.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            # Add noise for exploration (adaptive noise scale)
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            # Renormalize to ensure weights sum to 1
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)
            
            # Constraint: no single stock > 40%
            max_weight = 0.4
            if np.max(action) > max_weight:
                excess = np.max(action) - max_weight
                action[np.argmax(action)] = max_weight
                # Redistribute excess proportionally
                other_indices = np.arange(len(action)) != np.argmax(action)
                action[other_indices] += excess * action[other_indices] / (action[other_indices].sum() + 1e-8)
                action = action / (action.sum() + 1e-8)
        
        self.actor.train()
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=128):
        """Train agent on a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()


def calculate_portfolio_reward_improved(weights, returns, cov_matrix, omega=1.0, target_return=0.1, prices=None, ohlcv=None, previous_weights=None, transaction_cost_rate=TRANSACTION_COST_RATE, current_wealth=None, W_vol=None):
    """
    Reward function theo công thức: Reward = w^T R – λ*(Rủi ro) – Chi phí Giao dịch
    
    Trong đó:
    - w^T R: Portfolio return (expected return) = w^T * returns
    - λ*(Rủi ro): Risk penalty = λ * Portfolio volatility = λ * sqrt(w^T Σ w)
    - Chi phí Giao dịch: Transaction cost = Turnover * Transaction Cost Rate
    - W_vol: Maximum acceptable volatility (từ IPO) - dùng để điều chỉnh penalty
    
    Args:
        weights: Portfolio weights (w)
        returns: Mean returns (R) - vector of expected returns
        cov_matrix: Covariance matrix (Σ)
        omega: Risk aversion parameter (λ)
        target_return: Target return (not used, kept for compatibility)
        prices: Prices DataFrame (not used, kept for compatibility)
        ohlcv: OHLCV data (not used, kept for compatibility)
        previous_weights: Previous weights (for transaction cost calculation)
        transaction_cost_rate: Transaction cost rate (default: 0.3%)
        current_wealth: Current wealth (not used, kept for compatibility)
        W_vol: Maximum acceptable volatility từ IPO (optional, để điều chỉnh risk penalty)
        
    Returns:
        reward: Portfolio reward = w^T R – λ*(Rủi ro) – Chi phí Giao dịch
    """
    # Handle both mean returns and full returns DataFrame
    if isinstance(returns, pd.DataFrame):
        mean_returns = returns.mean().values
    else:
        mean_returns = returns
    
    # Check for NaN in inputs
    if np.any(np.isnan(weights)) or np.any(np.isnan(mean_returns)) or np.any(np.isnan(cov_matrix)):
        return -100.0
    
    # w^T R: Portfolio return (expected return)
    portfolio_return = np.dot(weights, mean_returns)
    
    # λ*(Rủi ro): Risk penalty = λ * Portfolio volatility
    # Portfolio volatility = sqrt(w^T Σ w)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0.0
    
    # Điều chỉnh risk penalty nếu có W_vol (maximum acceptable volatility)
    # Nếu portfolio_vol > W_vol, tăng penalty
    if W_vol is not None and W_vol > 0:
        # Scale penalty dựa trên W_vol
        vol_ratio = portfolio_vol / (W_vol + 1e-8)
        if vol_ratio > 1.0:
            # Vượt quá W_vol → Penalty tăng
            excess_ratio = vol_ratio - 1.0
            risk_penalty = omega * portfolio_vol * (1.0 + excess_ratio * 1.0)
        else:
            # Trong giới hạn W_vol → Penalty nhẹ hơn
            risk_penalty = omega * portfolio_vol * 0.6
    else:
        # Không có W_vol, dùng penalty thông thường (giảm nhẹ)
        risk_penalty = omega * portfolio_vol * 0.6
    
    # Chi phí Giao dịch: Transaction cost
    transaction_cost = 0.0
    if previous_weights is not None:
        # Tính turnover: tổng thay đổi weights
        turnover = float(np.abs(weights - previous_weights).sum())
        transaction_cost = turnover * transaction_cost_rate
    else:
        # Ước tính turnover từ concentration (heuristic)
        concentration = np.sum(weights ** 2)
        estimated_turnover = (1 - concentration) * 0.2
        transaction_cost = estimated_turnover * transaction_cost_rate
    
    # Check for NaN in calculations
    if np.isnan(portfolio_return) or np.isnan(risk_penalty) or np.isnan(transaction_cost):
        return -100.0
    
    # Reward = w^T R – λ*(Rủi ro) – Chi phí Giao dịch
    # Điều chỉnh để reward dương hơn và dễ học hơn
    # Scale return lớn hơn và giảm penalty scale
    # Điều chỉnh scale để reward dương hơn
    scaled_return = portfolio_return * 20000  # Tăng scale return từ 15000 lên 20000
    scaled_risk_penalty = risk_penalty * 600  # Giảm scale risk penalty từ 800 xuống 600
    scaled_transaction_cost = transaction_cost * 600  # Giảm scale transaction cost từ 800 xuống 600
    
    reward = scaled_return - scaled_risk_penalty - scaled_transaction_cost
    
    # Thêm baseline offset để reward dương hơn (giúp model học tốt hơn)
    # Baseline = expected return của equal weights portfolio
    baseline_return = np.mean(mean_returns) * 20000  # Baseline return (scale giống scaled_return)
    # Tăng offset đáng kể để reward dương hơn
    reward = reward - baseline_return + 50.0  # Offset để reward dương hơn
    
    # Check for NaN
    if np.isnan(reward) or np.isinf(reward):
        return -100.0
    
    return reward


def train_robo_advisor(returns, n_episodes=2000, stock_code=None, cache_manager=None, prices=None, ohlcv=None, load_existing_model=False):
    """
    Train improved DDPG agent for portfolio optimization (ENHANCED with OHLCV)
    
    Args:
        returns: Returns DataFrame
        n_episodes: Number of training episodes (increased default)
        stock_code: Stock code (for saving, optional)
        cache_manager: Cache manager (optional)
        prices: Prices DataFrame (optional, for improved reward)
        ohlcv: Dictionary with OHLCV data (optional, for enhanced features)
        load_existing_model: If True, load existing model and continue training (continual learning)
        
    Returns:
        agent: Trained DDPG agent
        history: Training history
    """
    n_stocks = len(returns.columns)
    
    # Calculate statistics
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # Check and fix NaN values in mean_returns and cov_matrix
    if np.any(np.isnan(mean_returns)):
        mean_returns = np.nan_to_num(mean_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(cov_matrix)):
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure covariance matrix is positive semi-definite
        cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Make symmetric
    
    # Check if data is valid
    if np.all(mean_returns == 0) and np.all(cov_matrix == 0):
        return None, None
    
    # Initialize agent with improved state dimension (11 features with OHLCV + 4 news features)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_features = 6
    ohlcv_features = 5 if ohlcv is not None else 0
    news_features_count = 4  # Always add news features (will be zeros if no news)
    state_dim = base_features + ohlcv_features + news_features_count  # 6 base + 5 OHLCV + 4 news = 15 total
    agent = DDPGAgent(n_stocks, state_dim=state_dim, device=device)
    
    # Load existing model if requested (continual learning)
    if load_existing_model:
        model_path = Path('models') / 'trained_model.pth'
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                # Check if model dimensions match
                if checkpoint.get('n_stocks') == n_stocks and checkpoint.get('state_dim') == state_dim:
                    agent.actor.load_state_dict(checkpoint['actor'])
                    agent.critic.load_state_dict(checkpoint['critic'])
                    agent.target_actor.load_state_dict(checkpoint.get('target_actor', checkpoint['actor']))
                    agent.target_critic.load_state_dict(checkpoint.get('target_critic', checkpoint['critic']))
                    # Load hyperparameters if available
                    if 'omega' in checkpoint:
                        agent.omega = checkpoint['omega']
                    if 'target_return' in checkpoint:
                        agent.target_return = checkpoint['target_return']
                    print("✅ Đã load model cũ - Tiếp tục học (Continual Learning)")
                else:
                    print("⚠️ Model dimensions không khớp - Train từ đầu")
            except Exception as e:
                print(f"⚠️ Không thể load model cũ: {e} - Train từ đầu")
        else:
            print("⚠️ Không tìm thấy model cũ - Train từ đầu")
    
    # Initialize IPO agent để lấy W (maximum acceptable volatility) và risk parameters
    ipo_agent = IPOAgent(n_stocks=n_stocks)
    initial_weights = np.ones(n_stocks) / n_stocks
    ipo_agent.learn_risk_preference(initial_weights, returns, cov_matrix, prices=prices)
    
    # Set omega và target_return từ IPO agent
    # Điều chỉnh omega để không quá cao (giảm penalty)
    agent.omega = np.clip(ipo_agent.risk_tolerance, 0.5, 1.5)  # Giảm max từ 2.5 xuống 1.5
    agent.target_return = ipo_agent.target_return
    
    # Lấy W_vol (maximum acceptable volatility) từ IPO agent
    W_vol = ipo_agent.W_vol  # Maximum acceptable volatility
    
    print(f"Risk aversion (omega/λ): {agent.omega:.4f}")
    print(f"Target return: {agent.target_return*100:.2f}%")
    if W_vol is not None:
        print(f"Maximum acceptable volatility (W): {W_vol*100:.2f}% (daily), {W_vol*np.sqrt(252)*100:.2f}% (annual)")
    else:
        print("W (maximum acceptable volatility): Not set")
    
    # Training history
    history = {
        'episode': [],
        'reward': [],
        'portfolio_return': [],
        'portfolio_vol': [],
        'sharpe': [],
        'critic_loss': [],
        'actor_loss': []
    }
    
    
    # Early stopping parameters (improved for better convergence)
    best_reward = -np.inf
    patience = 800  # Increased significantly to allow much more training
    no_improve_count = 0
    min_episodes = 1500  # Minimum episodes before early stopping (increased significantly for deeper learning)
    improvement_threshold = 1.0  # Minimum improvement to count as progress (increased for new reward scale)
    
    # Track previous weights for transaction cost calculation
    previous_weights = None
    
    # Load news features for training (if available)
    news_features_dict = None
    aggregated_news = None
    try:
        from news_features import load_news_features_from_db, aggregate_news_features
        tickers_list = list(returns.columns)
        news_features_dict = load_news_features_from_db(tickers_list, days_back=7)
        if news_features_dict:
            aggregated_news = aggregate_news_features(news_features_dict, tickers_list)
            print("✅ Đã load news features từ database")
    except ImportError:
        print("⚠️  news_features module chưa được cài đặt. Bỏ qua news features.")
    except Exception as e:
        print(f"⚠️  Lỗi khi load news features: {e}")
    
    # Training loop
    for episode in range(n_episodes):
        # Extract rich state features (with OHLCV and News if available)
        state = extract_state_features(returns, prices, ohlcv=ohlcv, news_features=aggregated_news)
        
        # Final check and fix NaN in state (should not happen after fix, but just in case)
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Adaptive noise scale (decrease over time, slower decay for longer training)
        # Giảm noise chậm hơn để exploration lâu hơn
        noise_scale = max(0.05, 0.2 * (1 - episode / (n_episodes * 1.5)))
        
        # Select action (portfolio weights)
        action = agent.select_action(state, explore=True, noise_scale=noise_scale)
        
        # Check and fix NaN in action
        if np.any(np.isnan(action)):
            action = np.ones(n_stocks) / n_stocks
        
        # Normalize action to ensure it sums to 1
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(n_stocks) / n_stocks
        
        # Calculate reward theo công thức: Reward = w^T R – λ*(Rủi ro) – Chi phí Giao dịch
        # Kết nối với W (maximum acceptable volatility) từ IPO
        reward = calculate_portfolio_reward_improved(
            action, returns, cov_matrix, 
            omega=agent.omega, target_return=agent.target_return,
            prices=prices, ohlcv=ohlcv,
            previous_weights=previous_weights,
            transaction_cost_rate=TRANSACTION_COST_RATE,
            current_wealth=None,
            W_vol=W_vol  # Pass W (maximum acceptable volatility) từ IPO
        )
        
        # Update previous_weights for next iteration
        previous_weights = action.copy()
        
        # Check and fix NaN in reward
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        # Next state (same for this setup, but could be updated)
        next_state = state
        done = False
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Train agent (multiple steps per episode for faster learning)
        critic_losses = []
        actor_losses = []
        # Start training earlier (when buffer has at least batch_size samples)
        # This ensures training starts from episode 1
        if len(agent.replay_buffer) >= 128:
            # Train multiple times per episode (increased for better learning)
            # Train more times per episode to ensure proper convergence
            # Increase training steps for deeper learning (20 times per episode)
            train_steps = min(20, len(agent.replay_buffer) // 50)  # Train up to 20 times per episode, use smaller batch ratio for more training
            for _ in range(train_steps):
                cl, al = agent.train_step(batch_size=128)
                if cl is not None:
                    critic_losses.append(cl)
                if al is not None and not (np.isnan(al) or np.isinf(al)):
                    actor_losses.append(al)
        elif len(agent.replay_buffer) >= 32:
            # Early training with smaller batches when buffer is still small
            # This ensures we start learning immediately
            train_steps = min(8, len(agent.replay_buffer) // 32)  # Increased from 5 to 8
            for _ in range(train_steps):
                batch_size = min(32, len(agent.replay_buffer))
                cl, al = agent.train_step(batch_size=batch_size)
                if cl is not None:
                    critic_losses.append(cl)
                if al is not None and not (np.isnan(al) or np.isinf(al)):
                    actor_losses.append(al)
            
            # Update learning rates (less frequently for more stable learning)
            if (episode + 1) % 20 == 0:  # Changed from 10 to 20 for slower decay
                agent.actor_scheduler.step()
                agent.critic_scheduler.step()
        
        # Calculate metrics
        portfolio_return = np.dot(action, mean_returns)
        portfolio_vol = np.sqrt(np.dot(action, np.dot(cov_matrix, action)))
        # Sharpe ratio với risk-free rate: Sharpe = (R - Rf) / σ
        sharpe = (portfolio_return - RISK_FREE_RATE_DAILY) / (portfolio_vol + 1e-8)
        
        # Check and fix NaN in metrics
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            portfolio_return = 0.0
        if np.isnan(portfolio_vol) or np.isinf(portfolio_vol) or portfolio_vol < 0:
            portfolio_vol = 1e-8
        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = 0.0
        
        # Record history
        history['episode'].append(episode + 1)
        history['reward'].append(reward)
        history['portfolio_return'].append(portfolio_return)
        history['portfolio_vol'].append(portfolio_vol)
        history['sharpe'].append(sharpe)
        
        # Print progress mỗi 200 episodes (giảm frequency để không spam output khi training lâu)
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(history['reward'][-200:]) if len(history['reward']) >= 200 else np.mean(history['reward'])
            avg_sharpe = np.mean(history['sharpe'][-200:]) if len(history['sharpe']) >= 200 else (np.mean(history['sharpe']) if len(history['sharpe']) > 0 else sharpe)
            print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.6f}, Avg Sharpe: {avg_sharpe:.4f}, Current Sharpe: {sharpe:.4f}")
        # Store losses (use last loss if available, otherwise 0)
        # This gives better visibility into current training state
        if critic_losses:
            history['critic_loss'].append(critic_losses[-1])  # Use last loss for better visibility
        else:
            history['critic_loss'].append(0.0)
        
        if actor_losses:
            history['actor_loss'].append(actor_losses[-1])  # Use last loss for better visibility
        else:
            history['actor_loss'].append(0.0)
        
        # Early stopping check (only if reward is valid)
        if not (np.isnan(reward) or np.isinf(reward)):
            # Valid reward
            if best_reward == -np.inf:
                # First valid reward
                best_reward = reward
                no_improve_count = 0
            elif reward > best_reward + improvement_threshold:
                # Significant improvement (above threshold)
                best_reward = reward
                no_improve_count = 0
            elif reward > best_reward:
                # Small improvement (below threshold, but still improvement)
                best_reward = reward
                no_improve_count = max(0, no_improve_count - 2)  # Reduce counter more aggressively
            else:
                # No improvement
                no_improve_count += 1
        else:
            # Invalid reward (NaN or Inf) - don't count towards early stopping
            # Only increment counter slightly to avoid infinite training
            if episode > min_episodes:
                no_improve_count += 0.3  # Reduced increment for invalid rewards
        
        
        # Early stopping (improved logic for better convergence)
        # Only stop early if:
        # 1. We've trained at least min_episodes
        # 2. No significant improvement for patience episodes
        # 3. Current reward is valid (not NaN)
        # 4. We have at least one valid best_reward
        if (episode >= min_episodes and
            no_improve_count >= patience and 
            not (np.isnan(reward) or np.isinf(reward)) and
            not (np.isnan(best_reward) or np.isinf(best_reward)) and
            best_reward != -np.inf and
            best_reward > 0):  # Only stop if best reward is positive (điều chỉnh cho reward mới)
            print(f"\nEarly stopping at episode {episode+1}: No improvement for {patience} episodes")
            print(f"Best reward: {best_reward:.6f}, Current reward: {reward:.6f}")
            break
    
    return agent, history
