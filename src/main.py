"""
Portfolio Optimization Advisor using Inverse Portfolio Optimization (IPO) and Deep Reinforcement Learning (DDPG)

Hệ thống tư vấn đầu tư tự động sử dụng:
1. IPO Agent - Học khẩu vị rủi ro từ phân bổ danh mục hiện tại
2. DDPG Agent - Tối ưu hóa phân bổ vốn đa kỳ
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure imports work regardless of current working directory
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from get_data import download_stock_data, save_data
from robo_agent import IPOAgent, train_robo_advisor

# 30 mã cổ phiếu VN Index + 1 chỉ số VN-Index mặc định
AVAILABLE_STOCKS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
    '^VNINDEX'
]

TRADING_DAYS_PER_YEAR = 252  # 252 trading days per year
ROLLING_WINDOW_DAYS = 126  # Rolling window 126 ngày (6 tháng)
RISK_FREE_RATE_ANNUAL = 0.045  # Risk-free rate: 4.5% annual (trái phiếu chính phủ VN)
# Transaction cost: ~0.3% theo tài liệu
TRANSACTION_COST_RATE = 0.003  # Transaction cost: 0.3% (phí giao dịch VN theo tài liệu)


def compute_weights_from_capital(tickers, capital_amounts, prices):
    """
    Convert capital amounts to portfolio weights using latest prices
    
    Args:
        tickers: List of stock tickers
        capital_amounts: List of capital amounts (in currency) for each ticker
        prices: DataFrame with prices
    
    Returns:
        weights: Portfolio weights (normalized to sum to 1)
    """
    last_prices = prices.iloc[-1]
    values = []
    
    for ticker, capital in zip(tickers, capital_amounts):
        price = last_prices.get(ticker)
        if price is None or np.isnan(price):
            price = 1.0  # fallback nếu thiếu giá
        values.append(float(capital))  # Dùng trực tiếp giá trị vốn
    
    total_value = sum(values)
    if total_value <= 0:
        return np.ones(len(tickers)) / len(tickers)
    
    weights = np.array(values) / total_value
    return weights


def classify_risk_level(risk_tolerance):
    """Phân loại mức độ rủi ro"""
    if risk_tolerance < 0.7:
        return "Conservative (Bảo thủ)"
    elif risk_tolerance < 1.3:
        return "Balanced (Cân bằng)"
    else:
        return "Aggressive (Mạo hiểm)"


def format_percentage(value, decimals=2):
    """Format percentage"""
    return f"{value*100:.{decimals}f}%"


def format_currency(value, decimals=0):
    """Format currency"""
    return f"{value:,.{decimals}f}"


def generate_recommendation(selected_tickers, capital_amounts, start_date, end_date, n_episodes=150):
    """
    Tạo đề xuất phân bổ vốn tối ưu
    
    Args:
        selected_tickers: List of 5 selected stock tickers
        capital_amounts: List of capital amounts for each ticker
        start_date: Start date for data
        end_date: End date for data
        n_episodes: Number of RL training episodes
    
    Returns:
        dict: Recommendation results
    """
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION ADVISOR")
    print("="*70)
    print(f"Selected Stocks: {selected_tickers}")
    print(f"Capital Amounts: {[format_currency(c) for c in capital_amounts]}")
    print(f"Total Capital: {format_currency(sum(capital_amounts))}")
    print(f"Period: {start_date} to {end_date}")
    print("="*70)
    
    # Step 1: Load data từ CSV (KHÔNG DOWNLOAD)
    prices, returns, ohlcv = download_stock_data(
        selected_tickers,
        start_date,
        end_date,
        use_cache=True,
        cache_manager=None,
        data_source='csv'  # Chỉ đọc từ CSV
    )
    
    # Check if data was loaded successfully
    if prices is None or returns is None:
        error_msg = "❌ Không thể tải dữ liệu từ file Data_test.csv.\n"
        error_msg += "   Vui lòng:\n"
        error_msg += "   1. Kiểm tra file data/Data_test.csv có tồn tại không\n"
        error_msg += "   2. Kiểm tra file CSV có chứa dữ liệu cho các mã này không\n"
        error_msg += f"   3. Các mã đã chọn: {', '.join(selected_tickers)}\n"
        error_msg += f"   4. Khoảng thời gian: {start_date} đến {end_date}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Filter for selected tickers (only if they exist in data)
    available_tickers = [t for t in selected_tickers if t in prices.columns]
    missing_tickers = [t for t in selected_tickers if t not in prices.columns]
    
    # Nếu có mã thiếu, chỉ báo lỗi (KHÔNG TẢI TỪ NGUỒN KHÁC)
    if missing_tickers:
        if len(available_tickers) < 2:
            error_msg = f"❌ Không đủ dữ liệu (cần ít nhất 2 mã, chỉ có {len(available_tickers)} mã).\n"
            error_msg += f"   Các mã có dữ liệu: {available_tickers}\n"
            error_msg += f"   Các mã thiếu dữ liệu: {missing_tickers}\n"
            error_msg += f"   Vui lòng:\n"
            error_msg += f"   1. Chọn các mã khác có trong file CSV (data/Data_test.csv)\n"
            error_msg += f"   2. Các mã có sẵn trong CSV: ACB, BID, BVH, CTG, FPT, GAS, GVR, HDB, HPG, KDH, MBB, MSN, MWG, NVL, PLX, PNJ, POW, SAB, SSI, STB, TCB, TCH, TPB, VCB, VHM, VIC, VJC, VNM, VPB, VRE"
            raise ValueError(error_msg)
    
    # Chỉ lấy các mã có dữ liệu
    prices = prices[available_tickers]
    returns = returns[available_tickers]
    
    # Filter OHLCV for available tickers
    if ohlcv:
        for key in ohlcv:
            if key in ohlcv and ohlcv[key] is not None:
                available_ohlcv_cols = [t for t in available_tickers if t in ohlcv[key].columns]
                if available_ohlcv_cols:
                    ohlcv[key] = ohlcv[key][available_ohlcv_cols]
    
    
    # Step 2: Convert capital amounts -> weights (chỉ với available_tickers)
    # Đảm bảo capital_amounts tương ứng với available_tickers
    available_capitals = []
    for ticker in available_tickers:
        idx = selected_tickers.index(ticker)
        available_capitals.append(capital_amounts[idx])
    
    current_weights = compute_weights_from_capital(available_tickers, available_capitals, prices)
    
    total_capital = sum(available_capitals)
    
    # Step 3: IPO Agent - learn risk preference from current allocation
    ipo_agent = IPOAgent(n_stocks=len(available_tickers))
    cov_matrix = returns.cov().values
    target_return = ipo_agent.learn_risk_preference(current_weights, returns, cov_matrix, prices=prices)
    risk_level = classify_risk_level(ipo_agent.risk_tolerance)
    
    # Step 4: RL Agent (train on returns of available tickers with OHLCV)
    from robo_agent import extract_state_features
    agent, history = train_robo_advisor(
        returns,
        n_episodes=n_episodes,
        stock_code=None,
        cache_manager=None,
        prices=prices,
        ohlcv=ohlcv
    )
    
    # Step 5: Generate recommended allocation
    state = extract_state_features(returns, prices, ohlcv=ohlcv)
    recommended_weights = agent.select_action(state, explore=False)
    
    # Step 6: Calculate recommended capital allocation
    recommended_capitals = recommended_weights * total_capital
    
    # Step 7: Estimate expected metrics
    daily_mean = returns.mean().values
    daily_cov = returns.cov().values
    
    current_return = current_weights @ daily_mean * TRADING_DAYS_PER_YEAR
    current_vol = np.sqrt(current_weights @ daily_cov @ current_weights) * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Sharpe ratio với risk-free rate: Sharpe = (R - Rf) / σ
    current_sharpe = (current_return - RISK_FREE_RATE_ANNUAL) / (current_vol + 1e-8)
    
    expected_return = recommended_weights @ daily_mean * TRADING_DAYS_PER_YEAR
    portfolio_vol = np.sqrt(recommended_weights @ daily_cov @ recommended_weights) * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Sharpe ratio với risk-free rate: Sharpe = (R - Rf) / σ
    sharpe = (expected_return - RISK_FREE_RATE_ANNUAL) / (portfolio_vol + 1e-8)
    
    # Step 8: Build recommendation summary
    improvement_return = expected_return - current_return
    improvement_sharpe = sharpe - current_sharpe
    
    return {
        "risk_level": risk_level,
        "risk_tolerance": float(ipo_agent.risk_tolerance),
        "target_return": float(target_return),
        "current_weights": dict(zip(available_tickers, current_weights)),
        "recommended_weights": dict(zip(available_tickers, recommended_weights)),
        "current_capitals": dict(zip(available_tickers, available_capitals)),
        "recommended_capitals": dict(zip(available_tickers, recommended_capitals)),
        "current_return": float(current_return),
        "recommended_return": float(expected_return),
        "current_volatility": float(current_vol),
        "recommended_volatility": float(portfolio_vol),
        "current_sharpe": float(current_sharpe),
        "recommended_sharpe": float(sharpe),
        "improvement_return": float(improvement_return),
        "improvement_sharpe": float(improvement_sharpe)
    }


def prompt_user_inputs():
    """Nhận input từ người dùng"""
    print("="*70)
    print("PORTFOLIO OPTIMIZATION ADVISOR")
    print("="*70)
    print(f"\nAvailable Stocks ({len(AVAILABLE_STOCKS)}): {', '.join(AVAILABLE_STOCKS)}")
    print("\nPlease select 5 stocks from the list above.")
    
    # Select 5 stocks
    print("\n" + "-"*70)
    print("STEP 1: Select 5 Stocks")
    print("-"*70)
    
    selected = []
    available = AVAILABLE_STOCKS.copy()
    
    for i in range(5):
        print(f"\nAvailable stocks: {', '.join(available)}")
        ticker = input(f"Select stock {i+1}/5 (enter ticker): ").strip().upper()
        
        if ticker not in available:
            print(f"❌ {ticker} is not in available list or already selected. Please try again.")
            i -= 1
            continue
        
        selected.append(ticker)
        available.remove(ticker)
        print(f"✅ Selected: {ticker}")
    
    print(f"\n✅ Selected stocks: {', '.join(selected)}")
    
    # Input capital amounts
    print("\n" + "-"*70)
    print("STEP 2: Input Capital Allocation")
    print("-"*70)
    print("Enter the capital amount (in currency) you want to allocate to each stock.")
    print("You can enter 0 if you don't want to invest in that stock initially.")
    
    capital_amounts = []
    for ticker in selected:
        while True:
            capital_str = input(f"\nCapital for {ticker}: ").strip()
            try:
                capital = float(capital_str) if capital_str else 0.0
                if capital < 0:
                    print("❌ Capital cannot be negative. Please enter a positive number.")
                    continue
                capital_amounts.append(capital)
                print(f"✅ {ticker}: {format_currency(capital)}")
                break
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    
    total_capital = sum(capital_amounts)
    if total_capital == 0:
        print("\n⚠️  Warning: Total capital is 0. Using equal allocation.")
        capital_amounts = [1.0] * 5
    
    print(f"\n✅ Total Capital: {format_currency(sum(capital_amounts))}")
    
    # Date range - Mặc định 2 năm gần nhất
    print("\n" + "-"*70)
    print("STEP 3: Data Period")
    print("-"*70)
    from datetime import datetime, timedelta
    today = datetime.now()
    default_end = today.strftime('%Y-%m-%d')
    default_start = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # ~2 years
    
    print(f"Default: Last 2 years ({default_start} to {default_end})")
    start_date = input(f"Start date (YYYY-MM-DD, default {default_start}): ").strip() or default_start
    end_date = input(f"End date (YYYY-MM-DD, default {default_end}): ").strip() or default_end
    
    # Training episodes
    print("\n" + "-"*70)
    print("STEP 4: Training Parameters")
    print("-"*70)
    episodes_input = input("Number of RL training episodes (default 150): ").strip()
    n_episodes = int(episodes_input) if episodes_input else 150
    
    return selected, capital_amounts, start_date, end_date, n_episodes


def train_initial_models():
    """Train models với 30 mã cổ phiếu - Lấy dữ liệu 2 năm gần nhất"""
    print("="*70)
    print("INITIAL MODEL TRAINING")
    print("="*70)
    print(f"Training with {len(AVAILABLE_STOCKS)} stocks: {', '.join(AVAILABLE_STOCKS)}")
    
    # Tính 2 năm gần nhất từ ngày hiện tại
    from datetime import datetime, timedelta
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # ~2 years (730 days)
    
    print(f"\n📅 Data Period: {start_date} to {end_date} (2 years)")
    print(f"📥 Downloading data for all {len(AVAILABLE_STOCKS)} stocks...")
    
    prices, returns, ohlcv = download_stock_data(
        AVAILABLE_STOCKS,
        start_date,
        end_date,
        use_cache=True,
        cache_manager=None
    )
    
    # Check if data was downloaded successfully
    if prices is None or returns is None:
        error_msg = "❌ Không thể tải dữ liệu cho các mã cổ phiếu.\n"
        error_msg += "   Vui lòng:\n"
        error_msg += "   1. Kiểm tra kết nối internet\n"
        error_msg += "   2. Thử lại sau\n"
        error_msg += "   3. Hoặc sử dụng file CSV (Data_test.csv) nếu có"
        print(error_msg)
        raise ValueError(error_msg)
    
    print(f"✅ Data downloaded successfully!")
    print(f"   Total days: {len(prices)}")
    print(f"   Stocks: {list(prices.columns)}")
    
    # Save data
    save_data(prices, returns, output_dir='data')
    
    print("\n✅ Initial data preparation completed!")
    print("   You can now use the portfolio advisor with your selected stocks.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--train':
        # Train initial models
        train_initial_models()
    else:
        # Main application
        try:
            selected_tickers, capital_amounts, start_date, end_date, n_episodes = prompt_user_inputs()
            
            print("\n" + "="*70)
            print("STARTING PORTFOLIO ANALYSIS")
            print("="*70)
            
            result = generate_recommendation(
                selected_tickers,
                capital_amounts,
                start_date,
                end_date,
                n_episodes
            )
            
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE")
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

