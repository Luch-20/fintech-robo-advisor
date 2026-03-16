"""
Script để chia dataset thành train/test và evaluate model với baseline

Chia dataset:
- Train: 80% (hoặc configurable)
- Test: 20% (hoặc configurable)

Test strategies:
1. IPO-DRL (model của chúng ta)
2. Buy&Hold (baseline)
3. Quarterly MV (Equal Weight rebalanced every 126 days) - baseline
4. Benchmark (VN-Index) - baseline
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
from typing import Tuple, Dict, List

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

from Get_data import download_stock_data
from robo_agent import IPOAgent, ActorNetwork, train_robo_advisor, extract_state_features
from rebalance import backtest_rebalance
from report_figures import pick_tickers, load_trained_drl_model

# Configuration
TRAIN_RATIO = 0.7  # 70% train, 30% test (tăng test set để đánh giá chính xác hơn)
TEST_RATIO = 1.0 - TRAIN_RATIO

# 30 mã cổ phiếu VN Index
AVAILABLE_STOCKS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
    '^VNINDEX'
]

TRANSACTION_COST_RATE = 0.003  # 0.3%


def split_train_test(prices: pd.DataFrame, returns: pd.DataFrame, 
                     train_ratio: float = TRAIN_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dataset thành train và test sets
    
    Args:
        prices: DataFrame với prices
        returns: DataFrame với returns
        train_ratio: Tỷ lệ train (mặc định 0.8 = 80%)
    
    Returns:
        train_prices, train_returns, test_prices, test_returns
    """
    # Sort by date để đảm bảo thứ tự thời gian
    prices = prices.sort_index()
    returns = returns.sort_index()
    
    # Tính số ngày cho train
    n_total = len(prices)
    n_train = int(n_total * train_ratio)
    
    # Chia theo thời gian (không random để giữ tính liên tục)
    train_prices = prices.iloc[:n_train].copy()
    train_returns = returns.iloc[:n_train].copy()
    
    test_prices = prices.iloc[n_train:].copy()
    test_returns = returns.iloc[n_train:].copy()
    
    print(f"📊 Dataset Split:")
    print(f"   Total days: {n_total}")
    print(f"   Train: {n_train} days ({train_ratio*100:.1f}%)")
    print(f"   Test: {n_total - n_train} days ({TEST_RATIO*100:.1f}%)")
    print(f"   Train period: {train_prices.index[0]} → {train_prices.index[-1]}")
    print(f"   Test period: {test_prices.index[0]} → {test_prices.index[-1]}")
    
    return train_prices, train_returns, test_prices, test_returns


def train_model_on_train_set(train_returns: pd.DataFrame, train_prices: pd.DataFrame, 
                             ohlcv: Dict = None, n_episodes: int = 2000) -> Tuple:
    """
    Train model trên train set
    
    Returns:
        (agent, history)
    """
    print("\n" + "="*70)
    print("🚀 TRAINING MODEL ON TRAIN SET")
    print("="*70)
    
    agent, history = train_robo_advisor(
        train_returns,
        n_episodes=n_episodes,
        stock_code='train_set',
        cache_manager=None,
        prices=train_prices,
        ohlcv=ohlcv
    )
    
    if agent is None:
        raise RuntimeError("Training thất bại - agent is None")
    
    # Save model
    Path('models').mkdir(parents=True, exist_ok=True)
    model_path = Path('models') / 'trained_model.pth'
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'omega': agent.omega,
        'target_return': agent.target_return,
        'n_stocks': agent.n_stocks,
        'state_dim': agent.state_dim,
        'stock_names': list(train_returns.columns)
    }, model_path)
    
    print(f"✅ Model đã được lưu tại: {model_path}")
    
    return agent, history


def evaluate_strategies(test_prices: pd.DataFrame, test_returns: pd.DataFrame,
                       tickers: List[str], drl_actor=None, drl_info=None) -> Dict:
    """
    Evaluate tất cả strategies trên test set
    
    Args:
        test_prices: Test prices
        test_returns: Test returns
        tickers: List of tickers to evaluate
        drl_actor: Trained DRL actor (optional)
        drl_info: Model info (optional)
    
    Returns:
        Dictionary với results cho mỗi strategy
    """
    print("\n" + "="*70)
    print("📈 EVALUATING STRATEGIES ON TEST SET")
    print("="*70)
    
    # Filter prices và returns cho tickers
    p = test_prices[tickers]
    r = test_returns[tickers]
    
    results = {}
    
    # 1. Benchmark (VN-Index) nếu có
    if '^VNINDEX' in test_prices.columns:
        print("\n1️⃣  Benchmark (VN-Index)...")
        bench_prices = test_prices[['^VNINDEX']].dropna()
        bench_rets = test_returns[['^VNINDEX']].dropna()
        common_idx = p.index.intersection(bench_prices.index)
        if len(common_idx) > 0:
            bench_result = backtest_rebalance(
                bench_prices.loc[common_idx],
                bench_rets.loc[common_idx],
                strategy_name="Benchmark",
                rebalance_every=10_000,  # Không rebalance (Buy&Hold cho benchmark)
                transaction_cost_rate=0.0,
            )
            results['Benchmark'] = bench_result
            print(f"   ✅ Benchmark: Sharpe={bench_result.sharpe:.4f}, Return={bench_result.annual_return*100:.2f}%")
    
    # 2. Buy&Hold Baseline
    print("\n2️⃣  Buy&Hold Baseline...")
    buyhold_result = backtest_rebalance(
        p,
        r,
        strategy_name="Buy&Hold",
        rebalance_every=10_000,  # Không rebalance (effectively Buy&Hold)
        transaction_cost_rate=0.0,  # Buy&Hold không có transaction cost (chỉ mua 1 lần)
    )
    results['Buy&Hold'] = buyhold_result
    print(f"   ✅ Buy&Hold: Sharpe={buyhold_result.sharpe:.4f}, Return={buyhold_result.annual_return*100:.2f}%")
    print(f"      Final wealth: {buyhold_result.wealth.iloc[-1]:.4f}, Max drawdown: {buyhold_result.max_drawdown*100:.2f}%")
    
    # 3. Quarterly MV (Equal Weight rebalanced every 126 days)
    print("\n3️⃣  Quarterly MV (Equal Weight)...")
    # Tính số lần rebalance sẽ có trong test period
    n_test_days = len(p)
    n_rebalances = n_test_days // 126
    print(f"      Test period: {n_test_days} days, Expected rebalances: {n_rebalances}")
    
    quarterly_mv_result = backtest_rebalance(
        p,
        r,
        strategy_name="Quarterly MV",
        rebalance_every=126,  # Rebalance mỗi 126 ngày (6 tháng)
        transaction_cost_rate=TRANSACTION_COST_RATE,
    )
    results['Quarterly MV'] = quarterly_mv_result
    print(f"   ✅ Quarterly MV: Sharpe={quarterly_mv_result.sharpe:.4f}, Return={quarterly_mv_result.annual_return*100:.2f}%")
    print(f"      Final wealth: {quarterly_mv_result.wealth.iloc[-1]:.4f}, Max drawdown: {quarterly_mv_result.max_drawdown*100:.2f}%")
    print(f"      Avg turnover: {quarterly_mv_result.turnover:.4f}, Total tx cost: {quarterly_mv_result.tx_cost_cum:.4f}")
    
    # 4. IPO-DRL (nếu có model)
    if drl_actor is not None and drl_info is not None:
        print("\n4️⃣  IPO-DRL Strategy...")
        try:
            # IPO Agent: học risk preference từ equal weights ban đầu
            ipo_agent = IPOAgent(n_stocks=len(tickers))
            initial_weights = np.ones(len(tickers)) / len(tickers)
            cov_matrix = r.cov().values
            if np.any(np.isnan(cov_matrix)):
                cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            ipo_agent.learn_risk_preference(initial_weights, r, cov_matrix, prices=p)
            
            # Cache để smoothing weights
            previous_weights = None
            smoothing_alpha = 0.08
            rebalance_threshold = 0.12
            min_rebalance_interval = 2
            rebalance_counter = 0
            
            def ipo_drl_weights(pr, rt):
                """Tính weights từ IPO optimal + DRL adjustment"""
                nonlocal previous_weights, rebalance_counter
                rebalance_counter += 1
                
                # Tính IPO optimal weights
                lookback_days = min(252, len(rt))
                recent_returns = rt.tail(lookback_days) if len(rt) >= lookback_days else rt
                
                mean_returns = recent_returns.mean().values
                cov_matrix_current = recent_returns.cov().values
                if np.any(np.isnan(cov_matrix_current)):
                    cov_matrix_current = np.nan_to_num(cov_matrix_current, nan=0.0, posinf=0.0, neginf=0.0)
                
                ipo_optimal = ipo_agent.calculate_optimal_weights(mean_returns, cov_matrix_current)
                
                # Extract state features cho DRL
                state = extract_state_features(rt, pr, ohlcv=None)
                if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure state dimension matches model
                model_state_dim = drl_info['state_dim']
                if len(state) != model_state_dim:
                    if len(state) < model_state_dim:
                        state = np.pad(state, (0, model_state_dim - len(state)), mode='constant')
                    else:
                        state = state[:model_state_dim]
                
                # Get DRL adjustment từ trained actor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    drl_weights_full = drl_actor(state_tensor).cpu().numpy()[0]
                
                # Map DRL weights to current tickers
                if len(drl_weights_full) >= len(tickers):
                    drl_weights = drl_weights_full[:len(tickers)]
                    drl_weights = drl_weights / (drl_weights.sum() + 1e-8)
                else:
                    drl_weights = drl_weights_full.copy()
                    n_missing = len(tickers) - len(drl_weights)
                    if len(ipo_optimal) >= n_missing:
                        drl_weights = np.concatenate([drl_weights, ipo_optimal[:n_missing]])
                    else:
                        drl_weights = np.concatenate([drl_weights, np.ones(n_missing) / n_missing])
                    drl_weights = drl_weights / (drl_weights.sum() + 1e-8)
                
                # Kết hợp IPO optimal (96%) với DRL adjustment (4%)
                final_weights = 0.96 * ipo_optimal + 0.04 * drl_weights
                final_weights = final_weights / (final_weights.sum() + 1e-8)
                
                # Smoothing với previous weights
                if previous_weights is not None:
                    final_weights = smoothing_alpha * final_weights + (1 - smoothing_alpha) * previous_weights
                    final_weights = final_weights / (final_weights.sum() + 1e-8)
                
                # Rebalance threshold check
                if previous_weights is not None:
                    total_change = np.abs(final_weights - previous_weights).sum()
                    if total_change < rebalance_threshold and rebalance_counter % min_rebalance_interval != 0:
                        final_weights = previous_weights.copy()
                
                previous_weights = final_weights.copy()
                return final_weights
            
            # Backtest IPO-DRL
            print(f"      Test period: {len(p)} days")
            ipo_drl_result = backtest_rebalance(
                p,
                r,
                strategy_name="IPO-DRL",
                rebalance_every=1,  # Rebalance mỗi ngày (DRL có thể điều chỉnh liên tục)
                transaction_cost_rate=TRANSACTION_COST_RATE,
                weight_function=ipo_drl_weights,
            )
            results['IPO-DRL'] = ipo_drl_result
            print(f"   ✅ IPO-DRL: Sharpe={ipo_drl_result.sharpe:.4f}, Return={ipo_drl_result.annual_return*100:.2f}%")
            print(f"      Final wealth: {ipo_drl_result.wealth.iloc[-1]:.4f}, Max drawdown: {ipo_drl_result.max_drawdown*100:.2f}%")
            print(f"      Avg turnover: {ipo_drl_result.turnover:.4f}, Total tx cost: {ipo_drl_result.tx_cost_cum:.4f}")
            
            # Validation: Check if return is realistic
            if ipo_drl_result.annual_return > 5.0:  # > 500%
                print(f"   ⚠️  Warning: IPO-DRL return seems very high. Please verify:")
                print(f"      - Check for look-ahead bias in weight function")
                print(f"      - Verify test period is long enough (>= 1 year recommended)")
                print(f"      - Check if model is overfitting")
        except Exception as e:
            print(f"   ❌ IPO-DRL failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n4️⃣  IPO-DRL: Model chưa được train, bỏ qua")
    
    return results


def generate_comparison_table(results: Dict) -> pd.DataFrame:
    """
    Tạo bảng so sánh kết quả các strategies
    """
    rows = []
    for strategy_name, result in results.items():
        rows.append({
            'Strategy': strategy_name,
            'Annual Return (%)': result.annual_return * 100,
            'Std Dev (%)': result.std_dev * 100,
            'Sharpe Ratio': result.sharpe,
            'Max Drawdown (%)': result.max_drawdown * 100,
            'Turnover': result.turnover,
            'Tx Cost (cum)': result.tx_cost_cum,
        })
    
    df = pd.DataFrame(rows)
    return df


def main():
    """Main function"""
    print("="*70)
    print("📊 TRAIN/TEST SPLIT VÀ EVALUATION")
    print("="*70)
    
    # 1. Load data
    print("\n1️⃣  Loading data...")
    prices, returns, ohlcv = download_stock_data(
        AVAILABLE_STOCKS,
        "",  # Load all data
        "",
        use_cache=True,
        cache_manager=None,
        data_source='csv'
    )
    
    if prices is None or returns is None:
        raise RuntimeError("Không load được dữ liệu từ CSV")
    
    print(f"   ✅ Đã load: {len(prices)} ngày, {len(prices.columns)} mã cổ phiếu")
    
    # 2. Split train/test
    print("\n2️⃣  Splitting dataset...")
    train_prices, train_returns, test_prices, test_returns = split_train_test(
        prices, returns, train_ratio=TRAIN_RATIO
    )
    
    # 3. Train model trên train set
    print("\n3️⃣  Training model...")
    agent, history = train_model_on_train_set(
        train_returns, train_prices, ohlcv=ohlcv, n_episodes=2000
    )
    
    # 4. Evaluate trên test set với các strategies
    print("\n4️⃣  Evaluating strategies...")
    
    # Chọn tickers để test (bỏ index ticker)
    test_tickers = [t for t in AVAILABLE_STOCKS if not t.startswith('^')]
    test_tickers = [t for t in test_tickers if t in test_prices.columns][:30]  # Tối đa 30 mã
    
    # Load trained model
    drl_actor, drl_info = load_trained_drl_model()
    
    results = evaluate_strategies(
        test_prices, test_returns, test_tickers,
        drl_actor=drl_actor, drl_info=drl_info
    )
    
    # 5. Generate comparison table
    print("\n5️⃣  Generating comparison table...")
    comparison_df = generate_comparison_table(results)
    
    # Save results
    output_path = Path('test_results_comparison.csv')
    comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Kết quả đã được lưu tại: {output_path}")
    
    # Print table
    print("\n" + "="*70)
    print("📊 KẾT QUẢ SO SÁNH CÁC STRATEGIES (TEST SET)")
    print("="*70)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ HOÀN TẤT")
    print("="*70)
    print(f"\n📝 Summary:")
    print(f"   Train ratio: {TRAIN_RATIO*100:.1f}%")
    print(f"   Test ratio: {TEST_RATIO*100:.1f}%")
    print(f"   Test period: {len(test_prices)} days ({len(test_prices)/252:.2f} years)")
    print(f"   Strategies tested: {len(results)}")
    print(f"   Results saved to: {output_path}")
    
    # Additional validation summary
    print(f"\n🔍 Validation Summary:")
    for strategy_name, result in results.items():
        if result.annual_return > 5.0:
            print(f"   ⚠️  {strategy_name}: Return {result.annual_return*100:.2f}% seems very high")
        elif result.annual_return < -0.5:
            print(f"   ⚠️  {strategy_name}: Return {result.annual_return*100:.2f}% is negative (loss)")
        else:
            print(f"   ✅ {strategy_name}: Return {result.annual_return*100:.2f}% seems reasonable")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

