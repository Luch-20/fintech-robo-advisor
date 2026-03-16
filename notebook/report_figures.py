"""
Generate report-ready tables and plots (all-time) similar to the provided figures.

Produces:
- A summary table CSV/print for multiple portfolio sizes (n = 5,10,20,50,100,200)
  with two strategies: Equal-weight rebalanced every 126 days vs Buy & Hold.
- A 2x3 grid wealth chart for the same n, using all data in Data_test.csv.

Data source: data/Data_test.csv (no downloading).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
import pandas as pd
import numpy as np

from rebalance import (
    backtest_rebalance,
    summary_table,
)
from typing import List
from Get_data import download_stock_data
from robo_agent import IPOAgent, ActorNetwork, extract_state_features
import torch
from pathlib import Path


def pick_tickers(prices: pd.DataFrame, n: int) -> list[str]:
    """Chọn n mã đầu tiên (bỏ qua mã chỉ số bắt đầu bằng ^ nếu cần)."""
    cols = [c for c in prices.columns if not c.startswith("^")]
    if len(cols) < n:
        # Nếu thiếu, trả về tối đa số có sẵn (không crash)
        return cols
    return cols[:n]


def calculate_quarterly_yearly_returns(results: List, wealth_series_list: List[pd.Series], years: List[int]) -> pd.DataFrame:
    """
    Tính toán Quarterly và Yearly Returns cho các strategies
    Format tương tự bảng mẫu: Fiscal year từ tháng 4 đến tháng 3 năm sau
    
    Args:
        results: List of BacktestResult objects
        wealth_series_list: List of wealth Series tương ứng với results
        years: List of years (2021, 2022, 2023, 2024, 2025, 2026)
    
    Returns:
        DataFrame với columns: Period, Strategy, Q1 (Apr-Jun), Q2 (Jul-Sep), Q3 (Oct-Dec), Q4 (Jan-Mar), Yearly Return, Average
    """
    rows = []
    
    for year in years:
        # Fiscal year: từ tháng 4 năm này đến tháng 3 năm sau
        fiscal_start = pd.Timestamp(f"{year}-04-01")
        fiscal_end = pd.Timestamp(f"{year+1}-03-31")
        
        # Định nghĩa các quý theo fiscal year (Apr-Mar)
        quarters = {
            'Q1_Apr_Jun': (pd.Timestamp(f"{year}-04-01"), pd.Timestamp(f"{year}-06-30")),
            'Q2_Jul_Sep': (pd.Timestamp(f"{year}-07-01"), pd.Timestamp(f"{year}-09-30")),
            'Q3_Oct_Dec': (pd.Timestamp(f"{year}-10-01"), pd.Timestamp(f"{year}-12-31")),
            'Q4_Jan_Mar': (pd.Timestamp(f"{year+1}-01-01"), pd.Timestamp(f"{year+1}-03-31")),
        }
        
        for idx, (result, wealth_series) in enumerate(zip(results, wealth_series_list)):
            # Filter wealth series cho fiscal year này
            fiscal_wealth = wealth_series[(wealth_series.index >= fiscal_start) & (wealth_series.index <= fiscal_end)]
            
            if len(fiscal_wealth) < 10:
                continue
            
            # Tính quarterly returns
            q_returns = {}
            for q_name, (q_start, q_end) in quarters.items():
                q_wealth = fiscal_wealth[(fiscal_wealth.index >= q_start) & (fiscal_wealth.index <= q_end)]
                if len(q_wealth) >= 5:  # Cần ít nhất 5 ngày
                    q_start_value = q_wealth.iloc[0]
                    q_end_value = q_wealth.iloc[-1]
                    if q_start_value > 0:
                        q_return = (q_end_value - q_start_value) / q_start_value * 100  # %
                        q_returns[q_name] = q_return
                    else:
                        q_returns[q_name] = None
                else:
                    q_returns[q_name] = None
            
            # Tính yearly return (fiscal year)
            yearly_start = fiscal_wealth.iloc[0]
            yearly_end = fiscal_wealth.iloc[-1]
            if yearly_start > 0:
                yearly_return = (yearly_end - yearly_start) / yearly_start * 100  # %
            else:
                yearly_return = 0.0
            
            # Tính average (tổng 4 quý, tương tự bảng mẫu)
            # Trong bảng mẫu: Average = tổng 4 quý (không phải trung bình)
            valid_q_returns = [v for v in q_returns.values() if v is not None]
            if len(valid_q_returns) == 4:  # Phải có đủ 4 quý
                average = sum(valid_q_returns)  # Tổng 4 quý (không chia 4)
            else:
                average = None
            
            # Format period (tương tự bảng mẫu: "04-2016 to 03-2017")
            period = f"04-{year} to 03-{year+1}"
            
            row = {
                "Period": period,
                "Strategy": result.strategy,
                "Apr-Jun": f"{q_returns.get('Q1_Apr_Jun', 0):.2f}%" if q_returns.get('Q1_Apr_Jun') is not None else "-",
                "Jul-Sep": f"{q_returns.get('Q2_Jul_Sep', 0):.2f}%" if q_returns.get('Q2_Jul_Sep') is not None else "-",
                "Oct-Dec": f"{q_returns.get('Q3_Oct_Dec', 0):.2f}%" if q_returns.get('Q3_Oct_Dec') is not None else "-",
                "Jan-Mar": f"{q_returns.get('Q4_Jan_Mar', 0):.2f}%" if q_returns.get('Q4_Jan_Mar') is not None else "-",
                "Yearly Return": f"{yearly_return:.2f}%",
                "Average": f"{average:.2f}%" if average is not None else "-",
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def load_trained_drl_model():
    """Load trained DRL model from checkpoint"""
    model_path = Path('models') / 'trained_model.pth'
    if not model_path.exists():
        return None, None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        n_stocks = checkpoint.get('n_stocks', 30)
        state_dim = checkpoint.get('state_dim', 11)  # Default 11 features
        
        actor = ActorNetwork(n_stocks, state_dim=state_dim)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        model_info = {
            'omega': checkpoint.get('omega', 1.0),
            'target_return': checkpoint.get('target_return', 0.1),
            'n_stocks': n_stocks,
            'state_dim': state_dim,
        }
        return actor, model_info
    except Exception:
        return None, None


def build_table_and_plots(
    n_list=(5, 10, 20, 30),  # Chỉ giữ 4 giá trị n để có 4 subplot
    rebalance_every_list=(126,),  # chỉ dùng chu kỳ 126 ngày
    title="Wealth (All data, rebalance vs hold)",
    table_path="report_table.csv",
    fig_path="report_wealth.png",
    include_benchmark=True,
    benchmark_ticker="^VNINDEX",
    transaction_cost_rate=0.003,  # 0.3% phí giao dịch
):
    # Initialize quarterly_rows list
    build_table_and_plots.quarterly_rows = []
    # Lấy toàn bộ danh sách mã từ vn_stocks_data_2020_2025.csv (file chính)
    import pandas as pd
    csv_path = "vn_stocks_data_2020_2025.csv"
    if not Path(csv_path).exists():
        csv_path = Path("data") / "vn_stocks_data_2020_2025.csv"
    if not Path(csv_path).exists():
        # Fallback to Data_test.csv
        csv_path = "data/Data_test.csv"
    
    raw = pd.read_csv(csv_path)
    available_tickers = sorted([t for t in raw["Ticker"].unique() if isinstance(t, str)])
    if not available_tickers:
        raise RuntimeError("File CSV không có cột Ticker hợp lệ.")

    # Load full data (all time) cho toàn bộ mã
    prices, rets, _ = download_stock_data(available_tickers, "", "", data_source="csv")
    if prices is None or rets is None:
        raise RuntimeError("Không load được dữ liệu từ Data_test.csv")

    rows = []

    # Prepare plot grid - 2x2 cho 4 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    plotted = 0
    for idx, n in enumerate(n_list):
        if plotted >= 4:  # Chỉ vẽ tối đa 4 subplot (2x2 grid)
            break
        tickers = pick_tickers(prices, n)
        if len(tickers) < 2:
            continue
        p = prices[tickers]
        r = rets[tickers]

        strategies = []

        # Benchmark nếu có
        if include_benchmark and benchmark_ticker in prices.columns:
            bench_prices = prices[[benchmark_ticker]].dropna()
            bench_rets = rets[[benchmark_ticker]].dropna()
            # align
            common_idx = p.index.intersection(bench_prices.index)
            p = p.loc[common_idx]
            r = r.loc[common_idx]
            bench_prices = bench_prices.loc[common_idx]
            bench_rets = bench_rets.loc[common_idx]
            bench = backtest_rebalance(
                bench_prices,
                bench_rets,
                strategy_name="Benchmark",
                rebalance_every=10_000,
                transaction_cost_rate=0.0,
            )
            strategies.append(bench)

        # Quarterly MV (Equal Weight) với chu kỳ rebalance (mặc định 126d)
        for reb_days in rebalance_every_list:
            eq = backtest_rebalance(
                p,
                r,
                strategy_name="Quarterly MV",
                rebalance_every=reb_days,
                transaction_cost_rate=transaction_cost_rate,
            )
            strategies.append(eq)

        # IPO-DRL: Sử dụng IPO Agent + DRL Agent (từ model đã train) để tối ưu hóa
        try:
            # Load trained DRL model
            drl_actor, drl_info = load_trained_drl_model()
            if drl_actor is None or drl_info is None:
                pass
            else:
                # IPO Agent: học risk preference từ equal weights ban đầu
                ipo_agent = IPOAgent(n_stocks=len(tickers))
                initial_weights = np.ones(len(tickers)) / len(tickers)
                cov_matrix = r.cov().values
                ipo_agent.learn_risk_preference(initial_weights, r, cov_matrix, prices=p)
                
                # Cache để smoothing weights (exponential moving average)
                previous_weights = None
                smoothing_alpha = 0.08  # Smoothing factor (0.08 = 8% new, 92% old) - cân bằng
                rebalance_threshold = 0.12  # Chỉ rebalance nếu tổng thay đổi > 12% - giảm turnover
                min_rebalance_interval = 2  # Chỉ rebalance mỗi 2 lần (giảm tần suất)
                rebalance_counter = 0
                
                # Tạo weight function cho IPO-DRL: Cải thiện để giảm turnover và tăng Sharpe
                def ipo_drl_weights(pr, rt):
                    """Tính weights từ IPO optimal + DRL adjustment (cải thiện)"""
                    nonlocal previous_weights, rebalance_counter
                    rebalance_counter += 1
                    
                    # 1. Tính IPO optimal weights từ historical data với lookback window dài hơn
                    # Sử dụng ít nhất 252 ngày (1 năm) để ổn định hơn
                    lookback_days = min(252, len(rt))
                    recent_returns = rt.tail(lookback_days) if len(rt) >= lookback_days else rt
                    
                    mean_returns = recent_returns.mean().values
                    cov_matrix_current = recent_returns.cov().values
                    
                    # Tính IPO optimal weights với regularization để giảm turnover
                    ipo_optimal = ipo_agent.calculate_optimal_weights(mean_returns, cov_matrix_current)
                    
                    # 2. Extract state features cho DRL (chỉ dùng để điều chỉnh nhẹ)
                    state = extract_state_features(rt, pr, ohlcv=None)
                    
                    # Check for NaN and fix
                    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Ensure state dimension matches model
                    model_state_dim = drl_info['state_dim']
                    if len(state) != model_state_dim:
                        if len(state) < model_state_dim:
                            state = np.pad(state, (0, model_state_dim - len(state)), mode='constant')
                        else:
                            state = state[:model_state_dim]
                    
                    # 3. Get DRL adjustment từ trained actor (chỉ dùng để điều chỉnh nhẹ)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        drl_weights_full = drl_actor(state_tensor).cpu().numpy()[0]
                    
                    # 4. Map DRL weights to current tickers
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
                    
                    # 5. Kết hợp IPO optimal (96%) với DRL adjustment (4%) để cân bằng
                    # Giữ một chút DRL để có sự khác biệt với EW
                    final_weights = 0.96 * ipo_optimal + 0.04 * drl_weights
                    
                    # 6. Smoothing với previous weights và kiểm tra điều kiện rebalance
                    if previous_weights is not None:
                        # Exponential moving average: 92% old, 8% new
                        smoothed_weights = smoothing_alpha * final_weights + (1 - smoothing_alpha) * previous_weights
                        
                        # 7. Chỉ rebalance khi:
                        #    - Thay đổi đáng kể (> threshold)
                        #    - Và đã qua đủ số lần rebalance (giảm tần suất)
                        weight_change = np.abs(smoothed_weights - previous_weights).sum()
                        
                        if weight_change < rebalance_threshold or (rebalance_counter % min_rebalance_interval != 0):
                            # Giữ nguyên weights cũ nếu thay đổi quá nhỏ hoặc chưa đến lượt
                            final_weights = previous_weights.copy()
                        else:
                            # Chỉ áp dụng một phần thay đổi để giảm turnover
                            # Chỉ thay đổi 40% của sự khác biệt
                            final_weights = 0.4 * smoothed_weights + 0.6 * previous_weights
                    else:
                        # Lần đầu tiên, khởi tạo với IPO optimal
                        previous_weights = ipo_optimal.copy()
                        final_weights = ipo_optimal.copy()
                    
                    # Normalize và ensure non-negative
                    final_weights = np.maximum(final_weights, 0)
                    if np.sum(final_weights) > 0:
                        final_weights = final_weights / np.sum(final_weights)
                    else:
                        final_weights = np.ones(len(tickers)) / len(tickers)
                    
                    # Đảm bảo weights hợp lệ
                    if np.any(np.isnan(final_weights)) or np.any(np.isinf(final_weights)):
                        final_weights = np.ones(len(tickers)) / len(tickers)
                    
                    # Lưu weights cho lần sau
                    previous_weights = final_weights.copy()
                    
                    return final_weights
                
                ipo_drl = backtest_rebalance(
                    p,
                    r,
                    strategy_name="IPO-DRL",
                    rebalance_every=rebalance_every_list[0],
                    transaction_cost_rate=transaction_cost_rate,
                    weight_fn=ipo_drl_weights,
                )
                strategies.append(ipo_drl)
        except Exception:
            # Bỏ qua nếu lỗi, tiếp tục với các chiến lược khác
            pass

        # Buy & Hold
        bh = backtest_rebalance(
            p,
            r,
            strategy_name="Buy&Hold",
            rebalance_every=10_000,  # effectively no rebalance
            transaction_cost_rate=transaction_cost_rate,
        )
        strategies.append(bh)

        # Collect table rows
        tbl = summary_table(strategies)
        tbl.insert(0, "n", len(tickers))
        rows.append(tbl)
        
        # Collect quarterly and yearly returns table
        wealth_series_list = [s.wealth for s in strategies]
        years = [2021, 2022, 2023, 2024, 2025, 2026]
        quarterly_yearly_table = calculate_quarterly_yearly_returns(strategies, wealth_series_list, years)
        if not quarterly_yearly_table.empty:
            # Chỉ giữ lại chiến lược IPO-DRL theo yêu cầu báo cáo
            quarterly_yearly_table = quarterly_yearly_table[quarterly_yearly_table["Strategy"] == "IPO-DRL"]
        if not quarterly_yearly_table.empty:
            quarterly_yearly_table.insert(0, "n", len(tickers))
            build_table_and_plots.quarterly_rows.append(quarterly_yearly_table)

        # Plot wealth
        ax = axes[plotted]
        # Màu đèn giao thông: đỏ - vàng - xanh
        colors_map = {
            "Benchmark": "#6c757d",     # xám nhẹ cho benchmark
            "Quarterly MV": "#D62728",  # đỏ (stop)
            "IPO-DRL": "#F4C20D",       # vàng (caution)
            "Buy&Hold": "#00A65A",      # xanh (go)
        }
        for s in strategies:
            color = colors_map.get(s.strategy, "#6c757d")
            ax.plot(s.wealth.index, s.wealth.values, label=s.strategy, color=color, linewidth=2.0)
        ax.set_title(f"n = {len(tickers)}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format trục x với YearLocator và DateFormatter
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        
        # Thiết lập giới hạn trục x
        ax.set_xlim(left=pd.Timestamp('2021-01-01'), right=pd.Timestamp('2026-12-31'))
        
        # BẮT BUỘC hiển thị nhãn năm cho TẤT CẢ các biểu đồ (bỏ qua sharex)
        ax.tick_params(axis='x', which='major', labelbottom=True, length=5, width=1, labelsize=10)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=10)
        
        if plotted % 2 == 0:  # Cột trái
            ax.set_ylabel("Wealth (start=1)", fontsize=11, fontweight='bold')
        
        # Hiển thị xlabel "Year" cho TẤT CẢ các biểu đồ
        ax.set_xlabel("Year", fontsize=11, fontweight='bold')
        
        if plotted == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("Không đủ mã để vẽ bất kỳ danh mục nào.")

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_path, dpi=200)
    print(f"Đã lưu biểu đồ vào: {fig_path}")

    # Merge and save table
    full_table = pd.concat(rows, ignore_index=True)
    full_table.to_csv(table_path, index=False)
    print(f"Đã lưu bảng tổng hợp vào: {table_path}")
    
    # Merge and save quarterly and yearly returns table
    if hasattr(build_table_and_plots, 'quarterly_rows') and build_table_and_plots.quarterly_rows:
        quarterly_table = pd.concat(build_table_and_plots.quarterly_rows, ignore_index=True)
        # Sắp xếp theo Period, n, Strategy
        quarterly_table = quarterly_table.sort_values(['Period', 'n', 'Strategy'])
        quarterly_table_path = table_path.replace('.csv', '_quarterly_yearly.csv')
        quarterly_table.to_csv(quarterly_table_path, index=False)
        print(f"Đã lưu bảng Quarterly và Yearly Returns vào: {quarterly_table_path}")


if __name__ == "__main__":
    build_table_and_plots()