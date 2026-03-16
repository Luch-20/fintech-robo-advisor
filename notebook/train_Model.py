"""
Training Script for DDPG Model

Train DDPG agent với 30 mã cổ phiếu VN Index
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

from Get_data import download_stock_data, save_data
from robo_agent import train_robo_advisor
import torch

# 30 mã cổ phiếu VN Index + 1 chỉ số VN-Index mặc định
AVAILABLE_STOCKS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
    '^VNINDEX'
]


def train_all_models(use_train_test_split=False, train_ratio=0.8):
    """
    Train model cho tất cả 30 mã cổ phiếu
    
    Args:
        use_train_test_split: Nếu True, chia dataset thành train/test và chỉ train trên train set
        train_ratio: Tỷ lệ train (chỉ dùng khi use_train_test_split=True)
    """
    print("="*70)
    print("TRAINING MODELS FOR ALL STOCKS")
    if use_train_test_split:
        print(f"Using Train/Test Split: {train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test")
    else:
        print("Using Full Dataset (no split)")
    print("="*70)
    
    # Tính 2 năm gần nhất từ thời điểm hiện tại (theo giờ)
    from datetime import datetime, timedelta
    now = datetime.now()
    
    # End date: Hôm nay (yfinance sẽ lấy đến giờ gần nhất có data)
    end_date = now.strftime('%Y-%m-%d')
    
    # Start date: 730 ngày (2 năm) trước từ hôm nay
    start_date = (now - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Download data cho tất cả stocks (with OHLCV)
    # Use CSV data source (vn_stocks_data_2020_2025.csv)
    print("Đang load dữ liệu từ CSV...")
    prices, returns, ohlcv = download_stock_data(
        AVAILABLE_STOCKS,
        start_date,
        end_date,
        use_cache=True,
        cache_manager=None,
        data_source='csv'  # Use CSV data from Data_test.csv
    )
    
    if prices is None or returns is None:
        raise RuntimeError("Không load được dữ liệu từ CSV")
    
    print(f"Đã load dữ liệu: {len(prices)} ngày, {len(prices.columns)} mã cổ phiếu")
    
    # Split train/test nếu cần
    if use_train_test_split:
        print("\n📊 Chia dataset thành train/test...")
        n_total = len(prices)
        n_train = int(n_total * train_ratio)
        
        prices = prices.sort_index()
        returns = returns.sort_index()
        
        train_prices = prices.iloc[:n_train].copy()
        train_returns = returns.iloc[:n_train].copy()
        
        print(f"   Train: {n_train} days ({train_ratio*100:.1f}%)")
        print(f"   Test: {n_total - n_train} days ({(1-train_ratio)*100:.1f}%)")
        print(f"   Train period: {train_prices.index[0]} → {train_prices.index[-1]}")
        
        # Use train set for training
        prices = train_prices
        returns = train_returns
        
        # Filter OHLCV for train set
        if ohlcv:
            for key in ohlcv:
                if key in ohlcv and ohlcv[key] is not None:
                    ohlcv[key] = ohlcv[key].iloc[:n_train].copy()
    
    # Save data
    print("Đang lưu dữ liệu...")
    save_data(prices, returns, output_dir='data')
    
    # Train model với tất cả stocks (with OHLCV)
    print("Bắt đầu training model...")
    print(f"Số episodes: 2000")
    agent, history = train_robo_advisor(
        returns,
        n_episodes=2000,  # Increased to 2000 for much deeper learning with large dataset
        stock_code='all_stocks',
        cache_manager=None,
        prices=prices,
        ohlcv=ohlcv
    )
    
    if agent is None:
        raise RuntimeError("Training thất bại - agent is None")
    
    print("Training hoàn tất!")
    
    # Save model
    Path('models').mkdir(parents=True, exist_ok=True)
    
    model_path = Path('models') / 'trained_model.pth'
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'omega': agent.omega,
        'target_return': agent.target_return,
        'n_stocks': agent.n_stocks,
        'state_dim': agent.state_dim,  # Save state_dim for proper model loading
        'stock_names': AVAILABLE_STOCKS
    }, model_path)
    
    return model_path


if __name__ == "__main__":
    try:
        train_all_models()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
