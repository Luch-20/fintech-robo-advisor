"""
Get Data Module - Lấy dữ liệu giá cổ phiếu theo ngày (daily data)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Không cần import data_source vì chỉ đọc từ CSV
HAS_CUSTOM_DATA_SOURCE = False

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)


def check_ticker_validity(ticker, data_source='vnstock'):
    """
    Kiểm tra xem mã cổ phiếu có hợp lệ và có dữ liệu trên nguồn không
    
    Args:
        ticker: Mã cổ phiếu cần kiểm tra
        data_source: Nguồn dữ liệu để kiểm tra
    
    Returns:
        (is_valid, reason): Tuple (True/False, lý do)
    """
    ticker_clean = ticker.upper().replace('^', '')
    
    # Mã chỉ số không phải cổ phiếu
    if ticker.startswith('^'):
        return False, "Đây là chỉ số, không phải cổ phiếu"
    
    # Kiểm tra với vnstock nếu có
    if data_source == 'vnstock' and HAS_CUSTOM_DATA_SOURCE:
        try:
            # Thử lấy danh sách mã để kiểm tra
            # Note: Có thể cần điều chỉnh tùy API vnstock
            return True, "Mã hợp lệ"
        except Exception:
            pass
    
    # Mặc định coi là hợp lệ, sẽ kiểm tra khi download
    return True, "Chưa kiểm tra được"


def validate_tickers_list(tickers):
    """
    Validate danh sách mã cổ phiếu và phân loại
    
    Args:
        tickers: List các mã cổ phiếu
    
    Returns:
        dict với keys: 'valid', 'index_symbols', 'potentially_invalid'
    """
    result = {
        'valid': [],
        'index_symbols': [],
        'potentially_invalid': []
    }
    
    for ticker in tickers:
        ticker_clean = ticker.upper().replace('^', '')
        
        # Phân loại mã chỉ số
        if ticker.startswith('^'):
            result['index_symbols'].append(ticker)
        else:
            # Kiểm tra format cơ bản (3-4 ký tự chữ cái)
            if len(ticker_clean) >= 3 and len(ticker_clean) <= 4 and ticker_clean.isalpha():
                result['valid'].append(ticker)
            else:
                result['potentially_invalid'].append(ticker)
    
    return result


def load_data_from_csv(csv_path, tickers, start_date=None, end_date=None):
    """
    Load dữ liệu từ file CSV (Data_test.csv format) - Daily data
    
    Args:
        csv_path (str): Đường dẫn đến file CSV
        tickers (list): Danh sách stock symbols cần load
        start_date (str): Ngày bắt đầu filter (optional)
        end_date (str): Ngày kết thúc filter (optional)
    
    Returns:
        prices (DataFrame): Giá đóng cửa theo ngày
        returns (DataFrame): Returns theo ngày (%)
        ohlcv (dict): Dictionary chứa Open, High, Low, Close, Volume DataFrames
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Remove timezone if present to avoid comparison issues
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Filter by tickers
    available_tickers = df['Ticker'].unique()
    requested_tickers = [t for t in tickers if t in available_tickers]
    
    if not requested_tickers:
        return None, None, None
    
    df = df[df['Ticker'].isin(requested_tickers)]
    
    # Filter by date range if provided
    if start_date:
        start_dt = pd.to_datetime(start_date)
        # Remove timezone if present
        if start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
        df = df[df['Date'] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        # Remove timezone if present
        if end_dt.tz is not None:
            end_dt = end_dt.tz_localize(None)
        df = df[df['Date'] <= end_dt]
    
    if len(df) == 0:
        return None, None, None
    
    # Create daily DataFrame (one row per day per ticker)
    daily_data = []
    for ticker in requested_tickers:
        ticker_df = df[df['Ticker'] == ticker].sort_values('Date')
        
        for _, row in ticker_df.iterrows():
            daily_data.append({
                'Date': pd.to_datetime(row['Date']),
                'Ticker': ticker,
                'Close': row['Close'],
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Volume': row['Volume']
            })
    
    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.sort_values(['Ticker', 'Date'])
    
    # Create prices DataFrame (pivot by ticker)
    prices = daily_df.pivot(index='Date', columns='Ticker', values='Close')
    prices.index.name = None
    
    # Create OHLCV dictionaries
    ohlcv = {
        'Open': daily_df.pivot(index='Date', columns='Ticker', values='Open'),
        'High': daily_df.pivot(index='Date', columns='Ticker', values='High'),
        'Low': daily_df.pivot(index='Date', columns='Ticker', values='Low'),
        'Close': prices.copy(),
        'Volume': daily_df.pivot(index='Date', columns='Ticker', values='Volume')
    }
    
    # Calculate returns (daily returns)
    returns = prices.pct_change().dropna()
    
    # Clean data
    prices = prices.dropna()
    for key in ohlcv:
        ohlcv[key] = ohlcv[key].reindex(prices.index).ffill().bfill()
    
    return prices, returns, ohlcv


def merge_dataframes(df1, df2):
    """
    Merge hai DataFrame theo index (Date), ưu tiên df1 nếu có conflict
    
    Args:
        df1: DataFrame đầu tiên
        df2: DataFrame thứ hai
    
    Returns:
        merged: DataFrame đã merge
    """
    if df1 is None or df1.empty:
        return df2
    if df2 is None or df2.empty:
        return df1
    
    # Lấy union của tất cả dates
    all_dates = df1.index.union(df2.index).sort_values()
    
    # Tạo merged DataFrame với tất cả dates
    merged = pd.DataFrame(index=all_dates)
    
    # Thêm các cột từ df1 trước
    for col in df1.columns:
        merged[col] = df1[col]
    
    # Thêm các cột từ df2 (chỉ nếu chưa có trong df1)
    for col in df2.columns:
        if col not in merged.columns:
            merged[col] = df2[col]
        else:
            # Nếu có conflict, giữ df1 nhưng fill NaN từ df2
            merged[col] = merged[col].fillna(df2[col])
    
    return merged.sort_index()


def merge_ohlcv(ohlcv1, ohlcv2):
    """
    Merge hai OHLCV dictionaries
    
    Args:
        ohlcv1: OHLCV dict đầu tiên
        ohlcv2: OHLCV dict thứ hai
    
    Returns:
        merged: OHLCV dict đã merge
    """
    if ohlcv1 is None or not ohlcv1:
        return ohlcv2
    if ohlcv2 is None or not ohlcv2:
        return ohlcv1
    
    merged = {}
    for key in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if key in ohlcv1 and ohlcv1[key] is not None:
            merged[key] = merge_dataframes(ohlcv1[key], ohlcv2.get(key))
        elif key in ohlcv2 and ohlcv2[key] is not None:
            merged[key] = ohlcv2[key]
    
    return merged


def download_stock_data(tickers, start_date, end_date, use_cache=True, cache_manager=None, data_source='csv'):
    """
    Load dữ liệu giá cổ phiếu từ file CSV Data_test.csv (KHÔNG DOWNLOAD)
    
    Args:
        tickers (list): Danh sách stock symbols ['AAPL', 'GOOGL', ...]
        start_date (str): Ngày bắt đầu '2018-01-01'
        end_date (str): Ngày kết thúc '2024-01-01'
        use_cache (bool): Không sử dụng (giữ lại để tương thích)
        cache_manager: Không sử dụng (giữ lại để tương thích)
        data_source (str): Luôn dùng 'csv' - chỉ đọc từ CSV
    
    Returns:
        prices (DataFrame): Giá đóng cửa theo ngày
        returns (DataFrame): Returns theo ngày (%)
        ohlcv (dict): Dictionary chứa Open, High, Low, Close, Volume DataFrames
    """
    # CHỈ ĐỌC TỪ CSV - KHÔNG DOWNLOAD
    # Tìm file vn_stocks_data_2020_2025.csv ở data/ hoặc thư mục gốc
    csv_path = Path('data') / 'vn_stocks_data_2020_2025.csv'
    if not csv_path.exists():
        csv_path = Path('vn_stocks_data_2020_2025.csv')
    if not csv_path.exists():
        # Fallback cũ nếu chưa có file mới
        csv_path = Path('data') / 'Data_test.csv'
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"   Vui lòng đảm bảo file vn_stocks_data_2020_2025.csv tồn tại trong thư mục data/")
        return None, None, None
    
    # Load data từ CSV
    prices_csv, returns_csv, ohlcv_csv = load_data_from_csv(str(csv_path), tickers, start_date, end_date)
    
    if prices_csv is None or prices_csv.empty:
        return None, None, None
    
    # Kiểm tra xem có mã nào thiếu không
    available_tickers = list(prices_csv.columns)
    
    # Chỉ trả về các mã có sẵn
    available_cols = [t for t in tickers if t in prices_csv.columns]
    
    if not available_cols:
        return None, None, None
    
    prices = prices_csv[available_cols] if available_cols else prices_csv
    returns = returns_csv[available_cols] if available_cols else returns_csv
    
    # Filter OHLCV for available tickers
    ohlcv = {}
    if ohlcv_csv:
        for key in ohlcv_csv:
            if ohlcv_csv[key] is not None and not ohlcv_csv[key].empty:
                ohlcv_cols = [t for t in available_cols if t in ohlcv_csv[key].columns]
                if ohlcv_cols:
                    ohlcv[key] = ohlcv_csv[key][ohlcv_cols]
                else:
                    ohlcv[key] = pd.DataFrame()
    
    return prices, returns, ohlcv


def generate_synthetic_data(tickers, start_date, end_date, n_days=None):
    """
    Tạo dữ liệu synthetic theo ngày nếu không download được (với OHLCV)
    
    Dữ liệu có tính chất realistic:
    - Drift (xu hướng tăng)
    - Volatility (biến động ngẫu nhiên)
    - Correlation giữa các stocks
    
    Args:
        tickers (list): Danh sách stocks
        start_date (str): Ngày bắt đầu
        end_date (str): Ngày kết thúc
        n_days (int): Số ngày data (None = tự động tính từ start_date đến end_date)
    
    Returns:
        prices (DataFrame): Giá synthetic theo ngày
        returns (DataFrame): Returns synthetic theo ngày
        ohlcv (dict): Dictionary chứa Open, High, Low, Close, Volume DataFrames
    """
    from datetime import datetime, timedelta
    
    # Tính số ngày trading từ start_date đến end_date
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Tạo business days (trading days) trong khoảng thời gian
    business_days = pd.bdate_range(start=start_date, end=end_date)
    n_trading_days = len(business_days)
    
    # Tính số ngày
    if n_days is None:
        n_days = n_trading_days
    
    
    np.random.seed(42)  # Reproducible
    n_stocks = len(tickers)
    n_days = int(n_days)
    
    # Tạo daily timestamps từ start_date
    dates = business_days[:n_days]
    
    # Giá ban đầu (realistic cho từng stock)
    initial_prices_map = {
        'AAPL': 100,
        'GOOGL': 750,
        'MSFT': 50,
        'AMZN': 600,
        'META': 100
    }
    initial_prices = np.array([initial_prices_map.get(ticker, 100) for ticker in tickers])
    
    # Parameters cho realistic daily returns
    # Mean return: ~0.0005-0.0015 per day (tương đương ~12-38% annual)
    mean_returns = np.random.uniform(0.0005, 0.0015, n_stocks)  # Daily drift
    # Volatility: ~0.01-0.02 per day (tương đương ~16-25% annual)
    volatility = np.random.uniform(0.01, 0.02, n_stocks)        # Daily vol
    
    # Tạo correlation matrix
    correlation = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            corr = np.random.uniform(0.3, 0.7)
            correlation[i, j] = correlation[j, i] = corr
    
    # Cholesky decomposition cho correlated returns
    try:
        cholesky = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        cholesky = np.eye(n_stocks)
    
    # Generate prices
    prices = np.zeros((n_days, n_stocks))
    prices[0] = initial_prices
    
    for i in range(1, n_days):
        # Generate correlated random shocks
        random_shocks = np.random.randn(n_stocks)
        correlated_shocks = cholesky @ random_shocks
        
        # Calculate returns
        daily_returns = mean_returns + volatility * correlated_shocks
        
        # Update prices
        prices[i] = prices[i-1] * (1 + daily_returns)
    
    # Convert to DataFrames
    prices_df = pd.DataFrame(prices, index=dates, columns=tickers)
    returns_df = prices_df.pct_change().dropna()
    
    # Generate OHLCV data from prices
    ohlcv = {}
    close_prices = prices_df.copy()
    
    # Generate Open (slightly different from previous close)
    open_prices = close_prices.shift(1) * (1 + np.random.normal(0, 0.002, close_prices.shape))
    open_prices.iloc[0] = close_prices.iloc[0] * (1 + np.random.normal(0, 0.002, len(tickers)))
    
    # Generate High (higher than Open and Close)
    high_prices = close_prices.copy()
    for i in range(len(close_prices)):
        for j in range(len(tickers)):
            high_prices.iloc[i, j] = max(open_prices.iloc[i, j], close_prices.iloc[i, j]) * \
                                    (1 + abs(np.random.normal(0, 0.01)))
    
    # Generate Low (lower than Open and Close)
    low_prices = close_prices.copy()
    for i in range(len(close_prices)):
        for j in range(len(tickers)):
            low_prices.iloc[i, j] = min(open_prices.iloc[i, j], close_prices.iloc[i, j]) * \
                                   (1 - abs(np.random.normal(0, 0.01)))
    
    # Generate Volume (realistic volume based on price movement)
    volume_base = np.random.uniform(1000000, 10000000, (len(close_prices), len(tickers)))
    price_change = abs(close_prices.pct_change().fillna(0))
    volume = volume_base * (1 + price_change * 2)  # Higher volume on larger moves
    
    ohlcv = {
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': pd.DataFrame(volume, index=dates, columns=tickers)
    }
    
    return prices_df, returns_df, ohlcv


def save_data(prices, returns, output_dir='data'):
    """
    Lưu data vào CSV files
    
    Args:
        prices (DataFrame): Prices data
        returns (DataFrame): Returns data
        output_dir (str): Output directory
    """
    # Tạo directory nếu chưa có
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save prices
    prices_file = f"{output_dir}/prices.csv"
    prices.to_csv(prices_file)
    
    # Save returns
    returns_file = f"{output_dir}/returns.csv"
    returns.to_csv(returns_file)
    
    # Save info file
    info_file = f"{output_dir}/info.txt"
    with open(info_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DATA INFORMATION\n")
        f.write("="*70 + "\n\n")
        
        # Trading days per year: 252 days
        trading_days_per_year = 252
        
        f.write(f"Stocks: {list(prices.columns)}\n")
        f.write(f"Total trading days: {len(prices)}\n")
        f.write(f"Date range: {prices.index[0]} to {prices.index[-1]}\n\n")
        
        f.write("Daily Statistics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean daily return: {returns.mean().mean()*100:.4f}%\n")
        f.write(f"Mean daily volatility: {returns.std().mean()*100:.4f}%\n\n")
        
        f.write("Annualized Statistics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean annual return: {returns.mean().mean()*trading_days_per_year*100:.2f}%\n")
        f.write(f"Mean annual volatility: {returns.std().mean()*np.sqrt(trading_days_per_year)*100:.2f}%\n\n")
        
        f.write("Per Stock Statistics:\n")
        f.write("-"*70 + "\n")
        for col in returns.columns:
            annual_ret = returns[col].mean() * trading_days_per_year * 100
            annual_vol = returns[col].std() * np.sqrt(trading_days_per_year) * 100
            sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
            f.write(f"{col}:\n")
            f.write(f"  Annual Return: {annual_ret:>8.2f}%\n")
            f.write(f"  Annual Vol:    {annual_vol:>8.2f}%\n")
            f.write(f"  Sharpe Ratio:  {sharpe:>8.2f}\n")


def print_data_summary(prices, returns):
    """
    In summary của data ra console
    
    Args:
        prices (DataFrame): Prices data
        returns (DataFrame): Returns data
    """
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    # Trading days per year: 252 days
    trading_days_per_year = 252
    
    print(f"\n📊 PRICES DATA:")
    print(f"   Shape: {prices.shape}")
    print(f"   Columns: {list(prices.columns)}")
    print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\n   Latest 3 days:")
    print(prices.tail(3).to_string())
    
    print(f"\n📈 RETURNS DATA:")
    print(f"   Shape: {returns.shape}")
    print(f"\n   Basic Statistics:")
    print(returns.describe().to_string())
    
    print(f"\n📅 ANNUALIZED STATISTICS:")
    print(f"   {'Stock':<10} {'Annual Return':<15} {'Annual Vol':<15} {'Sharpe':<10}")
    print(f"   {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    
    for col in returns.columns:
        daily_mean = returns[col].mean()
        daily_std = returns[col].std()
        
        annual_return = daily_mean * trading_days_per_year * 100
        annual_vol = daily_std * np.sqrt(trading_days_per_year) * 100
        sharpe = annual_return / (annual_vol + 1e-8)
        
        print(f"   {col:<10} {annual_return:>13.2f}% {annual_vol:>13.2f}% {sharpe:>10.2f}")
    
    # Correlation matrix
    print(f"\n📊 CORRELATION MATRIX:")
    print(returns.corr().to_string())


def download_all_tickers_guaranteed(tickers, start_date, end_date, max_retries=3):
    """
    Download đảm bảo đủ tất cả các mã cổ phiếu bằng cách thử từng mã một từ nhiều nguồn
    
    Args:
        tickers: List các mã cổ phiếu cần download
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        max_retries: Số lần thử lại tối đa cho mỗi mã
    
    Returns:
        prices: DataFrame với tất cả các mã đã download được
        returns: DataFrame với returns
        ohlcv: Dict với OHLCV data
        successful_tickers: List các mã đã download thành công
    """
    print(f"\n🔄 Bắt đầu download đảm bảo đủ {len(tickers)} mã...")
    print(f"   Sẽ thử từng mã một từ nhiều nguồn với {max_retries} lần retry\n")
    
    all_prices = {}
    all_ohlcv = {'Open': {}, 'High': {}, 'Low': {}, 'Close': {}, 'Volume': {}}
    successful_tickers = []
    failed_tickers = []
    
    # Danh sách các nguồn để thử (theo thứ tự ưu tiên)
    sources = ['vnstock', 'yahoo', 'auto']
    
    # Download từng mã một
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Đang tải {ticker}...")
        ticker_success = False
        
        # Thử từng nguồn
        for source in sources:
            if ticker_success:
                break
                
            # Retry cho mỗi nguồn
            for retry in range(max_retries):
                try:
                    prices_single, returns_single, ohlcv_single = download_stock_data(
                        [ticker], start_date, end_date, 
                        use_cache=False, 
                        data_source=source
                    )
                    
                    # Kiểm tra dữ liệu
                    if prices_single is not None and not prices_single.empty:
                        if ticker in prices_single.columns:
                            ticker_data = prices_single[ticker].dropna()
                            if len(ticker_data) >= 50:  # Tối thiểu 50 ngày
                                all_prices[ticker] = prices_single[ticker]
                                
                                # Lưu OHLCV nếu có
                                if ohlcv_single:
                                    for key in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                        if key in ohlcv_single and ohlcv_single[key] is not None:
                                            if not ohlcv_single[key].empty and ticker in ohlcv_single[key].columns:
                                                all_ohlcv[key][ticker] = ohlcv_single[key][ticker]
                                
                                successful_tickers.append(ticker)
                                ticker_success = True
                                print(f"   ✅ {ticker}: Thành công từ {source} ({len(ticker_data)} ngày)")
                                break
                            else:
                                print(f"   ⚠️  {ticker}: {source} chỉ có {len(ticker_data)} ngày (quá ít), thử nguồn khác...")
                        else:
                            print(f"   ⚠️  {ticker}: {source} không có cột {ticker}, thử lại...")
                    else:
                        if retry < max_retries - 1:
                            print(f"   ⚠️  {ticker}: {source} lần thử {retry + 1}/{max_retries} thất bại, thử lại...")
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"   ⚠️  {ticker}: {source} lỗi (lần {retry + 1}): {str(e)[:50]}...")
                    continue
        
        if not ticker_success:
            failed_tickers.append(ticker)
            print(f"   ❌ {ticker}: Thất bại sau khi thử tất cả nguồn")
    
    # Tổng hợp dữ liệu
    if not all_prices:
        print(f"\n❌ Không download được mã nào!")
        return None, None, None, []
    
    # Tạo DataFrame từ tất cả dữ liệu
    all_dates = set()
    for series in all_prices.values():
        all_dates.update(series.index)
    all_dates = sorted(all_dates)
    
    prices = pd.DataFrame(index=all_dates)
    for ticker, series in all_prices.items():
        prices[ticker] = series
    
    # Tạo OHLCV
    ohlcv = {}
    for key in ['Open', 'High', 'Low', 'Close', 'Volume']:
        ohlcv[key] = pd.DataFrame(index=all_dates)
        for ticker, series in all_ohlcv[key].items():
            ohlcv[key][ticker] = series
    
    # Clean data
    prices = prices.dropna(how='all')
    if prices.empty:
        return None, None, None, []
    
    # Forward fill và backward fill
    prices = prices.ffill().bfill()
    for key in ohlcv:
        if not ohlcv[key].empty:
            ohlcv[key] = ohlcv[key].reindex(prices.index).ffill().bfill()
    
    # Tính returns
    returns = prices.pct_change().dropna()
    
    print(f"\n📊 Kết quả download:")
    print(f"   ✅ Thành công: {len(successful_tickers)}/{len(tickers)} mã")
    print(f"   ❌ Thất bại: {len(failed_tickers)}/{len(tickers)} mã")
    if successful_tickers:
        print(f"   Các mã thành công: {', '.join(successful_tickers)}")
    if failed_tickers:
        print(f"   Các mã thất bại: {', '.join(failed_tickers)}")
    
    return prices, returns, ohlcv, successful_tickers


def main():
    """Main function - Lấy dữ liệu 30 mã cổ phiếu trong 2 năm gần nhất"""
    print("="*70)
    print("ROBO-ADVISOR: BƯỚC 1 - LẤY DATA")
    print("="*70)
 
    
    # 30 mã cổ phiếu VN Index + chỉ số VN-Index mặc định
    TICKERS = [
        'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
        '^VNINDEX'  # Chỉ số VN-Index (có thể không có trên vnstock)
    ]
    
    # Validate danh sách mã
    print(f"\n🔍 Kiểm tra danh sách {len(TICKERS)} mã cổ phiếu...")
    validation = validate_tickers_list(TICKERS)
    
    print(f"   ✅ Mã cổ phiếu hợp lệ: {len(validation['valid'])} mã")
    if validation['index_symbols']:
        print(f"   📊 Mã chỉ số (có thể không có trên vnstock): {validation['index_symbols']}")
    if validation['potentially_invalid']:
        print(f"   ⚠️  Mã có thể không hợp lệ: {validation['potentially_invalid']}")
    
    # Lưu ý về mã chỉ số
    if validation['index_symbols']:
        print(f"\n💡 Lưu ý về mã chỉ số:")
        for idx_symbol in validation['index_symbols']:
            print(f"   - {idx_symbol}: Đây là chỉ số, không phải cổ phiếu")
            print(f"     → vnstock có thể không hỗ trợ download chỉ số")
            print(f"     → Yahoo Finance có thể hỗ trợ tốt hơn")
    
    # Sử dụng tất cả mã để thử download
    TICKERS_TO_DOWNLOAD = TICKERS.copy()
    
    # Tính 2 năm gần nhất từ thời điểm hiện tại
    from datetime import datetime, timedelta
    now = datetime.now()
    
    # End date: Hôm nay
    end_date = now.strftime('%Y-%m-%d')
    
    # Start date: 730 ngày (2 năm) trước từ hôm nay
    start_date = (now - timedelta(days=730)).strftime('%Y-%m-%d')
    
    OUTPUT_DIR = 'data'
    
    print(f"\n📅 Data Period: {start_date} to {end_date} (2 years)")
    print(f"⏰ Data Type: DAILY (1-day intervals)")
    print(f"📊 Stocks: {len(TICKERS_TO_DOWNLOAD)} stocks")
    print(f"   {', '.join(TICKERS_TO_DOWNLOAD)}")
    print(f"\n💡 Data Source: CHỈ ĐỌC TỪ CSV (Data_test.csv)")
    print(f"   KHÔNG DOWNLOAD từ nguồn online!")
    # Tính số ngày trading kỳ vọng (2 năm = ~500 trading days, ít nhất 400 ngày)
    expected_trading_days = 400  # Tối thiểu 400 ngày trading trong 2 năm
    expected_tickers = len(TICKERS_TO_DOWNLOAD)  # Phải có đủ 30 mã
    
    # CHỈ ĐỌC TỪ CSV - KHÔNG DOWNLOAD
    csv_path = Path('data') / 'Data_test.csv'
    
    if not csv_path.exists():
        print(f"\n❌ LỖI: Không tìm thấy file CSV: {csv_path}")
        print(f"   Vui lòng đảm bảo file Data_test.csv tồn tại trong thư mục data/")
        return
    
    print(f"\n📂 Loading dữ liệu từ CSV: {csv_path}")
    prices, returns, ohlcv = load_data_from_csv(str(csv_path), TICKERS_TO_DOWNLOAD, start_date, end_date)
    
    # VALIDATION: Kiểm tra đảm bảo đủ dữ liệu
    if prices is None or returns is None:
        print(f"\n❌ LỖI: Không thể load dữ liệu từ CSV!")
        print(f"   Vui lòng kiểm tra:")
        print(f"   1. File Data_test.csv có tồn tại và có dữ liệu không")
        print(f"   2. File CSV có chứa dữ liệu cho các mã cổ phiếu đã chọn không")
        print(f"   3. Khoảng thời gian yêu cầu có dữ liệu trong CSV không")
        return
    
    # Kiểm tra số mã
    available_tickers = list(prices.columns)
    missing_tickers = [t for t in TICKERS_TO_DOWNLOAD if t not in available_tickers]
    
    if len(available_tickers) < expected_tickers:
        print(f"\n⚠️  CẢNH BÁO: Chỉ có {len(available_tickers)}/{expected_tickers} mã có dữ liệu")
        if missing_tickers:
            print(f"   Các mã thiếu: {missing_tickers}")
            
            # Phân tích lý do thiếu mã
            print(f"\n🔍 Phân tích các mã thiếu:")
            for ticker in missing_tickers:
                if ticker.startswith('^'):
                    print(f"   - {ticker}: Mã chỉ số (không phải cổ phiếu)")
                    print(f"     → vnstock thường không hỗ trợ download chỉ số")
                    print(f"     → Thử Yahoo Finance với format: {ticker.replace('^', '')}.VN")
                else:
                    print(f"   - {ticker}: Có thể:")
                    print(f"     → Mã không tồn tại hoặc đã ngừng giao dịch")
                    print(f"     → Format mã khác trên vnstock (thử {ticker}.HM)")
                    print(f"     → Rate limit hoặc lỗi API")
        
        print(f"\n💡 Gợi ý để có đủ {expected_tickers} mã:")
        print(f"   1. Kiểm tra lại các mã cổ phiếu có đúng không")
        print(f"   2. Kiểm tra file Data_test.csv có chứa dữ liệu cho các mã thiếu không")
        print(f"   3. Các mã có sẵn trong CSV: {available_tickers}")
        print(f"\n   Tiếp tục với {len(available_tickers)} mã có sẵn trong CSV...")
    else:
        print(f"\n✅ Đã có đủ {len(available_tickers)}/{expected_tickers} mã!")
    
    # Kiểm tra số ngày dữ liệu
    actual_trading_days = len(prices)
    actual_date_range = (prices.index[-1] - prices.index[0]).days if len(prices) > 0 else 0
    
    if actual_trading_days < expected_trading_days:
        print(f"\n⚠️  CẢNH BÁO: Chỉ có {actual_trading_days} ngày trading (kỳ vọng: {expected_trading_days}+)")
        print(f"   Khoảng thời gian thực tế: {prices.index[0].date()} đến {prices.index[-1].date()} ({actual_date_range} ngày)")
        print(f"   Có thể thiếu dữ liệu cho một số ngày")
    else:
        print(f"\n✅ Đã có đủ {actual_trading_days} ngày trading (kỳ vọng: {expected_trading_days}+)")
        print(f"   Khoảng thời gian: {prices.index[0].date()} đến {prices.index[-1].date()}")
    
    # Kiểm tra từng mã có đủ dữ liệu không
    print(f"\n📊 Kiểm tra chi tiết từng mã:")
    tickers_with_issues = []
    for ticker in available_tickers:
        ticker_data = prices[ticker].dropna()
        ticker_days = len(ticker_data)
        ticker_pct = (ticker_days / actual_trading_days * 100) if actual_trading_days > 0 else 0
        
        if ticker_days < expected_trading_days * 0.8:  # Ít hơn 80% số ngày kỳ vọng
            print(f"   ⚠️  {ticker}: {ticker_days} ngày ({ticker_pct:.1f}%) - THIẾU DỮ LIỆU")
            tickers_with_issues.append(ticker)
        else:
            print(f"   ✅ {ticker}: {ticker_days} ngày ({ticker_pct:.1f}%)")
    
    if tickers_with_issues:
        print(f"\n⚠️  Có {len(tickers_with_issues)} mã thiếu dữ liệu: {tickers_with_issues}")
    
    # Đảm bảo thứ tự cột theo TICKERS_TO_DOWNLOAD (chỉ lấy các mã có sẵn)
    available_cols = [t for t in TICKERS_TO_DOWNLOAD if t in prices.columns]
    prices = prices[available_cols] if available_cols else prices
    returns = returns[available_cols] if available_cols else returns
    
    print_data_summary(prices, returns)
    
    save_data(prices, returns, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ BƯỚC 1 HOÀN TẤT!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  ✓ {OUTPUT_DIR}/prices.csv")
    print(f"  ✓ {OUTPUT_DIR}/returns.csv")
    print(f"  ✓ {OUTPUT_DIR}/info.txt")
    print(f"\nNext step:")
    print(f"  → Run: python Train_Model.py")
    print("="*70)


if __name__ == "__main__":
    main()
