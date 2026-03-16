"""
Data Source Module - Hỗ trợ nhiều nguồn data khác nhau
Lấy dữ liệu theo ngày (daily data) - CHỈ DÙNG DỮ LIỆU THẬT
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import vnstock (Vietnamese stock market library)
try:
    from vnstock import *
    HAS_VNSTOCK = True
except ImportError:
    HAS_VNSTOCK = False
    print("ℹ️  vnstock not installed. Install with: pip install vnstock")


def download_daily_data_yahoo(tickers, start_date, end_date):
    """
    Download daily data từ Yahoo Finance
    Thử nhiều format ticker cho mã VN (ví dụ: BCM, BCM.VN, BCM.HM)
    
    Args:
        tickers: List các mã cổ phiếu
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
    
    Returns:
        prices: DataFrame với daily prices
        returns: DataFrame với daily returns
        ohlcv: Dict với daily OHLCV data
    """
    print(f"   📥 Trying Yahoo Finance...")
    
    try:
        # Thử nhiều format ticker cho mã VN
        ticker_formats = []
        for ticker in tickers:
            ticker_clean = ticker.replace('^', '')
            # Thử các format: BCM, BCM.VN, BCM.HM
            formats_to_try = [ticker_clean, f"{ticker_clean}.VN", f"{ticker_clean}.HM"]
            ticker_formats.extend(formats_to_try)
        
        # Download daily data với tất cả formats
        data = yf.download(
            ticker_formats,
            start=start_date,
            end=end_date,
            interval='1d',  # Daily data
            progress=False,
            group_by='ticker'
        )
        
        if data.empty:
            return None, None, None
        
        # Handle single ticker vs multiple tickers
        if len(tickers) == 1:
            ticker = tickers[0]
            # Tìm format ticker đã dùng thành công
            ticker_clean = ticker.replace('^', '')
            formats_tried = [ticker_clean, f"{ticker_clean}.VN", f"{ticker_clean}.HM"]
            ticker_used = None
            
            # Kiểm tra xem format nào có dữ liệu
            if isinstance(data.columns, pd.MultiIndex):
                for fmt in formats_tried:
                    if (fmt, 'Close') in data.columns:
                        ticker_used = fmt
                        break
            else:
                for fmt in formats_tried:
                    if fmt in data.columns:
                        ticker_used = fmt
                        break
            
            if ticker_used is None:
                # Thử lấy cột đầu tiên nếu không tìm thấy
                if isinstance(data.columns, pd.MultiIndex):
                    if len(data.columns) > 0:
                        ticker_used = data.columns[0][0]
                else:
                    if len(data.columns) > 0:
                        ticker_used = data.columns[0]
            
            if ticker_used and 'Close' in str(data.columns):
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = (ticker_used, 'Close')
                    if close_col in data.columns:
                        prices = pd.DataFrame({ticker: data[close_col]})
                        ohlcv = {
                            'Open': pd.DataFrame({ticker: data[(ticker_used, 'Open')]}),
                            'High': pd.DataFrame({ticker: data[(ticker_used, 'High')]}),
                            'Low': pd.DataFrame({ticker: data[(ticker_used, 'Low')]}),
                            'Close': prices.copy(),
                            'Volume': pd.DataFrame({ticker: data[(ticker_used, 'Volume')]})
                        }
                    else:
                        return None, None, None
                else:
                    if 'Close' in data.columns:
                        prices = pd.DataFrame({ticker: data['Close']})
                        ohlcv = {
                            'Open': pd.DataFrame({ticker: data['Open']}),
                            'High': pd.DataFrame({ticker: data['High']}),
                            'Low': pd.DataFrame({ticker: data['Low']}),
                            'Close': prices.copy(),
                            'Volume': pd.DataFrame({ticker: data['Volume']})
                        }
                    else:
                        return None, None, None
            else:
                return None, None, None
        else:
            # Multiple tickers - data structure is different
            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex columns: (ticker, price_type)
                prices_dict = {}
                ohlcv_dict = {'Open': {}, 'High': {}, 'Low': {}, 'Close': {}, 'Volume': {}}
                
                for ticker in tickers:
                    ticker_clean = ticker.replace('^', '')
                    formats_tried = [ticker_clean, f"{ticker_clean}.VN", f"{ticker_clean}.HM"]
                    
                    for fmt in formats_tried:
                        if (fmt, 'Close') in data.columns:
                            prices_dict[ticker] = data[(fmt, 'Close')]
                            ohlcv_dict['Open'][ticker] = data[(fmt, 'Open')]
                            ohlcv_dict['High'][ticker] = data[(fmt, 'High')]
                            ohlcv_dict['Low'][ticker] = data[(fmt, 'Low')]
                            ohlcv_dict['Close'][ticker] = data[(fmt, 'Close')]
                            ohlcv_dict['Volume'][ticker] = data[(fmt, 'Volume')]
                            break
                
                if not prices_dict:
                    return None, None, None
                
                prices = pd.DataFrame(prices_dict)
                ohlcv = {key: pd.DataFrame(ohlcv_dict[key]) for key in ohlcv_dict}
            else:
                # Single level columns - all tickers combined
                if 'Close' in data.columns:
                    prices = data['Close'] if isinstance(data['Close'], pd.DataFrame) else pd.DataFrame(data['Close'])
                    ohlcv = {
                        'Open': data['Open'] if isinstance(data['Open'], pd.DataFrame) else pd.DataFrame(data['Open']),
                        'High': data['High'] if isinstance(data['High'], pd.DataFrame) else pd.DataFrame(data['High']),
                        'Low': data['Low'] if isinstance(data['Low'], pd.DataFrame) else pd.DataFrame(data['Low']),
                        'Close': prices.copy(),
                        'Volume': data['Volume'] if isinstance(data['Volume'], pd.DataFrame) else pd.DataFrame(data['Volume'])
                    }
                else:
                    return None, None, None
        
        # Clean data
        prices = prices.dropna()
        if prices.empty:
            return None, None, None
        
        for key in ohlcv:
            ohlcv[key] = ohlcv[key].dropna()
            ohlcv[key] = ohlcv[key].reindex(prices.index).ffill().bfill()
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(prices) < 10:  # Too few data points
            return None, None, None
        
        print(f"   ✅ Yahoo Finance: {len(prices)} days for {len(prices.columns)} tickers")
        return prices, returns, ohlcv
        
    except Exception as e:
        print(f"   ❌ Yahoo Finance failed: {str(e)[:100]}")
        return None, None, None


def download_daily_data_vnstock(tickers, start_date, end_date):
    """
    Download daily data từ vnstock (Vietnamese stock market)
    
    Args:
        tickers: List các mã cổ phiếu VN (ví dụ: 'ACB', 'VCB', etc.)
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
    
    Returns:
        prices: DataFrame với daily prices
        returns: DataFrame với daily returns
        ohlcv: Dict với daily OHLCV data
    """
    if not HAS_VNSTOCK:
        return None, None, None
    
    print(f"   📥 Trying vnstock...")
    
    try:
        prices_dict = {}
        ohlcv_dict = {'Open': {}, 'High': {}, 'Low': {}, 'Close': {}, 'Volume': {}}
        
        for ticker in tickers:
            try:
                # vnstock format: ticker needs to be uppercase, remove ^ if present
                ticker_clean = ticker.upper().replace('^', '')
                
                # Try different vnstock API methods
                df = None
                
                # Method 1: stock_historical_data (newer API)
                try:
                    df = stock_historical_data(
                        symbol=ticker_clean,
                        start_date=start_date,
                        end_date=end_date,
                        resolution='1D',
                        type='stock'
                    )
                except:
                    pass
                
                # Method 2: If method 1 fails, try alternative
                if df is None or df.empty:
                    try:
                        # Alternative: use different function if available
                        from vnstock import listing_companies
                        # Try to get data using different approach
                        pass  # Placeholder for alternative method
                    except:
                        pass
                
                if df is not None and not df.empty:
                    # Check column names (vnstock may use different naming)
                    cols_lower = [c.lower() for c in df.columns]
                    
                    # Find close price column
                    close_col = None
                    for col in df.columns:
                        if 'close' in col.lower() or 'gia_dong_cua' in col.lower():
                            close_col = col
                            break
                    
                    if close_col is not None:
                        # Set index to date if available
                        date_col = None
                        for col in df.columns:
                            if 'date' in col.lower() or 'time' in col.lower() or 'ngay' in col.lower():
                                date_col = col
                                break
                        
                        if date_col:
                            df = df.set_index(date_col)
                            df.index = pd.to_datetime(df.index)
                        
                        prices_dict[ticker] = df[close_col]
                        
                        # Find other OHLCV columns
                        open_col = next((c for c in df.columns if 'open' in c.lower() or 'gia_mo_cua' in c.lower()), close_col)
                        high_col = next((c for c in df.columns if 'high' in c.lower() or 'gia_cao_nhat' in c.lower()), close_col)
                        low_col = next((c for c in df.columns if 'low' in c.lower() or 'gia_thap_nhat' in c.lower()), close_col)
                        volume_col = next((c for c in df.columns if 'volume' in c.lower() or 'khoi_luong' in c.lower()), None)
                        
                        ohlcv_dict['Open'][ticker] = df[open_col] if open_col else df[close_col]
                        ohlcv_dict['High'][ticker] = df[high_col] if high_col else df[close_col]
                        ohlcv_dict['Low'][ticker] = df[low_col] if low_col else df[close_col]
                        ohlcv_dict['Close'][ticker] = df[close_col]
                        ohlcv_dict['Volume'][ticker] = df[volume_col] if volume_col is not None else pd.Series(0, index=df.index)
                    
            except Exception as e:
                print(f"      ⚠️  {ticker}: {str(e)[:50]}")
                continue
        
        if not prices_dict:
            return None, None, None
        
        # Convert to DataFrames and align dates
        all_dates = set()
        for series in prices_dict.values():
            all_dates.update(series.index)
        all_dates = sorted(all_dates)
        
        prices = pd.DataFrame(index=all_dates)
        for ticker, series in prices_dict.items():
            prices[ticker] = series
        
        ohlcv = {}
        for key in ohlcv_dict:
            ohlcv[key] = pd.DataFrame(index=all_dates)
            for ticker, series in ohlcv_dict[key].items():
                ohlcv[key][ticker] = series
        
        # Clean data
        prices = prices.dropna(how='all')  # Remove rows where all are NaN
        if prices.empty:
            return None, None, None
        
        # Forward fill and backward fill
        prices = prices.ffill().bfill()
        for key in ohlcv:
            ohlcv[key] = ohlcv[key].reindex(prices.index).ffill().bfill()
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(prices) < 10:
            return None, None, None
        
        print(f"   ✅ vnstock: {len(prices)} days for {len(prices.columns)} tickers")
        return prices, returns, ohlcv
        
    except Exception as e:
        print(f"   ❌ vnstock failed: {str(e)[:100]}")
        return None, None, None


def download_daily_data_multi_source(tickers, start_date, end_date, sources=None):
    """
    Download daily data từ nhiều nguồn, thử TỪNG MÃ MỘT từ nhiều nguồn
    
    Args:
        tickers: List các mã cổ phiếu
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        sources: List các nguồn để thử ['yahoo', 'vnstock', ...]
    
    Returns:
        prices: DataFrame với daily prices (chỉ các mã có dữ liệu)
        returns: DataFrame với daily returns
        ohlcv: Dict với daily OHLCV data
        successful_tickers: List các mã đã download thành công
    """
    if sources is None:
        sources = ['vnstock', 'yahoo']  # Thử vnstock trước cho thị trường VN
    
    print(f"\n📥 Downloading DAILY data from multiple sources (trying each ticker individually)...")
    print(f"   Tickers: {tickers}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Sources to try: {sources}")
    
    all_prices = {}
    all_ohlcv = {'Open': {}, 'High': {}, 'Low': {}, 'Close': {}, 'Volume': {}}
    successful_tickers = set()
    
    # Thử TỪNG MÃ MỘT từ nhiều nguồn
    for ticker in tickers:
        if ticker in successful_tickers:
            continue  # Đã có dữ liệu cho mã này
        
        ticker_found = False
        
        # Thử từng nguồn cho mã này
        for source in sources:
            try:
                if source == 'yahoo':
                    prices_single, returns_single, ohlcv_single = download_daily_data_yahoo(
                        [ticker], start_date, end_date
                    )
                elif source == 'vnstock':
                    prices_single, returns_single, ohlcv_single = download_daily_data_vnstock(
                        [ticker], start_date, end_date
                    )
                else:
                    continue
                
                # Kiểm tra xem có dữ liệu không và đủ số ngày không
                if prices_single is not None and not prices_single.empty and ticker in prices_single.columns:
                    ticker_data = prices_single[ticker].dropna()
                    ticker_days = len(ticker_data)
                    
                    # Tính số ngày kỳ vọng (2 năm = ~500 trading days, tối thiểu 300 ngày)
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    expected_days = int((end_dt - start_dt).days * 0.6)  # Ít nhất 60% số ngày
                    
                    if ticker_days >= max(50, expected_days):  # Tối thiểu 50 ngày hoặc 60% số ngày kỳ vọng
                        # Có dữ liệu đủ từ nguồn này
                        all_prices[ticker] = prices_single[ticker]
                        if ohlcv_single:
                            for key in ['Open', 'High', 'Low', 'Close', 'Volume']:
                                if key in ohlcv_single and ohlcv_single[key] is not None:
                                    if not ohlcv_single[key].empty and ticker in ohlcv_single[key].columns:
                                        all_ohlcv[key][ticker] = ohlcv_single[key][ticker]
                        successful_tickers.add(ticker)
                        ticker_found = True
                        print(f"   ✅ {ticker}: Found in {source} ({ticker_days} days)")
                        break  # Đã tìm thấy, không cần thử nguồn khác
                    else:
                        print(f"   ⚠️  {ticker}: Found in {source} but only {ticker_days} days (expected {expected_days}+), trying other sources...")
                        # Tiếp tục thử nguồn khác
            except Exception as e:
                # Thử nguồn tiếp theo
                continue
        
        if not ticker_found:
            print(f"   ❌ {ticker}: Not found in any source")
    
    if not all_prices:
        print(f"❌ Could not download data from any source")
        return None, None, None, []
    
    # Combine all data
    # Align all dates
    all_dates = set()
    for series in all_prices.values():
        all_dates.update(series.index)
    all_dates = sorted(all_dates)
    
    prices = pd.DataFrame(index=all_dates)
    for ticker, series in all_prices.items():
        prices[ticker] = series
    
    ohlcv = {}
    for key in ['Open', 'High', 'Low', 'Close', 'Volume']:
        ohlcv[key] = pd.DataFrame(index=all_dates)
        for ticker, series in all_ohlcv[key].items():
            ohlcv[key][ticker] = series
    
    # Clean data
    prices = prices.dropna(how='all')
    if prices.empty:
        return None, None, None, []
    
    # Forward fill and backward fill
    prices = prices.ffill().bfill()
    for key in ohlcv:
        if not ohlcv[key].empty:
            ohlcv[key] = ohlcv[key].reindex(prices.index).ffill().bfill()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    if len(prices) < 10:
        return None, None, None, []
    
    # Kiểm tra số ngày dữ liệu cho từng mã
    print(f"\n📊 Validation dữ liệu:")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    expected_days = int((end_dt - start_dt).days * 0.6)  # Ít nhất 60% số ngày
    
    tickers_with_insufficient_data = []
    for ticker in successful_tickers:
        ticker_data = prices[ticker].dropna()
        ticker_days = len(ticker_data)
        if ticker_days < expected_days:
            tickers_with_insufficient_data.append((ticker, ticker_days, expected_days))
    
    print(f"✅ Successfully downloaded {len(successful_tickers)}/{len(tickers)} tickers")
    print(f"   Successful: {sorted(successful_tickers)}")
    if len(successful_tickers) < len(tickers):
        missing = [t for t in tickers if t not in successful_tickers]
        print(f"   Missing: {missing}")
    if tickers_with_insufficient_data:
        print(f"   ⚠️  Mã có ít dữ liệu:")
        for ticker, actual, expected in tickers_with_insufficient_data:
            print(f"      {ticker}: {actual} days (expected {expected}+)")
    
    return prices, returns, ohlcv, list(successful_tickers)


def download_real_daily_data(tickers, start_date, end_date, data_source='auto'):
    """
    Download real daily data từ nhiều nguồn khác nhau
    CHỈ TRẢ VỀ DỮ LIỆU THẬT, KHÔNG DÙNG SYNTHETIC
    
    Args:
        tickers: List các mã cổ phiếu
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        data_source: 'auto' (thử nhiều nguồn), 'yahoo', 'vnstock', etc.
    
    Returns:
        prices: DataFrame với daily prices
        returns: DataFrame với daily returns
        ohlcv: Dict với daily OHLCV
    """
    if data_source == 'auto':
        # Try multiple sources
        prices, returns, ohlcv, successful = download_daily_data_multi_source(
            tickers, start_date, end_date, 
            sources=['yahoo', 'vnstock']
        )
        return prices, returns, ohlcv
    
    elif data_source == 'yahoo':
        prices, returns, ohlcv = download_daily_data_yahoo(tickers, start_date, end_date)
        return prices, returns, ohlcv
    
    elif data_source == 'vnstock':
        prices, returns, ohlcv = download_daily_data_vnstock(tickers, start_date, end_date)
        return prices, returns, ohlcv
    
    else:
        print(f"❌ Unknown data source: {data_source}")
        return None, None, None
