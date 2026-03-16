"""
Daily Data Fetcher - Tự động lấy data realtime và lưu vào database

Chạy hàng ngày để:
1. Lấy data realtime từ API
2. Lưu vào database
3. Trigger retrain nếu cần
"""

import requests
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time
import pandas as pd
import numpy as np

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False
    print("⚠️  pymongo not installed. Install with: pip install pymongo")

# SQLite imports (fallback)
try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

# Configuration
BASE_URL = "http://localhost:5001"  # URL của Flask app
USE_MONGODB = True  # Set False để dùng SQLite
MONGODB_URI = "mongodb://localhost:27017/"  # MongoDB connection string
MONGODB_DB_NAME = "stock_data"
MONGODB_COLLECTION_NAME = "daily_stock_data"
SQLITE_DB_PATH = Path('data') / 'realtime_data.db'
REALTIME_API_URL = None  # URL API realtime của bạn (sẽ được set)

# Global flag để track database type
_db_initialized = False
_use_mongodb = USE_MONGODB

# Import Get_data module để tái sử dụng (chủ yếu cho fallback CSV)
try:
    from Get_data import download_stock_data, load_data_from_csv
except ImportError:
    download_stock_data = None
    load_data_from_csv = None

# Import các hàm lấy dữ liệu từ VNDirect / TCBS trong Script.py
# → Đây là "cách lấy data" bạn đã dùng để tạo vn_stocks_data_2020_2025.csv
try:
    from Script import get_vn_stock_data_vndirect, get_vn_stock_data_ssi
    HAS_VNDIRECT_SSI = True
except ImportError:
    HAS_VNDIRECT_SSI = False

# Import vnstock (API thay thế)
try:
    from vnstock import stock_historical_data
    HAS_VNSTOCK = True
except ImportError:
    HAS_VNSTOCK = False

# Import yfinance (backup API)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def get_mongodb_client():
    """Get MongoDB client connection"""
    if not HAS_MONGODB:
        raise ImportError("pymongo not installed. Install with: pip install pymongo")
    
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        return client
    except ConnectionFailure:
        raise ConnectionError(f"Cannot connect to MongoDB at {MONGODB_URI}")


def init_database():
    """Initialize MongoDB database hoặc SQLite database"""
    global _use_mongodb
    
    if _use_mongodb and HAS_MONGODB:
        # Initialize MongoDB
        try:
            client = get_mongodb_client()
            db = client[MONGODB_DB_NAME]
            collection = db[MONGODB_COLLECTION_NAME]
            
            # Create unique index on (date, ticker)
            collection.create_index([("date", 1), ("ticker", 1)], unique=True)
            
            # Create index on date for faster queries
            collection.create_index([("date", -1)])
            
            print(f"✅ MongoDB initialized: {MONGODB_DB_NAME}.{MONGODB_COLLECTION_NAME}")
            client.close()
            return True
        except Exception as e:
            print(f"❌ MongoDB initialization error: {e}")
            if HAS_SQLITE:
                print("⚠️  Falling back to SQLite...")
                _use_mongodb = False
            else:
                raise
    
    # Fallback to SQLite
    if HAS_SQLITE:
        Path('data').mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                returns REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, ticker)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ SQLite initialized: {SQLITE_DB_PATH}")
        return True
    
    raise RuntimeError("Neither MongoDB nor SQLite available")


def get_data_from_vnstock(ticker, start_date, end_date):
    """
    Lấy data từ vnstock API
    """
    if not HAS_VNSTOCK:
        return None
    
    try:
        df = stock_historical_data(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            resolution='1D',
            type='stock'
        )
        
        if df is None or df.empty:
            return None
        
        # Reset index nếu cần
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Tìm các cột OHLCV
        cols_lower = {col.lower(): col for col in df.columns}
        date_col = next((cols_lower[k] for k in ['date', 'time', 'ngay', 'timestamp'] if k in cols_lower), None)
        open_col = next((cols_lower[k] for k in ['open', 'gia_mo_cua', 'o'] if k in cols_lower), None)
        high_col = next((cols_lower[k] for k in ['high', 'gia_cao_nhat', 'h'] if k in cols_lower), None)
        low_col = next((cols_lower[k] for k in ['low', 'gia_thap_nhat', 'l'] if k in cols_lower), None)
        close_col = next((cols_lower[k] for k in ['close', 'gia_dong_cua', 'c', 'price'] if k in cols_lower), None)
        volume_col = next((cols_lower[k] for k in ['volume', 'khoi_luong', 'vol', 'kl'] if k in cols_lower), None)
        
        if close_col is None:
            return None
        
        # Tạo Date column và remove timezone nếu có
        if date_col:
            dates = pd.to_datetime(df[date_col])
        else:
            dates = pd.to_datetime(df.index)
        
        # Remove timezone để tránh lỗi so sánh
        if hasattr(dates, 'dt') and dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        elif isinstance(dates, pd.DatetimeIndex) and dates.tz is not None:
            dates = dates.tz_localize(None)
        
        result_df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Open': df[open_col].values if open_col else df[close_col].values,
            'High': df[high_col].values if high_col else df[close_col].values,
            'Low': df[low_col].values if low_col else df[close_col].values,
            'Close': df[close_col].values,
            'Volume': df[volume_col].values if volume_col else 0
        })
        
        result_df['Log_Returns'] = np.log(result_df['Close'] / result_df['Close'].shift(1))
        result_df = result_df.sort_values('Date')
        return result_df
        
    except Exception:
        return None


def get_data_from_yfinance(ticker, start_date, end_date):
    """
    Lấy data từ yfinance (backup, có thể không có data VN)
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        # yfinance dùng format ticker.VN cho VN stocks
        symbol = f"{ticker}.VN"
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            return None
        
        # Convert index to datetime và remove timezone nếu có
        dates = pd.to_datetime(df.index)
        if hasattr(dates, 'dt') and dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        elif isinstance(dates, pd.DatetimeIndex) and dates.tz is not None:
            dates = dates.tz_localize(None)
        
        result_df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Open': df['Open'].values,
            'High': df['High'].values,
            'Low': df['Low'].values,
            'Close': df['Close'].values,
            'Volume': df['Volume'].values
        })
        
        result_df['Log_Returns'] = np.log(result_df['Close'] / result_df['Close'].shift(1))
        return result_df
        
    except Exception:
        return None


def fetch_realtime_data_from_api(tickers, date=None, days_back=30):
    """
    Lấy realtime data 30 NGÀY GẦN NHẤT theo **cách của Script.py**:
    - Ưu tiên VNDirect
    - Nếu lỗi thì fallback sang TCBS (SSI)
    - Merge với data hiện tại từ CSV
    
    Args:
        tickers: List of stock tickers
        date: Date string (YYYY-MM-DD), default: today
        days_back: Số ngày gần nhất cần lấy (default: 30)
    
    Returns:
        DataFrame với columns: Date, Ticker, Open, High, Low, Close, Volume, Returns
        (Format giống vn_stocks_data_2020_2025.csv)
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Nếu bạn có API realtime riêng thì ưu tiên dùng
    if REALTIME_API_URL:
        try:
            response = requests.get(
                f"{REALTIME_API_URL}/api/stocks/daily",
                params={"tickers": ",".join(tickers), "date": date},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️  API realtime error: {e}")
    
    # Cách lấy dữ liệu chính: Thử nhiều API theo thứ tự ưu tiên
    # VNDirect → SSI (TCBS) → vnstock → yfinance → CSV fallback
    if not HAS_VNDIRECT_SSI and not HAS_VNSTOCK and not HAS_YFINANCE:
        print("⚠️  Không có API nào available. Cần cài đặt:")
        print("   - Script.py (VNDirect/SSI) hoặc")
        print("   - pip install vnstock hoặc")
        print("   - pip install yfinance")
        return pd.DataFrame()
    
    print(f"📥 Đang lấy {days_back} NGÀY GẦN NHẤT từ VNDirect/TCBS cho {len(tickers)} mã...")
    
    target_date = datetime.strptime(date, '%Y-%m-%d')
    start_date_short = (target_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date_short = date
    
    # Lưu tất cả data 30 ngày (không chỉ 1 ngày)
    all_daily_data = []
    
    for i, ticker in enumerate(tickers, 1):
        # Bỏ qua chỉ số
        if ticker.startswith('^'):
            continue
        
        print(f"   [{i}/{len(tickers)}] {ticker}:", end=" ", flush=True)
        
        # Logic giống Script.py: Thử VNDirect trước, nếu fail thì thử TCBS (SSI)
        df = None
        vndirect_success = False
        
        # Bước 1: Thử VNDirect trước với retry logic
        # Lưu ý: Script.py sẽ tự catch exception và return None, không throw exception
        max_retries = 2  # Giảm retry để chuyển sang SSI nhanh hơn
        for retry in range(max_retries):
            df = get_vn_stock_data_vndirect(ticker, start_date_short, end_date_short)
            if df is not None and len(df) > 0:
                vndirect_success = True
                break  # Thành công, thoát retry loop
            else:
                # VNDirect fail (timeout hoặc lỗi), Script.py đã print error message
                # Nhưng chúng ta cần check xem có cần retry không
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # 2s, 4s
                    print(f"⏳ retry {retry+1}/{max_retries}...", end=" ", flush=True)
                    time.sleep(wait_time)
        
        # Bước 2: Nếu VNDirect thất bại, thử các API khác theo thứ tự: SSI → vnstock → yfinance
        if not vndirect_success:
            # Thử SSI (TCBS)
            sys.stdout.flush()
            print("→SSI", end=" ", flush=True)
            sys.stdout.flush()
            
            ssi_success = False
            for retry in range(max_retries):
                df = get_vn_stock_data_ssi(ticker, start_date_short, end_date_short)
                if df is not None and len(df) > 0:
                    ssi_success = True
                    break
                elif retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    print(f"⏳ retry {retry+1}/{max_retries}...", end=" ", flush=True)
                    sys.stdout.flush()
                    time.sleep(wait_time)
            
            # Nếu SSI cũng fail, thử vnstock
            if not ssi_success and HAS_VNSTOCK:
                print("→vnstock", end=" ", flush=True)
                sys.stdout.flush()
                df = get_data_from_vnstock(ticker, start_date_short, end_date_short)
                if df is not None and len(df) > 0:
                    ssi_success = True  # Dùng flag này để đánh dấu thành công
            
            # Nếu vnstock cũng fail, thử yfinance (backup)
            if not ssi_success and HAS_YFINANCE:
                print("→yfinance", end=" ", flush=True)
                sys.stdout.flush()
                df = get_data_from_yfinance(ticker, start_date_short, end_date_short)
                if df is not None and len(df) > 0:
                    ssi_success = True
        
        if df is None or len(df) == 0:
            print("❌ (tất cả API đều fail)")
            if i < len(tickers):
                time.sleep(1)
            continue
        
        # Chuẩn hóa Date và lấy TẤT CẢ các ngày trong khoảng
        df['Date'] = pd.to_datetime(df['Date'])
        # Remove timezone nếu có để tránh lỗi so sánh
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        # Convert target_date thành pandas Timestamp để so sánh
        target_date_ts = pd.Timestamp(target_date)
        df = df[df['Date'] <= target_date_ts].copy()
        
        if len(df) == 0:
            print("⚠️  Không có data")
            continue
        
        # Thêm tất cả các ngày vào all_daily_data
        for _, row in df.iterrows():
            returns_value = 0.0
            if 'Log_Returns' in df.columns and not pd.isna(row['Log_Returns']):
                returns_value = float(row['Log_Returns'])
                if not np.isfinite(returns_value):
                    returns_value = 0.0
            else:
                # Tính returns từ ngày trước
                prev_rows = df[df['Date'] < row['Date']]
                if len(prev_rows) > 0:
                    prev_close = prev_rows.iloc[-1]['Close']
                    if prev_close and prev_close > 0:
                        returns_value = float((row['Close'] - prev_close) / prev_close)
            
            all_daily_data.append({
                "Date": row['Date'],
                "Ticker": ticker,
                "Open": float(row['Open']),
                "High": float(row['High']),
                "Low": float(row['Low']),
                "Close": float(row['Close']),
                "Volume": int(row['Volume']) if pd.notna(row['Volume']) else 0,
                "Returns": float(returns_value)
            })
        
        print(f"✅ ({len(df)} ngày)")
        
        # Delay để tránh rate limit
        if i < len(tickers):
            time.sleep(2)
    
    if all_daily_data:
        # Convert sang DataFrame (format giống vn_stocks_data_2020_2025.csv)
        new_df = pd.DataFrame(all_daily_data)
        new_df = new_df.sort_values(['Ticker', 'Date'])
        print(f"✅ Đã lấy được {len(new_df)} records từ {len(set(new_df['Ticker']))} mã")
        return new_df
    
    # Fallback: nếu không lấy được từ API, thử đọc từ CSV hiện tại
    print("⚠️  Không lấy được data từ VNDirect/TCBS, thử đọc từ CSV...")
    csv_path = Path('vn_stocks_data_2020_2025.csv')
    if csv_path.exists() and load_data_from_csv:
        try:
            # Lấy 30 ngày gần nhất từ CSV
            target_date_dt = datetime.strptime(date, '%Y-%m-%d')
            start_date_csv = (target_date_dt - timedelta(days=30)).strftime('%Y-%m-%d')
            
            prices, returns_df, ohlcv = load_data_from_csv(
                csv_path=str(csv_path),
                tickers=tickers,
                start_date=start_date_csv,
                end_date=date
            )
            
            if prices is not None and not prices.empty:
                csv_result = []
                for ticker in tickers:
                    if ticker in prices.columns:
                        # Lấy tất cả các ngày trong khoảng
                        ticker_prices = prices[ticker].dropna()
                        ticker_prices = ticker_prices[ticker_prices.index <= target_date_dt]
                        
                        for date_idx in ticker_prices.index:
                            close_price = ticker_prices.loc[date_idx]
                            
                            open_price = ohlcv.get('Open', pd.DataFrame()).loc[date_idx, ticker] if 'Open' in ohlcv and ticker in ohlcv['Open'].columns else close_price
                            high_price = ohlcv.get('High', pd.DataFrame()).loc[date_idx, ticker] if 'High' in ohlcv and ticker in ohlcv['High'].columns else close_price
                            low_price = ohlcv.get('Low', pd.DataFrame()).loc[date_idx, ticker] if 'Low' in ohlcv and ticker in ohlcv['Low'].columns else close_price
                            volume = ohlcv.get('Volume', pd.DataFrame()).loc[date_idx, ticker] if 'Volume' in ohlcv and ticker in ohlcv['Volume'].columns else 0
                            
                            # Tính returns
                            prev_dates = ticker_prices.index[ticker_prices.index < date_idx]
                            if len(prev_dates) > 0:
                                prev_close = ticker_prices.loc[prev_dates[-1]]
                                stock_return = (close_price - prev_close) / prev_close if prev_close > 0 else 0.0
                            else:
                                stock_return = 0.0
                            
                            csv_result.append({
                                "Date": date_idx,
                                "Ticker": ticker,
                                "Open": float(open_price),
                                "High": float(high_price),
                                "Low": float(low_price),
                                "Close": float(close_price),
                                "Volume": int(volume) if pd.notna(volume) else 0,
                                "Returns": float(stock_return)
                            })
                
                if csv_result:
                    csv_df = pd.DataFrame(csv_result)
                    csv_df = csv_df.sort_values(['Ticker', 'Date'])
                    print(f"✅ Đã lấy {len(csv_df)} records từ CSV (30 ngày gần nhất)")
                    return csv_df
        except Exception as e:
            print(f"⚠️  Lỗi khi đọc CSV: {e}")
    
    print("❌ Không thể lấy data từ bất kỳ nguồn nào (API và CSV đều fail)")
    return pd.DataFrame()


def save_data_to_db(data):
    """Lưu data vào MongoDB hoặc SQLite"""
    global _use_mongodb
    
    if _use_mongodb and HAS_MONGODB:
        return save_data_to_mongodb(data)
    elif HAS_SQLITE:
        return save_data_to_sqlite(data)
    else:
        raise RuntimeError("No database available")


def save_data_to_mongodb(data):
    """Lưu data vào MongoDB"""
    client = get_mongodb_client()
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION_NAME]
    
    saved_count = 0
    for record in data:
        try:
            # Prepare document
            document = {
                "date": record['date'],
                "ticker": record['ticker'],
                "open": record.get('open'),
                "high": record.get('high'),
                "low": record.get('low'),
                "close": record.get('close'),
                "volume": record.get('volume'),
                "returns": record.get('returns'),
                "created_at": datetime.now()
            }
            
            # Upsert (insert or update)
            collection.update_one(
                {"date": record['date'], "ticker": record['ticker']},
                {"$set": document},
                upsert=True
            )
            saved_count += 1
        except DuplicateKeyError:
            # Should not happen due to upsert, but handle anyway
            print(f"⚠️  Duplicate key for {record.get('ticker')} on {record.get('date')}")
            saved_count += 1
        except Exception as e:
            print(f"❌ Error saving {record.get('ticker')} on {record.get('date')}: {e}")
    
    client.close()
    return saved_count


def save_data_to_sqlite(data):
    """Lưu data vào SQLite (fallback)"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    saved_count = 0
    for record in data:
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO daily_stock_data 
                (date, ticker, open, high, low, close, volume, returns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['date'],
                record['ticker'],
                record.get('open'),
                record.get('high'),
                record.get('low'),
                record.get('close'),
                record.get('volume'),
                record.get('returns')
            ))
            saved_count += 1
        except Exception as e:
            print(f"❌ Error saving {record.get('ticker')} on {record.get('date')}: {e}")
    
    conn.commit()
    conn.close()
    
    return saved_count


def send_data_to_api(data):
    """Gửi data đến Flask API để lưu vào DB"""
    try:
        url = f"{BASE_URL}/api/realtime_data"
        response = requests.post(url, json={"data": data}, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('saved_count', 0), None
        else:
            return 0, f"API error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return 0, "Cannot connect to Flask API. Is app.py running?"
    except Exception as e:
        return 0, str(e)


def trigger_retrain(force=False):
    """Trigger retrain model"""
    try:
        url = f"{BASE_URL}/api/retrain"
        response = requests.post(url, json={"force": force}, timeout=10)
        
        if response.status_code == 200:
            return True, None
        else:
            return False, f"Retrain error: {response.status_code}"
    except Exception as e:
        return False, str(e)


def check_retrain_status():
    """Kiểm tra trạng thái retrain"""
    try:
        url = f"{BASE_URL}/api/retrain_status"
        response = requests.get(url, timeout=5)
        return response.json()
    except Exception as e:
        return None


def merge_with_existing_csv(new_data_df, csv_path='vn_stocks_data_2020_2025.csv'):
    """
    Merge data mới (30 ngày gần nhất) với data hiện tại trong CSV
    
    Args:
        new_data_df: DataFrame mới với columns: Date, Ticker, Open, High, Low, Close, Volume, Returns
        csv_path: Đường dẫn đến file CSV hiện tại
    
    Returns:
        DataFrame đã merge và sort
    """
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"⚠️  File {csv_path} không tồn tại. Tạo file mới...")
        return new_data_df
    
    try:
        # Load CSV hiện tại
        existing_df = pd.read_csv(csv_file)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        print(f"📂 Đã load {len(existing_df)} records từ {csv_path}")
        
        # Merge: loại bỏ duplicate (ưu tiên data mới)
        # Tạo key để identify duplicate: (Date, Ticker)
        existing_df['_key'] = existing_df['Date'].astype(str) + '_' + existing_df['Ticker']
        new_data_df['_key'] = new_data_df['Date'].astype(str) + '_' + new_data_df['Ticker']
        
        # Loại bỏ các record trong existing_df có key trùng với new_data_df
        existing_df = existing_df[~existing_df['_key'].isin(new_data_df['_key'])]
        
        # Merge
        merged_df = pd.concat([existing_df.drop('_key', axis=1), new_data_df.drop('_key', axis=1)], ignore_index=True)
        merged_df = merged_df.sort_values(['Ticker', 'Date'])
        
        # Đảm bảo thứ tự cột
        columns_order = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Returns' in merged_df.columns:
            columns_order.append('Returns')
        merged_df = merged_df[columns_order]
        
        print(f"✅ Đã merge: {len(existing_df)} records cũ + {len(new_data_df)} records mới = {len(merged_df)} records")
        
        return merged_df
        
    except Exception as e:
        print(f"⚠️  Lỗi khi merge CSV: {e}")
        import traceback
        traceback.print_exc()
        return new_data_df


def daily_workflow(tickers=None, auto_retrain=True, retrain_after_days=1, update_csv=True, use_yesterday=True, scrape_news=True):
    """
    Workflow hàng ngày: Lấy 30 ngày gần nhất → Merge với CSV → Lưu DB → Scrape News → Retrain (nếu cần)
    
    Args:
        tickers: List of tickers to fetch (None = use all available)
        auto_retrain: Tự động retrain sau khi lưu data
        retrain_after_days: Retrain sau bao nhiêu ngày có data mới
        update_csv: Có cập nhật file CSV không
        scrape_news: Có scrape news không
    """
    print("="*70)
    print(f"DAILY DATA FETCHER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize database
    init_database()
    
    # Default tickers
    if tickers is None:
        tickers = [
            'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
            'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
            'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
        ]
    
    # Xác định ngày target: ngày hôm trước (D-1) để chạy lúc 6h30 sáng
    if use_yesterday:
        target_dt = datetime.now() - timedelta(days=1)
    else:
        target_dt = datetime.now()
    target_date_str = target_dt.strftime('%Y-%m-%d')
    
    # Step 1: Fetch 30 ngày gần nhất (kết thúc tại ngày hôm trước)
    print(f"\n📊 Step 1: Fetching 30 ngày gần nhất (tới {target_date_str}) cho {len(tickers)} tickers...")
    
    new_data_df = fetch_realtime_data_from_api(tickers, target_date_str, days_back=30)
    
    if new_data_df.empty:
        print("⚠️  No data fetched. Check your API implementation.")
        return False
    
    print(f"✅ Fetched {len(new_data_df)} records (30 ngày gần nhất)")
    
    # Step 2: Merge với CSV hiện tại
    if update_csv:
        print(f"\n🔄 Step 2: Merging với CSV hiện tại...")
        merged_df = merge_with_existing_csv(new_data_df, 'vn_stocks_data_2020_2025.csv')
        
        # Lưu lại CSV
        output_file = 'vn_stocks_data_2020_2025.csv'
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã cập nhật {output_file} với {len(merged_df)} records")
    
    # Step 3: Convert sang format để lưu DB (chỉ lấy NGÀY HÔM TRƯỚC)
    target_date_dt = pd.to_datetime(target_date_str)
    today_data = new_data_df[new_data_df['Date'] == target_date_dt]
    if today_data.empty:
        # Lấy ngày gần nhất trước đó
        today_data = new_data_df[new_data_df['Date'] == new_data_df['Date'].max()]
    
    stock_data = []
    for _, row in today_data.iterrows():
        stock_data.append({
            "ticker": row['Ticker'],
            "date": row['Date'].strftime('%Y-%m-%d'),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close']),
            "volume": int(row['Volume']),
            "returns": float(row.get('Returns', 0.0))
        })
    
    # Step 4: Save to database via API (không block nếu fail)
    print("\n💾 Step 3: Saving data to database...")
    saved_count, error = send_data_to_api(stock_data)
    
    if error:
        print(f"⚠️  Warning: {error}")
        print("   → Tiếp tục workflow (news scraping và retrain vẫn chạy)")
        # Không return False, tiếp tục workflow
    else:
        print(f"✅ Saved {saved_count} records to database")
    
    # Step 5: Scrape news để phân tích độ ảnh hưởng (theo ngày cụ thể)
    if scrape_news:
        print(f"\n📰 Step 4: Scraping news để phân tích độ ảnh hưởng (ngày {target_date_str})...")
        try:
            from news_scraper import scrape_news_for_tickers, save_news_to_database
            
            # Scrape news cho ngày cụ thể (ngày hôm trước)
            all_news = scrape_news_for_tickers(tickers, target_date=target_date_str, days_back=7)
            
            # Lưu vào database
            for ticker, news_list in all_news.items():
                if news_list:
                    save_news_to_database(news_list, ticker)
            
            total_news = sum(len(news_list) for news_list in all_news.values())
            print(f"✅ Đã scrape {total_news} tin tức cho {len([t for t in all_news.keys() if all_news[t]])} mã")
            
        except ImportError:
            print("⚠️  news_scraper module chưa được cài đặt. Bỏ qua bước scrape news.")
        except Exception as e:
            print(f"⚠️  Lỗi khi scrape news: {e}")
    
    # Step 6: Trigger retrain sau khi có data mới (để model học tiếp và tăng độ chính xác)
    if auto_retrain and update_csv:
        print("\n🔄 Step 5: Triggering retrain với data mới để model học tiếp...")
        status = check_retrain_status()
        
        if status and not status.get('is_training'):
            # Luôn trigger retrain sau khi có data mới để model học tiếp (fine-tuning)
            print("📚 Đang trigger retrain để model học tiếp với data mới (fine-tuning)...")
            success, error = trigger_retrain(force=False)
            if success:
                print("✅ Retrain triggered successfully - Model sẽ học tiếp với data mới để tăng độ chính xác")
            else:
                print(f"❌ Retrain error: {error}")
        elif status and status.get('is_training'):
            print("⏭️  Model đang training. Sẽ retrain sau khi hoàn thành...")
        else:
            print("🔄 Không thể check retrain status. Thử trigger retrain...")
            success, error = trigger_retrain(force=False)
            if success:
                print("✅ Retrain triggered successfully")
            else:
                print(f"❌ Retrain error: {error}")
    
    print("\n" + "="*70)
    print("✅ DAILY WORKFLOW COMPLETED")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    tickers = None
    auto_retrain = True
    retrain_after_days = 1
    
    if len(sys.argv) > 1:
        if '--no-retrain' in sys.argv:
            auto_retrain = False
        if '--retrain-days' in sys.argv:
            idx = sys.argv.index('--retrain-days')
            if idx + 1 < len(sys.argv):
                retrain_after_days = int(sys.argv[idx + 1])
    
    # Parse --no-news flag
    scrape_news = True
    if '--no-news' in sys.argv:
        scrape_news = False
    
    success = daily_workflow(
        tickers=tickers,
        auto_retrain=auto_retrain,
        retrain_after_days=retrain_after_days,
        scrape_news=scrape_news
    )
    
    sys.exit(0 if success else 1)

