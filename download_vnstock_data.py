"""
Script để tải dữ liệu từ vnstock cho 30 mã cổ phiếu VN Index
Lưu vào file CSV format giống Data_test.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import vnstock
try:
    from vnstock import Trading
    HAS_VNSTOCK = True
except ImportError:
    HAS_VNSTOCK = False
    print("❌ vnstock not installed. Install with: pip install vnstock")
    exit(1)

# 30 mã cổ phiếu VN Index + chỉ số VN-Index
TICKERS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
    '^VNINDEX'  # Chỉ số VN-Index (có thể không có trong vnstock)
]

def download_ticker_data(ticker, start_date, end_date, retry=3):
    """
    Tải dữ liệu cho một mã cổ phiếu từ vnstock
    
    Args:
        ticker: Mã cổ phiếu
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        retry: Số lần thử lại nếu lỗi
    
    Returns:
        DataFrame với columns: Date, Ticker, Open, High, Low, Close, Volume
    """
    ticker_clean = ticker.replace('^', '').upper()
    
    # Bỏ qua ^VNINDEX vì vnstock không hỗ trợ chỉ số
    if ticker == '^VNINDEX':
        print(f"   ⚠️  Bỏ qua {ticker} (vnstock không hỗ trợ chỉ số)")
        return None
    
    for attempt in range(retry):
        try:
            print(f"   📥 Đang tải {ticker}... (lần thử {attempt + 1}/{retry})")
            
            # Tải dữ liệu từ vnstock
            df = stock_historical_data(
                symbol=ticker_clean,
                start_date=start_date,
                end_date=end_date,
                resolution='1D',  # Daily data
                type='stock'
            )
            
            if df is None or df.empty:
                print(f"   ❌ {ticker}: Không có dữ liệu")
                return None
            
            # Kiểm tra và chuẩn hóa tên cột
            # vnstock có thể dùng tên cột khác nhau
            cols_lower = {col.lower(): col for col in df.columns}
            
            # Tìm cột Date
            date_col = None
            for key in ['date', 'time', 'ngay', 'timestamp']:
                if key in cols_lower:
                    date_col = cols_lower[key]
                    break
            
            if date_col:
                df = df.set_index(date_col)
                df.index = pd.to_datetime(df.index)
            elif df.index.name:
                df.index = pd.to_datetime(df.index)
            else:
                # Nếu không có date column, tạo từ index
                df.index = pd.to_datetime(df.index)
            
            # Tìm các cột OHLCV
            open_col = next((cols_lower[k] for k in ['open', 'gia_mo_cua', 'o'] if k in cols_lower), None)
            high_col = next((cols_lower[k] for k in ['high', 'gia_cao_nhat', 'h'] if k in cols_lower), None)
            low_col = next((cols_lower[k] for k in ['low', 'gia_thap_nhat', 'l'] if k in cols_lower), None)
            close_col = next((cols_lower[k] for k in ['close', 'gia_dong_cua', 'c', 'price'] if k in cols_lower), None)
            volume_col = next((cols_lower[k] for k in ['volume', 'khoi_luong', 'vol', 'kl'] if k in cols_lower), None)
            
            # Nếu không tìm thấy close, thử dùng cột đầu tiên
            if close_col is None:
                if len(df.columns) > 0:
                    close_col = df.columns[0]
                else:
                    print(f"   ❌ {ticker}: Không tìm thấy cột giá")
                    return None
            
            # Tạo DataFrame kết quả
            result_data = []
            for date in df.index:
                row_data = {
                    'Date': date,
                    'Ticker': ticker,
                    'Open': df.loc[date, open_col] if open_col else df.loc[date, close_col],
                    'High': df.loc[date, high_col] if high_col else df.loc[date, close_col],
                    'Low': df.loc[date, low_col] if low_col else df.loc[date, close_col],
                    'Close': df.loc[date, close_col],
                    'Volume': df.loc[date, volume_col] if volume_col else 0
                }
                result_data.append(row_data)
            
            result_df = pd.DataFrame(result_data)
            
            # Loại bỏ NaN
            result_df = result_df.dropna(subset=['Close'])
            
            if len(result_df) > 0:
                # Kiểm tra số ngày dữ liệu
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                expected_days = int((end_dt - start_dt).days * 0.6)  # Ít nhất 60% số ngày
                actual_days = len(result_df)
                
                if actual_days >= max(50, expected_days):
                    print(f"   ✅ {ticker}: {actual_days} ngày dữ liệu ({result_df['Date'].min().date()} đến {result_df['Date'].max().date()})")
                    return result_df
                else:
                    print(f"   ⚠️  {ticker}: Chỉ có {actual_days} ngày (kỳ vọng {expected_days}+), có thể thiếu dữ liệu")
                    # Vẫn trả về dữ liệu nhưng cảnh báo
                return result_df
            else:
                print(f"   ❌ {ticker}: Không có dữ liệu hợp lệ")
                return None
                
        except Exception as e:
            if attempt < retry - 1:
                wait_time = (attempt + 1) * 2  # Tăng thời gian chờ mỗi lần thử
                print(f"   ⚠️  {ticker}: Lỗi (lần thử {attempt + 1}): {str(e)[:100]}")
                print(f"   ⏳ Chờ {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ {ticker}: Thất bại sau {retry} lần thử: {str(e)[:100]}")
                return None
    
    return None


def download_all_tickers(tickers, start_date, end_date, output_file='data/Data_test.csv'):
    """
    Tải dữ liệu cho tất cả các mã cổ phiếu
    
    Args:
        tickers: List các mã cổ phiếu
        start_date: Ngày bắt đầu 'YYYY-MM-DD'
        end_date: Ngày kết thúc 'YYYY-MM-DD'
        output_file: Đường dẫn file CSV để lưu
    """
    print("="*70)
    print("TẢI DỮ LIỆU TỪ VNSTOCK")
    print("="*70)
    print(f"\n📅 Khoảng thời gian: {start_date} đến {end_date}")
    print(f"📊 Số mã cổ phiếu: {len(tickers)}")
    print(f"📁 File output: {output_file}")
    print(f"\n🚀 Bắt đầu tải dữ liệu...\n")
    
    all_data = []
    successful = []
    failed = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        
        data = download_ticker_data(ticker, start_date, end_date)
        
        if data is not None and len(data) > 0:
            all_data.append(data)
            successful.append(ticker)
        else:
            failed.append(ticker)
        
        # Chờ một chút giữa các lần tải để tránh rate limit
        if i < len(tickers):
            time.sleep(1)  # Chờ 1 giây giữa các mã
    
    # Kết hợp tất cả dữ liệu
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['Ticker', 'Date'])
        
        # Đảm bảo thứ tự cột
        columns_order = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        combined_df = combined_df[columns_order]
        
        # Tạo thư mục nếu chưa có
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Lưu vào CSV
        combined_df.to_csv(output_file, index=False)
        
        print("\n" + "="*70)
        print("✅ HOÀN TẤT!")
        print("="*70)
        print(f"\n📊 Kết quả:")
        print(f"   ✅ Thành công: {len(successful)}/{len(tickers)} mã")
        print(f"   ❌ Thất bại: {len(failed)}/{len(tickers)} mã")
        
        if successful:
            print(f"\n   Các mã thành công: {', '.join(successful)}")
        if failed:
            print(f"\n   Các mã thất bại: {', '.join(failed)}")
        
        print(f"\n📁 Dữ liệu đã được lưu vào: {output_file}")
        print(f"   Tổng số dòng: {len(combined_df)}")
        print(f"   Khoảng thời gian: {combined_df['Date'].min().date()} đến {combined_df['Date'].max().date()}")
        print(f"   Số mã có dữ liệu: {combined_df['Ticker'].nunique()}")
        
        # Validation: Kiểm tra số ngày dữ liệu cho từng mã
        print(f"\n📊 Validation dữ liệu từng mã:")
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        expected_days = int((end_dt - start_dt).days * 0.6)  # Ít nhất 60% số ngày
        
        tickers_with_issues = []
        for ticker in combined_df['Ticker'].unique():
            ticker_df = combined_df[combined_df['Ticker'] == ticker]
            ticker_days = len(ticker_df)
            ticker_pct = (ticker_days / expected_days * 100) if expected_days > 0 else 0
            
            if ticker_days < expected_days:
                print(f"   ⚠️  {ticker}: {ticker_days} ngày ({ticker_pct:.1f}%) - THIẾU DỮ LIỆU")
                tickers_with_issues.append(ticker)
            else:
                print(f"   ✅ {ticker}: {ticker_days} ngày ({ticker_pct:.1f}%)")
        
        if tickers_with_issues:
            print(f"\n⚠️  CẢNH BÁO: {len(tickers_with_issues)} mã thiếu dữ liệu: {tickers_with_issues}")
        else:
            print(f"\n✅ Tất cả mã đều có đủ dữ liệu!")
        
        return combined_df
    else:
        print("\n❌ Không tải được dữ liệu cho mã nào!")
        return None


def main():
    """Main function"""
    # Tính 2 năm gần nhất
    now = datetime.now()
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Output file
    output_file = 'data/Data_test.csv'
    
    # Tải dữ liệu
    result = download_all_tickers(TICKERS, start_date, end_date, output_file)
    
    if result is not None:
        print("\n✅ Script hoàn tất!")
        print(f"   Bạn có thể sử dụng file {output_file} cho các bước tiếp theo.")
    else:
        print("\n❌ Script thất bại!")
        print("   Vui lòng kiểm tra:")
        print("   1. Kết nối internet")
        print("   2. vnstock đã được cài đặt: pip install vnstock")
        print("   3. Có thể cần API key cho vnstock (kiểm tra https://vnstocks.com/)")


if __name__ == "__main__":
    main()

