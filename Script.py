import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import sys

# Danh sách mã cổ phiếu
tickers = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
           'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
           'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']

def get_vn_stock_data_vndirect(ticker, start_date='2020-01-01', end_date='2025-12-31'):
    """
    Lấy dữ liệu cổ phiếu từ VNDirect API
    """
    try:
        url = 'https://finfo-api.vndirect.com.vn/v4/stock_prices'
        params = {
            'sort': 'date',
            'q': f'code:{ticker}~date:gte:{start_date}~date:lte:{end_date}',
            'size': 9999,
            'page': 1
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'])
                df = df[['date', 'close', 'high', 'low', 'open', 'nmVolume', 'code']]
                df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker']
                
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df.sort_values('Date')
                
                return df
        return None
            
    except Exception as e:
        print(f"  Lỗi VNDirect: {str(e)}")
        return None

def get_vn_stock_data_ssi(ticker, start_date='2020-01-01', end_date='2025-12-31'):
    """
    Lấy dữ liệu từ SSI API (dự phòng)
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        url = 'https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term'
        params = {
            'ticker': ticker,
            'type': 'stock',
            'resolution': 'D',
            'from': int(start.timestamp()),
            'to': int(end.timestamp())
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'])
                df['Date'] = pd.to_datetime(df['tradingDate'])
                df = df.rename(columns={
                    'close': 'Close',
                    'high': 'High', 
                    'low': 'Low',
                    'open': 'Open',
                    'volume': 'Volume'
                })
                df['Ticker'] = ticker
                df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker']]
                
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df = df.sort_values('Date')
                
                return df
        return None
            
    except Exception as e:
        print(f"  Lỗi TCBS: {str(e)}")
        return None

def main():
    """
    Hàm chính
    """
    print("🚀 BẮT ĐẦU TẢI DỮ LIỆU CỔ PHIẾU VIỆT NAM")
    print("=" * 70)
    print(f"Số lượng mã: {len(tickers)}")
    print(f"Thời gian: 2020-01-01 đến 2025-12-31")
    print("=" * 70)
    
    all_data = []
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}:", end=' ')
        sys.stdout.flush()
        
        # Thử VNDirect trước
        df = get_vn_stock_data_vndirect(ticker, '2020-01-01', '2025-12-31')
        
        # Nếu thất bại, thử TCBS
        if df is None or len(df) == 0:
            print("VNDirect failed, trying TCBS...", end=' ')
            sys.stdout.flush()
            df = get_vn_stock_data_ssi(ticker, '2020-01-01', '2025-12-31')
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"✅ OK ({len(df)} dòng)")
            success_count += 1
        else:
            print(f"❌ THẤT BẠI")
            fail_count += 1
        
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ: ✅ {success_count} thành công | ❌ {fail_count} thất bại")
    print("=" * 70)
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(['Ticker', 'Date'])
        
        output_file = 'vn_stocks_data_2020_2025.csv'
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ HOÀN THÀNH!")
        print(f"📁 File: {output_file}")
        print(f"📊 Tổng số dòng: {len(final_df):,}")
        print(f"📈 Số mã cổ phiếu: {final_df['Ticker'].nunique()}")
        print(f"📅 Từ {final_df['Date'].min()} đến {final_df['Date'].max()}")
        
        print("\n📋 MẪU DỮ LIỆU:")
        print(final_df.head(10).to_string(index=False))
        
        print(f"\n💾 Dữ liệu đã được lưu tại: {output_file}")
        
    else:
        print("\n❌ KHÔNG CÓ DỮ LIỆU NÀO!")
        print("Vui lòng kiểm tra kết nối internet hoặc thử lại sau.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)