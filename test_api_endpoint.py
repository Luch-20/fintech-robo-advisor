"""
Test script cho endpoint /api
"""

import requests
import json
import time

BASE_URL = "http://localhost:5001"
ENDPOINT = f"{BASE_URL}/api"

# Test data
test_data = {
    "investing": [
        {"ticker": "ACB", "amount": 1000000},
        {"ticker": "BCM", "amount": 2000000},
        {"ticker": "BID", "amount": 1500000},
        {"ticker": "CTG", "amount": 1200000},
        {"ticker": "DGC", "amount": 800000}
    ]
}

def test_api_endpoint():
    """Test API endpoint /api"""
    print("="*70)
    print("🧪 TESTING /api ENDPOINT")
    print("="*70)
    
    print(f"\n📤 Request:")
    print(f"   URL: {ENDPOINT}")
    print(f"   Method: POST")
    print(f"   Body:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    try:
        # Send request
        print(f"\n⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT,
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minutes timeout (vì có thể mất thời gian để load model và tính toán)
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"📥 Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed_time:.2f} seconds")
        
        # Parse response
        try:
            result = response.json()
        except:
            print(f"   ❌ Invalid JSON response:")
            print(f"   {response.text[:500]}")
            return False
        
        if result.get('success'):
            print(f"   ✅ Success: {result.get('success')}")
            
            data = result.get('data', {})
            
            print(f"\n📊 Recommended Weights:")
            recommended_weights = data.get('recommended_weights', {})
            for ticker, weight in list(recommended_weights.items())[:5]:
                print(f"      {ticker}: {weight:.4f} ({weight*100:.2f}%)")
            
            print(f"\n💰 Recommended Capitals:")
            recommended_capitals = data.get('recommended_capitals', {})
            for ticker, capital in list(recommended_capitals.items())[:5]:
                print(f"      {ticker}: {capital:,.0f} VNĐ")
            
            print(f"\n📈 Metrics:")
            metrics = data.get('metrics', {})
            if metrics:
                print(f"      Total Capital: {metrics.get('total_capital', 0):,.0f} VNĐ")
                if 'recommended_return' in metrics:
                    print(f"      Recommended Return: {metrics.get('recommended_return', 0)*100:.2f}%")
                if 'recommended_sharpe' in metrics:
                    print(f"      Recommended Sharpe: {metrics.get('recommended_sharpe', 0):.4f}")
                if 'period_expected_return' in metrics:
                    print(f"      Period Expected Return: {metrics.get('period_expected_return', 0)*100:.2f}%")
            
            print(f"\n📰 Tickers Analysis (first 3):")
            tickers = data.get('tickers', [])
            for ticker_info in tickers[:3]:
                print(f"\n   {ticker_info.get('ticker')}:")
                print(f"      Action: {ticker_info.get('action')}")
                print(f"      Current Weight: {ticker_info.get('current_weight', 0):.4f}")
                print(f"      Recommended Weight: {ticker_info.get('recommended_weight', 0):.4f}")
                print(f"      Change: {ticker_info.get('change_percent', 0):.2f}%")
                if ticker_info.get('recent_news'):
                    print(f"      News: {len(ticker_info.get('recent_news', []))} articles")
            
            print(f"\n✅ API Test PASSED")
            print(f"   Total response time: {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"   ❌ Error: {result.get('error')}")
            print(f"\n❌ API Test FAILED")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Cannot connect to {BASE_URL}")
        print(f"   Please make sure Flask app is running:")
        print(f"   python3 app.py")
        return False
    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: Request timeout (120s)")
        print(f"   The API might be processing. Try again later.")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_endpoint()
    exit(0 if success else 1)

