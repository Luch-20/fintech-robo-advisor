"""
Test script cho Agent Service API
Test endpoint /api và /agent/analyze mà không thay đổi JSON format
"""

import requests
import json
from datetime import datetime

# API endpoint
API_URL = "http://localhost:5002/api"
AGENT_ANALYZE_URL = "http://localhost:5002/agent/analyze"

# Test data - không thay đổi format
test_request = {
    "investing": [
        {"ticker": "ACB", "amount": 1000000},
        {"ticker": "BCM", "amount": 2000000},
        {"ticker": "BID", "amount": 1500000},
        {"ticker": "CTG", "amount": 1200000},
        {"ticker": "FPT", "amount": 800000}
    ]
}


def test_api_endpoint(url, endpoint_name):
    """Test một API endpoint"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing {endpoint_name}")
    print(f"{'='*70}")
    print(f"URL: {url}")
    print(f"Request JSON:")
    print(json.dumps(test_request, indent=2, ensure_ascii=False))
    print(f"\n⏳ Sending request...")
    
    try:
        response = requests.post(url, json=test_request, timeout=60)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            
            print(f"\n✅ Response received successfully!")
            print(f"\n📋 Response Structure:")
            print(f"  - success: {response_data.get('success', 'N/A')}")
            
            if 'analysis_result' in response_data:
                print(f"  - analysis_result: ✅")
                print(f"    * current_return: {response_data['analysis_result'].get('current_return', 'N/A')}")
                print(f"    * recommend_return: {response_data['analysis_result'].get('recommend_return', 'N/A')}")
                print(f"    * improvement: {response_data['analysis_result'].get('improvement', 'N/A')}")
            
            if 'capital_allocation' in response_data:
                print(f"  - capital_allocation: ✅ ({len(response_data['capital_allocation'])} items)")
                for item in response_data['capital_allocation'][:2]:  # Show first 2
                    print(f"    * {item.get('ticker', 'N/A')}: {item.get('current', 'N/A')} → {item.get('recommended', 'N/A')} ({item.get('action', 'N/A')})")
            
            if 'ticker_details' in response_data:
                print(f"  - ticker_details: ✅ ({len(response_data['ticker_details'])} items)")
                for item in response_data['ticker_details'][:2]:  # Show first 2
                    print(f"    * {item.get('ticker', 'N/A')}: {item.get('action', 'N/A')} - {len(item.get('reason', []))} reasons")
            
            if 'chart' in response_data:
                print(f"  - chart: ✅ ({len(response_data['chart'].get('results', []))} items)")
            
            if 'performance_comparison' in response_data:
                print(f"  - performance_comparison: ✅")
            
            if 'news_info' in response_data:
                print(f"  - news_info: ✅")
            
            if 'risk_profile' in response_data:
                print(f"  - risk_profile: ✅")
            
            # Save full response to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_response_{endpoint_name.replace('/', '_')}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Full response saved to: {filename}")
            
            # Print full JSON (formatted)
            print(f"\n📄 Full Response JSON:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
            
        else:
            print(f"\n❌ Error Response:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error: Cannot connect to {url}")
        print("   Make sure the service is running on port 5002")
    except requests.exceptions.Timeout:
        print(f"\n❌ Timeout: Request took too long")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def test_health_check():
    """Test health check endpoint"""
    print(f"\n{'='*70}")
    print(f"🏥 Testing Health Check")
    print(f"{'='*70}")
    
    try:
        response = requests.get("http://localhost:5002/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            print("✅ Service is healthy")
        else:
            print("❌ Service health check failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    print("="*70)
    print("🧪 AGENT SERVICE API TEST")
    print("="*70)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test health check first
    test_health_check()
    
    # Test /api endpoint
    test_api_endpoint(API_URL, "/api")
    
    # Test /agent/analyze endpoint
    test_api_endpoint(AGENT_ANALYZE_URL, "/agent/analyze")
    
    print(f"\n{'='*70}")
    print("✅ Testing completed!")
    print(f"{'='*70}")
