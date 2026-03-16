"""
Test script cho agent_service.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:5002"
ENDPOINTS = [
    ("/health", "GET"),
    ("/api", "POST"),
    ("/agent/analyze", "POST")
]

# Test data
test_data = {
    "investing": [
        {"ticker": "ACB", "amount": 1000000},
        {"ticker": "BCM", "amount": 2000000},
        {"ticker": "BID", "amount": 1500000}
    ]
}

def test_health():
    """Test health endpoint"""
    print("="*70)
    print("🏥 TESTING /health")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_endpoint(endpoint, method="POST"):
    """Test API endpoint"""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING {endpoint}")
    print("="*70)
    
    print(f"\n📤 Request:")
    print(f"   URL: {BASE_URL}{endpoint}")
    print(f"   Method: {method}")
    if method == "POST":
        print(f"   Body:")
        print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
        else:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📥 Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed_time:.2f} seconds")
        
        try:
            result = response.json()
            if result.get('success'):
                print(f"   ✅ Success: {result.get('success')}")
                
                if endpoint == "/api":
                    data = result.get('data', {})
                    print(f"\n   📊 Metrics:")
                    metrics = data.get('metrics', {})
                    if metrics:
                        print(f"      Recommended Return: {metrics.get('recommended_return', 0)*100:.2f}%")
                        print(f"      Recommended Sharpe: {metrics.get('recommended_sharpe', 0):.4f}")
                elif endpoint == "/agent/analyze":
                    print(f"\n   📊 Metrics:")
                    metrics = result.get('metrics', {})
                    if metrics:
                        print(f"      Recommended Return: {metrics.get('recommended_return', 0)*100:.2f}%")
                        print(f"      Recommended Sharpe: {metrics.get('recommended_sharpe', 0):.4f}")
                
                return True
            else:
                print(f"   ❌ Error: {result.get('error')}")
                return False
        except:
            print(f"   ❌ Invalid JSON response:")
            print(f"   {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Cannot connect to {BASE_URL}")
        print(f"   Please make sure agent_service.py is running:")
        print(f"   python3 agent_service.py")
        return False
    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: Request timeout (120s)")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 TESTING AGENT SERVICE")
    print("="*70)
    
    # Test health
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Service may not be running.")
        return False
    
    # Test API endpoints
    results = []
    for endpoint, method in ENDPOINTS:
        if method == "GET":
            continue  # Skip health, already tested
        success = test_endpoint(endpoint, method)
        results.append((endpoint, success))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"   Health: {'✅ PASS' if health_ok else '❌ FAIL'}")
    for endpoint, success in results:
        print(f"   {endpoint}: {'✅ PASS' if success else '❌ FAIL'}")
    
    all_passed = health_ok and all(success for _, success in results)
    print("\n" + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

