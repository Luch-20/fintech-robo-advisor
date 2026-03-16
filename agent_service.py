"""
Agent Service - API-only service cho AI Agent
Chỉ chứa API endpoints, không có web UI
"""

from flask import Flask, request, jsonify
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

from Get_data import download_stock_data
from robo_agent import IPOAgent, ActorNetwork

# Import generate_recommendation từ app.py
try:
    from app import (
        generate_recommendation,
        AVAILABLE_STOCKS,
        safe_float
    )
except ImportError:
    print("❌ Error: Cannot import from app.py")
    print("   Please ensure app.py is in the same directory")
    exit(1)

app = Flask(__name__)

# Disable Flask's default logging
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - Service information"""
    return jsonify({
        'service': 'Agent API Service',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'health': 'GET /health',
            'api': 'POST /api',
            'agent_analyze': 'POST /agent/analyze'
        },
        'usage': {
            'example_request': {
                'investing': [
                    {'ticker': 'ACB', 'amount': 1000000},
                    {'ticker': 'BCM', 'amount': 2000000}
                ]
            }
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'agent-api',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api', methods=['POST'])
def api():
    """
    Main API endpoint - Nhận input từ agent và trả về recommendations
    
    Request JSON:
    {
        "investing": [
            {"ticker": "ACB", "amount": 1000000},
            {"ticker": "BCM", "amount": 2000000}
        ],
        "fromDate": "2025-12-21T02:06:49.637Z",  // Optional: ISO 8601 or YYYY-MM-DD format, default: today
        "toDate": "2025-12-31T02:06:49.637Z"     // Optional: ISO 8601 or YYYY-MM-DD format, default: today + 10 days
    }
    
    Response JSON:
    {
        "success": true,
        "analysis_result": {...},
        "capital_allocation": [...],
        "ticker_details": [...],
        "chart": {...},
        "risk_profile": {...}  // Ở cuối cùng
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'Missing request body'}), 400
        
        # Parse request format
        investing_list = data.get('investing', [])
        
        if not investing_list:
            return jsonify({
                'success': False,
                'error': 'Missing "investing" array in request body'
            }), 400
        
        if len(investing_list) < 2:
            return jsonify({
                'success': False,
                'error': 'Need at least 2 stocks in investing array'
            }), 400
        
        # Convert sang format của hệ thống
        selected_tickers = []
        capital_amounts = []
        
        for item in investing_list:
            ticker = item.get('ticker', '').strip().upper()
            amount = item.get('amount', 0)
            
            if not ticker:
                return jsonify({
                    'success': False,
                    'error': f'Invalid ticker in investing array: {item}'
                }), 400
            
            try:
                amount = float(amount)
                if amount < 0:
                    return jsonify({
                        'success': False,
                        'error': f'Amount cannot be negative for ticker {ticker}'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Invalid amount for ticker {ticker}: {item.get("amount")}'
                }), 400
            
            selected_tickers.append(ticker)
            capital_amounts.append(amount)
        
        # Validate tickers
        invalid_tickers = [t for t in selected_tickers if t not in AVAILABLE_STOCKS]
        if invalid_tickers:
            return jsonify({
                'success': False,
                'error': f'Invalid tickers: {", ".join(invalid_tickers)}. Available tickers: {", ".join(AVAILABLE_STOCKS[:10])}...'
            }), 400
        
        # Get date range from request (optional) - support fromDate/toDate (ISO 8601 or YYYY-MM-DD)
        from_date_str = data.get('fromDate', '') or data.get('start_date', '')  # Support both formats
        to_date_str = data.get('toDate', '') or data.get('end_date', '')  # Support both formats
        
        # Default dates: today to 10 days later if not provided
        today = datetime.now()
        if not from_date_str or not to_date_str:
            from_date_str = today.strftime('%Y-%m-%d')
            to_date_str = (today + timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Parse dates - support ISO 8601 format (with T and Z) or YYYY-MM-DD
        def parse_date(date_str):
            """Parse date from ISO 8601 or YYYY-MM-DD format"""
            if not date_str:
                return None
            # Remove timezone info if present (Z or +HH:MM)
            date_str = date_str.split('T')[0] if 'T' in date_str else date_str
            date_str = date_str.split('+')[0] if '+' in date_str else date_str
            date_str = date_str.replace('Z', '')
            return datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
        
        # Validate date range: fromDate must be today or later, toDate max 10 days from fromDate
        try:
            today_date = today.date()
            start_dt = parse_date(from_date_str)
            end_dt = parse_date(to_date_str)
            
            if not start_dt or not end_dt:
                raise ValueError("Invalid date format")
            
            # Start date should be today or later
            if start_dt < today_date:
                return jsonify({
                    'success': False,
                    'error': 'Ngày bắt đầu phải là hôm nay hoặc sau đó'
                }), 400
            
            # End date should be max 10 days from start
            max_end = start_dt + timedelta(days=10)
            if end_dt > max_end:
                return jsonify({
                    'success': False,
                    'error': 'Ngày kết thúc không được quá 10 ngày từ ngày bắt đầu'
                }), 400
            
            if end_dt <= start_dt:
                return jsonify({
                    'success': False,
                    'error': 'Ngày kết thúc phải sau ngày bắt đầu'
                }), 400
                
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid date format: {str(e)}. Please use YYYY-MM-DD or ISO 8601 format (e.g., 2025-12-21 or 2025-12-21T02:06:49.637Z)'
            }), 400
        
        # Convert back to YYYY-MM-DD format for generate_recommendation
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        
        # Generate recommendation
        result = generate_recommendation(selected_tickers, capital_amounts, start_date, end_date)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        # GIỮ ĐÚNG THỨ TỰ NHƯ REQUEST
        results = result.get('results', [])
        
        # Tạo mapping từ results để lookup nhanh
        results_map = {item.get('ticker', ''): item for item in results}
        
        # Format từng ticker thành object đầy đủ thông tin - THEO ĐÚNG THỨ TỰ REQUEST
        tickers_array = []
        for ticker in selected_tickers:  # Giữ đúng thứ tự như request
            item = results_map.get(ticker)
            if item:
                current_weight = item.get('current_weight', 0.0)
                recommended_weight = item.get('recommended_weight', 0.0)
                current_capital = item.get('current_capital', 0.0)
                recommended_capital = item.get('recommended_capital', 0.0)
                change = item.get('change', 0.0)
                action = item.get('action', 'Keep')
                
                ticker_data = {
                    'ticker': ticker,
                    'current_weight': current_weight,
                    'current_weight_percent': current_weight * 100,
                    'recommended_weight': recommended_weight,
                    'recommended_weight_percent': recommended_weight * 100,
                    'current_capital': current_capital,
                    'recommended_capital': recommended_capital,
                    'change': change,
                    'change_percent': change * 100,
                    'change_amount': recommended_capital - current_capital,
                    'action': action,
                    'action_vn': 'Tăng' if action == 'Increase' else ('Giảm' if action == 'Decrease' else 'Giữ'),
                    'reasons': item.get('reasons', []),
                    'metrics': item.get('metrics', {})
                }
                tickers_array.append(ticker_data)
            else:
                ticker_data = {
                    'ticker': ticker,
                    'action': 'Keep',
                    'action_vn': 'Giữ',
                    'change': 0.0,
                    'change_percent': 0.0,
                    'change_amount': 0.0,
                    'current_capital': 0.0,
                    'current_weight': 0.0,
                    'current_weight_percent': 0.0,
                    'recommended_capital': 0.0,
                    'recommended_weight': 0.0,
                    'recommended_weight_percent': 0.0,
                    'reasons': [],
                    'metrics': {}
                }
                tickers_array.append(ticker_data)
        
        # Convert results list to dicts
        recommended_weights = {}
        recommended_capitals = {}
        current_weights = {}
        current_capitals = {}
        
        for item in results:
            ticker = item.get('ticker', '')
            if ticker:
                recommended_weights[ticker] = item.get('recommended_weight', 0.0)
                recommended_capitals[ticker] = item.get('recommended_capital', 0.0)
                current_weights[ticker] = item.get('current_weight', 0.0)
                current_capitals[ticker] = item.get('current_capital', 0.0)
        
        # 1. Capital Allocation
        capital_allocation = []
        for ticker in selected_tickers:
            if ticker in current_weights:
                current_weight = current_weights.get(ticker, 0.0)
                recommended_weight = recommended_weights.get(ticker, 0.0)
                change_weight = recommended_weight - current_weight
                
                if change_weight > 0.02:
                    action = "Tăng"
                elif change_weight < -0.02:
                    action = "Giảm"
                else:
                    action = "Giữ"
                
                change_str = f"{change_weight * 100:+.2f}%" if change_weight != 0 else "0.00%"
                
                capital_allocation.append({
                    'ticker': ticker,
                    'current': f"{current_weight * 100:.2f}%",
                    'recommended': f"{recommended_weight * 100:.2f}%",
                    'change': change_str,
                    'action': action
                })
        
        # 2. Risk Profile - Format đúng như app.py, đảm bảo luôn có
        risk_profile_raw = result.get('risk_profile', {})
        if not risk_profile_raw:
            # Default values nếu không có
            risk_profile = {
                'risk_level': 'Balanced (Cân bằng)',
                'risk_tolerance': 1.0,
                'target_return': 10.0
            }
        else:
            risk_profile = {
                'risk_level': risk_profile_raw.get('risk_level', 'Balanced (Cân bằng)'),
                'risk_tolerance': float(risk_profile_raw.get('risk_tolerance', 1.0)),
                'target_return': float(risk_profile_raw.get('target_return', 10.0))  # Đã là % từ generate_recommendation
            }
        
        # 3. Format analysis_result theo yêu cầu
        metrics = result.get('metrics', {})
        analysis_result = {
            'current_return': f"{metrics.get('current_return', 0):.2f}%",
            'recommend_return': f"{metrics.get('recommended_return', 0):.2f}%",
            'improvement': f"{metrics.get('improvement_return', 0):+.2f}%",
            'current_shape': f"{metrics.get('current_sharpe', 0):.2f}",
            'recommend_shape': f"{metrics.get('recommended_sharpe', 0):.2f}",
            'improvement_shape': f"{metrics.get('improvement_sharpe', 0):+.2f}",
            'turnover_rate': f"{metrics.get('turnover_rate', 0):.2f}%"
        }
        
        # 4. Format ticker_details theo yêu cầu
        formatted_ticker_details = []
        for ticker_data in tickers_array:
            ticker_metrics = ticker_data.get('metrics', {})
            reasons = ticker_data.get('reasons', [])
            
            # Format reasons thành array of objects
            formatted_reasons = [{'content': reason} for reason in reasons]
            
            formatted_ticker = {
                'ticker': ticker_data.get('ticker', ''),
                'action': ticker_data.get('action', 'Keep'),
                'reason': formatted_reasons,
                'expected_return': f"{ticker_metrics.get('expected_return', 0):.2f}%",
                'volatility': f"{ticker_metrics.get('volatility', 0):.2f}%",
                'shape_ratio': round(ticker_metrics.get('sharpe_ratio', 0), 2),
                'portfolio_correlation': round(ticker_metrics.get('correlation', 0), 2),
                'momentum': f"{ticker_metrics.get('momentum', 0):.2f}%",
                'up_day_ratio': f"{ticker_metrics.get('trend_ratio', 0):.1f}%"
            }
            formatted_ticker_details.append(formatted_ticker)
        
        # 5. Format chart theo yêu cầu
        chart_results = []
        for ticker in selected_tickers:
            chart_results.append({
                'ticker': ticker,
                'current_weight': current_weights.get(ticker, 0.0),
                'recommend_weight': recommended_weights.get(ticker, 0.0)
            })
        
        chart = {
            'results': chart_results
        }
        
        # Tổ chức response theo format yêu cầu - risk_profile ở cuối cùng
        response = {
            'success': True,
            'analysis_result': analysis_result,
            'capital_allocation': capital_allocation,
            'ticker_details': formatted_ticker_details,
            'chart': chart,
            'risk_profile': risk_profile  # Đặt ở cuối cùng
        }
        
        print(f"✅ DEBUG: Response created with chart: {len(chart_results)} items")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/agent/analyze', methods=['POST'])
def agent_analyze():
    """
    API endpoint cho AI Agent - Format giống /api
    
    Request JSON:
    {
        "investing": [
            {"ticker": "ACB", "amount": 1000000},
            {"ticker": "BCM", "amount": 2000000}
        ],
        "fromDate": "2025-12-21T02:06:49.637Z",  // Optional: ISO 8601 or YYYY-MM-DD format, default: today
        "toDate": "2025-12-31T02:06:49.637Z"     // Optional: ISO 8601 or YYYY-MM-DD format, default: today + 10 days
    }
    """
    # Sử dụng lại logic từ /api
    return api()


if __name__ == '__main__':
    # Get port from environment (for cloud platforms) or use default
    PORT = int(os.environ.get('PORT', 5002))
    
    print("="*70)
    print("🤖 AGENT SERVICE - API ONLY")
    print("="*70)
    print("\n📡 Endpoints:")
    print("   POST /api")
    print("   POST /agent/analyze")
    print("   GET  /health")
    print(f"\n🚀 Starting on port {PORT}...")
    print("   (app.py runs on port 5001 for web UI)")
    print("="*70)
    print()
    
    app.run(host='0.0.0.0', port=PORT, debug=False)