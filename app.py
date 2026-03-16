"""
Flask Web Application for Portfolio Optimization Advisor

Web interface cho hệ thống Robo-advisor sử dụng IPO và DDPG
"""

from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timedelta

# Set working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir:
    os.chdir(script_dir)

from Get_data import download_stock_data
from robo_agent import IPOAgent, ActorNetwork, train_robo_advisor
# Import generate_recommendation from main.py để tái sử dụng
try:
    from main import generate_recommendation as generate_recommendation_main
except ImportError:
    generate_recommendation_main = None

app = Flask(__name__)

# Removed retrain_api usage

# 30 mã cổ phiếu VN Index + 1 chỉ số VN-Index mặc định
AVAILABLE_STOCKS = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
    '^VNINDEX'
]

TRADING_DAYS_PER_YEAR = 252  # 252 trading days per year
ROLLING_WINDOW_DAYS = 126  # Rolling window 126 ngày (6 tháng)
RISK_FREE_RATE_ANNUAL = 0.045  # Risk-free rate: 4.5% annual (trái phiếu chính phủ VN)
# Transaction cost: ~0.3% theo tài liệu
TRANSACTION_COST_RATE = 0.003  # Transaction cost: 0.3% (phí giao dịch VN theo tài liệu)

# Global variables để cache model và data
trained_model = None
model_info = None


def safe_float(value, default=0.0):
    """
    Convert value to float, handling NaN and None values for JSON serialization
    
    Args:
        value: Value to convert
        default: Default value if value is NaN or None
    
    Returns:
        float: Safe float value for JSON
    """
    if value is None:
        return default
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (ValueError, TypeError):
        return default


def load_trained_model():
    """Load trained model"""
    global trained_model, model_info
    
    if trained_model is not None:
        return trained_model, model_info
    
    # Try multiple possible model paths
    possible_paths = [
        Path('models') / 'trained_model.pth',
        Path('models') / 'actor_model.pth'
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        return None, None
    
    try:
        # PyTorch 2.6+ requires weights_only=False for models with numpy arrays
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check if model has required keys
        if 'n_stocks' not in checkpoint or 'actor' not in checkpoint:
            print(f"Error: Model file {model_path} is missing required keys")
            return None, None
        
        n_stocks = checkpoint['n_stocks']
        
        # Get state_dim from checkpoint, or infer from model weights
        state_dim = checkpoint.get('state_dim', None)
        if state_dim is None:
            # Infer state_dim from the saved model weights (backward compatibility)
            actor_state = checkpoint['actor']
            if 'fc1.weight' in actor_state:
                state_dim = actor_state['fc1.weight'].shape[1]  # Input dimension of first layer
            else:
                state_dim = 6  # Default fallback
        
        actor = ActorNetwork(n_stocks, state_dim=state_dim)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        trained_model = actor
        model_info = {
            'omega': checkpoint.get('omega', 1.0),
            'target_return': checkpoint.get('target_return', 0.1),
            'n_stocks': n_stocks,
            'state_dim': state_dim,
            'stock_names': checkpoint.get('stock_names', AVAILABLE_STOCKS)
        }
        
        return trained_model, model_info
    except Exception:
        return None, None


def compute_weights_from_capital(tickers, capital_amounts, prices):
    """Convert capital amounts to portfolio weights"""
    last_prices = prices.iloc[-1]
    values = []
    
    for ticker, capital in zip(tickers, capital_amounts):
        price = last_prices.get(ticker)
        if price is None or np.isnan(price):
            price = 1.0
        values.append(float(capital))
    
    total_value = sum(values)
    if total_value <= 0:
        return np.ones(len(tickers)) / len(tickers)
    
    weights = np.array(values) / total_value
    return weights


def get_ticker_specific_news(ticker, days_back=7, db_path='data/news_data.db'):
    """
    Lấy news cụ thể cho một mã cổ phiếu từ database
    
    Args:
        ticker: Mã cổ phiếu
        days_back: Số ngày gần nhất
        db_path: Đường dẫn database
    
    Returns:
        List of news dicts với: title, sentiment_label, impact_score, date, link
    """
    from pathlib import Path
    import sqlite3
    from datetime import datetime, timedelta
    
    if not Path(db_path).exists():
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Lấy news trong days_back ngày gần nhất
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT title, sentiment_label, impact_score, date, link, source, compound_score
            FROM news_data
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY impact_score DESC, date DESC
            LIMIT 10
        ''', (ticker, start_date, end_date))
        
        results = cursor.fetchall()
        news_list = []
        for row in results:
            news_list.append({
                'title': row['title'],
                'sentiment_label': row['sentiment_label'],
                'impact_score': row['impact_score'],
                'date': row['date'],
                'link': row['link'],
                'source': row['source'],
                'compound_score': row['compound_score']
            })
        
        conn.close()
        return news_list
    except Exception as e:
        print(f"⚠️  Lỗi khi lấy news cho {ticker}: {e}")
        return []


def generate_ticker_news_explanation(ticker, delta, days_back=7):
    """
    Tạo explanation về news cụ thể cho một mã cổ phiếu
    
    Args:
        ticker: Mã cổ phiếu
        delta: Thay đổi weight (recommended - current)
        days_back: Số ngày gần nhất để lấy news
    
    Returns:
        List of news explanations
    """
    news_list = get_ticker_specific_news(ticker, days_back=days_back)
    
    if not news_list:
        return []
    
    explanations = []
    
    # Phân loại news tích cực và tiêu cực
    # Positive: sentiment = 'positive' hoặc impact_score > 0.1
    positive_news = [n for n in news_list if (n['sentiment_label'] == 'positive' or n.get('impact_score', 0) > 0.1) and n.get('impact_score', 0) > 0]
    # Negative: sentiment = 'negative' hoặc impact_score < -0.1
    negative_news = [n for n in news_list if (n['sentiment_label'] == 'negative' or n.get('impact_score', 0) < -0.1) and n.get('impact_score', 0) < 0]
    # Neutral: các news còn lại
    neutral_news = [n for n in news_list if n not in positive_news and n not in negative_news]
    
    # Nếu đề xuất tăng allocation
    if delta > 0.02:
        if positive_news:
            # Hiển thị top 2-3 news tích cực có impact cao nhất
            top_positive = sorted(positive_news, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:3]
            for news in top_positive:
                title_short = news['title'][:60] + '...' if len(news['title']) > 60 else news['title']
                impact = news.get('impact_score', 0)
                date_str = news.get('date', 'N/A')
                explanations.append(f"📰 Tin tích cực: \"{title_short}\" (Impact: {impact:.2f}, {date_str})")
        elif neutral_news:
            # Nếu không có positive news, hiển thị neutral news có impact cao
            top_neutral = sorted(neutral_news, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:2]
            for news in top_neutral:
                title_short = news['title'][:60] + '...' if len(news['title']) > 60 else news['title']
                explanations.append(f"ℹ️  Tin tức: \"{title_short}\" ({news.get('date', 'N/A')})")
        
        # Cảnh báo nếu có news tiêu cực mạnh
        if negative_news:
            top_negative = sorted(negative_news, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:1]
            for news in top_negative:
                title_short = news['title'][:60] + '...' if len(news['title']) > 60 else news['title']
                impact = news.get('impact_score', 0)
                explanations.append(f"⚠️  Lưu ý tin tiêu cực: \"{title_short}\" (Impact: {impact:.2f})")
    
    # Nếu đề xuất giảm allocation
    elif delta < -0.02:
        if negative_news:
            # Hiển thị top 2-3 news tiêu cực có impact cao nhất
            top_negative = sorted(negative_news, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:3]
            for news in top_negative:
                title_short = news['title'][:60] + '...' if len(news['title']) > 60 else news['title']
                impact = news.get('impact_score', 0)
                date_str = news.get('date', 'N/A')
                explanations.append(f"📰 Tin tiêu cực: \"{title_short}\" (Impact: {impact:.2f}, {date_str})")
        elif neutral_news:
            # Nếu không có negative news, hiển thị neutral news
            top_neutral = sorted(neutral_news, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:2]
            for news in top_neutral:
                title_short = news['title'][:60] + '...' if len(news['title']) > 60 else news['title']
                explanations.append(f"ℹ️  Tin tức: \"{title_short}\" ({news.get('date', 'N/A')})")
        
        # Nếu có news tích cực nhưng vẫn giảm (có thể do các yếu tố khác)
        if positive_news and len(positive_news) > len(negative_news):
            explanations.append(f"ℹ️  Có {len(positive_news)} tin tích cực nhưng vẫn khuyến nghị giảm do các yếu tố rủi ro khác (volatility, correlation, etc.)")
    
    return explanations


def generate_news_impact_explanation(news_features, tickers, recommended_weights):
    """
    Generate explanation về ảnh hưởng của news đến prediction
    
    Args:
        news_features: numpy array với 4 news features [avg_sentiment, avg_impact, positive_ratio, news_activity]
        tickers: List of tickers
        recommended_weights: Recommended portfolio weights
    
    Returns:
        Dict với news impact explanation
    """
    if news_features is None or len(news_features) != 4:
        return {
            'has_news': False,
            'explanation': 'Chưa có dữ liệu news để phân tích',
            'impact_level': 'neutral'
        }
    
    avg_sentiment = news_features[0]  # -1 to 1
    avg_impact = news_features[1]     # -1 to 1
    positive_ratio = news_features[2]  # 0 to 1
    news_activity = news_features[3]   # 0 to 1
    
    explanations = []
    impact_level = 'neutral'
    
    # Sentiment analysis
    if avg_sentiment > 0.2:
        explanations.append(f"Tin tức tích cực (sentiment: {avg_sentiment:.2f}) → Khuyến nghị tăng allocation")
        impact_level = 'positive'
    elif avg_sentiment < -0.2:
        explanations.append(f"Tin tức tiêu cực (sentiment: {avg_sentiment:.2f}) → Khuyến nghị giảm allocation")
        impact_level = 'negative'
    else:
        explanations.append(f"Tin tức trung tính (sentiment: {avg_sentiment:.2f})")
    
    # Impact score analysis
    if abs(avg_impact) > 0.3:
        if avg_impact > 0:
            explanations.append(f"Độ ảnh hưởng cao ({avg_impact:.2f}) → Tin tức có tác động mạnh đến giá")
        else:
            explanations.append(f"Độ ảnh hưởng âm ({avg_impact:.2f}) → Tin tức có thể làm giảm giá")
    
    # Positive ratio analysis
    if positive_ratio > 0.6:
        explanations.append(f"Tỷ lệ tin tích cực cao ({positive_ratio*100:.1f}%) → Xu hướng tốt")
        if impact_level == 'neutral':
            impact_level = 'positive'
    elif positive_ratio < 0.4:
        explanations.append(f"Tỷ lệ tin tiêu cực cao ({(1-positive_ratio)*100:.1f}%) → Xu hướng xấu")
        if impact_level == 'neutral':
            impact_level = 'negative'
    
    # News activity analysis
    if news_activity > 0.5:
        explanations.append(f"Hoạt động tin tức cao ({news_activity*100:.1f}%) → Thị trường đang quan tâm nhiều")
    elif news_activity < 0.2:
        explanations.append(f"Hoạt động tin tức thấp ({news_activity*100:.1f}%) → Ít thông tin mới")
    
    return {
        'has_news': True,
        'explanation': ' | '.join(explanations),
        'impact_level': impact_level,
        'sentiment': float(avg_sentiment),
        'impact_score': float(avg_impact),
        'positive_ratio': float(positive_ratio),
        'news_activity': float(news_activity)
    }


def calculate_period_expected_return(returns, weights, start_date, end_date):
    """
    Tính lợi nhuận kỳ vọng từ start_date đến end_date cho portfolio
    
    Args:
        returns: Returns DataFrame (có thể có hoặc không có Date index)
        weights: Portfolio weights array
        start_date: Ngày bắt đầu (string 'YYYY-MM-DD')
        end_date: Ngày kết thúc (string 'YYYY-MM-DD')
    
    Returns:
        Dict với expected_return_period (%), num_trading_days, period_days
    """
    from datetime import datetime, timedelta
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Tính số ngày calendar
        period_days = (end_dt - start_dt).days
        if period_days <= 0:
            period_days = 1
        
        # Ước tính số trading days: ~70% của calendar days (bỏ qua weekends và holidays)
        # VN stock market: Thứ 2-6, bỏ qua holidays
        num_trading_days = max(1, int(period_days * 0.7))
        
        # Tính daily mean return từ historical data
        daily_mean_return = returns.mean().values
        portfolio_daily_return = np.dot(weights, daily_mean_return)
        
        # Expected return cho period = daily_return * num_trading_days
        expected_return_period = portfolio_daily_return * num_trading_days
        
        # Annualized return để so sánh
        if num_trading_days > 0:
            annualized_return = portfolio_daily_return * TRADING_DAYS_PER_YEAR
        else:
            annualized_return = 0.0
        
        return {
            'expected_return_period': safe_float(expected_return_period * 100, 0.0),  # %
            'annualized_return': safe_float(annualized_return * 100, 0.0),  # %
            'num_trading_days': num_trading_days,
            'period_days': period_days,
            'daily_return': safe_float(portfolio_daily_return * 100, 0.0)  # %
        }
    except Exception as e:
        print(f"⚠️  Lỗi khi tính period return: {e}")
        return {
            'expected_return_period': 0.0,
            'annualized_return': 0.0,
            'num_trading_days': 0,
            'period_days': 0,
            'daily_return': 0.0
        }


def calculate_individual_period_return(returns, ticker, start_date, end_date):
    """
    Tính lợi nhuận kỳ vọng cho một mã cổ phiếu cụ thể từ start_date đến end_date
    
    Args:
        returns: Returns DataFrame
        ticker: Ticker symbol
        start_date: Ngày bắt đầu (string 'YYYY-MM-DD')
        end_date: Ngày kết thúc (string 'YYYY-MM-DD')
    
    Returns:
        Expected return cho period (%)
    """
    from datetime import datetime
    
    try:
        if ticker not in returns.columns:
            return 0.0
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        period_days = (end_dt - start_dt).days
        if period_days <= 0:
            period_days = 1
        
        # Ước tính số trading days
        num_trading_days = max(1, int(period_days * 0.7))
        
        # Tính daily mean return cho ticker này
        ticker_daily_return = returns[ticker].mean()
        
        # Expected return cho period
        expected_return_period = ticker_daily_return * num_trading_days
        
        return safe_float(expected_return_period * 100, 0.0)  # %
    except Exception as e:
        return 0.0


def generate_reasoning(selected_tickers, current_weights, recommended_weights, 
                       returns, prices, cov_matrix, daily_mean, news_features=None, 
                       start_date=None, end_date=None):
    """
    Generate reasoning for each recommendation based on stock metrics and news
    
    Args:
        selected_tickers: List of stock tickers
        current_weights: Current portfolio weights
        recommended_weights: Recommended portfolio weights
        returns: Returns DataFrame
        prices: Prices DataFrame
        cov_matrix: Covariance matrix
        daily_mean: Mean returns array
        news_features: numpy array với news features (optional)
    
    Returns:
        List of reasoning dictionaries for each stock
    """
    reasons = []
    
    # Calculate individual stock metrics
    # Annualize returns: use simple multiplication for daily returns
    # If daily return > 5% (0.05), it's likely an outlier or data issue
    # For realistic daily returns (< 5%), simple multiplication is accurate enough
    individual_returns = daily_mean * TRADING_DAYS_PER_YEAR  # Annualized (simple)
    
    # Flag unrealistic returns (likely data issues) - > 5% per day is unrealistic
    
    individual_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(TRADING_DAYS_PER_YEAR)  # Annualized
    individual_sharpes = individual_returns / (individual_vols + 1e-8)
    
    # Calculate correlation with current portfolio
    current_portfolio_returns = returns @ current_weights
    correlations = []
    for ticker in selected_tickers:
        stock_returns = returns[ticker]
        corr = stock_returns.corr(current_portfolio_returns)
        correlations.append(corr if not np.isnan(corr) else 0.0)
    
    # Calculate contribution to portfolio risk
    portfolio_vol = np.sqrt(current_weights @ cov_matrix @ current_weights) * np.sqrt(TRADING_DAYS_PER_YEAR)
    risk_contributions = []
    for i, ticker in enumerate(selected_tickers):
        # Marginal contribution to risk
        marginal_risk = (cov_matrix[i] @ current_weights) / (portfolio_vol + 1e-8)
        risk_contributions.append(marginal_risk * current_weights[i])
    
    # Calculate momentum (recent performance)
    lookback = min(20, len(returns))
    recent_returns = returns.iloc[-lookback:]
    recent_mean = recent_returns.mean().values
    # Annualize momentum using simple multiplication (realistic for daily returns)
    momentum = recent_mean * TRADING_DAYS_PER_YEAR
    
    # Calculate trend (positive vs negative days)
    positive_days = (recent_returns > 0).sum().values
    trend_ratio = positive_days / lookback
    
    for i, ticker in enumerate(selected_tickers):
        delta = recommended_weights[i] - current_weights[i]
        
        # Build reasoning
        reason_parts = []
        
        if abs(delta) < 0.02:
            reason_parts.append("Phân bổ hiện tại đã tối ưu cho mã này.")
        else:
            # Return-based reasoning
            stock_return = individual_returns[i] * 100
            avg_return = np.mean(individual_returns) * 100
            
            # Cap unrealistic returns for display (if > 100%, likely data issue)
            if stock_return > 100:
                stock_return_display = 100
                reason_parts.append(f"⚠️ Lợi nhuận kỳ vọng rất cao (có thể do dữ liệu bất thường)")
            elif stock_return > 50:
                stock_return_display = min(stock_return, 100)
                reason_parts.append(f"Lợi nhuận kỳ vọng rất cao ({stock_return_display:.2f}%/năm)")
            else:
                stock_return_display = stock_return
            
            if delta > 0.02:  # Increase
                if stock_return > avg_return * 1.1 and stock_return <= 100:
                    reason_parts.append(f"Lợi nhuận kỳ vọng cao ({stock_return_display:.2f}%/năm, cao hơn trung bình {min(avg_return, 100):.2f}%/năm)")
                elif stock_return > 0 and stock_return <= 100:
                    reason_parts.append(f"Lợi nhuận kỳ vọng tích cực ({stock_return_display:.2f}%/năm)")
                
                # Sharpe ratio reasoning
                stock_sharpe = individual_sharpes[i]
                avg_sharpe = np.mean(individual_sharpes)
                if stock_sharpe > avg_sharpe * 1.2:
                    reason_parts.append(f"Sharpe ratio tốt ({stock_sharpe:.2f}, cao hơn trung bình {avg_sharpe:.2f})")
                
                # Momentum reasoning
                stock_momentum = momentum[i] * 100
                if stock_momentum > 5:
                    reason_parts.append(f"Xu hướng tăng mạnh gần đây ({stock_momentum:.2f}%/năm)")
                elif stock_momentum > 0:
                    reason_parts.append(f"Xu hướng tăng tích cực ({stock_momentum:.2f}%/năm)")
                
                # Correlation reasoning (diversification)
                stock_corr = correlations[i]
                if stock_corr < 0.5:
                    reason_parts.append(f"Tương quan thấp với portfolio ({stock_corr:.2f}), giúp đa dạng hóa rủi ro")
                elif stock_corr < 0.7:
                    reason_parts.append(f"Tương quan vừa phải ({stock_corr:.2f}), cân bằng rủi ro")
                
                # Volatility reasoning
                stock_vol = individual_vols[i] * 100
                avg_vol = np.mean(individual_vols) * 100
                if stock_vol < avg_vol * 0.8:
                    reason_parts.append(f"Độ biến động thấp ({stock_vol:.2f}% vs trung bình {avg_vol:.2f}%), rủi ro kiểm soát được")
                
                # Trend reasoning
                if trend_ratio[i] > 0.6:
                    reason_parts.append(f"Tỷ lệ ngày tăng cao ({trend_ratio[i]*100:.1f}%), xu hướng ổn định")
            
            else:  # Decrease
                stock_return_display = min(stock_return, 100) if stock_return > 0 else stock_return
                avg_return_display = min(avg_return, 100) if avg_return > 0 else avg_return
                if stock_return < avg_return * 0.9:
                    reason_parts.append(f"Lợi nhuận kỳ vọng thấp ({stock_return_display:.2f}%/năm, thấp hơn trung bình {avg_return_display:.2f}%/năm)")
        
        # Add period expected return explanation if dates provided (cho từng mã cổ phiếu)
        if start_date and end_date:
            ticker_period_return = calculate_individual_period_return(returns, ticker, start_date, end_date)
            from datetime import datetime
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                period_days = (end_dt - start_dt).days
                
                if abs(delta) > 0.02:
                    if delta > 0.02:  # Increase
                        if ticker_period_return > 2.0:
                            reason_parts.append(f"💰 Lợi nhuận kỳ vọng trong {period_days} ngày: +{ticker_period_return:.2f}% → Khuyến nghị tăng allocation để tận dụng cơ hội")
                        elif ticker_period_return > 0.5:
                            reason_parts.append(f"💰 Lợi nhuận kỳ vọng trong {period_days} ngày: +{ticker_period_return:.2f}% → Tiềm năng tích cực trong ngắn hạn")
                        elif ticker_period_return > 0:
                            reason_parts.append(f"💰 Lợi nhuận kỳ vọng trong {period_days} ngày: +{ticker_period_return:.2f}% → Có tiềm năng tăng trưởng")
                    else:  # Decrease
                        if ticker_period_return < -1.0:
                            reason_parts.append(f"💰 Lợi nhuận kỳ vọng trong {period_days} ngày: {ticker_period_return:.2f}% → Khuyến nghị giảm allocation để tránh rủi ro giảm giá")
                        elif ticker_period_return < 0:
                            reason_parts.append(f"💰 Lợi nhuận kỳ vọng trong {period_days} ngày: {ticker_period_return:.2f}% → Tiềm năng tiêu cực trong ngắn hạn")
                        elif ticker_period_return < 0.5:
                            reason_parts.append(f"💰 Lợi nhuận kỳ vọng trong {period_days} ngày: {ticker_period_return:.2f}% → Tiềm năng thấp, nên giảm allocation")
            except Exception:
                pass  # Skip nếu không parse được date
        
        # Add specific news articles explanation for this ticker
        ticker_news_explanations = generate_ticker_news_explanation(ticker, delta, days_back=7)
        if ticker_news_explanations:
            reason_parts.extend(ticker_news_explanations)
        
        # Add aggregate news impact explanation if available
        if news_features is not None and len(news_features) == 4:
            news_impact = generate_news_impact_explanation(news_features, selected_tickers, recommended_weights)
            if news_impact['has_news']:
                # Add aggregate news impact summary (nếu chưa có news cụ thể)
                if not ticker_news_explanations:
                    if news_impact['impact_level'] == 'positive' and delta > 0.02:
                        reason_parts.append(f"📰 News tích cực tổng thể: {news_impact['explanation']}")
                    elif news_impact['impact_level'] == 'negative' and delta < -0.02:
                        reason_parts.append(f"📰 News tiêu cực tổng thể: {news_impact['explanation']}")
                    elif abs(delta) > 0.02:
                        reason_parts.append(f"📰 News impact tổng thể: {news_impact['explanation']}")
                
                # Sharpe ratio reasoning
                stock_sharpe = individual_sharpes[i]
                avg_sharpe = np.mean(individual_sharpes)
                if stock_sharpe < avg_sharpe * 0.8:
                    reason_parts.append(f"Sharpe ratio thấp ({stock_sharpe:.2f}, thấp hơn trung bình {avg_sharpe:.2f})")
                
                # Momentum reasoning
                stock_momentum = momentum[i] * 100
                if stock_momentum < -5:
                    reason_parts.append(f"Xu hướng giảm gần đây ({stock_momentum:.2f}%/năm)")
                elif stock_momentum < 0:
                    reason_parts.append(f"Xu hướng yếu ({stock_momentum:.2f}%/năm)")
                
                # Correlation reasoning (over-concentration)
                stock_corr = correlations[i]
                if stock_corr > 0.8:
                    reason_parts.append(f"Tương quan cao với portfolio ({stock_corr:.2f}), tập trung rủi ro")
                
                # Volatility reasoning
                stock_vol = individual_vols[i] * 100
                avg_vol = np.mean(individual_vols) * 100
                if stock_vol > avg_vol * 1.2:
                    reason_parts.append(f"Độ biến động cao ({stock_vol:.2f}% vs trung bình {avg_vol:.2f}%), rủi ro lớn")
                
                # Risk contribution reasoning
                risk_contrib = risk_contributions[i]
                if risk_contrib > 0.25:  # Contributing more than 25% to portfolio risk
                    reason_parts.append(f"Đóng góp rủi ro lớn ({risk_contrib*100:.1f}%), cần giảm để cân bằng")
                
                # Trend reasoning
                if trend_ratio[i] < 0.4:
                    reason_parts.append(f"Tỷ lệ ngày tăng thấp ({trend_ratio[i]*100:.1f}%), xu hướng yếu")
        
        # If no specific reasons found, provide general reasoning
        if not reason_parts:
            if delta > 0.02:
                reason_parts.append("Để tối ưu hóa lợi nhuận và cân bằng rủi ro của portfolio")
            elif delta < -0.02:
                reason_parts.append("Để giảm rủi ro tập trung và cải thiện đa dạng hóa")
            else:
                reason_parts.append("Phân bổ hiện tại đã phù hợp")
        
        # Cap unrealistic returns for display (if > 100%, likely data issue)
        expected_return_display = min(individual_returns[i] * 100, 100) if individual_returns[i] * 100 > 0 else individual_returns[i] * 100
        momentum_display = min(momentum[i] * 100, 100) if momentum[i] * 100 > 0 else momentum[i] * 100
        
        # Add news metrics if available
        metrics_dict = {
            'expected_return': safe_float(expected_return_display, 0.0),  # Annualized
            'volatility': safe_float(individual_vols[i] * 100, 0.0),
            'sharpe_ratio': safe_float(individual_sharpes[i], 0.0),
            'correlation': safe_float(correlations[i], 0.0),
            'momentum': safe_float(momentum_display, 0.0),
            'trend_ratio': safe_float(trend_ratio[i] * 100, 0.0)
        }
        
        # Add period expected return for this individual stock if dates provided
        if start_date and end_date:
            ticker_period_return = calculate_individual_period_return(returns, ticker, start_date, end_date)
            metrics_dict['period_expected_return'] = ticker_period_return
        
        # Add news metrics if available
        if news_features is not None and len(news_features) == 4:
            news_impact = generate_news_impact_explanation(news_features, selected_tickers, recommended_weights)
            if news_impact['has_news']:
                metrics_dict['news_sentiment'] = safe_float(news_impact['sentiment'], 0.0)
                metrics_dict['news_impact_score'] = safe_float(news_impact['impact_score'], 0.0)
                metrics_dict['news_positive_ratio'] = safe_float(news_impact['positive_ratio'], 0.5)
                metrics_dict['news_activity'] = safe_float(news_impact['news_activity'], 0.0)
        
        reasons.append({
            'ticker': ticker,
            'reasons': reason_parts,
            'metrics': metrics_dict
        })
    
    return reasons


def generate_recommendation(selected_tickers, capital_amounts, start_date, end_date):
    """Generate portfolio recommendation"""
    
    # Load model
    actor, info = load_trained_model()
    if actor is None or info is None:
        return {
            'error': 'Model not found. Please run train_model.py first.'
        }
    
    # IMPORTANT: Use historical data (2 years) for accurate return estimation
    # The start_date/end_date from user is for investment period, not for data analysis
    # We need longer historical data to estimate expected returns accurately
    from datetime import datetime, timedelta
    today = datetime.now()
    historical_end = today.strftime('%Y-%m-%d')
    historical_start = (today - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
    
    # Load data từ CSV (KHÔNG DOWNLOAD) - use 2 years for accurate metrics
    # Use CSV data source (vn_stocks_data_2020_2025.csv)
    prices, returns, ohlcv = download_stock_data(
        selected_tickers,
        historical_start,  # Use historical period, not user's investment period
        historical_end,
        use_cache=True,
        cache_manager=None,
        data_source='csv'  # Chỉ đọc từ CSV Data_test.csv
    )
    
    # Check if data was loaded successfully
    if prices is None or returns is None:
        return {
            'error': 'Không thể tải dữ liệu từ file Data_test.csv. Vui lòng kiểm tra file và đảm bảo có dữ liệu cho các mã đã chọn.'
        }
    
    # Filter for selected tickers (only if they exist in data)
    available_tickers = [t for t in selected_tickers if t in prices.columns]
    missing_tickers = [t for t in selected_tickers if t not in prices.columns]
    
    # Nếu có mã thiếu, chỉ báo lỗi (KHÔNG TẢI TỪ NGUỒN KHÁC)
    if missing_tickers:
        if len(available_tickers) < 2:
            return {
                'error': f'Không đủ dữ liệu (cần ít nhất 2 mã, chỉ có {len(available_tickers)} mã trong CSV). Các mã thiếu: {", ".join(missing_tickers)}. Vui lòng chọn các mã khác có trong file Data_test.csv.',
                'available_tickers': available_tickers,
                'missing_tickers': missing_tickers
            }
    
    # Đảm bảo chỉ lấy các mã có dữ liệu
    available_tickers = [t for t in selected_tickers if t in prices.columns]
    prices = prices[available_tickers]
    returns = returns[available_tickers]
    
    # Filter OHLCV for available tickers
    if ohlcv:
        for key in ohlcv:
            if key in ohlcv and ohlcv[key] is not None:
                available_ohlcv_cols = [t for t in available_tickers if t in ohlcv[key].columns]
                if available_ohlcv_cols:
                    ohlcv[key] = ohlcv[key][available_ohlcv_cols]
    
    
    # Convert capital to weights (use latest prices) - chỉ với các mã có dữ liệu
    # Đảm bảo capital_amounts tương ứng với available_tickers
    available_capitals = []
    for ticker in available_tickers:
        idx = selected_tickers.index(ticker)
        available_capitals.append(capital_amounts[idx])
    
    current_weights = compute_weights_from_capital(available_tickers, available_capitals, prices)
    total_capital = sum(available_capitals)
    
    # IPO Agent - learn risk preference (using historical data for accuracy)
    ipo_agent = IPOAgent(n_stocks=len(available_tickers))
    cov_matrix = returns.cov().values
    
    # Check for NaN in covariance matrix
    if np.any(np.isnan(cov_matrix)):
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    target_return = ipo_agent.learn_risk_preference(current_weights, returns, cov_matrix, prices=prices)
    target_return = safe_float(target_return, 0.1)  # Default 10% if invalid
    
    # Train RL model with available stocks (with OHLCV)
    # We train a new model for the available stocks using historical data
    from robo_agent import extract_state_features
    agent, history = train_robo_advisor(
        returns,
        n_episodes=150,  # Increased for better accuracy
        stock_code=None,
        cache_manager=None,
        prices=prices,
        ohlcv=ohlcv
    )
    
    # NEW: Scrape và load news features cho các mã cổ phiếu
    news_features = None
    try:
        from news_scraper import scrape_news_for_tickers, save_news_to_database
        from news_features import load_news_features_from_db, aggregate_news_features
        from datetime import datetime
        
        # Filter bỏ chỉ số (^VNINDEX) vì không cần news cho chỉ số
        tickers_for_news = [t for t in available_tickers if not t.startswith('^')]
        
        if tickers_for_news:
            print(f"📰 Đang lấy news cho {len(tickers_for_news)} mã cổ phiếu...")
            
            # Scrape news cho các mã (ngày hiện tại)
            today_str = datetime.now().strftime('%Y-%m-%d')
            all_news = scrape_news_for_tickers(tickers_for_news, target_date=today_str, days_back=7)
            
            # Lưu news vào database
            news_saved_count = 0
            for ticker, news_list in all_news.items():
                if news_list:
                    save_news_to_database(news_list, ticker)
                    news_saved_count += len(news_list)
            
            if news_saved_count > 0:
                print(f"✅ Đã lưu {news_saved_count} tin tức vào database")
            
            # Load news features từ database
            news_features_dict = load_news_features_from_db(tickers_for_news, target_date=today_str, days_back=7)
            market_news_features = aggregate_news_features(news_features_dict, tickers_for_news, target_date=today_str)
            
            print(f"✅ Đã load news features: sentiment={market_news_features[0]:.3f}, impact={market_news_features[1]:.3f}, positive_ratio={market_news_features[2]:.3f}, activity={market_news_features[3]:.3f}")
            
            news_features = market_news_features
        else:
            print("⚠️  Không có mã cổ phiếu để lấy news (chỉ có chỉ số)")
            news_features = None
        
    except ImportError as e:
        print(f"⚠️  Không thể import news modules: {e}")
        print("   → Tiếp tục prediction không có news features")
        news_features = None
    except Exception as e:
        print(f"⚠️  Lỗi khi lấy news: {e}")
        print("   → Tiếp tục prediction không có news features")
        news_features = None
    
    # Get recommendation from trained model with improved state (with OHLCV + News)
    state = extract_state_features(returns, prices, ohlcv=ohlcv, news_features=news_features)
    
    # Check for NaN in state
    if np.any(np.isnan(state)):
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    
    recommended_weights = agent.select_action(state, explore=False)
    
    # Ensure recommended_weights are valid
    if recommended_weights is None or len(recommended_weights) != len(available_tickers):
        return {
            'error': 'Không thể tạo đề xuất phân bổ. Vui lòng thử lại.'
        }
    
    # Normalize weights and check for NaN
    recommended_weights = np.array(recommended_weights)
    
    if np.any(np.isnan(recommended_weights)):
        recommended_weights = np.ones(len(available_tickers)) / len(available_tickers)
    else:
        # Normalize to sum to 1
        weight_sum = np.sum(recommended_weights)
        if weight_sum > 0:
            recommended_weights = recommended_weights / weight_sum
        else:
            recommended_weights = np.ones(len(available_tickers)) / len(available_tickers)
    
    # Tránh phân bổ dồn hết vào 1 mã (biểu đồ 1 màu)
    # - Đặt floor nhỏ cho tất cả mã
    # - Giới hạn trần cho 1 mã
    floor = 1e-3
    cap = 0.7
    recommended_weights = np.maximum(recommended_weights, floor)
    # Nếu có mã vượt trần, hạ xuống cap và phân bổ phần dư đều
    if recommended_weights.max() > cap:
        excess = np.maximum(recommended_weights - cap, 0.0)
        recommended_weights = np.minimum(recommended_weights, cap)
        # Phân bổ phần dư cho các mã còn lại (không vượt trần)
        remain_mask = recommended_weights < cap - 1e-9
        remain_sum = recommended_weights[remain_mask].sum()
        if remain_sum > 0 and excess.sum() > 0:
            recommended_weights[remain_mask] += excess.sum() * (recommended_weights[remain_mask] / remain_sum)
    # Chuẩn hóa lại
    recommended_weights = recommended_weights / recommended_weights.sum()
    
    # Calculate metrics
    if len(returns) == 0:
        return {
            'error': 'Không có dữ liệu returns để tính toán. Vui lòng kiểm tra dữ liệu.'
        }
    
    daily_mean = returns.mean().values
    daily_cov = returns.cov().values
    
    # Check for NaN in covariance matrix and mean returns
    if np.any(np.isnan(daily_cov)):
        daily_cov = np.nan_to_num(daily_cov, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(daily_mean)):
        daily_mean = np.nan_to_num(daily_mean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Annualize returns using simple multiplication (realistic for daily returns)
    # For realistic daily returns (< 5%), simple multiplication is accurate
    current_return = (current_weights @ daily_mean) * TRADING_DAYS_PER_YEAR
    current_vol = np.sqrt(current_weights @ daily_cov @ current_weights) * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Sharpe ratio với risk-free rate: Sharpe = (R - Rf) / σ
    current_sharpe = safe_float((current_return - RISK_FREE_RATE_ANNUAL) / (current_vol + 1e-8), 0.0)
    
    expected_return = (recommended_weights @ daily_mean) * TRADING_DAYS_PER_YEAR
    portfolio_vol = np.sqrt(recommended_weights @ daily_cov @ recommended_weights) * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Sharpe ratio với risk-free rate: Sharpe = (R - Rf) / σ
    sharpe = safe_float((expected_return - RISK_FREE_RATE_ANNUAL) / (portfolio_vol + 1e-8), 0.0)
    
    recommended_capitals = recommended_weights * total_capital
    
    # Calculate Turnover Rate (chỉ với available_tickers)
    # Công thức đúng: Turnover rate = (1/(N-1)) * Σ(i=0 to N-2) Σ(j=1 to n) |v^(j)_(i+1) - v^(j)_i| / W_(i+1) * 100%
    # Where:
    #   N = number of time periods
    #   n = number of stocks
    #   v^(j)_(i+1) = dollar value invested in stock j at time i+1
    #   v^(j)_(i) = dollar value invested in stock j at time i
    #   W_(i+1) = total portfolio value at time i+1
    # 
    # For our case (single transition from current to recommended):
    #   N = 2, so (N-1) = 1
    #   i = 0 (only one transition)
    #   Turnover rate = (1/1) * Σ(j=1 to n) |v^(j)_(1) - v^(j)_(0)| / W_(1) * 100%
    #   = Σ(j=1 to n) |recommended_capital[j] - current_capital[j]| / total_capital * 100%
    #   = Σ(j=1 to n) |recommended_weight[j] - current_weight[j]| * 100%
    weight_changes = np.abs(recommended_weights - current_weights)
    turnover_rate = np.sum(weight_changes)  # Sum of absolute weight changes
    turnover_rate = safe_float(turnover_rate * 100, 0.0)  # Convert to percentage
    
    # Calculate Transaction Cost
    # Công thức: Transaction cost = (c / W_0) * Σ(i=0 to N-2) Σ(j=1 to n) |v^(j)_(i+1) - v^(j)_i| * 100%
    # Where:
    #   c = transaction cost rate (0.15% = 0.0015)
    #   W_0 = initial portfolio value (total_capital)
    # For our case (single transition):
    #   Transaction cost = (c / W_0) * Σ(j=1 to n) |v^(j)_(1) - v^(j)_(0)| * 100%
    #   = (c / total_capital) * Σ(j=1 to n) |recommended_capital[j] - current_capital[j]| * 100%
    #   = c * Σ(j=1 to n) |recommended_weight[j] - current_weight[j]| * 100%
    capital_changes = np.abs(recommended_capitals - available_capitals)
    transaction_cost = TRANSACTION_COST_RATE * np.sum(capital_changes) / total_capital * 100
    transaction_cost = safe_float(transaction_cost, 0.0)  # Already in percentage
    
    # Calculate Cumulative Transaction Cost (for this single rebalance, same as transaction_cost)
    cumulative_transaction_cost = transaction_cost
    
    # Calculate Maximum Drawdown (MDD) for current and recommended portfolios
    def calculate_mdd(weights, returns_data, prices_data):
        """
        Calculate Maximum Drawdown (MDD) for a portfolio
        
        MDD = max over t of [(Peak_t - Value_t) / Peak_t]
        """
        if prices_data is None or len(prices_data) == 0:
            return 0.0
        
        try:
            # Calculate portfolio value over time
            portfolio_values = (prices_data * weights).sum(axis=1)
            
            # Calculate cumulative returns
            portfolio_returns = portfolio_values.pct_change().fillna(0)
            cumulative = (1 + portfolio_returns).cumprod()
            
            # Calculate running maximum (peak)
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / (running_max + 1e-8)
            
            # Maximum drawdown (most negative value)
            mdd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            return safe_float(mdd * 100, 0.0)  # Convert to percentage
        except Exception:
            return 0.0
    
    # Calculate MDD for current portfolio
    current_mdd = calculate_mdd(current_weights, returns, prices)
    
    # Calculate MDD for recommended portfolio (simulated)
    # Note: We use historical data to estimate MDD for recommended portfolio
    recommended_mdd = calculate_mdd(recommended_weights, returns, prices)
    
    # Calculate period expected return (từ hôm nay đến ngày user chọn)
    period_return_info = None
    if start_date and end_date:
        try:
            period_return_info = calculate_period_expected_return(returns, recommended_weights, start_date, end_date)
        except Exception as e:
            print(f"⚠️  Lỗi khi tính period return: {e}")
            period_return_info = None
    
    # Classify risk level
    risk_tolerance = ipo_agent.risk_tolerance
    if risk_tolerance < 0.7:
        risk_level = "Conservative (Bảo thủ)"
    elif risk_tolerance < 1.3:
        risk_level = "Balanced (Cân bằng)"
    else:
        risk_level = "Aggressive (Mạo hiểm)"
    
    # Generate reasoning for recommendations (sử dụng available_tickers) - với news features và period return
    reasoning = generate_reasoning(
        available_tickers, 
        current_weights, 
        recommended_weights,
        returns,
        prices,
        cov_matrix,
        daily_mean,
        news_features=news_features,  # Pass news features để giải thích impact
        start_date=start_date,  # Pass dates để tính period return
        end_date=end_date
    )
    
    # Create a mapping from ticker to reasoning
    reasoning_map = {r['ticker']: r for r in reasoning}
    
    # Prepare results with reasoning (chỉ với available_tickers)
    results = []
    for ticker, curr_w, rec_w, curr_cap, rec_cap in zip(
        available_tickers, current_weights, recommended_weights,
        available_capitals, recommended_capitals
    ):
        delta = rec_w - curr_w
        action = "Increase" if delta > 0.02 else ("Decrease" if delta < -0.02 else "Keep")
        
        # Get reasoning for this ticker
        ticker_reasoning = reasoning_map.get(ticker, {'reasons': [], 'metrics': {}})
        
        rec_w_safe = safe_float(rec_w, 0.0)
        curr_w_safe = safe_float(curr_w, 0.0)
        
        results.append({
            'ticker': ticker,
            'current_weight': curr_w_safe,
            'recommended_weight': rec_w_safe,
            'current_capital': safe_float(curr_cap, 0.0),
            'recommended_capital': safe_float(rec_cap, 0.0),
            'change': safe_float(delta, 0.0),
            'action': action if action else 'Keep',
            'reasons': ticker_reasoning.get('reasons', []),
            'metrics': ticker_reasoning.get('metrics', {})
        })
    
    return {
        'success': True,
        'results': results,
        'metrics': {
            'current_return': safe_float(current_return * 100, 0.0),  # Annualized
            'recommended_return': safe_float(expected_return * 100, 0.0),  # Annualized
            'current_volatility': safe_float(current_vol * 100, 0.0),
            'recommended_volatility': safe_float(portfolio_vol * 100, 0.0),
            'current_sharpe': safe_float(current_sharpe, 0.0),
            'recommended_sharpe': safe_float(sharpe, 0.0),
            'improvement_return': safe_float((expected_return - current_return) * 100, 0.0),
            'improvement_sharpe': safe_float(sharpe - current_sharpe, 0.0),
            'turnover_rate': safe_float(turnover_rate, 0.0),  # Already in percentage
            'transaction_cost': safe_float(transaction_cost, 0.0),  # Transaction cost in percentage
            'cumulative_transaction_cost': safe_float(cumulative_transaction_cost, 0.0),  # Cumulative transaction cost in percentage
            'current_mdd': safe_float(current_mdd, 0.0),  # Maximum Drawdown for current portfolio (%)
            'recommended_mdd': safe_float(recommended_mdd, 0.0),  # Maximum Drawdown for recommended portfolio (%)
            # Period expected return (từ hôm nay đến ngày user chọn)
            'period_expected_return': period_return_info['expected_return_period'] if period_return_info else None,
            'period_days': period_return_info['period_days'] if period_return_info else None,
            'period_trading_days': period_return_info['num_trading_days'] if period_return_info else None,
            'period_daily_return': period_return_info['daily_return'] if period_return_info else None
        },
        'risk_profile': {
            'risk_tolerance': safe_float(risk_tolerance, 1.0),
            'risk_level': risk_level if risk_level else 'Balanced (Cân bằng)',
            'target_return': safe_float(target_return * 100, 10.0)
        },
        'total_capital': safe_float(total_capital, 0.0),
        'news_info': generate_news_impact_explanation(news_features, available_tickers, recommended_weights) if news_features is not None else {
            'has_news': False,
            'explanation': 'Chưa có dữ liệu news để phân tích',
            'impact_level': 'neutral'
        }
    }


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', stocks=AVAILABLE_STOCKS)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze portfolio"""
    try:
        data = request.json
        
        # Get selected stocks (5 stocks)
        selected_tickers = data.get('stocks', [])
        if len(selected_tickers) != 5:
            return jsonify({'error': 'Please select exactly 5 stocks'}), 400
        
        # Get capital amounts
        capital_amounts = [float(data.get(f'capital_{i}', 0)) for i in range(5)]
        
        # Get date range
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        
        if not start_date or not end_date:
            # Default: today to 10 days later
            today = datetime.now()
            start_date = today.strftime('%Y-%m-%d')
            end_date = (today + timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Validate date range: start_date must be today or later, end_date max 10 days from start
        today = datetime.now().date()
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Start date should be today or later
        if start_dt < today:
            return jsonify({'error': 'Ngày bắt đầu phải là hôm nay hoặc sau đó'}), 400
        
        # End date should be max 10 days from start
        max_end = start_dt + timedelta(days=10)
        if end_dt > max_end:
            return jsonify({'error': 'Ngày kết thúc không được quá 10 ngày từ ngày bắt đầu'}), 400
        
        if end_dt <= start_dt:
            return jsonify({'error': 'Ngày kết thúc phải sau ngày bắt đầu'}), 400
        
        # Generate recommendation
        result = generate_recommendation(selected_tickers, capital_amounts, start_date, end_date)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check_model', methods=['GET'])
def check_model():
    """Check if model is loaded"""
    # Check for model files first
    possible_paths = [
        Path('models') / 'trained_model.pth',
        Path('models') / 'actor_model.pth'
    ]
    
    model_exists = any(path.exists() for path in possible_paths)
    existing_paths = [str(p) for p in possible_paths if p.exists()]
    
    if not model_exists:
        return jsonify({
            'loaded': False, 
            'message': 'Model file not found. Please run train_model.py first.',
            'checked_paths': [str(p) for p in possible_paths]
        })
    
    actor, info = load_trained_model()
    if actor is None or info is None:
        return jsonify({
            'loaded': False, 
            'message': 'Model file exists but could not be loaded. Please check the model file format.',
            'existing_files': existing_paths
        })
    
    return jsonify({
        'loaded': True,
        'message': 'Model loaded successfully',
        'n_stocks': safe_float(info.get('n_stocks', 0), 0),
        'stocks': info.get('stock_names', []),
        'model_files': existing_paths
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
        ]
    }
    
    Response JSON:
    {
        "success": true,
        "data": {
            "recommended_weights": {...},
            "recommended_capitals": {...},
            ...
        }
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
        
        # Default dates: today to 10 days later
        today = datetime.now()
        start_date = today.strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Generate recommendation
        result = generate_recommendation(selected_tickers, capital_amounts, start_date, end_date)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        # Format response
        response = {
            'success': True,
            'data': {
                'recommended_weights': result.get('recommended_weights', {}),
                'recommended_capitals': result.get('recommended_capitals', {}),
                'current_weights': result.get('current_weights', {}),
                'current_capitals': result.get('current_capitals', {}),
                'metrics': result.get('metrics', {}),
                'tickers': result.get('tickers', []),
                'reasoning': result.get('reasoning', ''),
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/agent/analyze', methods=['POST'])
def agent_analyze():
    """
    API endpoint cho AI Agent - Format request như hình 2
    
    Request JSON:
    {
        "investing": [
            {"ticker": "ACB", "amount": 1000000},
            {"ticker": "BCM", "amount": 2000000},
            {"ticker": "BID", "amount": 1500000},
            {"ticker": "CTG", "amount": 1200000},
            {"ticker": "DGC", "amount": 800000}
        ]
    }
    
    Response JSON:
    {
        "success": true,
        "recommended_weights": {...},
        "recommended_capitals": {...},
        "metrics": {...},
        "tickers": [...]
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'Missing request body'}), 400
        
        # Parse request format từ hình 2
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
        
        # Default dates: today to 10 days later
        today = datetime.now()
        start_date = today.strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=10)).strftime('%Y-%m-%d')
        
        # Generate recommendation
        result = generate_recommendation(selected_tickers, capital_amounts, start_date, end_date)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        # Format response
        response = {
            'success': True,
            'recommended_weights': result.get('recommended_weights', {}),
            'recommended_capitals': result.get('recommended_capitals', {}),
            'current_weights': result.get('current_weights', {}),
            'current_capitals': result.get('current_capitals', {}),
            'metrics': result.get('metrics', {}),
            'tickers': result.get('tickers', []),
            'reasoning': result.get('reasoning', ''),
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/portfolio_optimize', methods=['POST'])
def portfolio_optimize():
    """
    API endpoint để nhận input từ FE và trả về kết quả tối ưu hóa portfolio
    
    Request JSON:
    {
        "stocks": ["ACB", "BCM", "BID", "CTG", "DGC"],
        "capitals": [1000000, 2000000, 1500000, 1200000, 800000],
        "start_date": "2022-01-01",
        "end_date": "2024-01-01",
        "n_episodes": 150
    }
    
    Response JSON:
    {
        "success": true,
        "risk_level": "Balanced (Cân bằng)",
        "risk_tolerance": 1.0,
        "target_return": 0.1,
        "current_weights": {"ACB": 0.15, ...},
        "recommended_weights": {"ACB": 0.12, ...},
        "current_capitals": {"ACB": 1000000, ...},
        "recommended_capitals": {"ACB": 780000, ...},
        "current_return": 0.125,
        "recommended_return": 0.152,
        "current_volatility": 0.183,
        "recommended_volatility": 0.168,
        "current_sharpe": 0.68,
        "recommended_sharpe": 0.75,
        "improvement_return": 0.027,
        "improvement_sharpe": 0.07
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'Missing request body'}), 400
        
        # Validate required fields
        selected_tickers = data.get('stocks', [])
        capital_amounts = data.get('capitals', [])
        
        if not selected_tickers or not capital_amounts:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: "stocks" and "capitals"'
            }), 400
        
        if len(selected_tickers) != len(capital_amounts):
            return jsonify({
                'success': False,
                'error': f'Length mismatch: stocks ({len(selected_tickers)}) != capitals ({len(capital_amounts)})'
            }), 400
        
        if len(selected_tickers) < 2:
            return jsonify({
                'success': False,
                'error': 'Need at least 2 stocks'
            }), 400
        
        # Get optional fields
        start_date = data.get('start_date', '')
        end_date = data.get('end_date', '')
        n_episodes = data.get('n_episodes', 150)
        
        # Default dates: last 2 years
        if not start_date or not end_date:
            today = datetime.now()
            end_date = today.strftime('%Y-%m-%d')
            start_date = (today - timedelta(days=730)).strftime('%Y-%m-%d')
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if end_dt <= start_dt:
                return jsonify({
                    'success': False,
                    'error': 'end_date must be after start_date'
                }), 400
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid date format: {str(e)}. Use YYYY-MM-DD'
            }), 400
        
        # Validate capitals are numbers
        try:
            capital_amounts = [float(c) for c in capital_amounts]
            if any(c < 0 for c in capital_amounts):
                return jsonify({
                    'success': False,
                    'error': 'Capitals cannot be negative'
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Capitals must be numbers'
            }), 400
        
        # Call generate_recommendation from main.py (hàm tính toán chính)
        if generate_recommendation_main is None:
            return jsonify({
                'success': False,
                'error': 'Cannot import generate_recommendation from main.py'
            }), 500
        
        result = generate_recommendation_main(
            selected_tickers,
            capital_amounts,
            start_date,
            end_date,
            n_episodes=int(n_episodes)
        )
        
        # Format response
        response = {
            'success': True,
            'risk_level': result.get('risk_level', 'Unknown'),
            'risk_tolerance': safe_float(result.get('risk_tolerance', 1.0), 1.0),
            'target_return': safe_float(result.get('target_return', 0.1), 0.1),
            'current_weights': result.get('current_weights', {}),
            'recommended_weights': result.get('recommended_weights', {}),
            'current_capitals': result.get('current_capitals', {}),
            'recommended_capitals': result.get('recommended_capitals', {}),
            'current_return': safe_float(result.get('current_return', 0.0), 0.0),
            'recommended_return': safe_float(result.get('recommended_return', 0.0), 0.0),
            'current_volatility': safe_float(result.get('current_volatility', 0.0), 0.0),
            'recommended_volatility': safe_float(result.get('recommended_volatility', 0.0), 0.0),
            'current_sharpe': safe_float(result.get('current_sharpe', 0.0), 0.0),
            'recommended_sharpe': safe_float(result.get('recommended_sharpe', 0.0), 0.0),
            'improvement_return': safe_float(result.get('improvement_return', 0.0), 0.0),
            'improvement_sharpe': safe_float(result.get('improvement_sharpe', 0.0), 0.0)
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

