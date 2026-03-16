"""
News Features Module - Load và xử lý news features để tích hợp vào DRL model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3


def load_news_features_from_db(tickers, target_date=None, days_back=7, db_path='data/news_data.db'):
    """
    Load news features từ database cho các tickers theo ngày cụ thể
    
    Args:
        tickers: List of tickers
        target_date: Ngày cụ thể để lấy news (datetime object hoặc string 'YYYY-MM-DD')
                     Nếu None thì lấy days_back ngày gần nhất
        days_back: Số ngày gần nhất để lấy news (chỉ dùng nếu target_date=None)
        db_path: Đường dẫn database
    
    Returns:
        dict với key là ticker, value là dict chứa features:
        {
            'avg_sentiment': float,
            'avg_impact_score': float,
            'positive_count': int,
            'negative_count': int,
            'total_news': int
        }
    """
    news_features = {}
    
    if not Path(db_path).exists():
        # Database chưa tồn tại, return default values
        for ticker in tickers:
            news_features[ticker] = {
                'avg_sentiment': 0.0,
                'avg_impact_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'total_news': 0
            }
        return news_features
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Xác định date range để query
        if target_date is not None:
            # Lấy news cho ngày cụ thể
            if isinstance(target_date, str):
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            else:
                target_dt = target_date
            
            # Lấy news trong khoảng ±1 ngày để có đủ context
            start_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Lấy news trong days_back ngày gần nhất
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for ticker in tickers:
            # Query news trong khoảng thời gian cụ thể
            # Date được lưu dạng 'YYYY-MM-DD' nên có thể so sánh trực tiếp
            cursor.execute('''
                SELECT sentiment_label, compound_score, impact_score, date
                FROM news_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date DESC
            ''', (ticker, start_date, end_date))
            
            results = cursor.fetchall()
            
            if not results:
                # Không có news, dùng default values
                news_features[ticker] = {
                    'avg_sentiment': 0.0,
                    'avg_impact_score': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'total_news': 0
                }
                continue
            
            # Tính toán features
            sentiments = [r[1] for r in results if r[1] is not None]
            impact_scores = [r[2] for r in results if r[2] is not None]
            sentiment_labels = [r[0] for r in results if r[0] is not None]
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            avg_impact_score = np.mean(impact_scores) if impact_scores else 0.0
            positive_count = sum(1 for label in sentiment_labels if label == 'positive')
            negative_count = sum(1 for label in sentiment_labels if label == 'negative')
            total_news = len(results)
            
            news_features[ticker] = {
                'avg_sentiment': float(avg_sentiment),
                'avg_impact_score': float(avg_impact_score),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_news': total_news
            }
        
        conn.close()
        
    except Exception as e:
        print(f"⚠️  Lỗi khi load news features: {e}")
        # Return default values
        for ticker in tickers:
            news_features[ticker] = {
                'avg_sentiment': 0.0,
                'avg_impact_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'total_news': 0
            }
    
    return news_features


def aggregate_news_features(news_features_dict, tickers, target_date=None):
    """
    Aggregate news features từ tất cả tickers thành market-level features
    
    Args:
        news_features_dict: Dict từ load_news_features_from_db
        tickers: List of tickers (để đảm bảo thứ tự)
        target_date: Ngày cụ thể (optional, để logging)
    
    Returns:
        numpy array với 4 features:
        [avg_market_sentiment, avg_market_impact, positive_ratio, news_activity]
    """
    if not news_features_dict:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    all_sentiments = []
    all_impacts = []
    total_positive = 0
    total_negative = 0
    total_news = 0
    
    for ticker in tickers:
        if ticker in news_features_dict:
            features = news_features_dict[ticker]
            if features['total_news'] > 0:
                all_sentiments.append(features['avg_sentiment'])
                all_impacts.append(features['avg_impact_score'])
                total_positive += features['positive_count']
                total_negative += features['negative_count']
                total_news += features['total_news']
    
    # Tính toán aggregate features
    avg_market_sentiment = np.mean(all_sentiments) if all_sentiments else 0.0
    avg_market_impact = np.mean(all_impacts) if all_impacts else 0.0
    
    # Positive ratio (tỷ lệ tin tích cực)
    total_sentiment_news = total_positive + total_negative
    positive_ratio = total_positive / total_sentiment_news if total_sentiment_news > 0 else 0.5  # Default neutral
    
    # News activity (số lượng tin tức normalized)
    # Normalize về [0, 1] với max = 100 news per ticker
    news_activity = min(1.0, total_news / (len(tickers) * 10))  # Max 10 news per ticker
    
    return np.array([
        float(avg_market_sentiment),
        float(avg_market_impact),
        float(positive_ratio),
        float(news_activity)
    ])


def get_ticker_specific_news_features(news_features_dict, ticker):
    """
    Lấy news features cho một ticker cụ thể
    
    Args:
        news_features_dict: Dict từ load_news_features_from_db
        ticker: Ticker symbol
    
    Returns:
        numpy array với 2 features: [avg_sentiment, avg_impact_score]
    """
    if ticker not in news_features_dict:
        return np.array([0.0, 0.0])
    
    features = news_features_dict[ticker]
    return np.array([
        features['avg_sentiment'],
        features['avg_impact_score']
    ])

