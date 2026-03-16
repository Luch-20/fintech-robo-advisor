"""
News Scraper - Tự động cào tin tức theo keyword từng mã cổ phiếu
và phân tích độ ảnh hưởng đến giá cổ phiếu
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import re
from pathlib import Path
import json

# Try to import news libraries
try:
    from GoogleNews import GoogleNews
    HAS_GOOGLENEWS = True
except ImportError:
    HAS_GOOGLENEWS = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

# Configuration
NEWS_LIMIT_PER_TICKER = 10  # Số lượng tin tức tối đa mỗi mã
NEWS_DAYS_BACK = 7  # Lấy tin tức trong N ngày gần nhất
MIN_SENTIMENT_SCORE = 0.1  # Điểm sentiment tối thiểu để coi là có ảnh hưởng


def get_news_from_google(ticker, days_back=NEWS_DAYS_BACK, lang='vi', region='VN', target_date=None):
    """
    Lấy tin tức từ Google News theo ngày cụ thể
    
    Args:
        ticker: Mã cổ phiếu (ví dụ: 'ACB', 'VCB')
        days_back: Số ngày gần nhất
        lang: Ngôn ngữ ('vi' cho tiếng Việt)
        region: Vùng ('VN' cho Việt Nam)
        target_date: Ngày cụ thể để lấy news (datetime object)
    
    Returns:
        List of dicts với keys: title, link, date, snippet
    """
    if not HAS_GOOGLENEWS:
        return []
    
    try:
        googlenews = GoogleNews(lang=lang, region=region)
        
        # Tìm kiếm với keyword: mã cổ phiếu + "cổ phiếu" hoặc "chứng khoán"
        keywords = [
            f"{ticker} cổ phiếu",
            f"{ticker} chứng khoán",
            f"{ticker} VN",
            f"mã {ticker}"
        ]
        
        # Nếu có target_date, thêm vào keyword để tìm chính xác hơn
        if target_date:
            date_str = target_date.strftime('%d/%m/%Y')
            keywords = [
                f"{ticker} cổ phiếu {date_str}",
                f"{ticker} chứng khoán {date_str}"
            ]
        
        all_news = []
        for keyword in keywords[:2]:  # Chỉ lấy 2 keyword đầu để tránh spam
            try:
                googlenews.search(keyword)
                results = googlenews.results(sort=True)[:NEWS_LIMIT_PER_TICKER * 2]  # Lấy nhiều hơn để filter
                
                for result in results:
                    # Parse date từ Google News
                    try:
                        date_str = result.get('date', '')
                        if date_str:
                            # Google News có thể trả về: "2 hours ago", "1 day ago", "Dec 19, 2025", etc.
                            news_date = parse_google_news_date(date_str, target_date)
                        else:
                            # Nếu không có date, dùng target_date hoặc ngày hiện tại - 1
                            news_date = target_date if target_date else (datetime.now() - timedelta(days=1))
                    except:
                        news_date = target_date if target_date else (datetime.now() - timedelta(days=1))
                    
                    # Filter theo date nếu có target_date
                    if target_date:
                        date_diff = abs((news_date.date() - target_date.date()).days)
                        if date_diff > 1:  # Chỉ lấy news trong ±1 ngày
                            continue
                    
                    all_news.append({
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'date': news_date,
                        'snippet': result.get('desc', ''),
                        'source': 'Google News',
                        'keyword': keyword
                    })
            except Exception as e:
                print(f"⚠️  Lỗi khi search keyword '{keyword}': {e}")
                continue
            
            time.sleep(1)  # Delay để tránh rate limit
        
        return all_news[:NEWS_LIMIT_PER_TICKER]
        
    except Exception as e:
        print(f"⚠️  Lỗi Google News cho {ticker}: {e}")
        return []


def parse_google_news_date(date_str, reference_date=None):
    """
    Parse date string từ Google News thành datetime object
    
    Args:
        date_str: Date string từ Google News (ví dụ: "2 hours ago", "1 day ago", "Dec 19, 2025")
        reference_date: Ngày tham chiếu (thường là target_date hoặc datetime.now())
    
    Returns:
        datetime object
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    date_str_lower = date_str.lower().strip()
    
    # Parse các format phổ biến
    try:
        # "X hours ago" hoặc "X hour ago"
        if 'hour' in date_str_lower:
            hours = int(re.search(r'(\d+)', date_str_lower).group(1))
            return reference_date - timedelta(hours=hours)
        
        # "X days ago" hoặc "X day ago"
        if 'day' in date_str_lower:
            days = int(re.search(r'(\d+)', date_str_lower).group(1))
            return reference_date - timedelta(days=days)
        
        # "X weeks ago" hoặc "X week ago"
        if 'week' in date_str_lower:
            weeks = int(re.search(r'(\d+)', date_str_lower).group(1))
            return reference_date - timedelta(weeks=weeks)
        
        # Format "Dec 19, 2025" hoặc "19 Dec 2025"
        try:
            from dateparser import parse as dateparse
            parsed = dateparse(date_str)
            if parsed:
                return parsed
        except:
            pass
        
        # Format "YYYY-MM-DD"
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            pass
        
        # Format "DD/MM/YYYY"
        try:
            return datetime.strptime(date_str, '%d/%m/%Y')
        except:
            pass
        
    except:
        pass
    
    # Default: trả về reference_date - 1 day
    return reference_date - timedelta(days=1)


def get_news_from_rss_feeds(ticker, days_back=NEWS_DAYS_BACK, target_date=None):
    """
    Lấy tin tức từ RSS feeds của các trang tin tài chính VN theo ngày cụ thể
    
    Args:
        ticker: Mã cổ phiếu
        days_back: Số ngày gần nhất
        target_date: Ngày cụ thể để lấy news (datetime object)
    
    Returns:
        List of dicts với keys: title, link, date, snippet
    """
    if not HAS_FEEDPARSER:
        return []
    
    # RSS feeds của các trang tin tài chính VN
    rss_feeds = [
        'https://cafef.vn/rss.chn',  # CafeF
        'https://vneconomy.vn/rss.xml',  # VnEconomy
        'https://vnexpress.net/rss/kinh-doanh.rss',  # VnExpress Kinh doanh
    ]
    
    all_news = []
    
    # Xác định date range
    if target_date:
        start_date = (target_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = (target_date + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:30]:  # Lấy 30 tin mới nhất từ mỗi feed để filter tốt hơn
                # Check nếu tin có chứa mã cổ phiếu
                title = entry.get('title', '').upper()
                summary = entry.get('summary', '').upper()
                
                if ticker.upper() in title or ticker.upper() in summary:
                    # Parse date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            # Thử parse từ published string
                            pub_date_str = entry.get('published', '')
                            if pub_date_str:
                                try:
                                    from dateparser import parse as dateparse
                                    pub_date = dateparse(pub_date_str)
                                    if not pub_date:
                                        pub_date = datetime.now() - timedelta(days=1)
                                except:
                                    pub_date = datetime.now() - timedelta(days=1)
                            else:
                                pub_date = datetime.now() - timedelta(days=1)
                    except:
                        pub_date = datetime.now() - timedelta(days=1)
                    
                    # Filter theo date range
                    if start_date <= pub_date <= end_date:
                        all_news.append({
                            'title': entry.get('title', ''),
                            'link': entry.get('link', ''),
                            'date': pub_date,
                            'snippet': entry.get('summary', ''),
                            'source': feed.feed.get('title', 'RSS Feed'),
                            'keyword': ticker
                        })
            
            time.sleep(0.5)  # Delay
            
        except Exception as e:
            print(f"⚠️  Lỗi RSS feed {feed_url}: {e}")
            continue
    
    return all_news[:NEWS_LIMIT_PER_TICKER]


def analyze_sentiment(text):
    """
    Phân tích sentiment của text (positive/negative/neutral)
    
    Args:
        text: Text cần phân tích
    
    Returns:
        dict với keys: compound, pos, neu, neg, sentiment_label
    """
    if not HAS_VADER:
        # Fallback: Simple keyword-based sentiment
        return simple_sentiment_analysis(text)
    
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        
        # Xác định label
        if scores['compound'] >= 0.05:
            sentiment_label = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        scores['sentiment_label'] = sentiment_label
        return scores
        
    except Exception as e:
        return simple_sentiment_analysis(text)


def simple_sentiment_analysis(text):
    """
    Simple keyword-based sentiment analysis (fallback)
    """
    text_lower = text.lower()
    
    # Positive keywords
    positive_keywords = [
        'tăng', 'tăng trưởng', 'tốt', 'tích cực', 'lợi nhuận', 'tăng giá',
        'phát triển', 'thành công', 'mạnh', 'khả quan', 'tích cực', 'cải thiện'
    ]
    
    # Negative keywords
    negative_keywords = [
        'giảm', 'sụt giảm', 'xấu', 'tiêu cực', 'lỗ', 'giảm giá',
        'suy giảm', 'thất bại', 'yếu', 'bi quan', 'tiêu cực', 'khó khăn'
    ]
    
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    if pos_count > neg_count:
        compound = min(0.5, pos_count * 0.1)
        sentiment_label = 'positive'
    elif neg_count > pos_count:
        compound = max(-0.5, -neg_count * 0.1)
        sentiment_label = 'negative'
    else:
        compound = 0.0
        sentiment_label = 'neutral'
    
    return {
        'compound': compound,
        'pos': pos_count / max(len(text.split()), 1),
        'neu': 1.0 - abs(compound),
        'neg': neg_count / max(len(text.split()), 1),
        'sentiment_label': sentiment_label
    }


def calculate_impact_score(news_item, sentiment_scores):
    """
    Tính điểm ảnh hưởng của tin tức đến giá cổ phiếu
    
    Args:
        news_item: Dict chứa thông tin tin tức
        sentiment_scores: Dict chứa sentiment scores
    
    Returns:
        float: Điểm ảnh hưởng (từ -1 đến 1)
    """
    # Base score từ sentiment
    base_score = sentiment_scores.get('compound', 0.0)
    
    # Weight theo độ mới của tin (tin mới hơn = ảnh hưởng lớn hơn)
    news_date = news_item.get('date', datetime.now())
    days_old = (datetime.now() - news_date).days
    time_weight = max(0.5, 1.0 - (days_old / NEWS_DAYS_BACK))
    
    # Weight theo source (một số nguồn uy tín hơn)
    source = news_item.get('source', '').lower()
    source_weight = 1.0
    if 'cafef' in source or 'vneconomy' in source:
        source_weight = 1.2
    elif 'google' in source:
        source_weight = 1.0
    
    # Weight theo độ dài title/snippet (tin chi tiết hơn = ảnh hưởng lớn hơn)
    title_len = len(news_item.get('title', ''))
    snippet_len = len(news_item.get('snippet', ''))
    content_weight = min(1.2, 1.0 + (title_len + snippet_len) / 500)
    
    # Tính điểm cuối cùng
    impact_score = base_score * time_weight * source_weight * content_weight
    
    # Normalize về [-1, 1]
    impact_score = max(-1.0, min(1.0, impact_score))
    
    return impact_score


def scrape_news_for_ticker(ticker, target_date=None, days_back=NEWS_DAYS_BACK):
    """
    Cào tin tức cho một mã cổ phiếu theo ngày cụ thể
    
    Args:
        ticker: Mã cổ phiếu
        target_date: Ngày cụ thể để lấy news (datetime hoặc string 'YYYY-MM-DD')
                     Nếu None thì lấy days_back ngày gần nhất
        days_back: Số ngày gần nhất (chỉ dùng nếu target_date=None)
    
    Returns:
        List of dicts với đầy đủ thông tin: title, link, date, snippet, source,
        sentiment_scores, impact_score
    """
    if target_date:
        if isinstance(target_date, str):
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        else:
            target_dt = target_date
        date_str = target_dt.strftime('%Y-%m-%d')
        print(f"   📰 Đang lấy tin tức cho {ticker} (ngày {date_str})...", end=" ", flush=True)
        # Lấy news trong khoảng ±1 ngày để có đủ context
        days_back_calc = 2
    else:
        print(f"   📰 Đang lấy tin tức cho {ticker}...", end=" ", flush=True)
        days_back_calc = days_back
        target_dt = None
    
    all_news = []
    
    # Lấy từ Google News (truyền target_date để filter chính xác hơn)
    if HAS_GOOGLENEWS:
        google_news = get_news_from_google(ticker, days_back_calc, target_date=target_dt)
        all_news.extend(google_news)
    
    # Lấy từ RSS feeds (truyền target_date để filter chính xác hơn)
    if HAS_FEEDPARSER:
        rss_news = get_news_from_rss_feeds(ticker, days_back_calc, target_date=target_dt)
        all_news.extend(rss_news)
    
    # Loại bỏ duplicate (dựa trên link)
    seen_links = set()
    unique_news = []
    for news in all_news:
        link = news.get('link', '')
        if link and link not in seen_links:
            seen_links.add(link)
            unique_news.append(news)
    
    # Phân tích sentiment và tính impact score cho mỗi tin
    processed_news = []
    for news in unique_news[:NEWS_LIMIT_PER_TICKER]:
        # Combine title và snippet để phân tích
        text = f"{news.get('title', '')} {news.get('snippet', '')}"
        
        # Phân tích sentiment
        sentiment_scores = analyze_sentiment(text)
        news['sentiment_scores'] = sentiment_scores
        
        # Tính impact score
        impact_score = calculate_impact_score(news, sentiment_scores)
        news['impact_score'] = impact_score
        
        processed_news.append(news)
    
    # Sort theo impact score (ảnh hưởng lớn nhất trước)
    processed_news.sort(key=lambda x: abs(x.get('impact_score', 0)), reverse=True)
    
    print(f"✅ ({len(processed_news)} tin)")
    
    return processed_news


def scrape_news_for_tickers(tickers, target_date=None, days_back=NEWS_DAYS_BACK):
    """
    Cào tin tức cho nhiều mã cổ phiếu theo ngày cụ thể
    
    Args:
        tickers: List of tickers
        target_date: Ngày cụ thể để lấy news (datetime hoặc string 'YYYY-MM-DD')
                     Nếu None thì lấy days_back ngày gần nhất
        days_back: Số ngày gần nhất (chỉ dùng nếu target_date=None)
    
    Returns:
        Dict với key là ticker, value là list of news
    """
    all_ticker_news = {}
    
    for i, ticker in enumerate(tickers, 1):
        if ticker.startswith('^'):
            continue  # Bỏ qua chỉ số
        
        news = scrape_news_for_ticker(ticker, target_date=target_date, days_back=days_back)
        all_ticker_news[ticker] = news
        
        # Delay để tránh rate limit
        if i < len(tickers):
            time.sleep(2)
    
    return all_ticker_news


def get_news_by_date(tickers, target_date, db_path='data/news_data.db'):
    """
    Lấy news từ database theo ngày cụ thể để dùng cho prediction
    
    Args:
        tickers: List of tickers
        target_date: Ngày cụ thể (datetime hoặc string 'YYYY-MM-DD')
        db_path: Đường dẫn database
    
    Returns:
        dict với key là ticker, value là list of news dicts
    """
    if not Path(db_path).exists():
        return {ticker: [] for ticker in tickers}
    
    try:
        import sqlite3
        
        # Parse target_date
        if isinstance(target_date, str):
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        else:
            target_dt = target_date
        
        date_str = target_dt.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Để truy cập bằng tên cột
        cursor = conn.cursor()
        
        news_by_ticker = {}
        
        for ticker in tickers:
            # Query news cho ngày cụ thể (±1 ngày để có context)
            start_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            
            cursor.execute('''
                SELECT ticker, title, link, date, snippet, source, 
                       sentiment_label, compound_score, impact_score
                FROM news_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY impact_score DESC
            ''', (ticker, start_date, end_date))
            
            results = cursor.fetchall()
            news_list = []
            for row in results:
                news_list.append({
                    'ticker': row['ticker'],
                    'title': row['title'],
                    'link': row['link'],
                    'date': row['date'],
                    'snippet': row['snippet'],
                    'source': row['source'],
                    'sentiment_label': row['sentiment_label'],
                    'compound_score': row['compound_score'],
                    'impact_score': row['impact_score']
                })
            
            news_by_ticker[ticker] = news_list
        
        conn.close()
        return news_by_ticker
        
    except Exception as e:
        print(f"⚠️  Lỗi khi lấy news theo ngày: {e}")
        return {ticker: [] for ticker in tickers}


def save_news_to_database(news_data, ticker, db_path='data/news_data.db'):
    """
    Lưu tin tức vào SQLite database
    
    Args:
        news_data: List of news dicts
        ticker: Mã cổ phiếu
        db_path: Đường dẫn database
    """
    try:
        import sqlite3
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tạo table nếu chưa có
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                title TEXT,
                link TEXT UNIQUE,
                date TEXT,
                snippet TEXT,
                source TEXT,
                sentiment_label TEXT,
                compound_score REAL,
                impact_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert news
        for news in news_data:
            try:
                # Lưu date chỉ lấy phần ngày (YYYY-MM-DD) để dễ query theo ngày
                news_date = news.get('date', datetime.now())
                if isinstance(news_date, str):
                    try:
                        news_date = datetime.strptime(news_date, '%Y-%m-%d')
                    except:
                        news_date = datetime.now()
                date_str = news_date.strftime('%Y-%m-%d')  # Chỉ lưu ngày, không có time
                
                cursor.execute('''
                    INSERT OR REPLACE INTO news_data 
                    (ticker, title, link, date, snippet, source, sentiment_label, compound_score, impact_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    news.get('title', ''),
                    news.get('link', ''),
                    date_str,  # Lưu dạng YYYY-MM-DD để query theo ngày dễ dàng
                    news.get('snippet', ''),
                    news.get('source', ''),
                    news.get('sentiment_scores', {}).get('sentiment_label', 'neutral'),
                    news.get('sentiment_scores', {}).get('compound', 0.0),
                    news.get('impact_score', 0.0)
                ))
            except sqlite3.IntegrityError:
                # Link đã tồn tại, skip
                continue
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"⚠️  Lỗi khi lưu news vào database: {e}")


def get_news_summary_for_ticker(ticker, days_back=NEWS_DAYS_BACK):
    """
    Lấy summary tin tức cho một mã cổ phiếu
    
    Returns:
        dict với keys: total_news, positive_count, negative_count, 
        avg_impact_score, top_news
    """
    news_list = scrape_news_for_ticker(ticker, days_back)
    
    if not news_list:
        return {
            'total_news': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'avg_impact_score': 0.0,
            'top_news': []
        }
    
    positive_count = sum(1 for n in news_list if n.get('sentiment_scores', {}).get('sentiment_label') == 'positive')
    negative_count = sum(1 for n in news_list if n.get('sentiment_scores', {}).get('sentiment_label') == 'negative')
    neutral_count = len(news_list) - positive_count - negative_count
    
    avg_impact_score = sum(n.get('impact_score', 0.0) for n in news_list) / len(news_list)
    
    # Top 3 news có impact lớn nhất
    top_news = sorted(news_list, key=lambda x: abs(x.get('impact_score', 0)), reverse=True)[:3]
    
    return {
        'total_news': len(news_list),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'avg_impact_score': avg_impact_score,
        'top_news': [
            {
                'title': n.get('title', ''),
                'impact_score': n.get('impact_score', 0.0),
                'sentiment': n.get('sentiment_scores', {}).get('sentiment_label', 'neutral')
            }
            for n in top_news
        ]
    }


if __name__ == "__main__":
    # Test với một vài mã
    test_tickers = ['ACB', 'VCB', 'VIC']
    
    print("="*70)
    print("NEWS SCRAPER TEST")
    print("="*70)
    
    for ticker in test_tickers:
        summary = get_news_summary_for_ticker(ticker)
        print(f"\n{ticker}:")
        print(f"  Total news: {summary['total_news']}")
        print(f"  Positive: {summary['positive_count']}, Negative: {summary['negative_count']}, Neutral: {summary['neutral_count']}")
        print(f"  Avg impact score: {summary['avg_impact_score']:.3f}")
        print(f"  Top news:")
        for news in summary['top_news']:
            print(f"    - {news['title'][:60]}... (impact: {news['impact_score']:.3f}, sentiment: {news['sentiment']})")

