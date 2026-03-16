"""
API Endpoints for Realtime Data and Retraining

- /api/realtime_data: Nhận realtime data và lưu vào database
- /api/retrain: Retrain model với dữ liệu mới nhất
- /api/retrain_status: Kiểm tra trạng thái retrain
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import threading
import time

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
    HAS_MONGODB = True
except ImportError:
    HAS_MONGODB = False

# SQLite imports (fallback)
try:
    import sqlite3
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

from Get_data import download_stock_data, save_data
from robo_agent import train_robo_advisor

retrain_bp = Blueprint('retrain', __name__)

# Global variables
retrain_status = {
    'is_training': False,
    'last_training': None,
    'training_progress': 0,
    'error': None
}

# Database configuration
USE_MONGODB = True  # Set False để dùng SQLite
MONGODB_URI = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "stock_data"
MONGODB_COLLECTION_NAME = "daily_stock_data"
SQLITE_DB_PATH = Path('data') / 'realtime_data.db'

# Global flag
_use_mongodb = USE_MONGODB


def get_mongodb_client():
    """Get MongoDB client connection"""
    if not HAS_MONGODB:
        raise ImportError("pymongo not installed. Install with: pip install pymongo")
    
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except ConnectionFailure:
        raise ConnectionError(f"Cannot connect to MongoDB at {MONGODB_URI}")


def init_database():
    """Initialize MongoDB hoặc SQLite database"""
    global _use_mongodb
    
    if _use_mongodb and HAS_MONGODB:
        # Initialize MongoDB
        try:
            client = get_mongodb_client()
            db = client[MONGODB_DB_NAME]
            collection = db[MONGODB_COLLECTION_NAME]
            
            # Create unique index on (date, ticker)
            collection.create_index([("date", 1), ("ticker", 1)], unique=True)
            collection.create_index([("date", -1)])
            
            # Training history collection
            training_collection = db["training_history"]
            training_collection.create_index([("training_date", -1)])
            
            client.close()
            return True
        except Exception as e:
            print(f"MongoDB initialization error: {e}")
            if HAS_SQLITE:
                print("Falling back to SQLite...")
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_date TEXT NOT NULL,
                n_episodes INTEGER,
                status TEXT,
                metrics TEXT,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    
    raise RuntimeError("Neither MongoDB nor SQLite available")


def save_realtime_data_to_db(data):
    """
    Save realtime data to MongoDB hoặc SQLite database
    
    Args:
        data: List of dictionaries with stock data
        Format: [
            {
                "ticker": "ACB",
                "date": "2024-01-15",
                "open": 25000,
                "high": 25500,
                "low": 24800,
                "close": 25200,
                "volume": 1000000,
                "returns": 0.008
            },
            ...
        ]
    """
    global _use_mongodb
    
    if _use_mongodb and HAS_MONGODB:
        return save_realtime_data_to_mongodb(data)
    elif HAS_SQLITE:
        return save_realtime_data_to_sqlite(data)
    else:
        raise RuntimeError("No database available")


def save_realtime_data_to_mongodb(data):
    """Save data to MongoDB"""
    client = get_mongodb_client()
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION_NAME]
    
    saved_count = 0
    for record in data:
        try:
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
            
            collection.update_one(
                {"date": record['date'], "ticker": record['ticker']},
                {"$set": document},
                upsert=True
            )
            saved_count += 1
        except Exception as e:
            print(f"Error saving {record.get('ticker')} on {record.get('date')}: {e}")
    
    client.close()
    return saved_count


def save_realtime_data_to_sqlite(data):
    """Save data to SQLite (fallback)"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
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
        except Exception as e:
            print(f"Error saving {record['ticker']} on {record['date']}: {e}")
    
    conn.commit()
    conn.close()
    return len(data)


def load_data_from_db(start_date=None, end_date=None, tickers=None):
    """
    Load data from MongoDB hoặc SQLite database and convert to DataFrame format
    
    Returns:
        prices: DataFrame with date index and ticker columns
        returns: DataFrame with date index and ticker columns
        ohlcv: Dictionary with OHLCV data
    """
    global _use_mongodb
    
    if _use_mongodb and HAS_MONGODB:
        return load_data_from_mongodb(start_date, end_date, tickers)
    elif HAS_SQLITE:
        return load_data_from_sqlite(start_date, end_date, tickers)
    else:
        raise RuntimeError("No database available")


def load_data_from_mongodb(start_date=None, end_date=None, tickers=None):
    """Load data from MongoDB"""
    client = get_mongodb_client()
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION_NAME]
    
    # Build query
    query = {}
    if start_date:
        query['date'] = {'$gte': start_date}
    if end_date:
        if 'date' in query:
            query['date']['$lte'] = end_date
        else:
            query['date'] = {'$lte': end_date}
    if tickers:
        query['ticker'] = {'$in': tickers}
    
    # Fetch data
    cursor = collection.find(query).sort([("date", 1), ("ticker", 1)])
    data = list(cursor)
    client.close()
    
    if not data:
        return None, None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Drop MongoDB _id column
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    if df.empty:
        return None, None, None
    
    # Convert to wide format (pivot)
    prices_df = df.pivot(index='date', columns='ticker', values='close')
    returns_df = df.pivot(index='date', columns='ticker', values='returns')
    
    # Convert date index to datetime
    prices_df.index = pd.to_datetime(prices_df.index)
    returns_df.index = pd.to_datetime(returns_df.index)
    
    # Create OHLCV dictionary
    ohlcv = {}
    for col in ['open', 'high', 'low', 'volume']:
        if col in df.columns:
            ohlcv[col] = df.pivot(index='date', columns='ticker', values=col)
            ohlcv[col].index = pd.to_datetime(ohlcv[col].index)
    
    return prices_df, returns_df, ohlcv


def load_data_from_sqlite(start_date=None, end_date=None, tickers=None):
    """Load data from SQLite (fallback)"""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    
    query = "SELECT date, ticker, open, high, low, close, volume, returns FROM daily_stock_data"
    conditions = []
    params = []
    
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)
    
    if tickers:
        placeholders = ','.join(['?'] * len(tickers))
        conditions.append(f"ticker IN ({placeholders})")
        params.extend(tickers)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY date, ticker"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return None, None, None
    
    # Convert to wide format (pivot)
    prices_df = df.pivot(index='date', columns='ticker', values='close')
    returns_df = df.pivot(index='date', columns='ticker', values='returns')
    
    # Convert date index to datetime
    prices_df.index = pd.to_datetime(prices_df.index)
    returns_df.index = pd.to_datetime(returns_df.index)
    
    # Create OHLCV dictionary
    ohlcv = {}
    for col in ['open', 'high', 'low', 'volume']:
        ohlcv[col] = df.pivot(index='date', columns='ticker', values=col)
        ohlcv[col].index = pd.to_datetime(ohlcv[col].index)
    
    return prices_df, returns_df, ohlcv


def train_model_async():
    """Train model in background thread"""
    global retrain_status
    
    retrain_status['is_training'] = True
    retrain_status['training_progress'] = 0
    retrain_status['error'] = None
    
    try:
        # Load data from database (last 2 years)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        retrain_status['training_progress'] = 10
        
        prices, returns, ohlcv = load_data_from_db(start_date=start_date, end_date=end_date)
        
        if prices is None or returns is None:
            raise ValueError("Không có dữ liệu trong database")
        
        retrain_status['training_progress'] = 20
        
        # Try to load existing model for continual learning
        model_path = Path('models') / 'trained_model.pth'
        load_existing_model = model_path.exists()
        
        if load_existing_model:
            print("📚 Loading existing model for continual learning...")
            retrain_status['training_progress'] = 25
        
        # Train model (with or without existing model)
        agent, history = train_robo_advisor(
            returns,
            n_episodes=2000,
            stock_code='realtime_retrain',
            cache_manager=None,
            prices=prices,
            ohlcv=ohlcv,
            load_existing_model=load_existing_model  # Pass flag to load existing model
        )
        
        if agent is None:
            raise ValueError("Training thất bại")
        
        retrain_status['training_progress'] = 90
        
        # Save model
        Path('models').mkdir(parents=True, exist_ok=True)
        model_path = Path('models') / 'trained_model.pth'
        
        # Get stock names from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT ticker FROM daily_stock_data ORDER BY ticker")
        stock_names = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'omega': agent.omega,
            'target_return': agent.target_return,
            'n_stocks': agent.n_stocks,
            'state_dim': agent.state_dim,
            'stock_names': stock_names,
            'retrain_date': datetime.now().isoformat()
        }, model_path)
        
        retrain_status['training_progress'] = 100
        
        # Save training history
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO training_history 
            (training_date, n_episodes, status, model_path)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            2000,
            'success',
            str(model_path)
        ))
        conn.commit()
        conn.close()
        
        retrain_status['last_training'] = datetime.now().isoformat()
        retrain_status['is_training'] = False
        
    except Exception as e:
        retrain_status['error'] = str(e)
        retrain_status['is_training'] = False
        retrain_status['training_progress'] = 0
        
        # Save failed training
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO training_history 
            (training_date, n_episodes, status)
            VALUES (?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            2000,
            f'failed: {str(e)}'
        ))
        conn.commit()
        conn.close()


@retrain_bp.route('/api/realtime_data', methods=['POST'])
def receive_realtime_data():
    """
    Receive realtime stock data and save to database
    
    Request JSON:
    {
        "data": [
            {
                "ticker": "ACB",
                "date": "2024-01-15",
                "open": 25000,
                "high": 25500,
                "low": 24800,
                "close": 25200,
                "volume": 1000000,
                "returns": 0.008
            },
            ...
        ]
    }
    
    Response JSON:
    {
        "success": true,
        "saved_count": 5,
        "message": "Đã lưu 5 bản ghi"
    }
    """
    try:
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "data" field in request'
            }), 400
        
        stock_data = data['data']
        
        if not isinstance(stock_data, list):
            return jsonify({
                'success': False,
                'error': '"data" must be an array'
            }), 400
        
        # Validate required fields
        required_fields = ['ticker', 'date', 'close']
        for record in stock_data:
            for field in required_fields:
                if field not in record:
                    return jsonify({
                        'success': False,
                        'error': f'Missing required field "{field}" in record'
                    }), 400
        
        # Save to database
        save_realtime_data_to_db(stock_data)
        
        return jsonify({
            'success': True,
            'saved_count': len(stock_data),
            'message': f'Đã lưu {len(stock_data)} bản ghi vào database'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@retrain_bp.route('/api/retrain', methods=['POST'])
def retrain_model():
    """
    Trigger model retraining with latest data from database
    
    Request JSON (optional):
    {
        "force": false  // Force retrain even if already training
    }
    
    Response JSON:
    {
        "success": true,
        "message": "Đã bắt đầu retrain model",
        "status": "training"
    }
    """
    global retrain_status
    
    try:
        data = request.json or {}
        force = data.get('force', False)
        
        if retrain_status['is_training'] and not force:
            return jsonify({
                'success': False,
                'error': 'Model đang được train. Vui lòng đợi hoàn tất.',
                'status': 'training'
            }), 400
        
        # Start training in background thread
        thread = threading.Thread(target=train_model_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Đã bắt đầu retrain model',
            'status': 'training'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@retrain_bp.route('/api/retrain_status', methods=['GET'])
def get_retrain_status():
    """
    Get current retraining status
    
    Response JSON:
    {
        "is_training": false,
        "last_training": "2024-01-15T10:30:00",
        "training_progress": 0,
        "error": null
    }
    """
    return jsonify(retrain_status)


@retrain_bp.route('/api/data_stats', methods=['GET'])
def get_data_stats():
    """
    Get statistics about data in database
    
    Response JSON:
    {
        "total_records": 15000,
        "date_range": {
            "start": "2020-01-01",
            "end": "2024-01-15"
        },
        "tickers": ["ACB", "BCM", ...],
        "ticker_count": 30
    }
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM daily_stock_data")
        total_records = cursor.fetchone()[0]
        
        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_stock_data")
        min_date, max_date = cursor.fetchone()
        
        # Tickers
        cursor.execute("SELECT DISTINCT ticker FROM daily_stock_data ORDER BY ticker")
        tickers = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'total_records': total_records,
            'date_range': {
                'start': min_date,
                'end': max_date
            },
            'tickers': tickers,
            'ticker_count': len(tickers)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Initialize database on import
init_database()

