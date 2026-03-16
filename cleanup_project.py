"""
Script để clean up project, chỉ giữ lại các file cần thiết cho dự thi NCKH
"""

import os
from pathlib import Path
import shutil

# Files cần GIỮ (core files)
KEEP_FILES = {
    # Core Python files
    'app.py',
    'robo_agent.py',
    'Train_Model.py',
    'Get_data.py',
    'main.py',
    'report_figures.py',
    
    # News modules
    'news_scraper.py',
    'news_features.py',
    
    # Data fetching
    'Script.py',
    'daily_data_fetcher.py',
    'data_source.py',
    'download_vnstock_data.py',
    
    # API
    'retrain_api.py',
    
    # Config
    'requirements.txt',
    
    # Data files - giữ lại tất cả CSV
    'vn_stocks_data_2020_2025.csv',
    'report_table.csv',
    'report_table_quarterly_yearly.csv',
    'report_table_by_year.csv',
    'data/returns.csv',
    'data/prices.csv',
    'data/Data_test.csv',
    
    # Templates
    'templates/index.html',
}

# Documentation cần GIỮ (cho NCKH)
KEEP_DOCS = {
    'CHI_TIET_TRAINING_REWARD.md',
    'QUY_TRINH_VIET_REPORT.md',
    'CONG_THUC_REPORT_FIGURES.md',
    'CONG_THUC_BANG_SO_SANH.md',
    'IPO_W_EXPLANATION.md',
    'README.md',
    'ACTOR_CRITIC_EXPLANATION.md',  # Giải thích thuật toán
}

# Directories cần GIỮ
KEEP_DIRS = {
    'models',
    'templates',
    'data',  # Giữ data directory nhưng có thể clean bên trong
}

# Patterns để XÓA
DELETE_PATTERNS = [
    'test_*.py',
    'check_*.py',
    '*_old.csv',
    '*.backup_*',
    'merge_*.py',
    'update_*.py',
    'scheduled_*.py',
    'rebalance.py',
    'portfolio_advisor.py',
    '*.sh',
    '*.log',
]

# Documentation để XÓA
DELETE_DOCS = [
    'API_SETUP_GUIDE.md',
    'API_DOCUMENTATION.md',
    'API_RETRAIN_DOCUMENTATION.md',
    'API_QUICK_REFERENCE.md',
    'MONGODB_SETUP.md',
    'DAILY_SCHEDULER_GUIDE.md',
    'AUTO_UPDATE_WORKFLOW.md',
    'COMPLETE_WORKFLOW.md',
    'NEWS_INTEGRATION_GUIDE.md',
    'NEWS_PREDICTION_GUIDE.md',
    'NEWS_SCRAPER_GUIDE.md',
    'RETRAIN_GUIDE.md',
    'START_GUIDE.md',
    'TROUBLESHOOTING.md',
    'setup_daily_scheduler.md',
]

# Files trong data/ để XÓA (có thể regenerate)
DELETE_DATA_FILES = [
    'data/info.txt',  # Chỉ xóa info.txt, giữ lại tất cả CSV và DB
    # Giữ lại tất cả CSV files và databases
]

# Report files để XÓA (có thể regenerate)
# Giữ lại report CSV files để có thể xem kết quả
DELETE_REPORT_FILES = [
    'report_wealth.png',  # Chỉ xóa PNG, giữ lại CSV
]


def cleanup_project(dry_run=True):
    """
    Clean up project files
    
    Args:
        dry_run: If True, chỉ list files sẽ xóa, không xóa thật
    """
    base_dir = Path('.')
    deleted_files = []
    deleted_dirs = []
    
    print("="*70)
    print("🧹 CLEANUP PROJECT - CHỈ GIỮ LẠI FILES CẦN THIẾT CHO NCKH")
    print("="*70)
    
    # 1. Xóa test và check files
    print("\n1️⃣  Xóa test và check files...")
    for pattern in ['test_*.py', 'check_*.py']:
        for file in base_dir.glob(pattern):
            if file.name not in KEEP_FILES:
                deleted_files.append(file)
                print(f"   ❌ {file.name}")
    
    # 2. Xóa backup và merge files
    print("\n2️⃣  Xóa backup và merge files...")
    for pattern in ['*_old.csv', '*.backup_*', 'merge_*.py']:
        for file in base_dir.glob(pattern):
            deleted_files.append(file)
            print(f"   ❌ {file.name}")
    
    # 3. Xóa utility scripts không cần thiết
    print("\n3️⃣  Xóa utility scripts không cần thiết...")
    for pattern in ['update_*.py', 'scheduled_*.py']:
        for file in base_dir.glob(pattern):
            if file.name not in KEEP_FILES:
                deleted_files.append(file)
                print(f"   ❌ {file.name}")
    
    # Xóa các file cụ thể
    for file_name in ['rebalance.py', 'portfolio_advisor.py']:
        file_path = base_dir / file_name
        if file_path.exists():
            deleted_files.append(file_path)
            print(f"   ❌ {file_name}")
    
    # 4. Xóa shell scripts
    print("\n4️⃣  Xóa shell scripts...")
    for file in base_dir.glob('*.sh'):
        deleted_files.append(file)
        print(f"   ❌ {file.name}")
    
    # 5. Xóa documentation không cần thiết
    print("\n5️⃣  Xóa documentation không cần thiết...")
    for doc in DELETE_DOCS:
        doc_path = base_dir / doc
        if doc_path.exists():
            deleted_files.append(doc_path)
            print(f"   ❌ {doc}")
    
    # 6. Xóa report files (có thể regenerate)
    print("\n6️⃣  Xóa report files (có thể regenerate)...")
    for file_name in DELETE_REPORT_FILES:
        file_path = base_dir / file_name
        if file_path.exists():
            deleted_files.append(file_path)
            print(f"   ❌ {file_name}")
    
    # 7. Xóa data files không cần thiết
    print("\n7️⃣  Xóa data files không cần thiết...")
    for file_name in DELETE_DATA_FILES:
        file_path = base_dir / file_name
        if file_path.exists():
            deleted_files.append(file_path)
            print(f"   ❌ {file_name}")
    
    # 8. Xóa __pycache__
    print("\n8️⃣  Xóa __pycache__ directories...")
    for pycache_dir in base_dir.rglob('__pycache__'):
        deleted_dirs.append(pycache_dir)
        print(f"   ❌ {pycache_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"   Files sẽ xóa: {len(deleted_files)}")
    print(f"   Directories sẽ xóa: {len(deleted_dirs)}")
    
    # List files sẽ GIỮ
    print("\n✅ FILES SẼ GIỮ LẠI:")
    print("   Core Python files:")
    for file in sorted(KEEP_FILES):
        if (base_dir / file).exists():
            print(f"      ✅ {file}")
    
    print("\n   Documentation:")
    for doc in sorted(KEEP_DOCS):
        if (base_dir / doc).exists():
            print(f"      ✅ {doc}")
    
    print("\n   Directories:")
    for dir_name in sorted(KEEP_DIRS):
        if (base_dir / dir_name).exists():
            print(f"      ✅ {dir_name}/")
    
    # Xác nhận xóa
    if not dry_run:
        print("\n" + "="*70)
        confirm = input("⚠️  Bạn có chắc chắn muốn xóa các files trên? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Hủy bỏ cleanup")
            return
        
        # Xóa files
        print("\n🗑️  Đang xóa files...")
        for file_path in deleted_files:
            try:
                file_path.unlink()
                print(f"   ✅ Đã xóa: {file_path.name}")
            except Exception as e:
                print(f"   ❌ Lỗi khi xóa {file_path.name}: {e}")
        
        # Xóa directories
        for dir_path in deleted_dirs:
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Đã xóa: {dir_path}")
            except Exception as e:
                print(f"   ❌ Lỗi khi xóa {dir_path}: {e}")
        
        print("\n✅ Cleanup hoàn tất!")
    else:
        print("\n💡 Đây là DRY RUN. Để xóa thật, chạy:")
        print("   python3 cleanup_project.py --execute")


if __name__ == '__main__':
    import sys
    dry_run = '--execute' not in sys.argv
    cleanup_project(dry_run=dry_run)

