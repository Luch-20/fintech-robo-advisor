"""
Script để organize project structure một cách khoa học
"""

import os
import shutil
from pathlib import Path

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Định nghĩa cấu trúc thư mục mới
STRUCTURE = {
    'docs': [
        # Documentation files
        'DEMO_AND_PRESENTATION_GUIDE.md',
        'TRAIN_TEST_SPLIT_GUIDE.md',
        'API_FRONTEND_INTEGRATION.md',
        'CHI_TIET_TRAINING_REWARD.md',
        'QUY_TRINH_VIET_REPORT.md',
        'CONG_THUC_REPORT_FIGURES.md',
        'CONG_THUC_BANG_SO_SANH.md',
        'IPO_W_EXPLANATION.md',
        'ACTOR_CRITIC_EXPLANATION.md',
        'ANALYSIS_TEST_RESULTS.md',
        'ANALYSIS_NEW_RESULTS.md',
        'CHANGELOG_TRAIN_TEST.md',
        'README_TRAIN_TEST.md',
    ],
    'data': [
        # Data files (chỉ di chuyển report CSVs, giữ CSV chính ở root)
        'report_table.csv',
        'report_table_quarterly_yearly.csv',
        'report_table_by_year.csv',
        'test_results_comparison.csv',
    ],
    'api': [
        # API examples
        'api_example_request.json',
        'api_example_response.json',
    ],
}

# Files giữ ở root (core files)
KEEP_IN_ROOT = [
    'README.md',
    'requirements.txt',
    'app.py',
    'Train_Model.py',
    'train_test_split.py',
    'main.py',
    'Get_data.py',
    'Script.py',
    'daily_data_fetcher.py',
    'report_figures.py',
    'rebalance.py',
    'robo_agent.py',
    'news_scraper.py',
    'news_features.py',
    'retrain_api.py',
    'data_source.py',
    'download_vnstock_data.py',
    'cleanup_project.py',
    'organize_project.py',  # Giữ lại script này
    'vn_stocks_data_2020_2025.csv',  # CSV chính giữ ở root
]

# Directories đã tồn tại (không di chuyển)
EXISTING_DIRS = [
    'data',
    'models',
    'templates',
    '__pycache__',
]


def create_directories():
    """Tạo các thư mục mới nếu chưa tồn tại"""
    for dir_name in STRUCTURE.keys():
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}/")


def move_files():
    """Di chuyển files vào đúng thư mục"""
    moved_count = 0
    
    for target_dir, files in STRUCTURE.items():
        target_path = Path(target_dir)
        
        for file_name in files:
            source_path = Path(file_name)
            
            if source_path.exists():
                dest_path = target_path / file_name
                
                # Nếu file đã tồn tại ở destination, backup
                if dest_path.exists():
                    backup_path = dest_path.with_suffix(dest_path.suffix + '.backup')
                    shutil.move(str(dest_path), str(backup_path))
                    print(f"   ⚠️  Backed up existing: {dest_path} → {backup_path}")
                
                # Di chuyển file
                shutil.move(str(source_path), str(dest_path))
                print(f"   ✅ Moved: {file_name} → {target_dir}/")
                moved_count += 1
            else:
                print(f"   ⚠️  File not found: {file_name}")
    
    return moved_count


def verify_structure():
    """Kiểm tra cấu trúc sau khi organize"""
    print("\n" + "="*70)
    print("📁 VERIFYING STRUCTURE")
    print("="*70)
    
    # Check root files
    print("\n✅ Root files (core files):")
    for file_name in KEEP_IN_ROOT:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ⚠️  Missing: {file_name}")
    
    # Check directories
    print("\n✅ Directories:")
    for dir_name in STRUCTURE.keys():
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            print(f"   ✅ {dir_name}/ ({len(files)} files)")
        else:
            print(f"   ⚠️  Missing: {dir_name}/")
    
    # Check existing directories
    print("\n✅ Existing directories:")
    for dir_name in EXISTING_DIRS:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob('*'))
            print(f"   ✅ {dir_name}/ ({len(files)} files)")
        else:
            print(f"   ⚠️  Missing: {dir_name}/")


def main():
    """Main function"""
    print("="*70)
    print("📁 ORGANIZING PROJECT STRUCTURE")
    print("="*70)
    
    # 1. Tạo directories
    print("\n1️⃣  Creating directories...")
    create_directories()
    
    # 2. Di chuyển files
    print("\n2️⃣  Moving files...")
    moved_count = move_files()
    
    # 3. Verify structure
    verify_structure()
    
    print("\n" + "="*70)
    print("✅ ORGANIZATION COMPLETE")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Files moved: {moved_count}")
    print(f"   Directories created: {len(STRUCTURE)}")
    print(f"\n📁 New structure:")
    print(f"   docs/          - Documentation files")
    print(f"   data/          - Data files (report CSVs)")
    print(f"   api/           - API examples")
    print(f"   models/        - Trained models")
    print(f"   templates/     - Web UI templates")
    print(f"   Root/          - Core Python scripts và main CSV")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

