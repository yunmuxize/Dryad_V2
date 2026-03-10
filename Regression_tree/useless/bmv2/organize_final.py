# -*- coding: utf-8 -*-
"""
最终整理bmv2文件夹 - 创建分层结构
"""

import os
import shutil
from pathlib import Path

def organize_bmv2():
    """整理bmv2文件夹"""
    bmv2_dir = Path(__file__).parent
    
    print("=" * 60)
    print("最终整理bmv2文件夹 - 创建分层结构")
    print("=" * 60)
    
    # 创建目录结构
    dirs = ["scripts", "docs"]
    for dir_name in dirs:
        dir_path = bmv2_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"[OK] 创建目录: {dir_name}/")
    
    # 移动文件到相应目录
    file_moves = {
        # 脚本文件 -> scripts/
        "monitor_training.py": "scripts",
        "train_optimized.py": "scripts",
        "test_lgbm_model.py": "scripts",
        "organize_bmv2.ps1": "scripts",
        "organize_final.ps1": "scripts",
        "organize_final.py": "scripts",
        "start_optimized_training.bat": "scripts",
        "start_training.bat": "scripts",
        
        # 文档文件 -> docs/
        "README.md": "docs",
        "README_TRAINING.md": "docs",
        "QUICK_START.md": "docs",
        "FINAL_RESULTS.md": "docs",
        "SUMMARY.md": "docs",
    }
    
    moved_count = 0
    for file_name, target_dir in file_moves.items():
        src_path = bmv2_dir / file_name
        if src_path.exists():
            dst_path = bmv2_dir / target_dir / file_name
            shutil.move(str(src_path), str(dst_path))
            print(f"[OK] 移动: {file_name} -> {target_dir}/")
            moved_count += 1
    
    print(f"\n[完成] 共移动 {moved_count} 个文件")
    
    # 显示最终目录结构
    print("\n" + "=" * 60)
    print("最终目录结构:")
    print("=" * 60)
    
    print("\n根目录文件:")
    for f in sorted(bmv2_dir.glob("*.py")):
        print(f"  - {f.name}")
    for f in sorted(bmv2_dir.glob("*.csv")):
        print(f"  - {f.name}")
    for f in sorted(bmv2_dir.glob("*.md")):
        print(f"  - {f.name}")
    for f in sorted(bmv2_dir.glob("*.bat")):
        print(f"  - {f.name}")
    
    print("\nscripts/ 目录:")
    scripts_dir = bmv2_dir / "scripts"
    if scripts_dir.exists():
        for f in sorted(scripts_dir.iterdir()):
            print(f"  - {f.name}")
    
    print("\ndocs/ 目录:")
    docs_dir = bmv2_dir / "docs"
    if docs_dir.exists():
        for f in sorted(docs_dir.iterdir()):
            print(f"  - {f.name}")
    
    print("\n其他目录:")
    for d in sorted(bmv2_dir.iterdir()):
        if d.is_dir() and d.name not in ["scripts", "docs", "__pycache__"]:
            file_count = len(list(d.iterdir()))
            print(f"  - {d.name}/ ({file_count} files)")

if __name__ == "__main__":
    organize_bmv2()

