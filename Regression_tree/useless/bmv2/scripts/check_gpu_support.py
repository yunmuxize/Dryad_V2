# -*- coding: utf-8 -*-
"""
检查LightGBM GPU支持
"""

try:
    from lightgbm import LGBMRegressor
    print("=" * 60)
    print("LightGBM GPU支持检查")
    print("=" * 60)
    
    # 尝试创建GPU模型
    try:
        model = LGBMRegressor(device='gpu', n_jobs=1)
        print("\n[OK] GPU支持: 可用")
        print("   可以使用 --gpu 参数启用GPU加速训练")
    except Exception as e:
        print(f"\n[警告] GPU支持: 不可用")
        print(f"   错误信息: {e}")
        print("\n   解决方案:")
        print("   1. 确保安装了GPU版本的LightGBM:")
        print("      pip uninstall lightgbm")
        print("      pip install lightgbm --install-option=--gpu")
        print("   2. 确保安装了CUDA Toolkit")
        print("   3. 确保GPU驱动已正确安装")
        print("\n   当前将使用CPU模式训练")
    
    # 检查CPU模式
    try:
        model_cpu = LGBMRegressor(device='cpu', n_jobs=-1)
        print("\n[OK] CPU模式: 可用")
    except Exception as e:
        print(f"\n[错误] CPU模式: 不可用 - {e}")
    
    print("\n" + "=" * 60)
    
except ImportError as e:
    print(f"[错误] 无法导入LightGBM: {e}")

