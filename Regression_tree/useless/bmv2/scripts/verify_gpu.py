# -*- coding: utf-8 -*-
"""
验证LightGBM GPU支持脚本
"""

import sys
import os

def check_gpu_support():
    """检查LightGBM是否支持GPU"""
    print("=" * 60)
    print("检查LightGBM GPU支持")
    print("=" * 60)
    
    try:
        import lightgbm as lgb
        print(f"LightGBM版本: {lgb.__version__}")
    except ImportError:
        print("[ERROR] LightGBM未安装")
        return False
    
    # 检查编译选项
    try:
        import numpy as np
        from sklearn.datasets import make_regression
        
        # 创建测试数据
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        print("\n测试1: 尝试使用GPU设备...")
        try:
            model = lgb.LGBMRegressor(
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                n_estimators=1,
                verbose=-1
            )
            model.fit(X, y)
            print("[OK] GPU设备可用！")
            gpu_available = True
        except Exception as e:
            print(f"[WARNING] GPU设备不可用: {e}")
            gpu_available = False
        
        print("\n测试2: 检查CPU多线程...")
        try:
            model_cpu = lgb.LGBMRegressor(
                device='cpu',
                n_estimators=10,
                n_jobs=-1,
                verbose=-1
            )
            model_cpu.fit(X, y)
            print("[OK] CPU模式可用")
        except Exception as e:
            print(f"[ERROR] CPU模式失败: {e}")
        
        print("\n" + "=" * 60)
        if gpu_available:
            print("结论: GPU支持已启用，可以使用GPU训练")
        else:
            print("结论: GPU不可用，将使用CPU训练")
            print("提示: 如果您的系统有GPU，可能需要:")
            print("  1. 安装GPU版本的LightGBM: pip install lightgbm --install-option=--gpu")
            print("  2. 或从源码编译支持GPU的版本")
        print("=" * 60)
        
        return gpu_available
        
    except Exception as e:
        print(f"[ERROR] 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_gpu_support()

