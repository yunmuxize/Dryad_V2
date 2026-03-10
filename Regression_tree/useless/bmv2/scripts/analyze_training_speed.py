# -*- coding: utf-8 -*-
"""
分析训练速度差异的原因
"""

import numpy as np
from functools import reduce
import operator

def calculate_combinations():
    """计算不同模式的参数组合数"""
    
    # 标准模式参数网格
    param_grid_memory_std = {
        'n_estimators': [300, 500, 700],  # 3
        'learning_rate': [0.01, 0.03, 0.05],  # 3
        'max_depth': [8, 10, 12],  # 3
        'num_leaves': [50, 100, 150],  # 3
        'min_child_samples': [15, 20, 25],  # 3
        'subsample': [0.8, 0.9],  # 2
        'colsample_bytree': [0.8, 0.9]  # 2
    }
    
    param_grid_cpu_std = {
        'n_estimators': [200, 300, 500],  # 3
        'learning_rate': [0.01, 0.03, 0.05],  # 3
        'max_depth': [7, 10, 12],  # 3
        'num_leaves': [31, 50, 100],  # 3
        'min_child_samples': [20, 30],  # 2
        'subsample': [0.8, 0.9],  # 2
        'colsample_bytree': [0.8, 0.9],  # 2
        'class_weight': [None, 'balanced']  # 2
    }
    
    # 扩展模式参数网格
    param_grid_memory_ext = {
        'n_estimators': [300, 500, 700, 1000],  # 4
        'learning_rate': [0.01, 0.03, 0.05, 0.08],  # 4
        'max_depth': [8, 10, 12, -1],  # 4
        'num_leaves': [50, 100, 150, 200],  # 4
        'min_child_samples': [15, 20, 25, 30],  # 4
        'subsample': [0.8, 0.9, 1.0],  # 3
        'colsample_bytree': [0.8, 0.9, 1.0],  # 3
        'reg_alpha': [0, 0.1, 0.5],  # 3
        'reg_lambda': [0, 0.1, 0.5]  # 3
    }
    
    param_grid_cpu_ext = {
        'n_estimators': [200, 300, 500, 700],  # 4
        'learning_rate': [0.01, 0.03, 0.05, 0.1],  # 4
        'max_depth': [7, 10, 12, -1],  # 4
        'num_leaves': [31, 50, 100, 150],  # 4
        'min_child_samples': [20, 30, 50],  # 3
        'subsample': [0.8, 0.9, 1.0],  # 3
        'colsample_bytree': [0.8, 0.9, 1.0],  # 3
        'class_weight': [None, 'balanced'],  # 2
        'reg_alpha': [0, 0.1],  # 2
        'reg_lambda': [0, 0.1]  # 2
    }
    
    # 计算组合数
    mem_std = reduce(operator.mul, [len(v) for v in param_grid_memory_std.values()], 1)
    cpu_std = reduce(operator.mul, [len(v) for v in param_grid_cpu_std.values()], 1)
    mem_ext = reduce(operator.mul, [len(v) for v in param_grid_memory_ext.values()], 1)
    cpu_ext = reduce(operator.mul, [len(v) for v in param_grid_cpu_ext.values()], 1)
    
    # 考虑5折交叉验证
    cv_folds = 5
    
    print("=" * 70)
    print("训练速度差异分析")
    print("=" * 70)
    
    print("\n1. 参数组合数对比:")
    print(f"   标准模式 Memory: {mem_std:,} 组合")
    print(f"   标准模式 CPU:    {cpu_std:,} 组合")
    print(f"   标准模式 总计:   {mem_std + cpu_std:,} 组合")
    
    print(f"\n   扩展模式 Memory: {mem_ext:,} 组合")
    print(f"   扩展模式 CPU:    {cpu_ext:,} 组合")
    print(f"   扩展模式 总计:   {mem_ext + cpu_ext:,} 组合")
    
    print(f"\n2. 扩展模式是标准模式的倍数:")
    ratio = (mem_ext + cpu_ext) / (mem_std + cpu_std)
    print(f"   总组合数倍数: {ratio:.1f}x")
    print(f"   Memory倍数:   {mem_ext / mem_std:.1f}x")
    print(f"   CPU倍数:      {cpu_ext / cpu_std:.1f}x")
    
    print(f"\n3. 考虑5折交叉验证后的实际训练次数:")
    print(f"   标准模式: {(mem_std + cpu_std) * cv_folds:,} 次模型训练")
    print(f"   扩展模式: {(mem_ext + cpu_ext) * cv_folds:,} 次模型训练")
    print(f"   倍数:     {ratio:.1f}x")
    
    print(f"\n4. 训练时间估算（假设每个组合训练需要0.1秒）:")
    time_std = (mem_std + cpu_std) * cv_folds * 0.1 / 60
    time_ext = (mem_ext + cpu_ext) * cv_folds * 0.1 / 60
    print(f"   标准模式: 约 {time_std:.1f} 分钟")
    print(f"   扩展模式: 约 {time_ext:.1f} 分钟")
    print(f"   倍数:     {ratio:.1f}x")
    
    print(f"\n5. 关键差异因素:")
    print(f"   - Memory模型: 从 {mem_std:,} 增加到 {mem_ext:,} 组合 ({mem_ext/mem_std:.1f}x)")
    print(f"   - CPU模型: 从 {cpu_std:,} 增加到 {cpu_ext:,} 组合 ({cpu_ext/cpu_std:.1f}x)")
    print(f"   - 新增参数: reg_alpha, reg_lambda (正则化参数)")
    print(f"   - 参数值范围扩大: n_estimators, learning_rate, max_depth等")
    
    print("\n" + "=" * 70)
    print("结论:")
    print(f"扩展模式的训练时间约为标准模式的 {ratio:.1f} 倍")
    print("主要原因是参数组合数大幅增加，特别是Memory模型")
    print("=" * 70)

if __name__ == "__main__":
    calculate_combinations()

