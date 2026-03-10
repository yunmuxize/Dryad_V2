# -*- coding: utf-8 -*-
"""
数据预处理脚本
功能：
1. 读取原始 ml_features.csv
2. 去除 CPU 列
3. 将 memory 列减去基准值 26.21 MB，转换为 KB
4. 保存为新的 CSV 文件
"""

import pandas as pd
import os

# 配置
INPUT_FILE = 'ml_features.csv'
OUTPUT_FILE = 'processed_features.csv'
BASE_RSS_MB = 26.00  # 基准值 (MB) - 使用最小值避免负数

def preprocess_data():
    print("=" * 60)
    print("  BMv2 数据预处理")
    print("=" * 60)
    
    # 1. 读取数据
    print(f"\n[1/4] 读取原始数据: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"  - 原始数据形状: {df.shape}")
    print(f"  - 原始列名: {df.columns.tolist()}")
    
    # 2. 去除 CPU 列
    print(f"\n[2/4] 去除 CPU 列")
    if 'cpu' in df.columns:
        df = df.drop('cpu', axis=1)
        print(f"  - CPU 列已删除")
    else:
        print(f"  - 警告: 未找到 CPU 列")
    
    # 3. 处理 memory 列
    print(f"\n[3/4] 处理 memory 列")
    print(f"  - 基准值: {BASE_RSS_MB} MB")
    
    # 减去基准值
    df['memory'] = df['memory'] - BASE_RSS_MB
    
    # 转换为 KB 并保留两位小数
    df['memory'] = (df['memory'] * 1024).round(2)
    
    print(f"  - Memory 范围 (KB): {df['memory'].min():.2f} 到 {df['memory'].max():.2f}")
    print(f"  - Memory 平均值 (KB): {df['memory'].mean():.2f}")
    
    # 4. 保存新数据
    print(f"\n[4/4] 保存处理后的数据: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  - 新数据形状: {df.shape}")
    print(f"  - 新列名: {df.columns.tolist()}")
    
    print("\n" + "=" * 60)
    print("  数据预处理完成！")
    print("=" * 60)
    
    # 显示前几行数据
    print("\n预览前5行数据:")
    print(df.head())

if __name__ == "__main__":
    preprocess_data()
