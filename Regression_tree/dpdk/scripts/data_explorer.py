# -*- coding: utf-8 -*-
"""
DPDK 数据探索工具
用于分析 dataset.csv 和 dataset.json，提取特征统计信息
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
import re

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


class DPDKDataExplorer:
    """DPDK 数据探索器"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.csv_path = os.path.join(self.base_dir, 'dataset.csv')
        self.json_path = os.path.join(self.base_dir, 'dataset.json')
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print("=" * 80)
        print("  DPDK 数据探索工具")
        print("=" * 80)
    
    def load_csv_data(self):
        """加载 CSV 数据"""
        print("\n[1/4] 加载 CSV 数据")
        print("-" * 80)
        
        self.df_csv = pd.read_csv(self.csv_path)
        print(f"[OK] 加载 {len(self.df_csv)} 条记录")
        print(f"\n列名: {list(self.df_csv.columns)}")
        print(f"\n前5行数据:")
        print(self.df_csv.head())
        
        # 提取匹配方式标志位
        print("\n提取匹配方式标志位...")
        match_types = []
        for p4_file in self.df_csv['p4_file']:
            # 从文件名提取: 1_2_1_1_2_2_0_0_2341.p4 -> [1,2,1,1,2,2,0,0]
            parts = p4_file.replace('.p4', '').split('_')
            if len(parts) >= 9:  # 8个标志位 + size
                types = [int(x) for x in parts[:8]]
                match_types.append(types)
            else:
                match_types.append([0] * 8)
        
        # 添加到 DataFrame
        for i in range(8):
            self.df_csv[f'match_type_{i+1}'] = [mt[i] for mt in match_types]
        
        print(f"[OK] 提取了 8 个匹配方式特征")
        
    def load_json_data(self):
        """加载 JSON 数据"""
        print("\n[2/4] 加载 JSON 数据")
        print("-" * 80)
        
        with open(self.json_path, 'r') as f:
            self.data_json = json.load(f)
        
        print(f"[OK] 加载 {len(self.data_json)} 条记录")
        print(f"\n第一条记录的键:")
        print(list(self.data_json[0].keys()))
        
        # 转换为 DataFrame
        self.df_json = pd.DataFrame(self.data_json)
        
    def analyze_features(self):
        """分析特征"""
        print("\n[3/4] 特征分析")
        print("-" * 80)
        
        # CSV 特征统计
        print("\n=== CSV 特征统计 ===")
        print(f"\ntable_size 范围: {self.df_csv['table_size'].min()} - {self.df_csv['table_size'].max()}")
        print(f"exact_count 范围: {self.df_csv['exact_count'].min()} - {self.df_csv['exact_count'].max()}")
        print(f"wildcard_count 范围: {self.df_csv['wildcard_count'].min()} - {self.df_csv['wildcard_count'].max()}")
        print(f"table_memory_estimate_bytes 范围: {self.df_csv['table_memory_estimate_bytes'].min()} - {self.df_csv['table_memory_estimate_bytes'].max()}")
        
        # 匹配方式分布
        print("\n=== 匹配方式分布 ===")
        for i in range(8):
            col = f'match_type_{i+1}'
            counts = self.df_csv[col].value_counts().sort_index()
            print(f"\n{col}:")
            for val, count in counts.items():
                print(f"  {val}: {count} ({count/len(self.df_csv)*100:.1f}%)")
        
        # JSON 额外特征
        if hasattr(self, 'df_json'):
            print("\n=== JSON 额外特征统计 ===")
            extra_features = ['total_key_width_bits', 'match_complexity_score', 
                            'action_count', 'key_instruction_count']
            for feat in extra_features:
                if feat in self.df_json.columns:
                    print(f"\n{feat}:")
                    print(f"  范围: {self.df_json[feat].min()} - {self.df_json[feat].max()}")
                    print(f"  均值: {self.df_json[feat].mean():.2f}")
                    print(f"  标准差: {self.df_json[feat].std():.2f}")
        
        # 目标变量分析
        print("\n=== 目标变量分析 ===")
        memory = self.df_csv['table_memory_estimate_bytes']
        print(f"内存使用量 (bytes):")
        print(f"  最小值: {memory.min()}")
        print(f"  最大值: {memory.max()}")
        print(f"  均值: {memory.mean():.2f}")
        print(f"  中位数: {memory.median():.2f}")
        print(f"  标准差: {memory.std():.2f}")
        
    def visualize_data(self):
        """可视化数据"""
        print("\n[4/4] 数据可视化")
        print("-" * 80)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('DPDK 数据集特征分析', fontsize=16, fontweight='bold')
        
        # 1. table_size 分布
        axes[0, 0].hist(self.df_csv['table_size'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Table Size 分布')
        axes[0, 0].set_xlabel('Table Size')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. exact_count vs wildcard_count
        axes[0, 1].scatter(self.df_csv['exact_count'], self.df_csv['wildcard_count'], 
                          alpha=0.5, c='coral')
        axes[0, 1].set_title('Exact Count vs Wildcard Count')
        axes[0, 1].set_xlabel('Exact Count')
        axes[0, 1].set_ylabel('Wildcard Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 内存使用量分布
        axes[0, 2].hist(self.df_csv['table_memory_estimate_bytes'], bins=50, 
                       color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('内存使用量分布')
        axes[0, 2].set_xlabel('Memory (bytes)')
        axes[0, 2].set_ylabel('频数')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. table_size vs memory
        axes[1, 0].scatter(self.df_csv['table_size'], 
                          self.df_csv['table_memory_estimate_bytes'],
                          alpha=0.5, c='purple')
        axes[1, 0].set_title('Table Size vs Memory')
        axes[1, 0].set_xlabel('Table Size')
        axes[1, 0].set_ylabel('Memory (bytes)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 匹配方式分布（堆叠柱状图）
        match_counts = []
        for i in range(8):
            counts = self.df_csv[f'match_type_{i+1}'].value_counts()
            match_counts.append(counts)
        
        match_types_summary = pd.DataFrame({
            'Exact (0)': [self.df_csv[[f'match_type_{i+1}' for i in range(8)]].apply(lambda x: (x==0).sum(), axis=1).sum()],
            'Ternary (1)': [self.df_csv[[f'match_type_{i+1}' for i in range(8)]].apply(lambda x: (x==1).sum(), axis=1).sum()],
            'Wildcard (2)': [self.df_csv[[f'match_type_{i+1}' for i in range(8)]].apply(lambda x: (x==2).sum(), axis=1).sum()],
        })
        
        match_types_summary.T.plot(kind='bar', ax=axes[1, 1], legend=False, color=['skyblue', 'coral', 'lightgreen'])
        axes[1, 1].set_title('匹配方式总体分布')
        axes[1, 1].set_xlabel('匹配类型')
        axes[1, 1].set_ylabel('总数')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. wildcard_count vs memory
        axes[1, 2].scatter(self.df_csv['wildcard_count'], 
                          self.df_csv['table_memory_estimate_bytes'],
                          alpha=0.5, c='orange')
        axes[1, 2].set_title('Wildcard Count vs Memory')
        axes[1, 2].set_xlabel('Wildcard Count')
        axes[1, 2].set_ylabel('Memory (bytes)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.plots_dir, 'data_exploration.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 保存可视化图表: {plot_path}")
        
        plt.show()
        
    def generate_feature_recommendations(self):
        """生成特征推荐"""
        print("\n" + "=" * 80)
        print("  特征工程推荐")
        print("=" * 80)
        
        print("\n[推荐使用的特征]")
        print("\n1. 基础特征（CSV）:")
        print("   ✓ table_size (归一化)")
        print("   ✓ exact_count")
        print("   ✓ wildcard_count")
        print("   ✓ 8个匹配方式标志位 (match_type_1 ~ match_type_8)")
        
        print("\n2. 派生特征:")
        print("   ✓ normalized_table_size = (table_size - min) / (max - min)")
        print("   ✓ match_complexity = exact_count + wildcard_count * 2")
        print("   ✓ wildcard_ratio = wildcard_count / (exact_count + wildcard_count)")
        print("   ✓ size_complexity_interaction = table_size × match_complexity")
        
        if hasattr(self, 'df_json'):
            print("\n3. JSON 额外特征（可选）:")
            print("   ✓ total_key_width_bits")
            print("   ✓ match_complexity_score")
            print("   ✓ action_count")
            print("   ✓ key_instruction_count")
        
        print("\n[目标变量]")
        print("   ✓ table_memory_estimate_bytes (内存使用量)")
        
        print("\n[数据集划分建议]")
        print("   ✓ 训练集: 80% (6400 条)")
        print("   ✓ 测试集: 20% (1600 条)")
        
        print("\n[模型推荐]")
        print("   ✓ Random Forest (基础模型)")
        print("   ✓ Gradient Boosting (捕捉非线性)")
        print("   ✓ Extra Trees (增强泛化)")
        print("   ✓ Weighted Ensemble (加权集成)")
        
        print("\n[训练环境]")
        print("   ✓ 使用 CPU + 多进程 (n_jobs=-1)")
        print("   ✓ 数据规模适中，CPU 训练已足够快")
        print("   ✓ 预计训练时间: 1-3 分钟")


def main():
    """主程序"""
    explorer = DPDKDataExplorer()
    
    # 执行探索流程
    explorer.load_csv_data()
    explorer.load_json_data()
    explorer.analyze_features()
    explorer.visualize_data()
    explorer.generate_feature_recommendations()
    
    print("\n" + "=" * 80)
    print("  数据探索完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
