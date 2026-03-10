# -*- coding: utf-8 -*-
"""
BMv2 P4 内存预测系统
目标：预测 memory (已减去基准值的增量，单位KB)
策略：
1. 使用预处理后的数据 (processed_features.csv)
2. Memory 已经是增量值 (KB)，无需再计算 Base_RSS
3. 输入特征：8个匹配类型 + Size
4. 优化：使用 Log 变换降低 MAPE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class BMv2Predictor:
    def __init__(self, csv_file="data/processed_features.csv"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.csv_file = csv_file
        
        self.df = None
        self.X = None
        self.y = None  # Memory 增量 (KB)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'total_len', 'protocol', 'flags', 'ttl', 
            'src_port', 'dst_port', 'tcp_flags_2', 'tcp_flags_1', 
            'size'
        ]
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_and_process_data(self):
        print("\n[1/6] 加载预处理后的数据...")
        csv_path = os.path.join(self.base_dir, self.csv_file)
        self.df = pd.read_csv(csv_path)
        
        # 1. 清洗数据
        initial_len = len(self.df)
        self.df = self.df.dropna()
        if len(self.df) < initial_len:
            print(f"  - 丢弃了 {initial_len - len(self.df)} 条缺失值数据")
            
        # 2. 准备特征和目标
        # Memory 列已经是增量值 (KB)，直接使用
        self.X = self.df[self.feature_names].values
        self.y = self.df['memory'].values
        
        print(f"  - 数据集形状: {self.df.shape}")
        print(f"  - 特征矩阵形状: {self.X.shape}")
        print(f"  - 目标变量 (Memory增量 KB) 范围: {self.y.min():.2f} 到 {self.y.max():.2f}")
        print(f"  - 目标变量平均值: {self.y.mean():.2f} KB")

    def split_data(self):
        print("\n[2/6] 数据集划分...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # 标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"  - 训练集: {len(self.X_train)}, 测试集: {len(self.X_test)}")

    def train_model(self):
        print("\n[3/6] 训练模型 (Log变换优化 MAPE)...")
        
        # 1. 处理目标变量: 对数变换
        # Memory 增量已经全部为正，直接使用 log1p
        y_train_log = np.log1p(self.y_train)
        
        print(f"  - 目标变量预处理: Log1p变换")
        
        # 2. 定义模型
        models = {
            'HistGradientBoosting': HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
        }
        
        best_score = float('inf')
        best_model_name = ""
        
        for name, model in models.items():
            print(f"  - 训练 {name}...")
            # 交叉验证 (使用 Log 变换后的数据)
            scores = cross_val_score(model, self.X_train_scaled, y_train_log, 
                                   cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            mae_log = -scores.mean()
            print(f"    CV MAE (Log Space): {mae_log:.4f}")
            
            if mae_log < best_score:
                best_score = mae_log
                self.model = model
                best_model_name = name
        
        print(f"  - 最佳模型: {best_model_name}")
        self.model.fit(self.X_train_scaled, y_train_log)

    def evaluate(self):
        print("\n[4/6] 模型评估...")
        
        # 1. 预测 (得到 Log 空间的值)
        y_pred_log = self.model.predict(self.X_test_scaled)
        
        # 2. 还原 (逆变换: expm1)
        y_pred = np.expm1(y_pred_log)
        y_test = self.y_test
        
        # 3. 计算指标 (单位: KB)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # 4. 计算 MAPE
        # 过滤掉极小值 (< 1 KB)
        threshold = 1.0  # 1 KB
        mask = np.abs(y_test) > threshold
        
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            sample_count = np.sum(mask)
        else:
            mape = 0.0
            sample_count = 0
            
        print(f"  === 测试集性能 ===")
        print(f"  MAE : {mae:.2f} KB")
        print(f"  RMSE: {rmse:.2f} KB")
        print(f"  R²  : {r2:.4f}")
        print(f"  MAPE: {mape:.2f}% (统计 {sample_count}/{len(y_test)} 个样本)")
        
        # 保存性能指标
        with open(os.path.join(self.models_dir, 'performance.txt'), 'w') as f:
            f.write(f"Base_RSS: 26.00 MB\n")
            f.write(f"MAE_KB: {mae}\n")
            f.write(f"RMSE_KB: {rmse}\n")
            f.write(f"R2: {r2}\n")
            f.write(f"MAPE: {mape}\n")

        return y_test, y_pred

    def visualize(self, y_real, y_pred):
        print("\n[5/6] 可视化结果...")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_real, y_pred, alpha=0.5, color='blue', label='预测值')
        plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2, label='理想线')
        plt.xlabel('实际 Memory 增量 (KB)')
        plt.ylabel('预测 Memory 增量 (KB)')
        plt.title('BMv2 内存增量预测: 实际 vs 预测')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'prediction_scatter.png'), dpi=300)
        plt.close()
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45)
            plt.title('特征重要性')
            plt.ylabel('重要性')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            print("  - 特征重要性 Top 5:")
            for i in range(min(5, len(indices))):
                idx = indices[i]
                print(f"    {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        else:
            print("  - 当前模型不支持直接获取特征重要性")

    def save(self):
        print("\n[6/6] 保存模型...")
        joblib.dump(self.model, os.path.join(self.models_dir, 'bmv2_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        print("  - 模型已保存至 models/ 目录")

if __name__ == "__main__":
    predictor = BMv2Predictor()
    predictor.load_and_process_data()
    predictor.split_data()
    predictor.train_model()
    y_real, y_pred = predictor.evaluate()
    predictor.visualize(y_real, y_pred)
    predictor.save()
    
    print("\n" + "=" * 60)
    print("  训练完成！")
    print("=" * 60)
