# -*- coding: utf-8 -*-
"""
DPDK P4 内存预测系统 - 基础版
简化版本，用于快速验证和对比
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import re
from tqdm import tqdm


class DPDKPredictor:
    """DPDK 内存预测器 - 基础版"""
    
    def __init__(self, csv_file="dataset.csv"):
        """初始化预测器"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.csv_file = csv_file
        
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        print("=" * 80)
        print("  DPDK P4 内存预测系统 - 基础版")
        print("=" * 80)
        
    def load_and_process_data(self):
        """加载和预处理数据"""
        print("\n[1/5] 加载数据...")
        
        csv_path = os.path.join(self.base_dir, self.csv_file)
        self.df = pd.read_csv(csv_path)
        print(f"[OK] 加载 {len(self.df)} 条记录")
        
        # 提取特征
        feature_matrix = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="提取特征"):
            features = []
            
            # 从文件名提取匹配方式
            p4_file = row['p4_file']
            parts = p4_file.replace('.p4', '').split('_')
            if len(parts) >= 9:
                match_types = [int(x) for x in parts[:8]]
            else:
                match_types = [0] * 8
            
            features.extend(match_types)
            
            # 基础特征
            table_size = row['table_size']
            exact_count = row['exact_count']
            wildcard_count = row['wildcard_count']
            
            normalized_size = (table_size - 65) / (8177 - 65)
            
            features.extend([normalized_size, exact_count, wildcard_count])
            
            # 简单派生特征
            match_complexity = exact_count + wildcard_count * 2
            features.append(match_complexity)
            
            feature_matrix.append(features)
        
        self.X = np.array(feature_matrix)
        self.y = self.df['table_memory_estimate_bytes'].values
        
        self.feature_names = (
            [f'match_type_{i+1}' for i in range(8)] +
            ['normalized_size', 'exact_count', 'wildcard_count', 'match_complexity']
        )
        
        print(f"[OK] 特征矩阵: {self.X.shape}")
        
    def train_model(self):
        """训练模型"""
        print("\n[2/5] 训练模型...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练随机森林
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 评估
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"\n[OK] 模型训练完成")
        print(f"  训练集: R² = {train_r2:.4f}, MAE = {train_mae:.2f} bytes")
        print(f"  测试集: R² = {test_r2:.4f}, MAE = {test_mae:.2f} bytes")
        
    def save_models(self):
        """保存模型"""
        print("\n[3/5] 保存模型...")
        
        joblib.dump(self.model, os.path.join(self.models_dir, "memory_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names.pkl"))
        
        print("[OK] 模型已保存")


def main():
    """主程序"""
    predictor = DPDKPredictor()
    predictor.load_and_process_data()
    predictor.train_model()
    predictor.save_models()
    
    print("\n" + "=" * 80)
    print("  基础版训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
