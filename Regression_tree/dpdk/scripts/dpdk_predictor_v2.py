# -*- coding: utf-8 -*-
"""
DPDK P4 内存预测系统 - V2 改进版
目标：预测 table_memory_estimate_bytes
策略：加权集成学习 + 正则化控制过拟合
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


class DPDKPredictorV2:
    """DPDK 内存预测器 - V2 改进版"""
    
    def __init__(self, csv_file="dataset.csv", use_json_features=False):
        """初始化预测器"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.csv_file = csv_file
        self.use_json_features = use_json_features
        
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.model_type = None
        self.weights = None
        self.feature_names = None
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print("=" * 80)
        print("  DPDK P4 内存预测系统 - V2 改进版")
        print("  目标: 预测 table_memory_estimate_bytes")
        print("  策略: 加权集成学习 + 正则化")
        print("=" * 80)
        
    def load_and_process_data(self):
        """加载和预处理数据"""
        print("\n[阶段 1/7] 加载和预处理数据")
        print("-" * 80)
        
        csv_path = os.path.join(self.base_dir, self.csv_file)
        self.df = pd.read_csv(csv_path)
        print(f"[OK] 加载 {len(self.df)} 条记录")
        
        # 提取特征
        feature_matrix = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="提取特征", ncols=70):
            features = []
            
            # 从文件名提取匹配方式标志位
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
            
            # 归一化 table_size
            normalized_size = (table_size - 65) / (8177 - 65)  # 根据数据范围
            
            features.extend([normalized_size, exact_count, wildcard_count])
            feature_matrix.append(features)
        
        self.X = np.array(feature_matrix)
        self._create_derived_features()
        self._extract_target()
        
        print(f"[OK] 特征矩阵: {self.X.shape}")
        print(f"[OK] 目标变量范围: {self.y.min():.0f} - {self.y.max():.0f} bytes")
        
    def _create_derived_features(self):
        """创建派生特征"""
        print("创建派生特征...")
        
        # 匹配方式特征
        match_types = self.X[:, :8]
        exact_match_count = np.sum(match_types == 0, axis=1)
        ternary_match_count = np.sum(match_types == 1, axis=1)
        wildcard_match_count = np.sum(match_types == 2, axis=1)
        
        # 基础特征
        normalized_size = self.X[:, 8]
        exact_count = self.X[:, 9]
        wildcard_count = self.X[:, 10]
        
        # 派生特征
        match_complexity = exact_count + wildcard_count * 2
        wildcard_ratio = np.where(
            (exact_count + wildcard_count) > 0,
            wildcard_count / (exact_count + wildcard_count),
            0
        )
        size_complexity_interaction = normalized_size * match_complexity
        
        # 匹配方式复杂度
        match_type_complexity = (
            exact_match_count * 1 + 
            ternary_match_count * 1.5 + 
            wildcard_match_count * 2
        )
        
        # 组合特征
        total_matches = exact_match_count + ternary_match_count + wildcard_match_count
        wildcard_match_ratio = np.where(
            total_matches > 0,
            wildcard_match_count / total_matches,
            0
        )
        
        # 拼接派生特征
        derived_features = np.column_stack([
            match_complexity,
            wildcard_ratio,
            size_complexity_interaction,
            match_type_complexity,
            exact_match_count,
            ternary_match_count,
            wildcard_match_count,
            wildcard_match_ratio
        ])
        
        self.X = np.column_stack([self.X, derived_features])
        
        # 特征名称
        self.feature_names = (
            [f'match_type_{i+1}' for i in range(8)] +
            ['normalized_size', 'exact_count', 'wildcard_count'] +
            ['match_complexity', 'wildcard_ratio', 'size_complexity_interaction',
             'match_type_complexity', 'exact_match_count', 'ternary_match_count',
             'wildcard_match_count', 'wildcard_match_ratio']
        )
        
        print(f"[OK] 总特征数: {len(self.feature_names)}")
        
    def _extract_target(self):
        """提取目标变量"""
        self.y = self.df['table_memory_estimate_bytes'].values
        
    def split_data(self, test_size=0.2, random_state=42):
        """分割数据集"""
        print("\n[阶段 2/7] 分割数据集")
        print("-" * 80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"[OK] 训练集: {len(self.X_train)}, 测试集: {len(self.X_test)}")
        
    def train_models_with_ensemble(self):
        """训练模型（加权集成）"""
        print("\n[阶段 3/7] 训练模型（加权集成）")
        print("-" * 80)
        
        models_to_test = {}
        
        # 1. Random Forest
        print("  [1/4] 训练随机森林...")
        rf_params = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5],
            'max_features': ['sqrt']
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_absolute_error', 
                              n_jobs=-1, verbose=0)
        rf_grid.fit(self.X_train_scaled, self.y_train)
        
        cv_scores = cross_val_score(rf_grid.best_estimator_, self.X_train_scaled, self.y_train,
                                    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['RandomForest'] = {
            'model': rf_grid.best_estimator_,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'params': rf_grid.best_params_
        }
        print(f"      CV MAE: {-cv_scores.mean():.2f} bytes (+/- {cv_scores.std():.2f})")
        
        # 2. Gradient Boosting
        print("  [2/4] 训练梯度提升...")
        gb_params = {
            'n_estimators': [150, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'min_samples_split': [10, 15]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='neg_mean_absolute_error',
                              n_jobs=-1, verbose=0)
        gb_grid.fit(self.X_train_scaled, self.y_train)
        
        cv_scores = cross_val_score(gb_grid.best_estimator_, self.X_train_scaled, self.y_train,
                                    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['GradientBoosting'] = {
            'model': gb_grid.best_estimator_,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'params': gb_grid.best_params_
        }
        print(f"      CV MAE: {-cv_scores.mean():.2f} bytes (+/- {cv_scores.std():.2f})")
        
        # 3. Extra Trees
        print("  [3/4] 训练 Extra Trees...")
        et = ExtraTreesRegressor(
            n_estimators=200, max_depth=18, min_samples_split=8,
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        )
        et.fit(self.X_train_scaled, self.y_train)
        
        cv_scores = cross_val_score(et, self.X_train_scaled, self.y_train,
                                    cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['ExtraTrees'] = {
            'model': et,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'params': {}
        }
        print(f"      CV MAE: {-cv_scores.mean():.2f} bytes (+/- {cv_scores.std():.2f})")
        
        # 4. 加权集成
        print("  [4/4] 构建加权集成...")
        sorted_models = sorted(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[:3]
        
        # 计算权重（基于 1/MAE）
        weights = []
        for name, info in sorted_models:
            weight = 1.0 / (info['cv_mae'] + 1e-6)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加权集成预测
        ensemble_pred = np.zeros(len(self.y_train))
        for i, (name, info) in enumerate(sorted_models):
            pred = info['model'].predict(self.X_train_scaled)
            ensemble_pred += weights[i] * pred
        
        ensemble_mae = mean_absolute_error(self.y_train, ensemble_pred)
        
        models_to_test['WeightedEnsemble'] = {
            'model': sorted_models,
            'weights': weights,
            'cv_mae': ensemble_mae,
            'cv_std': 0.0,
            'params': {}
        }
        
        print(f"      使用 {len(sorted_models)} 个模型加权集成")
        for i, (name, info) in enumerate(sorted_models):
            print(f"        {name}: 权重={weights[i]:.3f}, CV MAE={info['cv_mae']:.2f} bytes")
        print(f"      加权集成 MAE: {ensemble_mae:.2f} bytes")
        
        # 选择最佳模型
        best_name = min(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[0]
        best_info = models_to_test[best_name]
        
        if best_name == 'WeightedEnsemble':
            self.model = best_info['model']
            self.weights = best_info['weights']
            self.model_type = 'weighted_ensemble'
        else:
            self.model = best_info['model']
            self.model_type = 'single'
        
        print(f"\n[OK] 最佳模型: {best_name}")
        print(f"    训练集 MAE: {best_info['cv_mae']:.2f} bytes")
        if best_name == 'WeightedEnsemble':
            print(f"    集成策略: 加权平均 ({len(self.model)} 个模型)")
        
    def evaluate_model(self):
        """评估模型性能"""
        print("\n[阶段 4/7] 模型性能评估")
        print("-" * 80)
        
        # 预测
        if self.model_type == 'weighted_ensemble':
            train_pred = np.zeros(len(self.y_train))
            test_pred = np.zeros(len(self.y_test))
            for i, (name, info) in enumerate(self.model):
                train_pred += self.weights[i] * info['model'].predict(self.X_train_scaled)
                test_pred += self.weights[i] * info['model'].predict(self.X_test_scaled)
        else:
            train_pred = self.model.predict(self.X_train_scaled)
            test_pred = self.model.predict(self.X_test_scaled)
        
        # 计算指标
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        train_mae = mean_absolute_error(self.y_train, train_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        
        # 相对误差
        train_mape = np.mean(np.abs((self.y_train - train_pred) / self.y_train)) * 100
        test_mape = np.mean(np.abs((self.y_test - test_pred) / self.y_test)) * 100
        
        print("\n=== 模型性能 ===")
        print(f"  训练集: R² = {train_r2:.4f}, MAE = {train_mae:.2f} bytes, RMSE = {train_rmse:.2f} bytes, MAPE = {train_mape:.2f}%")
        print(f"  测试集: R² = {test_r2:.4f}, MAE = {test_mae:.2f} bytes, RMSE = {test_rmse:.2f} bytes, MAPE = {test_mape:.2f}%")
        print(f"  过拟合差距: R² = {train_r2 - test_r2:.4f}, MAE = {test_mae - train_mae:.2f} bytes")
        
        # 存储性能指标
        self.performance = {
            'train': {'r2': train_r2, 'mae': train_mae, 'rmse': train_rmse, 'mape': train_mape},
            'test': {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse, 'mape': test_mape}
        }
        
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n[阶段 5/7] 特征重要性分析")
        print("-" * 80)
        
        if self.model_type == 'weighted_ensemble':
            # 加权平均特征重要性
            importance = np.zeros(len(self.feature_names))
            for i, (name, info) in enumerate(self.model):
                if hasattr(info['model'], 'feature_importances_'):
                    importance += self.weights[i] * info['model'].feature_importances_
        else:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            else:
                print("[警告] 模型不支持特征重要性分析")
                return
        
        # 排序
        indices = np.argsort(importance)[::-1]
        
        print("\n前10个重要特征:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"  {i+1}. {self.feature_names[idx]}: {importance[idx]:.4f}")
        
        # 可视化
        plt.figure(figsize=(12, 6))
        top_n = 15
        top_indices = indices[:top_n]
        plt.barh(range(top_n), importance[top_indices], color='skyblue')
        plt.yticks(range(top_n), [self.feature_names[i] for i in top_indices])
        plt.xlabel('重要性')
        plt.title('特征重要性分析（Top 15）')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] 保存特征重要性图表: {plot_path}")
        plt.close()
        
    def visualize_predictions(self):
        """可视化预测结果"""
        print("\n[阶段 6/7] 可视化预测结果")
        print("-" * 80)
        
        # 预测
        if self.model_type == 'weighted_ensemble':
            test_pred = np.zeros(len(self.y_test))
            for i, (name, info) in enumerate(self.model):
                test_pred += self.weights[i] * info['model'].predict(self.X_test_scaled)
        else:
            test_pred = self.model.predict(self.X_test_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 预测 vs 实际
        axes[0].scatter(self.y_test, test_pred, alpha=0.5, c='coral')
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'k--', lw=2, label='理想预测')
        axes[0].set_xlabel('实际内存 (bytes)')
        axes[0].set_ylabel('预测内存 (bytes)')
        axes[0].set_title(f'预测 vs 实际 (R² = {self.performance["test"]["r2"]:.4f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 残差分布
        residuals = self.y_test - test_pred
        axes[1].hist(residuals, bins=50, color='lightgreen', edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('残差 (bytes)')
        axes[1].set_ylabel('频数')
        axes[1].set_title(f'残差分布 (MAE = {self.performance["test"]["mae"]:.2f} bytes)')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, 'prediction_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 保存预测分析图表: {plot_path}")
        plt.close()
        
    def save_models(self):
        """保存模型"""
        print("\n[阶段 7/7] 保存模型")
        print("-" * 80)
        
        joblib.dump(self.model, os.path.join(self.models_dir, "memory_model_v2.pkl"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler_v2.pkl"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names_v2.pkl"))
        joblib.dump(self.performance, os.path.join(self.models_dir, "performance_v2.pkl"))
        
        if self.model_type == 'weighted_ensemble':
            joblib.dump(self.weights, os.path.join(self.models_dir, "weights_v2.pkl"))
        
        print("[OK] 模型已保存")
        print(f"    memory_model_v2.pkl")
        print(f"    scaler_v2.pkl")
        print(f"    feature_names_v2.pkl")
        print(f"    performance_v2.pkl")
        if self.model_type == 'weighted_ensemble':
            print(f"    weights_v2.pkl")
    
    def predict(self, match_types, table_size, exact_count, wildcard_count):
        """预测内存使用量"""
        # 构建特征向量
        normalized_size = (table_size - 65) / (8177 - 65)
        features = match_types + [normalized_size, exact_count, wildcard_count]
        
        # 添加派生特征（需要与训练时一致）
        match_types_arr = np.array(match_types)
        exact_match_count = np.sum(match_types_arr == 0)
        ternary_match_count = np.sum(match_types_arr == 1)
        wildcard_match_count = np.sum(match_types_arr == 2)
        
        match_complexity = exact_count + wildcard_count * 2
        wildcard_ratio = wildcard_count / (exact_count + wildcard_count) if (exact_count + wildcard_count) > 0 else 0
        size_complexity_interaction = normalized_size * match_complexity
        match_type_complexity = exact_match_count * 1 + ternary_match_count * 1.5 + wildcard_match_count * 2
        total_matches = exact_match_count + ternary_match_count + wildcard_match_count
        wildcard_match_ratio = wildcard_match_count / total_matches if total_matches > 0 else 0
        
        features.extend([
            match_complexity, wildcard_ratio, size_complexity_interaction,
            match_type_complexity, exact_match_count, ternary_match_count,
            wildcard_match_count, wildcard_match_ratio
        ])
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'weighted_ensemble':
            pred = 0
            for i, (name, info) in enumerate(self.model):
                pred += self.weights[i] * info['model'].predict(X_scaled)[0]
        else:
            pred = self.model.predict(X_scaled)[0]
        
        return {
            'memory_bytes': pred,
            'memory_kb': pred / 1024,
            'memory_mb': pred / (1024 * 1024)
        }


def main():
    """主程序"""
    import time
    start_time = time.time()
    
    predictor = DPDKPredictorV2()
    
    # 执行训练流程
    predictor.load_and_process_data()
    predictor.split_data()
    predictor.train_models_with_ensemble()
    predictor.evaluate_model()
    predictor.analyze_feature_importance()
    predictor.visualize_predictions()
    predictor.save_models()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  训练完成！总用时: {elapsed:.2f} 秒")
    print(f"  测试集 R²: {predictor.performance['test']['r2']:.4f}")
    print(f"  测试集 MAE: {predictor.performance['test']['mae']:.2f} bytes")
    print(f"  测试集 MAPE: {predictor.performance['test']['mape']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
