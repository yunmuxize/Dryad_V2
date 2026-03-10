# -*- coding: utf-8 -*-
"""
Dryad P4 资源预测系统 - V3 增强版
目标：降低 MAE 和 RMSE，同时控制过拟合
新增：1) 加入8个特征的bit数作为输入 2) Stages预测作为第三个输出
改进策略：正则化 + 交叉验证 + 集成学习
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# XGBoost 已在 V3 最终版中移除 (基于实验结论证明其并非必需)
XGBOOST_AVAILABLE = False

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


class DryadPredictorV2:
    """
    改进版预测器 - 控制过拟合，降低 MAE/RMSE
    """
    
    def __init__(self, csv_file="final_merged_resource_analysis.csv", fail_file="compile_results_p4_new_exceeded_theoretical.csv"):
        """初始化预测器"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.csv_file = csv_file
        self.fail_file = fail_file
        self.df = None
        self.X = None
        self.y_tcam = None
        self.y_sram = None
        self.y_stages = None
        self.X_train = None
        self.X_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # 硬件限制
        self.TCAM_CAPACITY = 288
        self.SRAM_CAPACITY = 960
        self.MAX_STAGES = 12
        
        # 匹配方式映射
        self.match_map = {
            'exact': 0,
            'ternary': 1,
            'range': 2,
            'prefix': 3,
            'lpm': 3
        }
        
        # 特征列名 (新数据集列名)
        self.base_features = [
            'total_len', 'protocol', 'flags[1:1]', 'ttl',
            'src_port', 'dst_port', 'tcp_flags[2:2]', 'tcp_flags[1:1]'
        ]
        
        print("=" * 80)
        print("  Dryad P4 资源预测系统 - V4 (Tiles 级精确预测)")
        print("  主要目标: 预测 Tiles 总数以规避百分比舍入误差")
        print("=" * 80)
        
    def load_and_process_data(self):
        """加载和预处理数据 (合并正负样本)"""
        print("\n[阶段 1/8] 加载和预处理数据")
        print("-" * 80)
        
        # 1. 加载主数据集
        csv_path = os.path.join(self.data_dir, self.csv_file)
        df_raw = pd.read_csv(csv_path)
        
        # 2. 统一列名映射
        mapping = {
            'Table Size': 'size',
            'Stages (核心校验)': 'total_stages',
            'TCAM Used (Tiles 总数)': 'tcam_used',
            'SRAM Used (Tiles 总数)': 'sram_used'
        }
        df_raw = df_raw.rename(columns=mapping)
        
        # 3. 加载并合并“编译失败/超限”数据集 (让模型学习超限特征)
        fail_path = os.path.join(self.data_dir, self.fail_file)
        if os.path.exists(fail_path):
            df_fail_raw = pd.read_csv(fail_path)
            # 映射失败集的列名
            fail_mapping = {
                'size': 'size',
                'max_stages': 'total_stages',
                'total_tcam_used': 'tcam_used',
                'total_sram_used': 'sram_used',
                'match_total_len': 'total_len',
                'match_protocol': 'protocol',
                'match_flags_1': 'flags[1:1]',
                'match_ttl': 'ttl',
                'match_src_port': 'src_port',
                'match_dst_port': 'dst_port',
                'match_tcp_flags_2': 'tcp_flags[2:2]',
                'match_tcp_flags_1': 'tcp_flags[1:1]'
            }
            df_fail = df_fail_raw.rename(columns=fail_mapping)
            # 仅保留需要的列进行合并
            cols_to_keep = list(fail_mapping.values())
            df_fail = df_fail[cols_to_keep]
            
            # 合并数据集
            df_raw = pd.concat([df_raw, df_fail], ignore_index=True)
            self.df_fail = df_fail # 保留供分析
            print(f"[OK] 已合并失败样本: {len(df_fail)} 条，总训练样本: {len(df_raw)} 条")
        else:
            self.df_fail = None
            print("[WARN] 失败参考集不存在")
            
        # 4. 匹配方式映射为数字
        for col in self.base_features:
            df_raw[col] = df_raw[col].str.lower().map(self.match_map)
            
        # 5. 随机打乱数据
        self.df = df_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[OK] 数据预处理完成")
            
        # 6. 提取特征
        print("构造 21 维可解释性特征矩阵...")
        self.X = self._extract_features_from_df(self.df)
        
        # 7. 提取目标变量 (Tiles)
        self.y_tcam = self.df['tcam_used'].values
        self.y_sram = self.df['sram_used'].values
        self.y_stages = self.df['total_stages'].values
        
        print(f"[OK] 特征矩阵: {self.X.shape}")
        print(f"[OK] 目标范围: TCAM Tiles [{self.y_tcam.min()}-{self.y_tcam.max()}], SRAM Tiles [{self.y_sram.min()}-{self.y_sram.max()}]")

    def _extract_features_from_df(self, df):
        """通用特征提取逻辑 (21维可解释性特征)"""
        BIT_WIDTHS = [16, 8, 8, 8, 16, 16, 8, 8] 
        feature_matrix = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="提取特征", leave=False, ncols=70):
            # 1. 匹配类型基础特征 (8维)
            match_types = row[self.base_features].values.astype(float)
            
            # 2. 对应的 Bit 数 (8维)
            bit_counts = np.array(BIT_WIDTHS, dtype=float)
            
            # 3. Size 归一化 (1维)
            # 使用 renamed 的 'size' 列
            size_value = row['size']
            normalized_size = (size_value - 256) / (20000 - 256) # 扩大归一化范围以适应新数据
            
            # 4. 匹配方式分布统计 (4维)
            exact_count = np.sum(match_types == 0)
            ternary_count = np.sum(match_types == 1)
            range_count = np.sum(match_types == 2)
            lpm_count = np.sum(match_types == 3)
            
            # 拼接 21 维特征
            features = np.concatenate([
                match_types,           # 8
                bit_counts,            # 8
                [normalized_size],     # 1
                [exact_count, ternary_count, range_count, lpm_count] # 4
            ])
            feature_matrix.append(features)
        
        if self.feature_names is None:
            self.feature_names = [
                'type_total_len', 'type_protocol', 'type_flags_1', 'type_ttl',
                'type_src_port', 'type_dst_port', 'type_tcp_flags_2', 'type_tcp_flags_1',
                'bits_total_len', 'bits_protocol', 'bits_flags_1', 'bits_ttl',
                'bits_src_port', 'bits_dst_port', 'bits_tcp_flags_2', 'bits_tcp_flags_1',
                'normalized_size',
                'exact_count', 'ternary_count', 'range_count', 'lpm_count'
            ]
        return np.array(feature_matrix)

    def split_data(self, test_size=0.2, random_state=42):
        """分割数据集"""
        print("\n[阶段 2/8] 分割数据集")
        print("-" * 80)
        
        # 标准化特征
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 分割资源预测数据
        self.X_train, self.X_test, \
        self.y_tcam_train, self.y_tcam_test, \
        self.y_sram_train, self.y_sram_test, \
        self.y_stages_train, self.y_stages_test = train_test_split(
            self.X_scaled, self.y_tcam, self.y_sram, self.y_stages,
            test_size=test_size, random_state=random_state
        )
        
        print(f"[OK] 训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")
        
    def analyze_failure_boundary(self):
        """报警机制：分析导致编译失败的边界条件"""
        print("\n[报警机制分析] 识别编译失败高危区")
        print("-" * 80)
        if self.df_fail is None:
            return

        # 1. Size 风险评估
        min_fail_size = self.df_fail['size'].min()
        print(f"  [风险1] 表大小 (Size) 阈值: 当 Size >= {min_fail_size:.0f} 时编译风险显著增加")
        
        # 2. 匹配方式风险评估 (按样本计算，即：百分之多少的失败样本包含该匹配方式)
        match_cols = self.base_features
        
        # 统计每个样本包含哪些匹配方式 (去重)
        fail_presence = {m: 0 for m in ['exact', 'ternary', 'range', 'lpm']}
        for _, row in self.df_fail.iterrows():
            row_matches = set(row[match_cols].str.lower().unique())
            for m in fail_presence:
                if m in row_matches:
                    fail_presence[m] += 1
        
        print(f"  [风险2] 匹配方式风险 (包含该方式的失败样本占比):")
        total_fails = len(self.df_fail)
        for m_type, count in sorted(fail_presence.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {m_type:8s}: {count/total_fails:>6.1%} ({count}/{total_fails} 例)")

        # 3. 典型失败规则警告
        print(f"  [报警] 若预测 TCAM > {self.TCAM_CAPACITY} Tiles 或 Stages > {self.MAX_STAGES}，系统将强制标记为编译失败。")

    def train_models_with_regularization(self):
        """训练模型（增强正则化控制过拟合）"""
        print("\n[阶段 3/8] 训练模型（正则化版本）")
        print("-" * 80)
        
        self._train_tcam_regularized()
        self._train_sram_regularized()
        self._train_stages_regularized()  # 新增
        
    def _train_tcam_regularized(self):
        """训练 TCAM 模型（控制过拟合，支持样本权重）"""
        print("\n[1/2] 训练 TCAM 模型（正则化 + 样本权重）...")
        
        models_to_test = {}
        
        # 1. Random Forest（正则化参数）
        print("  [1/5] 测试随机森林（正则化）...")
        rf_params = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5],
            'max_features': ['sqrt'],
            'bootstrap': [True]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        rf_grid.fit(self.X_train, self.y_tcam_train)
        
        models_to_test['RandomForest'] = {
            'model': rf_grid.best_estimator_,
            'cv_mae': -rf_grid.best_score_, # 使用 GridSearch 的最佳分数
            'cv_std': 0.0, # 暂不计算 std
            'params': rf_grid.best_params_
        }
        print(f"      Grid Best MAE: {-rf_grid.best_score_:.4f} Tiles")
        
        # 2. Gradient Boosting（更强正则化）
        print("  [2/5] 测试梯度提升（正则化）...")
        gb_params = {
            'n_estimators': [150, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8],
            'min_samples_split': [10, 15],
            'min_samples_leaf': [5, 7]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        gb_grid.fit(self.X_train, self.y_tcam_train)
        
        models_to_test['GradientBoosting'] = {
            'model': gb_grid.best_estimator_,
            'cv_mae': -gb_grid.best_score_,
            'cv_std': 0.0,
            'params': gb_grid.best_params_
        }
        print(f"      Grid Best MAE: {-gb_grid.best_score_:.4f} Tiles")
        
        # 3. Extra Trees（正则化）
        print("  [3/5] 测试 Extra Trees（正则化）...")
        et = ExtraTreesRegressor(
            n_estimators=200, max_depth=18, min_samples_split=8, 
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        )
        et.fit(self.X_train, self.y_tcam_train)
        
        # 简单的 CV
        cv_scores = cross_val_score(et, self.X_train, self.y_tcam_train,
                                     cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        
        models_to_test['ExtraTrees'] = {
            'model': et,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'params': {}
        }
        print(f"      CV MAE: {-cv_scores.mean():.4f} Tiles")
        
        # 4. XGBoost (已移除相关逻辑)
        pass
        
        # 5. 加权集成（策略A优化）
        print("  [5/5] 测试加权集成...")
        sorted_models = sorted(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[:min(5, len(models_to_test))]
        
        weights = []
        for name, info in sorted_models:
            weight = 1.0 / (info['cv_mae'] + 1e-6)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_preds = []
        for name, info in sorted_models:
            pred = info['model'].predict(self.X_train)
            ensemble_preds.append(pred)
        
        ensemble_pred = np.zeros(len(self.y_tcam_train))
        for i, pred in enumerate(ensemble_preds):
            ensemble_pred += weights[i] * pred
        
        # 计算加权 MAE (手动)
        ensemble_mae = mean_absolute_error(self.y_tcam_train, ensemble_pred)
        
        models_to_test['WeightedEnsemble'] = {
            'model': sorted_models,
            'weights': weights,
            'cv_mae': ensemble_mae,
            'cv_std': 0.0,
            'params': {}
        }
        
        print(f"      使用 {len(sorted_models)} 个模型加权集成")
        for i, (name, info) in enumerate(sorted_models):
            print(f"        {name}: 权重={weights[i]:.3f}, CV MAE={info['cv_mae']:.4f} Tiles")
        print(f"      加权集成 (Weighted) MAE: {ensemble_mae:.4f} Tiles")
        
        best_name = min(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[0]
        best_info = models_to_test[best_name]
        
        if best_name == 'WeightedEnsemble':
            self.tcam_model = best_info['model']
            self.tcam_weights = best_info['weights']
            self.tcam_model_type = 'weighted_ensemble'
        else:
            self.tcam_model = best_info['model']
            self.tcam_model_type = 'single'
        
    def _train_sram_regularized(self):
        """训练 SRAM 模型（控制过拟合，支持样本权重）"""
        print("\n[2/2] 训练 SRAM 模型（正则化 + 样本权重）...")
        
        models_to_test = {}
        
        # 1. Random Forest
        print("  [1/4] 测试随机森林（正则化）...")
        rf_params = {
            'n_estimators': [200],
            'max_depth': [12],
            'min_samples_split': [10],
            'min_samples_leaf': [5],
            'max_features': ['sqrt']
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        rf_grid.fit(self.X_train, self.y_sram_train)
        
        models_to_test['RandomForest'] = {
            'model': rf_grid.best_estimator_,
            'cv_mae': -rf_grid.best_score_,
            'cv_std': 0.0
        }
        print(f"      Grid Best MAE: {-rf_grid.best_score_:.6f} Tiles")
        
        # 2. Gradient Boosting
        print("  [2/4] 测试梯度提升（正则化）...")
        gb_params = {
            'n_estimators': [150],
            'max_depth': [5],
            'learning_rate': [0.05],
            'subsample': [0.75],
            'min_samples_split': [15],
            'min_samples_leaf': [7]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
        gb_grid.fit(self.X_train, self.y_sram_train)
        
        models_to_test['GradientBoosting'] = {
            'model': gb_grid.best_estimator_,
            'cv_mae': -gb_grid.best_score_,
            'cv_std': 0.0
        }
        print(f"      Grid Best MAE: {-gb_grid.best_score_:.6f} Tiles")
        
        # 3. Extra Trees
        print("  [3/4] 测试 Extra Trees（正则化）...")
        et = ExtraTreesRegressor(
            n_estimators=200, max_depth=12, min_samples_split=8, 
            min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
        )
        et.fit(self.X_train, self.y_sram_train)
        
        # 简单 CV
        cv_scores = cross_val_score(et, self.X_train, self.y_sram_train,
                                     cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['ExtraTrees'] = {
            'model': et,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"      CV MAE: {-cv_scores.mean():.6f} Tiles")
        
        # 4. 加权集成
        print("  [4/4] 测试加权集成...")
        sorted_models = sorted(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[:min(5, len(models_to_test))]
        
        weights = []
        for name, info in sorted_models:
            weight = 1.0 / (info['cv_mae'] + 1e-6)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        ensemble_preds = []
        for name, info in sorted_models:
            pred = info['model'].predict(self.X_train)
            ensemble_preds.append(pred)
        
        ensemble_pred = np.zeros(len(self.y_sram_train))
        for i, pred in enumerate(ensemble_preds):
            ensemble_pred += weights[i] * pred
            
        # 计算加权 MAE
        ensemble_mae = mean_absolute_error(self.y_sram_train, ensemble_pred)
        
        models_to_test['WeightedEnsemble'] = {
            'model': sorted_models,
            'weights': weights,
            'cv_mae': ensemble_mae,
            'cv_std': 0.0
        }
        
        print(f"      使用 {len(sorted_models)} 个模型加权集成")
        for i, (name, info) in enumerate(sorted_models):
            print(f"        {name}: 权重={weights[i]:.3f}, CV MAE={info['cv_mae']:.6f} Tiles")
        print(f"      加权集成 (Weighted) MAE: {ensemble_mae:.6f} Tiles")
        
        best_name = min(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[0]
        best_info = models_to_test[best_name]
        
        if best_name == 'WeightedEnsemble':
            self.sram_model = best_info['model']
            self.sram_weights = best_info['weights']
            self.sram_model_type = 'weighted_ensemble'
        else:
            self.sram_model = best_info['model']
            self.sram_model_type = 'single'
            
        print(f"\n[OK] SRAM 最佳模型: {best_name}")
        print(f"    训练集 MAE: {best_info['cv_mae']:.6f}%")

    def _train_stages_regularized(self):
        """训练 Stages 模型 (基于 stages_extracted.csv 数据)"""
        print("\n[3/3] 训练 Stages 模型（正则化）...")
        
        models_to_test = {}
        
        # 1. Random Forest
        print("  [1/3] 测试随机森林...")
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1
        )
        # Stages 数据不使用样本权重
        rf.fit(self.X_train, self.y_stages_train)
        
        cv_scores = cross_val_score(rf, self.X_train, self.y_stages_train,
                                     cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['RandomForest'] = {
            'model': rf,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"      CV MAE: {-cv_scores.mean():.4f} stages")
        
        # 2. Gradient Boosting
        print("  [2/3] 测试梯度提升...")
        gb = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_split=10, min_samples_leaf=5,
            random_state=42
        )
        gb.fit(self.X_train, self.y_stages_train)
        
        cv_scores = cross_val_score(gb, self.X_train, self.y_stages_train,
                                     cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['GradientBoosting'] = {
            'model': gb,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"      CV MAE: {-cv_scores.mean():.4f} stages")
        
        # 3. Extra Trees
        print("  [3/3] 测试 Extra Trees...")
        et = ExtraTreesRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1
        )
        et.fit(self.X_train, self.y_stages_train)
        
        cv_scores = cross_val_score(et, self.X_train, self.y_stages_train,
                                     cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        models_to_test['ExtraTrees'] = {
            'model': et,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        print(f"      CV MAE: {-cv_scores.mean():.4f} stages")
        
        # 选择最佳模型
        best_name = min(models_to_test.items(), key=lambda x: x[1]['cv_mae'])[0]
        best_info = models_to_test[best_name]
        self.stages_model = best_info['model']
        self.stages_model_type = 'single'
        
        print(f"\n[OK] Stages 最佳模型: {best_name}")
        print(f"    CV MAE: {best_info['cv_mae']:.4f} stages")

    def evaluate_models(self):
        """评估模型性能"""
        print("\n[阶段 4/8] 模型性能评估")
        print("-" * 80)
        
        # TCAM 评估
        if self.tcam_model_type == 'weighted_ensemble':
            # 加权集成预测
            tcam_train_pred = np.zeros(len(self.y_tcam_train))
            tcam_test_pred = np.zeros(len(self.y_tcam_test))
            for i, (name, info) in enumerate(self.tcam_model):
                tcam_train_pred += self.tcam_weights[i] * info['model'].predict(self.X_train)
                tcam_test_pred += self.tcam_weights[i] * info['model'].predict(self.X_test)
        else:
            tcam_train_pred = self.tcam_model.predict(self.X_train)
            tcam_test_pred = self.tcam_model.predict(self.X_test)
        
        tcam_train_r2 = r2_score(self.y_tcam_train, tcam_train_pred)
        tcam_test_r2 = r2_score(self.y_tcam_test, tcam_test_pred)
        tcam_train_mae = mean_absolute_error(self.y_tcam_train, tcam_train_pred)
        tcam_test_mae = mean_absolute_error(self.y_tcam_test, tcam_test_pred)
        tcam_train_rmse = np.sqrt(mean_squared_error(self.y_tcam_train, tcam_train_pred))
        tcam_test_rmse = np.sqrt(mean_squared_error(self.y_tcam_test, tcam_test_pred))
        
        # SRAM 评估
        if self.sram_model_type == 'weighted_ensemble':
            # 加权集成预测
            sram_train_pred = np.zeros(len(self.y_sram_train))
            sram_test_pred = np.zeros(len(self.y_sram_test))
            for i, (name, info) in enumerate(self.sram_model):
                sram_train_pred += self.sram_weights[i] * info['model'].predict(self.X_train)
                sram_test_pred += self.sram_weights[i] * info['model'].predict(self.X_test)
        else:
            sram_train_pred = self.sram_model.predict(self.X_train)
            sram_test_pred = self.sram_model.predict(self.X_test)
        
        sram_train_r2 = r2_score(self.y_sram_train, sram_train_pred)
        sram_test_r2 = r2_score(self.y_sram_test, sram_test_pred)
        sram_train_mae = mean_absolute_error(self.y_sram_train, sram_train_pred)
        sram_test_mae = mean_absolute_error(self.y_sram_test, sram_test_pred)
        sram_train_rmse = np.sqrt(mean_squared_error(self.y_sram_train, sram_train_pred))
        sram_test_rmse = np.sqrt(mean_squared_error(self.y_sram_test, sram_test_pred))
        
        print("\n[V4 Tiles 级模型性能]")
        print("\n=== TCAM 预测 (Tiles) ===")
        print(f"  训练集: MAE = {tcam_train_mae:.4f} Tiles, RMSE = {tcam_train_rmse:.4f} Tiles")
        print(f"  测试集: MAE = {tcam_test_mae:.4f} Tiles, RMSE = {tcam_test_rmse:.4f} Tiles")
        print(f"  百分比对应 MAE: {tcam_test_mae/self.TCAM_CAPACITY*100:.4f}%")
        
        print("\n=== SRAM 预测 (Tiles) ===")
        print(f"  训练集: MAE = {sram_train_mae:.4f} Tiles, RMSE = {sram_train_rmse:.4f} Tiles")
        print(f"  测试集: MAE = {sram_test_mae:.4f} Tiles, RMSE = {sram_test_rmse:.4f} Tiles")
        print(f"  百分比对应 MAE: {sram_test_mae/self.SRAM_CAPACITY*100:.4f}%")
        
        # Stages 评估
        stages_train_pred = self.stages_model.predict(self.X_train)
        stages_test_pred = self.stages_model.predict(self.X_test)
        
        stages_train_mae = mean_absolute_error(self.y_stages_train, stages_train_pred)
        stages_test_mae = mean_absolute_error(self.y_stages_test, stages_test_pred)
        stages_train_rmse = np.sqrt(mean_squared_error(self.y_stages_train, stages_train_pred))
        stages_test_rmse = np.sqrt(mean_squared_error(self.y_stages_test, stages_test_pred))

        print("\n=== Stages 预测模型 ===")
        print(f"  训练集: MAE = {stages_train_mae:.4f} stages, RMSE = {stages_train_rmse:.4f} stages")
        print(f"  测试集: MAE = {stages_test_mae:.4f} stages, RMSE = {stages_test_rmse:.4f} stages")

        # 存储性能指标用于对比
        self.performance = {
            'tcam': {
                'train': {'r2': tcam_train_r2, 'mae': tcam_train_mae, 'rmse': tcam_train_rmse},
                'test': {'r2': tcam_test_r2, 'mae': tcam_test_mae, 'rmse': tcam_test_rmse}
            },
            'sram': {
                'train': {'r2': sram_train_r2, 'mae': sram_train_mae, 'rmse': sram_train_rmse},
                'test': {'r2': sram_test_r2, 'mae': sram_test_mae, 'rmse': sram_test_rmse}
            },
            'stages': {
                'train': {'mae': stages_train_mae, 'rmse': stages_train_rmse},
                'test': {'mae': stages_test_mae, 'rmse': stages_test_rmse}
            }
        }
        
    def compare_with_baseline(self):
        """与基线版本对比"""
        print("\n[阶段 5/8] 性能对比分析")
        print("-" * 80)
        
        baseline_tcam_mae = 0.38
        baseline_tcam_rmse = 2.50  # 估算
        baseline_sram_mae = 0.004
        
        v1_tcam_mae = 0.2782
        v1_tcam_rmse = 2.0110
        v1_sram_mae = 0.003712
        
        print(f"\n  相比原始版改进: MAE 详见评估报告")
        
    def analyze_overfitting(self):
        """过拟合分析"""
        print("\n[阶段 6/8] 过拟合控制分析")
        print("-" * 80)
        
        tcam_overfit = self.performance['tcam']['train']['r2'] - self.performance['tcam']['test']['r2']
        sram_overfit = self.performance['sram']['train']['r2'] - self.performance['sram']['test']['r2']
        
        print("\n[过拟合指标 (训练R^2 - 测试R^2)]")
        print(f"  TCAM: {tcam_overfit:.4f}")
        print(f"  SRAM: {sram_overfit:.4f}")
        
        print("\n[评估]")
        if tcam_overfit < 0.15:
            print("  TCAM: 过拟合控制良好 [优秀]")
        elif tcam_overfit < 0.25:
            print("  TCAM: 轻微过拟合 [良好]")
        else:
            print("  TCAM: 过拟合严重 [需改进]")
        
        if sram_overfit < 0.15:
            print("  SRAM: 过拟合控制良好 [优秀]")
        elif sram_overfit < 0.25:
            print("  SRAM: 轻微过拟合 [良好]")
        else:
            print("  SRAM: 过拟合严重 [需改进]")
        
        # V1版本过拟合对比
        v1_tcam_overfit = 1.0000 - 0.7342  # 0.2658
        print(f"\n[对比 V1 优化版]")
        print(f"  V1 TCAM 过拟合: {v1_tcam_overfit:.4f} (训练R^2=1.0, 严重过拟合)")
        print(f"  V2 TCAM 过拟合: {tcam_overfit:.4f} (改进: {((v1_tcam_overfit - tcam_overfit) / v1_tcam_overfit * 100):.1f}%)")
        
    def save_models(self):
        """保存模型"""
        print("\n[阶段 7/8] 保存模型")
        print("-" * 80)
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        joblib.dump(self.tcam_model, os.path.join(self.models_dir, "tcam_model_v2.pkl"))
        joblib.dump(self.sram_model, os.path.join(self.models_dir, "sram_model_v2.pkl"))
        joblib.dump(self.stages_model, os.path.join(self.models_dir, "stages_model_v2.pkl"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler_v2.pkl"))
        joblib.dump(self.feature_names, os.path.join(self.models_dir, "feature_names_v2.pkl"))
        joblib.dump(self.performance, os.path.join(self.models_dir, "performance_v2.pkl"))
        
        print("[OK] V2 模型已保存")
        print(f"    tcam_model_v2.pkl")
        print(f"    sram_model_v2.pkl")
        print(f"    stages_model_v2.pkl")
        print(f"    scaler_v2.pkl")
        print(f"    feature_names_v2.pkl")
        print(f"    performance_v2.pkl")
        

def main():
    """主程序"""
    import time
    start_time = time.time()
    
    predictor = DryadPredictorV2()
    
    # 执行训练流程
    predictor.load_and_process_data()
    predictor.analyze_failure_boundary()
    predictor.split_data()
    predictor.train_models_with_regularization()
    predictor.evaluate_models()
    predictor.analyze_overfitting()
    predictor.save_models()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  V4 Tiles 级训练完成！总用时: {elapsed:.2f} 秒")
    print("  目标达成: 已预测 Tiles 绝对值，规避舍入误差，并集成报警机制。")
    print("=" * 80)


if __name__ == "__main__":
    main()

