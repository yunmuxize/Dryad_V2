# -*- coding: utf-8 -*-
"""
Dryad BMv2 资源预测系统 - 增强版核心预测器
优化版本：包含日志记录、可视化、后台运行支持和性能优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix
import joblib
import os
import logging
import json
import time
from datetime import datetime
from functools import reduce
import operator
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体，解决乱码问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 配置日志
def setup_logging(log_dir="logs"):
    """设置日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DryadPredictorEnhanced:
    """
    Dryad BMv2 资源预测系统核心类 - 增强版
    使用混合模型：Memory使用LGBMRegressor回归，CPU使用LGBMClassifier分类
    包含日志记录、可视化和性能优化
    """
    
    def __init__(self, csv_file="ml_features.csv", log_dir="logs", plots_dir="plots", use_gpu=False):
        """
        初始化预测器
        
        Args:
            csv_file: 数据文件路径
            log_dir: 日志目录
            plots_dir: 图表目录
            use_gpu: 是否使用GPU加速训练（需要安装GPU版本的LightGBM）
        """
        self.csv_file = csv_file
        self.log_dir = log_dir
        self.plots_dir = plots_dir
        self.use_gpu = use_gpu
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        self.logger = setup_logging(log_dir)
        self.logger.info("=" * 60)
        self.logger.info("Dryad BMv2 Enhanced Predictor Initialized")
        if use_gpu:
            self.logger.info("GPU acceleration: ENABLED")
        else:
            self.logger.info("GPU acceleration: DISABLED (using CPU)")
        self.logger.info("=" * 60)
        
        self.df = None
        self.X = None
        self.y_memory = None  # Memory目标变量（连续值）
        self.y_cpu_categorical = None  # CPU目标变量（分类标签）
        self.X_train = None
        self.X_test = None
        self.y_train_memory = None
        self.y_test_memory = None
        self.y_train_cpu_cat = None
        self.y_test_cpu_cat = None
        self.scaler = RobustScaler()  # 使用RobustScaler，对异常值更鲁棒
        self.memory_model = None  # Memory回归模型
        self.cpu_model = None  # CPU分类模型
        self.cpu_label_encoder = LabelEncoder()  # CPU标签编码器
        self.feature_names = None
        self.training_history = {
            # Memory回归指标
            'memory_train_r2': [],
            'memory_test_r2': [],
            'memory_train_mae': [],
            'memory_test_mae': [],
            # CPU分类指标
            'cpu_train_accuracy': [],
            'cpu_test_accuracy': [],
            'cpu_train_f1': [],
            'cpu_test_f1': [],
            'cpu_train_mae': [],  # 从分类标签转换后计算
            'cpu_test_mae': [],   # 从分类标签转换后计算
            'cv_scores': []
        }
        
        # 特征名称定义（9个特征）
        self.base_features = [
            'total_len', 'protocol', 'flags', 'ttl', 
            'src_port', 'dst_port', 'tcp_flags_2', 'tcp_flags_1', 'size'
        ]
        
    def load_and_process_data(self):
        """加载和预处理数据"""
        self.logger.info("=== Loading and Processing Data ===")
        
        # 加载数据
        csv_path = os.path.join(os.path.dirname(__file__), self.csv_file)
        self.df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded {len(self.df)} records")
        
        # 数据质量检查
        self._check_data_quality()
        
        # 提取基础特征（9个特征）
        feature_matrix = []
        for _, row in self.df.iterrows():
            features = []
            for feature_name in self.base_features:
                features.append(row[feature_name])
            feature_matrix.append(features)
        
        self.X = np.array(feature_matrix)
        
        # 创建增强的派生特征
        self._create_enhanced_features()
        
        # 提取目标变量
        self._extract_targets()
        
        self.logger.info(f"Feature matrix shape: {self.X.shape}")
        self.logger.info(f"Memory target shape: {self.y_memory.shape}")
        self.logger.info(f"CPU categorical target shape: {self.y_cpu_categorical.shape}")
        self.logger.info(f"CPU range: {self.df['cpu'].min():.2f} - {self.df['cpu'].max():.2f}")
        self.logger.info(f"Memory range: {self.y_memory.min():.2f} - {self.y_memory.max():.2f}")
        
    def _check_data_quality(self):
        """检查数据质量"""
        self.logger.info("Checking data quality...")
        
        # 检查缺失值
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            self.logger.warning(f"Missing values found: {missing[missing > 0]}")
        
        # 检查目标变量分布
        cpu_stats = self.df['cpu'].describe()
        memory_stats = self.df['memory'].describe()
        self.logger.info(f"CPU stats: mean={cpu_stats['mean']:.3f}, std={cpu_stats['std']:.3f}")
        self.logger.info(f"Memory stats: mean={memory_stats['mean']:.3f}, std={memory_stats['std']:.3f}")
        
    def _create_enhanced_features(self):
        """创建增强的派生特征"""
        self.logger.info("Creating enhanced derived features...")
        
        range_counts = []
        ternary_counts = []
        lpm_counts = []
        exact_counts = []
        complexity_scores = []
        
        # 新增特征：匹配方式的交互特征
        range_ternary_interactions = []
        high_complexity_counts = []  # range + ternary 的总数
        
        for i in range(len(self.X)):
            matching_features = self.X[i, :8]
            range_count = np.sum(matching_features == 3)
            ternary_count = np.sum(matching_features == 2)
            lpm_count = np.sum(matching_features == 1)
            exact_count = np.sum(matching_features == 0)
            complexity_score = range_count * 4 + ternary_count * 3 + lpm_count * 2 + exact_count * 1
            
            range_counts.append(range_count)
            ternary_counts.append(ternary_count)
            lpm_counts.append(lpm_count)
            exact_counts.append(exact_count)
            complexity_scores.append(complexity_score)
            
            # 新增：交互特征
            range_ternary_interactions.append(range_count * ternary_count)
            high_complexity_counts.append(range_count + ternary_count)
        
        # 添加基础派生特征
        derived_features = np.column_stack([
            range_counts, ternary_counts, lpm_counts, exact_counts, complexity_scores
        ])
        
        # 添加增强特征
        enhanced_features = np.column_stack([
            range_ternary_interactions, high_complexity_counts
        ])
        
        self.X = np.column_stack([self.X, derived_features, enhanced_features])
        
        # 更新特征名称
        self.feature_names = self.base_features + [
            'range_count', 'ternary_count', 'lpm_count', 'exact_count', 'complexity_score',
            'range_ternary_interaction', 'high_complexity_count'
        ]
        
        self.logger.info(f"Added enhanced derived features. Final shape: {self.X.shape}")
        
    def _extract_targets(self):
        """提取目标变量：Memory为连续值，CPU转换为分类标签"""
        # Memory目标变量（保持连续值）
        self.y_memory = self.df['memory'].values
        
        # CPU目标变量：转换为分类标签
        cpu_values = self.df['cpu'].values
        # 将CPU值乘以100并转换为整数（例如 0.50 -> 50）
        cpu_int = (cpu_values * 100).astype(int)
        # 使用LabelEncoder将整数转换为类别标签 [0, 1, 2, ...]
        self.y_cpu_categorical = self.cpu_label_encoder.fit_transform(cpu_int)
        
        # 记录CPU类别信息
        unique_cpu_values = np.unique(cpu_int)
        self.logger.info(f"CPU unique values (×100): {unique_cpu_values}")
        self.logger.info(f"CPU number of classes: {len(unique_cpu_values)}")
        self.logger.info(f"CPU class distribution:")
        for cpu_val in unique_cpu_values:
            count = np.sum(cpu_int == cpu_val)
            percentage = count / len(cpu_int) * 100
            self.logger.info(f"  {cpu_val/100.0:.2f} ({cpu_val}): {count} samples ({percentage:.2f}%)")
        
    def split_data(self, test_size=0.2, random_state=42):
        """分割训练和测试数据"""
        # 分割特征X
        self.X_train, self.X_test = train_test_split(
            self.X, test_size=test_size, random_state=random_state
        )
        
        # 分别分割Memory和CPU目标变量
        self.y_train_memory, self.y_test_memory = train_test_split(
            self.y_memory, test_size=test_size, random_state=random_state
        )
        self.y_train_cpu_cat, self.y_test_cpu_cat = train_test_split(
            self.y_cpu_categorical, test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.logger.info(f"Training samples: {len(self.X_train)}")
        self.logger.info(f"Test samples: {len(self.X_test)}")
        self.logger.info(f"Memory training target shape: {self.y_train_memory.shape}")
        self.logger.info(f"Memory test target shape: {self.y_test_memory.shape}")
        self.logger.info(f"CPU training target shape: {self.y_train_cpu_cat.shape}")
        self.logger.info(f"CPU test target shape: {self.y_test_cpu_cat.shape}")
        
    def train_models(self, extensive_search=False):
        """
        训练混合模型：Memory使用LGBMRegressor，CPU使用LGBMClassifier
        
        Args:
            extensive_search: 是否使用扩展的超参数搜索
        """
        self.logger.info("\n=== Training Hybrid Models ===")
        total_start_time = time.time()
        
        # 定义超参数网格 - 优化版本，目标Memory R²达到90%
        if extensive_search:
            self.logger.info("Using extensive hyperparameter search (optimized for R²=0.90)...")
            # Memory模型：更精细的参数搜索，重点关注高R²
            param_grid_memory = {
                'n_estimators': [300, 500, 700, 1000],
                'learning_rate': [0.01, 0.03, 0.05, 0.08],
                'max_depth': [8, 10, 12, -1],
                'num_leaves': [50, 100, 150, 200],
                'min_child_samples': [15, 20, 25, 30],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            # CPU模型：优化分类准确率
            param_grid_cpu = {
                'n_estimators': [200, 300, 500, 700],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'max_depth': [7, 10, 12, -1],
                'num_leaves': [31, 50, 100, 150],
                'min_child_samples': [20, 30, 50],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'class_weight': [None, 'balanced'],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0, 0.1]
            }
        else:
            # 标准搜索 - 优化版本
            param_grid_memory = {
                'n_estimators': [300, 500, 700],
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [8, 10, 12],
                'num_leaves': [50, 100, 150],
                'min_child_samples': [15, 20, 25],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
            param_grid_cpu = {
                'n_estimators': [200, 300, 500],
                'learning_rate': [0.01, 0.03, 0.05],
                'max_depth': [7, 10, 12],
                'num_leaves': [31, 50, 100],
                'min_child_samples': [20, 30],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'class_weight': [None, 'balanced']
            }
        
        # ========== 训练Memory回归模型 ==========
        self.logger.info("\n--- Training Memory Regression Model (LGBMRegressor) ---")
        memory_start_time = time.time()
        
        # 计算参数组合总数（所有参数的笛卡尔积）
        memory_param_count = reduce(operator.mul, [len(v) for v in param_grid_memory.values()], 1)
        self.logger.info(f"Memory parameter grid size: {memory_param_count:,} combinations")
        
        # 配置GPU或CPU
        device = 'gpu' if self.use_gpu else 'cpu'
        gpu_params = {}
        if self.use_gpu:
            gpu_params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }
            self.logger.info("Using GPU for Memory model training...")
        else:
            self.logger.info("Using CPU for Memory model training...")
        
        memory_lgbm = LGBMRegressor(
            random_state=42, 
            n_jobs=-1 if not self.use_gpu else 1,  # GPU模式下不使用多进程
            verbose=-1,
            **gpu_params
        )
        memory_grid = GridSearchCV(
            memory_lgbm, param_grid_memory,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        self.logger.info("Starting Memory model grid search...")
        memory_grid.fit(self.X_train_scaled, self.y_train_memory)
        self.memory_model = memory_grid.best_estimator_
        
        memory_elapsed = time.time() - memory_start_time
        self.logger.info(f"Memory model training completed in {memory_elapsed:.2f} seconds")
        self.logger.info(f"Memory best parameters: {memory_grid.best_params_}")
        self.logger.info(f"Memory best CV score (R²): {memory_grid.best_score_:.4f}")
        
        # ========== 训练CPU分类模型 ==========
        self.logger.info("\n--- Training CPU Classification Model (LGBMClassifier) ---")
        cpu_start_time = time.time()
        
        # 计算参数组合总数（所有参数的笛卡尔积）
        cpu_param_count = reduce(operator.mul, [len(v) for v in param_grid_cpu.values()], 1)
        self.logger.info(f"CPU parameter grid size: {cpu_param_count} combinations")
        
        # 配置GPU或CPU
        if self.use_gpu:
            self.logger.info("Using GPU for CPU model training...")
        else:
            self.logger.info("Using CPU for CPU model training...")
        
        cpu_lgbm = LGBMClassifier(
            random_state=42, 
            n_jobs=-1 if not self.use_gpu else 1,  # GPU模式下不使用多进程
            verbose=-1,
            **gpu_params
        )
        cpu_grid = GridSearchCV(
            cpu_lgbm, param_grid_cpu,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        self.logger.info("Starting CPU model grid search...")
        cpu_grid.fit(self.X_train_scaled, self.y_train_cpu_cat)
        self.cpu_model = cpu_grid.best_estimator_
        
        cpu_elapsed = time.time() - cpu_start_time
        self.logger.info(f"CPU model training completed in {cpu_elapsed:.2f} seconds")
        self.logger.info(f"CPU best parameters: {cpu_grid.best_params_}")
        self.logger.info(f"CPU best CV score (Accuracy): {cpu_grid.best_score_:.4f}")
        
        total_elapsed = time.time() - total_start_time
        self.logger.info(f"\nTotal training time: {total_elapsed:.2f} seconds")
        
        # 记录CV结果
        self._log_cv_results(memory_grid, cpu_grid)
        
        # 评估模型性能
        self._evaluate_models()
        
        # 生成可视化
        self._create_visualizations(memory_grid, cpu_grid)
        
        # 保存模型
        self._save_models()
        
        # 保存训练历史
        self._save_training_history()
        
    def _log_cv_results(self, memory_grid, cpu_grid):
        """记录交叉验证结果"""
        self.logger.info("\n=== Cross-Validation Results ===")
        
        # Memory模型CV结果
        self.logger.info("\nMemory Model - Top 5 parameter combinations:")
        memory_results_df = pd.DataFrame(memory_grid.cv_results_)
        memory_top_5 = memory_results_df.nlargest(5, 'mean_test_score')
        for idx, row in memory_top_5.iterrows():
            self.logger.info(f"  R² Score: {row['mean_test_score']:.4f}, Params: {row['params']}")
        
        # CPU模型CV结果
        self.logger.info("\nCPU Model - Top 5 parameter combinations:")
        cpu_results_df = pd.DataFrame(cpu_grid.cv_results_)
        cpu_top_5 = cpu_results_df.nlargest(5, 'mean_test_score')
        for idx, row in cpu_top_5.iterrows():
            self.logger.info(f"  Accuracy: {row['mean_test_score']:.4f}, Params: {row['params']}")
        
    def _evaluate_models(self):
        """评估模型性能：分别评估Memory回归和CPU分类"""
        self.logger.info("\n=== Model Performance ===")
        
        # ========== Memory回归模型评估 ==========
        self.logger.info("\n--- Memory Regression Model (LGBMRegressor) ---")
        memory_train_pred = self.memory_model.predict(self.X_train_scaled)
        memory_test_pred = self.memory_model.predict(self.X_test_scaled)
        
        memory_train_r2 = r2_score(self.y_train_memory, memory_train_pred)
        memory_test_r2 = r2_score(self.y_test_memory, memory_test_pred)
        memory_train_mae = mean_absolute_error(self.y_train_memory, memory_train_pred)
        memory_test_mae = mean_absolute_error(self.y_test_memory, memory_test_pred)
        memory_train_rmse = np.sqrt(mean_squared_error(self.y_train_memory, memory_train_pred))
        memory_test_rmse = np.sqrt(mean_squared_error(self.y_test_memory, memory_test_pred))
        
        self.logger.info(f"  Training R²: {memory_train_r2:.4f}, MAE: {memory_train_mae:.4f}, RMSE: {memory_train_rmse:.4f}")
        self.logger.info(f"  Test R²: {memory_test_r2:.4f}, MAE: {memory_test_mae:.4f}, RMSE: {memory_test_rmse:.4f}")
        
        # ========== CPU分类模型评估 ==========
        self.logger.info("\n--- CPU Classification Model (LGBMClassifier) ---")
        cpu_train_pred_labels = self.cpu_model.predict(self.X_train_scaled)
        cpu_test_pred_labels = self.cpu_model.predict(self.X_test_scaled)
        
        # 分类指标
        cpu_train_accuracy = accuracy_score(self.y_train_cpu_cat, cpu_train_pred_labels)
        cpu_test_accuracy = accuracy_score(self.y_test_cpu_cat, cpu_test_pred_labels)
        cpu_train_f1 = f1_score(self.y_train_cpu_cat, cpu_train_pred_labels, average='weighted')
        cpu_test_f1 = f1_score(self.y_test_cpu_cat, cpu_test_pred_labels, average='weighted')
        
        # 混淆矩阵（测试集）
        cpu_test_cm = confusion_matrix(self.y_test_cpu_cat, cpu_test_pred_labels)
        
        self.logger.info(f"  Training Accuracy: {cpu_train_accuracy:.4f}, F1 (weighted): {cpu_train_f1:.4f}")
        self.logger.info(f"  Test Accuracy: {cpu_test_accuracy:.4f}, F1 (weighted): {cpu_test_f1:.4f}")
        self.logger.info(f"\n  Test Confusion Matrix:")
        self.logger.info(f"\n{cpu_test_cm}")
        
        # 转换回原始CPU值计算MAE/RMSE（参考指标）
        cpu_train_pred_int = self.cpu_label_encoder.inverse_transform(cpu_train_pred_labels)
        cpu_test_pred_int = self.cpu_label_encoder.inverse_transform(cpu_test_pred_labels)
        cpu_train_pred_float = cpu_train_pred_int / 100.0
        cpu_test_pred_float = cpu_test_pred_int / 100.0
        
        # 获取真实CPU值（原始浮点数）
        cpu_train_true_int = self.cpu_label_encoder.inverse_transform(self.y_train_cpu_cat)
        cpu_test_true_int = self.cpu_label_encoder.inverse_transform(self.y_test_cpu_cat)
        cpu_train_true_float = cpu_train_true_int / 100.0
        cpu_test_true_float = cpu_test_true_int / 100.0
        
        cpu_train_mae = mean_absolute_error(cpu_train_true_float, cpu_train_pred_float)
        cpu_test_mae = mean_absolute_error(cpu_test_true_float, cpu_test_pred_float)
        cpu_train_rmse = np.sqrt(mean_squared_error(cpu_train_true_float, cpu_train_pred_float))
        cpu_test_rmse = np.sqrt(mean_squared_error(cpu_test_true_float, cpu_test_pred_float))
        
        self.logger.info(f"\n  Reference Metrics (converted from classification):")
        self.logger.info(f"  Training MAE: {cpu_train_mae:.4f}, RMSE: {cpu_train_rmse:.4f}")
        self.logger.info(f"  Test MAE: {cpu_test_mae:.4f}, RMSE: {cpu_test_rmse:.4f}")
        
        # 记录到历史
        self.training_history['memory_train_r2'].append(memory_train_r2)
        self.training_history['memory_test_r2'].append(memory_test_r2)
        self.training_history['memory_train_mae'].append(memory_train_mae)
        self.training_history['memory_test_mae'].append(memory_test_mae)
        
        self.training_history['cpu_train_accuracy'].append(cpu_train_accuracy)
        self.training_history['cpu_test_accuracy'].append(cpu_test_accuracy)
        self.training_history['cpu_train_f1'].append(cpu_train_f1)
        self.training_history['cpu_test_f1'].append(cpu_test_f1)
        self.training_history['cpu_train_mae'].append(cpu_train_mae)
        self.training_history['cpu_test_mae'].append(cpu_test_mae)
        
    def _create_visualizations(self, memory_grid=None, cpu_grid=None):
        """创建可视化图表"""
        self.logger.info("Creating visualizations...")
        
        # 1. 预测vs真实值散点图
        self._plot_prediction_scatter()
        
        # 2. 特征重要性
        self._plot_feature_importance()
        
        # 3. 残差分析（Memory）和分类错误分析（CPU）
        self._plot_residuals()
        
        # 4. 混淆矩阵（CPU分类）
        self._plot_confusion_matrix()
        
        # 5. 学习曲线（如果时间允许）
        try:
            self._plot_learning_curves()
        except Exception as e:
            self.logger.warning(f"Could not create learning curves: {e}")
        
        self.logger.info("Visualizations saved to plots/ directory")
        
    def _plot_prediction_scatter(self):
        """绘制预测vs真实值散点图"""
        # Memory预测
        memory_train_pred = self.memory_model.predict(self.X_train_scaled)
        memory_test_pred = self.memory_model.predict(self.X_test_scaled)
        
        # CPU预测（转换为浮点数）
        cpu_train_pred_labels = self.cpu_model.predict(self.X_train_scaled)
        cpu_test_pred_labels = self.cpu_model.predict(self.X_test_scaled)
        cpu_train_pred_int = self.cpu_label_encoder.inverse_transform(cpu_train_pred_labels)
        cpu_test_pred_int = self.cpu_label_encoder.inverse_transform(cpu_test_pred_labels)
        cpu_train_pred_float = cpu_train_pred_int / 100.0
        cpu_test_pred_float = cpu_test_pred_int / 100.0
        
        # 获取真实CPU值
        cpu_train_true_int = self.cpu_label_encoder.inverse_transform(self.y_train_cpu_cat)
        cpu_test_true_int = self.cpu_label_encoder.inverse_transform(self.y_test_cpu_cat)
        cpu_train_true_float = cpu_train_true_int / 100.0
        cpu_test_true_float = cpu_test_true_int / 100.0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CPU训练集（转换后的值）
        axes[0, 0].scatter(cpu_train_true_float, cpu_train_pred_float, alpha=0.5)
        axes[0, 0].plot([cpu_train_true_float.min(), cpu_train_true_float.max()],
                       [cpu_train_true_float.min(), cpu_train_true_float.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True CPU')
        axes[0, 0].set_ylabel('Predicted CPU')
        cpu_train_mae = mean_absolute_error(cpu_train_true_float, cpu_train_pred_float)
        cpu_train_acc = accuracy_score(self.y_train_cpu_cat, cpu_train_pred_labels)
        axes[0, 0].set_title(f'CPU Training (Acc={cpu_train_acc:.4f}, MAE={cpu_train_mae:.4f})')
        axes[0, 0].grid(True)
        
        # CPU测试集
        axes[0, 1].scatter(cpu_test_true_float, cpu_test_pred_float, alpha=0.5, color='orange')
        axes[0, 1].plot([cpu_test_true_float.min(), cpu_test_true_float.max()],
                       [cpu_test_true_float.min(), cpu_test_true_float.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('True CPU')
        axes[0, 1].set_ylabel('Predicted CPU')
        cpu_test_mae = mean_absolute_error(cpu_test_true_float, cpu_test_pred_float)
        cpu_test_acc = accuracy_score(self.y_test_cpu_cat, cpu_test_pred_labels)
        axes[0, 1].set_title(f'CPU Test (Acc={cpu_test_acc:.4f}, MAE={cpu_test_mae:.4f})')
        axes[0, 1].grid(True)
        
        # Memory训练集
        axes[1, 0].scatter(self.y_train_memory, memory_train_pred, alpha=0.5)
        axes[1, 0].plot([self.y_train_memory.min(), self.y_train_memory.max()],
                       [self.y_train_memory.min(), self.y_train_memory.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('True Memory')
        axes[1, 0].set_ylabel('Predicted Memory')
        memory_train_r2 = r2_score(self.y_train_memory, memory_train_pred)
        axes[1, 0].set_title(f'Memory Training (R² = {memory_train_r2:.4f})')
        axes[1, 0].grid(True)
        
        # Memory测试集
        axes[1, 1].scatter(self.y_test_memory, memory_test_pred, alpha=0.5, color='orange')
        axes[1, 1].plot([self.y_test_memory.min(), self.y_test_memory.max()],
                       [self.y_test_memory.min(), self.y_test_memory.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('True Memory')
        axes[1, 1].set_ylabel('Predicted Memory')
        memory_test_r2 = r2_score(self.y_test_memory, memory_test_pred)
        axes[1, 1].set_title(f'Memory Test (R² = {memory_test_r2:.4f})')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_feature_importance(self):
        """绘制特征重要性：分别显示Memory和CPU模型"""
        # Memory模型特征重要性
        memory_importance = self.memory_model.feature_importances_
        memory_features = list(zip(self.feature_names, memory_importance))
        memory_features.sort(key=lambda x: x[1], reverse=True)
        
        # CPU模型特征重要性
        cpu_importance = self.cpu_model.feature_importances_
        cpu_features = list(zip(self.feature_names, cpu_importance))
        cpu_features.sort(key=lambda x: x[1], reverse=True)
        
        # 创建两个子图
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Memory特征重要性
        top_memory = memory_features[:15]
        names_mem, values_mem = zip(*top_memory)
        axes[0].barh(range(len(names_mem)), values_mem)
        axes[0].set_yticks(range(len(names_mem)))
        axes[0].set_yticklabels(names_mem)
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_title('Memory Model - Top 15 Feature Importance')
        axes[0].invert_yaxis()
        axes[0].grid(True, axis='x', alpha=0.3)
        
        # CPU特征重要性
        top_cpu = cpu_features[:15]
        names_cpu, values_cpu = zip(*top_cpu)
        axes[1].barh(range(len(names_cpu)), values_cpu, color='orange')
        axes[1].set_yticks(range(len(names_cpu)))
        axes[1].set_yticklabels(names_cpu)
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('CPU Model - Top 15 Feature Importance')
        axes[1].invert_yaxis()
        axes[1].grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_residuals(self):
        """绘制残差分析图：Memory残差和CPU分类错误"""
        # Memory预测和残差
        memory_train_pred = self.memory_model.predict(self.X_train_scaled)
        memory_test_pred = self.memory_model.predict(self.X_test_scaled)
        memory_train_residuals = self.y_train_memory - memory_train_pred
        memory_test_residuals = self.y_test_memory - memory_test_pred
        
        # CPU预测（转换为浮点数）和残差
        cpu_train_pred_labels = self.cpu_model.predict(self.X_train_scaled)
        cpu_test_pred_labels = self.cpu_model.predict(self.X_test_scaled)
        cpu_train_pred_int = self.cpu_label_encoder.inverse_transform(cpu_train_pred_labels)
        cpu_test_pred_int = self.cpu_label_encoder.inverse_transform(cpu_test_pred_labels)
        cpu_train_pred_float = cpu_train_pred_int / 100.0
        cpu_test_pred_float = cpu_test_pred_int / 100.0
        
        cpu_train_true_int = self.cpu_label_encoder.inverse_transform(self.y_train_cpu_cat)
        cpu_test_true_int = self.cpu_label_encoder.inverse_transform(self.y_test_cpu_cat)
        cpu_train_true_float = cpu_train_true_int / 100.0
        cpu_test_true_float = cpu_test_true_int / 100.0
        
        cpu_train_residuals = cpu_train_true_float - cpu_train_pred_float
        cpu_test_residuals = cpu_test_true_float - cpu_test_pred_float
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CPU残差（转换后的值）
        axes[0, 0].scatter(cpu_train_pred_float, cpu_train_residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted CPU')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('CPU Training Residuals (from classification)')
        axes[0, 0].grid(True)
        
        axes[0, 1].scatter(cpu_test_pred_float, cpu_test_residuals, alpha=0.5, color='orange')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted CPU')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('CPU Test Residuals (from classification)')
        axes[0, 1].grid(True)
        
        # Memory残差
        axes[1, 0].scatter(memory_train_pred, memory_train_residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Memory')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Memory Training Residuals')
        axes[1, 0].grid(True)
        
        axes[1, 1].scatter(memory_test_pred, memory_test_residuals, alpha=0.5, color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Memory')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Memory Test Residuals')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_confusion_matrix(self):
        """绘制CPU分类的混淆矩阵"""
        cpu_test_pred_labels = self.cpu_model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test_cpu_cat, cpu_test_pred_labels)
        
        # 获取类别标签（原始CPU值）
        class_labels = self.cpu_label_encoder.classes_ / 100.0
        class_labels_str = [f'{val:.2f}' for val in class_labels]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels_str, yticklabels=class_labels_str,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted CPU')
        plt.ylabel('True CPU')
        plt.title('CPU Classification Confusion Matrix (Test Set)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'cpu_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_learning_curves(self):
        """绘制学习曲线"""
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # CPU学习曲线（使用accuracy）
        train_sizes_cpu, train_scores_cpu, val_scores_cpu = learning_curve(
            self.cpu_model, self.X_train_scaled, self.y_train_cpu_cat,
            train_sizes=train_sizes, cv=5, n_jobs=-1, scoring='accuracy'
        )
        
        axes[0].plot(train_sizes_cpu, train_scores_cpu.mean(axis=1), 'o-', label='Training Score')
        axes[0].plot(train_sizes_cpu, val_scores_cpu.mean(axis=1), 'o-', label='Validation Score')
        axes[0].set_xlabel('Training Set Size')
        axes[0].set_ylabel('Accuracy Score')
        axes[0].set_title('CPU Learning Curve (Classification)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Memory学习曲线（使用R²）
        train_sizes_mem, train_scores_mem, val_scores_mem = learning_curve(
            self.memory_model, self.X_train_scaled, self.y_train_memory,
            train_sizes=train_sizes, cv=5, n_jobs=-1, scoring='r2'
        )
        
        axes[1].plot(train_sizes_mem, train_scores_mem.mean(axis=1), 'o-', label='Training Score')
        axes[1].plot(train_sizes_mem, val_scores_mem.mean(axis=1), 'o-', label='Validation Score')
        axes[1].set_xlabel('Training Set Size')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Memory Learning Curve (Regression)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_models(self):
        """保存训练好的模型"""
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(self.memory_model, "models/memory_model.pkl")
        joblib.dump(self.cpu_model, "models/cpu_model.pkl")
        joblib.dump(self.cpu_label_encoder, "models/cpu_label_encoder.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")
        joblib.dump(self.feature_names, "models/feature_names.pkl")
        
        self.logger.info("Models saved successfully!")
        
    def _save_training_history(self):
        """保存训练历史"""
        history_file = os.path.join(self.log_dir, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved to {history_file}")
        
    def load_models(self):
        """加载训练好的模型"""
        try:
            self.memory_model = joblib.load("models/memory_model.pkl")
            self.cpu_model = joblib.load("models/cpu_model.pkl")
            self.cpu_label_encoder = joblib.load("models/cpu_label_encoder.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            self.feature_names = joblib.load("models/feature_names.pkl")
            self.logger.info("Models loaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def predict(self, matching_pattern, size=4096):
        """预测资源使用率"""
        if self.memory_model is None or self.cpu_model is None:
            raise ValueError("Models not loaded. Please train or load models first.")
        
        if len(matching_pattern) != 8:
            raise ValueError("Matching pattern must contain 8 fields")
        
        features = np.array(matching_pattern)
        features = np.append(features, size)
        
        range_count = np.sum(np.array(matching_pattern) == 3)
        ternary_count = np.sum(np.array(matching_pattern) == 2)
        lpm_count = np.sum(np.array(matching_pattern) == 1)
        exact_count = np.sum(np.array(matching_pattern) == 0)
        complexity_score = range_count * 4 + ternary_count * 3 + lpm_count * 2 + exact_count * 1
        range_ternary_interaction = range_count * ternary_count
        high_complexity_count = range_count + ternary_count
        
        features = np.append(features, [
            range_count, ternary_count, lpm_count, exact_count, complexity_score,
            range_ternary_interaction, high_complexity_count
        ])
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Memory预测（回归）
        memory_pred = self.memory_model.predict(features_scaled)[0]
        
        # CPU预测（分类）
        cpu_pred_label = self.cpu_model.predict(features_scaled)[0]
        cpu_pred_int = self.cpu_label_encoder.inverse_transform([cpu_pred_label])[0]
        cpu_pred = cpu_pred_int / 100.0
        
        return {
            'cpu': round(cpu_pred, 2),
            'memory': round(memory_pred, 2),
            'matching_pattern': matching_pattern,
            'size': size
        }


def main(extensive_search=False, use_gpu=False):
    """主程序入口"""
    print("=" * 60)
    print("Dryad BMv2 Enhanced Resource Usage Prediction System")
    if use_gpu:
        print("GPU acceleration: ENABLED")
    if extensive_search:
        print("Mode: EXTENSIVE SEARCH (slower, better results)")
    else:
        print("Mode: STANDARD SEARCH (faster)")
    print("=" * 60)
    
    predictor = DryadPredictorEnhanced(use_gpu=use_gpu)
    
    predictor.load_and_process_data()
    predictor.split_data()
    predictor.train_models(extensive_search=extensive_search)
    
    print("\n" + "=" * 60)
    print("System initialization completed successfully!")
    print("=" * 60)
    print(f"Check logs/ directory for training logs")
    print(f"Check plots/ directory for visualizations")


if __name__ == "__main__":
    import sys
    extensive = '--extensive' in sys.argv
    use_gpu = '--gpu' in sys.argv or '--use-gpu' in sys.argv
    main(extensive_search=extensive, use_gpu=use_gpu)

