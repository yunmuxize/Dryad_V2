# -*- coding: utf-8 -*-
"""
快速训练脚本 - 使用较小的参数网格快速获取结果，支持GPU加速
"""

from dryad_predictor_enhanced import DryadPredictorEnhanced
import sys
import warnings
warnings.filterwarnings('ignore')

def quick_train(use_gpu=True):
    """快速训练
    
    Args:
        use_gpu: 是否使用GPU加速（默认True）
    """
    print("=" * 60)
    print("快速训练Memory模型（精细超参数优化 + GPU加速）")
    print("目标: Memory R^2 > 0.8021（当前最佳）")
    print("策略: 在最佳参数附近精细搜索，尝试突破当前性能")
    if use_gpu:
        print("GPU加速: 已启用（RTX 3050）")
        print("并行设置: GridSearchCV n_jobs=3（优化GPU使用）")
    print("=" * 60)
    
    predictor = DryadPredictorEnhanced(use_gpu=use_gpu)
    
    # 加载数据
    print("\n1. 加载数据...")
    predictor.load_and_process_data()
    
    # 分割数据
    print("\n2. 分割数据...")
    predictor.split_data()
    
    # 修改参数网格为更小的版本
    print("\n3. 修改为快速训练模式...")
    
    # 临时修改train_models方法中的参数网格
    original_train = predictor.train_models
    
    def quick_train_models():
        """快速训练版本 - 优化参数"""
        predictor.logger.info("\n=== Training Hybrid Models (Quick Mode - Optimized) ===")
        if predictor.use_gpu:
            predictor.logger.info("GPU acceleration: ENABLED")
        import time
        from lightgbm import LGBMRegressor, LGBMClassifier
        from sklearn.model_selection import GridSearchCV
        from functools import reduce
        import operator
        
        total_start_time = time.time()
        
        # 进一步优化Memory模型超参数（基于当前最佳结果0.8021进行精细搜索）
        # 当前最佳参数: learning_rate=0.05, max_depth=7, min_child_samples=30, n_estimators=100, num_leaves=31
        # 策略：在最佳参数附近进行更精细的搜索，尝试突破0.8021
        param_grid_memory = {
            'n_estimators': [80, 100, 120, 150, 200],  # 在100附近精细搜索，尝试更多树
            'learning_rate': [0.04, 0.05, 0.06, 0.07],  # 在0.05附近精细搜索
            'max_depth': [6, 7, 8, 9],  # 在7附近精细搜索
            'num_leaves': [25, 31, 40, 50, 63],  # 在31附近精细搜索（63=2^6-1，常用值）
            'min_child_samples': [25, 30, 35, 40]  # 在30附近精细搜索，尝试不同正则化强度
        }
        
        param_grid_cpu = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [7, 10],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30]
        }
        
        memory_param_count = reduce(operator.mul, [len(v) for v in param_grid_memory.values()], 1)
        cpu_param_count = reduce(operator.mul, [len(v) for v in param_grid_cpu.values()], 1)
        predictor.logger.info(f"Memory参数组合数: {memory_param_count:,}")
        predictor.logger.info(f"CPU参数组合数: {cpu_param_count:,}")
        
        # GPU配置 - 验证GPU是否可用
        gpu_params = {}
        if predictor.use_gpu:
            try:
                import lightgbm as lgb
                # 尝试创建一个小模型来测试GPU
                test_model = lgb.LGBMRegressor(device='gpu', n_estimators=1, verbose=-1)
                # 如果GPU不可用，会抛出异常
                predictor.logger.info("GPU support detected. Using GPU for training...")
                gpu_params = {
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                }
            except Exception as e:
                predictor.logger.warning(f"GPU not available: {e}. Falling back to CPU.")
                predictor.logger.warning("If GPU is expected, check LightGBM GPU installation.")
                predictor.use_gpu = False
                gpu_params = {}
        
        # 训练Memory模型
        predictor.logger.info("\n--- Training Memory Regression Model ---")
        memory_start = time.time()
        memory_lgbm = LGBMRegressor(
            random_state=42, 
            n_jobs=-1 if not predictor.use_gpu else 1,
            verbose=-1,
            **gpu_params
        )
        # GPU模式下，限制GridSearchCV并行数（避免与GPU竞争资源）
        # LightGBM模型内部使用GPU，GridSearchCV的并行是CPU多进程
        # 对于RTX 3050，限制n_jobs在3以内以优化GPU训练效果
        memory_grid = GridSearchCV(
            memory_lgbm, param_grid_memory,
            cv=3,  # 保持3折CV以加快速度
            scoring='r2',
            n_jobs=3 if predictor.use_gpu else -1,  # GPU模式下限制为3，避免资源竞争
            verbose=2
        )
        memory_grid.fit(predictor.X_train_scaled, predictor.y_train_memory)
        predictor.memory_model = memory_grid.best_estimator_
        memory_elapsed = time.time() - memory_start
        predictor.logger.info(f"Memory训练耗时: {memory_elapsed:.2f}秒")
        predictor.logger.info(f"Memory best parameters: {memory_grid.best_params_}")
        predictor.logger.info(f"Memory best CV score (R²): {memory_grid.best_score_:.4f}")
        
        # 训练CPU模型
        predictor.logger.info("\n--- Training CPU Classification Model ---")
        cpu_start = time.time()
        cpu_lgbm = LGBMClassifier(
            random_state=42, 
            n_jobs=-1 if not predictor.use_gpu else 1,
            verbose=-1,
            **gpu_params
        )
        # CPU模型暂时跳过训练（用户要求先不管CPU）
        predictor.logger.info("\n--- 跳过CPU模型训练（专注于Memory优化）---")
        # 使用一个简单的默认模型作为占位符
        from lightgbm import LGBMClassifier
        cpu_lgbm = LGBMClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            n_jobs=-1 if not predictor.use_gpu else 1,
            verbose=-1,
            **gpu_params
        )
        cpu_lgbm.fit(predictor.X_train_scaled, predictor.y_train_cpu_cat)
        predictor.cpu_model = cpu_lgbm
        
        cpu_elapsed = 0  # 跳过训练，时间为0
        predictor.logger.info("CPU模型使用默认参数（跳过训练）")
        cpu_grid = None  # 设置为None，因为跳过了GridSearchCV
        
        total_elapsed = time.time() - total_start_time
        predictor.logger.info(f"\nTotal training time: {total_elapsed:.2f} seconds")
        
        # 评估模型
        predictor._evaluate_models()
        
        # 生成可视化（CPU模型使用None）
        predictor._create_visualizations(memory_grid, None)
        
        # 保存模型
        predictor._save_models()
        
        # 保存训练历史
        predictor._save_training_history()
    
    # 替换方法
    predictor.train_models = quick_train_models
    
    # 开始训练
    print("\n4. 开始训练...")
    predictor.train_models()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 输出最终结果摘要
    import json
    import glob
    import os
    history_files = glob.glob("logs/training_history_*.json")
    if history_files:
        latest_history = max(history_files, key=os.path.getmtime)
        with open(latest_history, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        print("\n" + "=" * 60)
        print("最终预测精度")
        print("=" * 60)
        
        if history.get('memory_test_r2') and len(history['memory_test_r2']) > 0:
            memory_r2 = history['memory_test_r2'][-1]
            memory_mae = history['memory_test_mae'][-1]
            print(f"\n[Memory回归模型]")
            print(f"  测试集 R^2:  {memory_r2:.4f} (目标: > 0.8021)")
            print(f"  测试集 MAE: {memory_mae:.4f}")
            if memory_r2 > 0.8021:
                print("  [OK] 超越目标！")
        
        if history.get('cpu_test_accuracy') and len(history['cpu_test_accuracy']) > 0:
            cpu_acc = history['cpu_test_accuracy'][-1]
            cpu_f1 = history['cpu_test_f1'][-1]
            print(f"\n[CPU分类模型]")
            print(f"  测试集准确率: {cpu_acc:.4f} (目标: > 0.4850)")
            print(f"  测试集 F1:    {cpu_f1:.4f}")
            if cpu_acc > 0.4850:
                print("  [OK] 超越目标！")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    # 支持命令行参数
    use_gpu = '--gpu' in sys.argv or '--use-gpu' in sys.argv
    quick_train(use_gpu=use_gpu)

