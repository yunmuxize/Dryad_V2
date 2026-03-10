# BMv2 混合LGBM模型项目总结

## 项目状态
✅ **已完成训练和验证**

## 最终预测精度

### Memory回归模型 (LGBMRegressor)
- **测试集 R²**: 0.8021 ✅ (优秀)
- **测试集 MAE**: 0.1125
- **测试集 RMSE**: 0.1389

### CPU分类模型 (LGBMClassifier)
- **测试集准确率**: 0.4850 ⚠️ (需要改进)
- **测试集 F1-Score**: 0.3929
- **测试集 MAE** (转换后): 0.0583

详细结果请查看 `FINAL_RESULTS.md`

## 文件结构

### 核心文件
- `dryad_predictor_enhanced.py` - 混合LGBM模型预测器
- `test_lgbm_model.py` - 模型测试脚本
- `README.md` - 项目文档
- `FINAL_RESULTS.md` - 最终结果报告
- `ml_features.csv` - 训练数据

### 训练相关
- `train_background.py` - 后台训练脚本
- `monitor_training.py` - 训练监控脚本
- `start_training.bat` - Windows启动脚本
- `README_TRAINING.md` - 训练文档
- `QUICK_START.md` - 快速开始指南

### 模型文件 (models/)
- `memory_model.pkl` - Memory回归模型
- `cpu_model.pkl` - CPU分类模型
- `cpu_label_encoder.pkl` - CPU标签编码器
- `scaler.pkl` - 特征标准化器
- `feature_names.pkl` - 特征名称

### 可视化 (plots/)
- `prediction_scatter.png` - 预测vs真实值散点图
- `feature_importance.png` - 特征重要性
- `residuals.png` - 残差分析
- `cpu_confusion_matrix.png` - CPU混淆矩阵
- `learning_curves.png` - 学习曲线

### 日志 (logs/)
- `training_history_20251107_201702.json` - 最新训练历史
- 训练日志文件

## 使用方法

### 1. 训练模型
```bash
python dryad_predictor_enhanced.py
```

### 2. 测试模型
```bash
python test_lgbm_model.py
```

### 3. 使用模型预测
```python
from dryad_predictor_enhanced import DryadPredictorEnhanced

predictor = DryadPredictorEnhanced()
predictor.load_models()
result = predictor.predict([0, 1, 2, 3, 0, 1, 2, 3], size=4096)
print(f"CPU: {result['cpu']}, Memory: {result['memory']}")
```

## 技术架构

- **Memory预测**: LGBMRegressor (回归)
- **CPU预测**: LGBMClassifier (分类)
- **特征数量**: 16个 (9个基础 + 7个派生)
- **训练样本**: 2002
- **测试样本**: 501

## 改进方向

1. **CPU预测准确率提升**
   - 收集更多样化的数据
   - 添加系统相关特征
   - 处理类别不平衡问题

2. **模型优化**
   - 尝试其他分类算法
   - 进一步优化超参数
   - 特征选择优化

## 项目完成时间
2025-11-07

