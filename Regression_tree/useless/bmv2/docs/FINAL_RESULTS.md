# 最终预测精度报告

## 训练完成时间
2025-11-07 20:17:02

## Memory回归模型 (LGBMRegressor)

### 训练集性能
- **R² Score**: 0.8630
- **MAE**: 0.0957
- **RMSE**: 0.1183

### 测试集性能
- **R² Score**: 0.8021 ✅ (优秀)
- **MAE**: 0.1125
- **RMSE**: 0.1389

**评价**: Memory预测效果优秀，R²达到0.80以上，说明模型能够有效预测Memory使用率。

---

## CPU分类模型 (LGBMClassifier)

### 训练集性能
- **准确率 (Accuracy)**: 0.7762
- **F1-Score (weighted)**: 0.7620
- **MAE** (转换后): 0.0234
- **RMSE** (转换后): 0.0505

### 测试集性能
- **准确率 (Accuracy)**: 0.4850 ⚠️ (需要改进)
- **F1-Score (weighted)**: 0.3929 ⚠️
- **MAE** (转换后): 0.0583
- **RMSE** (转换后): 0.0859

**评价**: CPU分类准确率较低（48.5%），主要原因是：
1. CPU值高度离散化（只有10个值）
2. 51%的样本CPU值都是0.50，类别不平衡
3. 所有特征与CPU相关性极弱（<0.05）

**改进建议**:
- 收集更多样化的数据
- 添加系统相关特征（系统负载、时间等）
- 处理类别不平衡问题
- 尝试其他分类算法（XGBoost, CatBoost）

---

## 总结

### 成功方面
✅ **Memory预测优秀**: R² = 0.8021，模型能够有效预测Memory使用率
✅ **混合模型架构**: 针对不同目标变量特性选择合适模型
✅ **独立优化**: Memory和CPU分别进行超参数调优

### 需要改进
⚠️ **CPU预测准确率较低**: 48.5%，需要进一步优化
⚠️ **数据质量**: CPU值高度集中，需要更多样化的数据

---

## 模型文件
- `models/memory_model.pkl`: Memory回归模型
- `models/cpu_model.pkl`: CPU分类模型
- `models/cpu_label_encoder.pkl`: CPU标签编码器
- `models/scaler.pkl`: 特征标准化器
- `models/feature_names.pkl`: 特征名称

## 可视化文件
- `plots/prediction_scatter.png`: 预测vs真实值散点图
- `plots/feature_importance.png`: 特征重要性
- `plots/residuals.png`: 残差分析
- `plots/cpu_confusion_matrix.png`: CPU混淆矩阵
- `plots/learning_curves.png`: 学习曲线

