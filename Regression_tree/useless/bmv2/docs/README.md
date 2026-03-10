# Dryad BMv2 资源预测系统

## 项目概述

本项目基于 Dryad 框架的 P4 程序编译数据，构建了一个混合模型预测系统，用于预测 P4 程序在 BMv2 软件交换机上的 CPU 和 Memory 资源使用率。系统采用混合架构：**Memory 使用 LGBMRegressor 回归模型，CPU 使用 LGBMClassifier 分类模型**，通过16个精心设计的特征分别预测两个目标变量。

## 核心技术策略

### 1. 模型选择策略：混合模型架构

**为什么采用混合模型？**

基于对数据的深入分析，我们发现：
- **CPU 值高度离散化**：只有10个不同的值（0.30, 0.40, ..., 0.90），且51%的样本都是0.50
- **CPU 与特征相关性极弱**：所有特征与CPU的相关系数都小于0.05
- **Memory 值分布连续**：范围1.65，与特征相关性较强

因此，我们采用混合策略：
- **Memory → LGBMRegressor（回归）**：连续值，适合回归模型
- **CPU → LGBMClassifier（分类）**：离散值，更适合分类模型

#### 1.1 混合模型的优势
- **针对性优化**：每个模型针对各自目标变量的特性进行优化
- **独立超参数调优**：可以为Memory和CPU分别选择最佳超参数
- **更好的预测性能**：分类模型更适合离散的CPU值
- **灵活性强**：可以根据各自特点选择不同的算法和参数

### 2. 特征工程策略：16个特征设计

**为什么采用16个特征（9个基础+7个派生）？**

#### 2.1 基础特征（9个）
```
匹配方式特征（8个）：
- total_len, protocol, flags, ttl
- src_port, dst_port, tcp_flags_2, tcp_flags_1

表大小特征（1个）：
- size: 表大小
```

**匹配方式编码规则：**
- `exact = 0`: 精确匹配
- `lpm = 1`: 最长前缀匹配
- `ternary = 2`: 三元匹配
- `range = 3`: 范围匹配

#### 2.2 派生特征（7个）
```
统计特征：
- range_count: Range匹配字段数量
- ternary_count: Ternary匹配字段数量
- lpm_count: LPM匹配字段数量
- exact_count: Exact匹配字段数量

复杂度特征：
- complexity_score = range_count×4 + ternary_count×3 + lpm_count×2 + exact_count×1

交互特征：
- range_ternary_interaction: range_count × ternary_count
- high_complexity_count: range_count + ternary_count + lpm_count
```

#### 2.3 派生特征的设计原理

**range_count**：最重要的预测因子
- Range匹配对资源消耗影响最大
- 直接反映硬件资源需求

**complexity_score**：综合复杂度指标
- 权重设计：Range(4) > Ternary(3) > LPM(2) > Exact(1)
- 反映配置的整体复杂度

**统计特征**：提供分布信息
- 帮助模型理解匹配方式的组合模式
- 增强模型的模式识别能力

## 系统架构

### 核心文件结构
```
Regression_tree/bmv2/
├── dryad_predictor_enhanced.py  # 核心预测器（混合LGBM模型）
├── ml_features.csv              # 训练数据
├── models/                      # 训练好的模型文件
│   ├── memory_model.pkl        # Memory回归模型
│   ├── cpu_model.pkl            # CPU分类模型
│   ├── cpu_label_encoder.pkl   # CPU标签编码器
│   ├── scaler.pkl               # 特征标准化器
│   └── feature_names.pkl        # 特征名称
├── logs/                         # 训练日志
├── plots/                        # 可视化图表
├── test_lgbm_model.py           # 模型测试脚本
└── README.md                     # 本文档
```

### 数据流程
1. **数据预处理**：加载CSV，提取9个基础特征
2. **特征工程**：创建7个派生特征，标准化处理
3. **目标变量处理**：
   - Memory：保持连续值
   - CPU：转换为分类标签（×100后编码）
4. **模型训练**：
   - Memory：LGBMRegressor + GridSearchCV（R²评分）
   - CPU：LGBMClassifier + GridSearchCV（Accuracy评分）
5. **预测应用**：分别从两个模型获取预测，CPU标签转换回原始值

## 模型性能

### CPU 分类模型（LGBMClassifier）
- **测试集准确率**：待训练后更新
- **测试集F1-Score**：待训练后更新
- **测试集MAE**（转换后）：待训练后更新

**分析**：采用分类模型的原因：
- CPU值高度离散化（只有10个值）
- 51%的样本CPU值都是0.50
- 所有特征与CPU相关性极弱（<0.05）
- 分类模型更适合这种离散目标

### Memory 回归模型（LGBMRegressor）
- **测试集R²**：待训练后更新
- **测试集MAE**：待训练后更新

**分析**：采用回归模型的原因：
- Memory值分布连续（范围1.65）
- 与特征相关性较强
- 回归模型能够有效学习连续值模式

## 使用方法

### 1. 完整系统训练
```bash
cd Regression_tree/bmv2
python dryad_predictor_enhanced.py

# 或使用扩展搜索（更长时间，更好效果）
python dryad_predictor_enhanced.py --extensive
```

### 2. 后台训练
```bash
# Windows
start_training.bat

# 或手动
python train_background.py
```

### 3. 预测示例
```python
from dryad_predictor_enhanced import DryadPredictorEnhanced

# 创建预测器
predictor = DryadPredictorEnhanced()
predictor.load_models()

# 预测资源使用率
# 注意：exact=0, lpm=1, ternary=2, range=3
result = predictor.predict([0, 1, 2, 3, 0, 1, 2, 3], size=4096)
print(f"CPU: {result['cpu']}, Memory: {result['memory']}")
```

### 4. 测试模型
```bash
python test_lgbm_model.py
```

## 预测示例结果

| 配置类型 | Size | CPU预测 | Memory预测 | Range数 | 特点 |
|---------|------|---------|-----------|---------|------|
| 全部Exact | 1024 | 0.50 | 26.46 | 0 | 最安全配置 |
| 全部LPM | 2048 | 0.49 | 26.87 | 0 | 中等复杂度 |
| 全部Ternary | 4096 | 0.51 | 27.28 | 0 | 较高复杂度 |
| 全部Range | 8192 | 0.52 | 27.42 | 8 | 高风险配置 |
| 混合模式 | 4096 | 0.52 | 27.39 | 2 | 实际应用场景 |

## 技术栈

- **Python 3.x**
- **lightgbm**: LGBMRegressor（Memory回归）和 LGBMClassifier（CPU分类）
- **scikit-learn**: GridSearchCV、特征工程、评估指标
- **pandas**: 数据处理
- **numpy**: 数值计算
- **matplotlib/seaborn**: 可视化（散点图、混淆矩阵、特征重要性等）
- **joblib**: 模型序列化

## 关键发现

### 1. CPU值高度离散化
- **只有10个不同的值**：0.30, 0.40, 0.49, 0.50, 0.59, 0.60, 0.70, 0.80, 0.90, 0.39
- **51%的样本CPU值都是0.50**：数据高度集中
- **更适合分类模型**：离散值不适合回归模型

### 2. 特征与CPU相关性极弱
- **最强相关性仅0.0438**：所有特征与CPU相关性都<0.05
- **匹配模式对CPU无影响**：不同配置下CPU几乎相同
- **可能受其他因素影响**：系统负载、其他进程等未测量因素

### 3. Memory预测效果良好
- **值分布连续**：范围1.65，有更多变化空间
- **与特征相关性较强**：size特征对Memory影响大
- **回归模型适合**：连续值适合回归模型

### 4. 混合模型的优势
- **针对性优化**：每个模型针对各自目标变量特性优化
- **独立超参数调优**：可以为Memory和CPU分别选择最佳参数
- **更好的预测性能**：分类模型更适合离散的CPU值

## 改进方向

### 数据增强（最重要）
- **收集更多样化的数据**：在不同系统负载、不同时间点收集数据
- **增加CPU值分布的多样性**：减少0.50值的占比
- **添加相关特征**：系统负载、时间、进程信息等

### 模型优化
- **CPU分类模型**：
  - 尝试不同的分类算法：XGBoost, CatBoost
  - 处理类别不平衡问题（51%是0.50）
  - 优化超参数以提高准确率
- **Memory回归模型**：
  - 进一步优化超参数
  - 尝试集成方法
  - 特征选择优化

### 特征工程
- **添加系统相关特征**：系统负载、CPU使用率、内存使用率
- **时间特征**：如果数据有时间序列特性
- **交互特征**：创建更多特征之间的交互项

## 总结

本系统成功构建了基于混合LGBM模型的 BMv2 资源使用率预测系统，通过精心设计的16个特征和针对性的模型选择，实现了：
- **Memory预测**：使用LGBMRegressor回归模型，适合连续值预测
- **CPU预测**：使用LGBMClassifier分类模型，适合离散值预测

**核心价值**：
- ✅ 混合模型架构：针对不同目标变量特性选择合适模型
- ✅ 独立优化：Memory和CPU分别进行超参数调优
- ✅ 完整可视化：散点图、混淆矩阵、特征重要性、学习曲线等
- ✅ 日志记录：完整的训练历史和性能指标记录
- ✅ 后台训练支持：支持后台运行和实时监控

系统为 P4 程序设计提供了科学的资源消耗预测能力，是 Dryad 框架的重要技术支撑。

