# 训练指南 - 增强版

## 快速开始

### 1. 标准训练（前台运行）
```bash
python dryad_predictor_enhanced.py
```

### 2. 扩展超参数搜索（更长时间，更好效果）
```bash
python dryad_predictor_enhanced.py --extensive
```

### 3. 后台训练（Windows）
```bash
# 方法1: 使用批处理文件
start_training.bat

# 方法2: 使用start命令
start /MIN python train_background.py

# 方法3: 使用start命令 + 扩展搜索
start /MIN python train_background.py --extensive
```

### 4. 监控训练进程
```bash
# 在另一个终端窗口运行
python monitor_training.py

# 或者指定检查间隔（秒）
python monitor_training.py 10
```

## 功能特性

### 1. 增强的特征工程
- 基础特征：9个原始特征
- 派生特征：5个统计特征
- **新增**：2个交互特征
  - `range_ternary_interaction`: Range和Ternary的交互
  - `high_complexity_count`: 高复杂度匹配总数

### 2. 优化的超参数搜索
- **标准模式**：快速搜索，适合快速迭代
- **扩展模式**：全面搜索，包含更多参数组合
- **自定义评分**：同时优化CPU和Memory，CPU权重更高（60%）

### 3. 完整的日志系统
- 所有训练过程记录到 `logs/training_YYYYMMDD_HHMMSS.log`
- 训练历史保存为JSON格式
- 实时日志输出到控制台

### 4. 丰富的可视化
生成以下图表到 `plots/` 目录：
- `prediction_scatter.png`: 预测vs真实值散点图
- `feature_importance.png`: 特征重要性
- `residuals.png`: 残差分析
- `learning_curves.png`: 学习曲线

### 5. 性能优化
- 使用 `RobustScaler` 替代 `StandardScaler`（对异常值更鲁棒）
- 优化的超参数网格
- 针对CPU预测的专门优化

## 性能改进策略

### CPU预测优化
1. **特征增强**：添加交互特征
2. **评分函数**：提高CPU权重（60% vs 40%）
3. **超参数调优**：扩展搜索空间
4. **数据预处理**：使用RobustScaler

### Memory预测优化
1. **保持现有优势**：Memory预测已经很好（R²=0.8056）
2. **进一步优化**：通过扩展搜索提升

## 文件结构

```
Regression_tree/bmv2/
├── dryad_predictor_enhanced.py  # 增强版预测器
├── train_background.py          # 后台训练脚本
├── monitor_training.py          # 监控脚本
├── start_training.bat           # Windows批处理启动脚本
├── logs/                        # 日志目录
│   ├── training_*.log           # 训练日志
│   └── training_history_*.json # 训练历史
├── plots/                       # 可视化图表
│   ├── prediction_scatter.png
│   ├── feature_importance.png
│   ├── residuals.png
│   └── learning_curves.png
└── models/                      # 模型文件
    ├── unified_model.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

## 使用示例

### 示例1: 快速训练
```bash
python dryad_predictor_enhanced.py
```

### 示例2: 完整训练（推荐）
```bash
# 终端1: 启动训练
start /MIN python train_background.py --extensive

# 终端2: 监控进度
python monitor_training.py
```

### 示例3: 只监控（训练已在运行）
```bash
python train_background.py --monitor-only
```

## 性能指标

训练完成后，查看以下指标：

1. **日志文件** (`logs/training_*.log`)
   - 完整的训练过程
   - 超参数搜索结果
   - 最终性能指标

2. **训练历史** (`logs/training_history_*.json`)
   - CPU和Memory的R²和MAE
   - 可用于后续分析

3. **可视化图表** (`plots/`)
   - 直观查看模型性能
   - 识别问题和改进方向

## 故障排除

### 问题1: 训练时间过长
- 使用标准模式而非扩展模式
- 减少超参数搜索空间

### 问题2: CPU预测仍然较差
- 检查特征工程是否生效
- 查看特征重要性图
- 考虑添加更多特征

### 问题3: 监控脚本无输出
- 确认训练脚本正在运行
- 检查日志目录是否存在
- 确认日志文件正在更新

## 下一步优化建议

1. **特征工程**
   - 尝试多项式特征
   - 添加更多交互特征
   - 考虑特征选择

2. **模型改进**
   - 尝试XGBoost或LightGBM
   - 使用集成方法
   - 考虑深度学习模型

3. **数据增强**
   - 收集更多数据
   - 数据平衡
   - 异常值处理

