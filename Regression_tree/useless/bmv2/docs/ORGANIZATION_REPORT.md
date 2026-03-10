# BMv2文件夹整理报告

## 整理完成时间
2025-11-07

## 整理结果

### 文件夹结构

```
bmv2/
├── dryad_predictor_enhanced.py    # 核心预测器（保留在根目录）
├── ml_features.csv                 # 训练数据（保留在根目录）
│
├── scripts/                        # 脚本目录
│   ├── train_optimized.py         # 优化训练脚本
│   ├── test_lgbm_model.py         # 模型测试脚本
│   ├── monitor_training.py        # 训练监控脚本
│   ├── check_training_status.py   # 状态检查脚本
│   ├── start_optimized_training.bat  # 优化训练启动脚本
│   ├── start_training.bat         # 标准训练启动脚本
│   └── organize_*.py/ps1          # 整理脚本
│
├── docs/                           # 文档目录
│   ├── README.md                   # 项目主文档
│   ├── README_TRAINING.md          # 训练文档
│   ├── QUICK_START.md              # 快速开始指南
│   ├── FINAL_RESULTS.md            # 最终结果报告
│   ├── SUMMARY.md                  # 项目总结
│   └── ORGANIZATION_REPORT.md      # 本报告
│
├── models/                         # 模型文件目录
│   ├── memory_model.pkl
│   ├── cpu_model.pkl
│   ├── cpu_label_encoder.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── logs/                           # 日志目录
│   ├── training_*.log              # 训练日志
│   └── training_history_*.json     # 训练历史
│
├── plots/                          # 可视化图表目录
│   ├── prediction_scatter.png
│   ├── feature_importance.png
│   ├── residuals.png
│   ├── cpu_confusion_matrix.png
│   └── learning_curves.png
│
└── analysis/                       # 分析目录（空）
```

### 整理统计

- **移动文件数**: 13个
- **创建目录数**: 2个（scripts/, docs/）
- **根目录文件数**: 2个（核心文件）

## 训练状态

### 当前运行状态

- **train_optimized.py**: 未在运行
- **dryad_predictor_enhanced.py**: 已运行完成（21:36:19）

### 日志文件

1. **dryad_predictor_20251107_213615.log** (6.06 KB)
   - 最后修改: 2025-11-07 21:36:19
   - 状态: 标准训练已完成

2. **training_20251107_213618.log** (2.57 KB)
   - 最后修改: 2025-11-07 21:36:19
   - 状态: 训练进行中

3. **training_20251107_201307.log** (4.07 KB)
   - 最后修改: 2025-11-07 20:17:02
   - 状态: 快速训练已完成

### 模型文件

- **最后更新**: 2025-11-07 20:17:02
- **模型数量**: 5个
- **总大小**: ~4 MB

## 启动优化训练

要启动优化训练（目标Memory R² >= 0.90），请执行：

```bash
cd scripts
start_optimized_training.bat
```

或者：

```bash
python scripts/train_optimized.py
```

日志文件将保存在 `logs/training_optimized_YYYYMMDD_HHMMSS.log`

## 检查训练状态

使用以下命令检查训练状态：

```bash
python scripts/check_training_status.py
```

## 文件说明

### 核心文件（根目录）
- `dryad_predictor_enhanced.py`: 混合LGBM模型预测器核心类
- `ml_features.csv`: 训练数据集

### 脚本文件（scripts/）
- `train_optimized.py`: 优化训练脚本（extensive_search=True）
- `test_lgbm_model.py`: 模型测试脚本
- `monitor_training.py`: 实时监控训练进度
- `check_training_status.py`: 检查训练状态和进程
- `start_optimized_training.bat`: Windows后台启动脚本

### 文档文件（docs/）
- `README.md`: 项目主文档
- `README_TRAINING.md`: 训练指南
- `QUICK_START.md`: 快速开始
- `FINAL_RESULTS.md`: 最终结果报告
- `SUMMARY.md`: 项目总结

## 注意事项

1. **日志重定向**: 所有训练日志都保存在 `logs/` 目录
2. **后台训练**: 使用 `start_optimized_training.bat` 启动后台训练
3. **进程监控**: 使用 `check_training_status.py` 检查训练进程
4. **文件组织**: 所有脚本和文档已分类整理，根目录保持简洁

## 下一步

1. 启动优化训练: `scripts/start_optimized_training.bat`
2. 监控训练进度: `python scripts/monitor_training.py`
3. 检查训练状态: `python scripts/check_training_status.py`
4. 查看最终结果: 训练完成后查看 `docs/FINAL_RESULTS.md`

