# DPDK P4 内存预测系统

## 项目概述

本项目基于 DPDK P4 程序编译数据，构建回归树预测系统，用于预测 P4 程序在 DPDK 平台上的内存使用量（table_memory_estimate_bytes）。系统采用随机森林回归器及集成学习方法，通过精心设计的特征实现高精度预测。

## 核心技术策略

### 1. 模型选择策略：加权集成学习

参考 Tofino 框架的成功经验，采用以下模型组合：

- **Random Forest**: 基础集成模型，抗过拟合能力强
- **Gradient Boosting**: 梯度提升，捕捉复杂非线性关系
- **Extra Trees**: 极端随机树，增强泛化能力
- **Weighted Ensemble**: 基于交叉验证性能的加权集成

### 2. 特征工程策略

#### 2.1 基础特征（来自 CSV）
```
- table_size: 表大小（归一化处理）
- exact_count: Exact 匹配字段数量
- wildcard_count: Wildcard 匹配字段数量
- 8个匹配方式标志位（从文件名提取）
```

#### 2.2 可选增强特征（来自 JSON）
```
- total_key_width_bits: 键总位宽
- total_key_width_bytes: 键总字节数
- match_complexity_score: 匹配复杂度分数
- action_count: 动作数量
- key_instruction_count: 键指令数量
- metadata_width_bits: 元数据位宽
```

#### 2.3 派生特征
```
- normalized_table_size: 归一化表大小
- match_complexity: exact_count + wildcard_count * 2
- size_complexity_interaction: table_size × match_complexity
- wildcard_ratio: wildcard_count / (exact_count + wildcard_count)
```

## 系统架构

### 文件夹结构
```
dpdk/
├── dataset.csv                 # CSV 格式数据集（8000条）
├── dataset.json                # JSON 格式数据集（包含详细元数据）
├── README.md                   # 本文档
├── scripts/                    # Python脚本
│   ├── dpdk_predictor.py       # 核心预测器（训练+预测）
│   ├── dpdk_predictor_v2.py    # V2改进版（加权集成）
│   └── data_explorer.py        # 数据探索工具
├── models/                     # 训练好的模型文件
│   ├── memory_model.pkl
│   ├── memory_model_v2.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── plots/                      # 可视化图表
    ├── feature_importance.png
    └── prediction_analysis.png
```

### 数据格式

**CSV 格式** (推荐用于快速训练):
```csv
p4_file,table_size,exact_count,wildcard_count,table_memory_estimate_bytes
1_2_1_1_2_2_0_0_2341.p4,2341,2,6,65548
```

**JSON 格式** (用于提取额外特征):
```json
{
  "table_size": 2341,
  "exact_count": 2,
  "wildcard_count": 6,
  "total_key_width_bits": 88,
  "table_memory_estimate_bytes": 65548,
  ...
}
```

### 匹配方式编码

从 P4 文件名提取 8 个匹配方式标志位：
- `0`: Exact 精确匹配
- `1`: Ternary 三态匹配
- `2`: Wildcard 通配符匹配

示例: `1_2_1_1_2_2_0_0_2341.p4` → `[1,2,1,1,2,2,0,0]`

## 使用方法

### 1. 数据探索
```bash
python scripts/data_explorer.py
```

### 2. 训练模型

**基础版本**:
```bash
python scripts/dpdk_predictor.py
```

**V2 改进版（推荐）**:
```bash
python scripts/dpdk_predictor_v2.py
```

### 3. 预测示例
```python
from dpdk_predictor_v2 import DPDKPredictorV2

predictor = DPDKPredictorV2()
predictor.load_models()

# 预测内存使用
result = predictor.predict(
    match_types=[1,2,1,1,2,2,0,0],
    table_size=2341,
    exact_count=2,
    wildcard_count=6
)
print(f"预测内存: {result['memory_bytes']} bytes")
```

## 训练环境配置

### Windows 环境（当前）
- **Python**: `C:\Users\86177\anaconda3\envs\linc_env\python.exe`
- **计算资源**: CPU + 多进程（`n_jobs=-1`）
- **推荐理由**: 
  - 数据规模适中（8000条），CPU 已足够快
  - scikit-learn 树模型主要依赖 CPU
  - 避免 GPU 配置复杂性

### Linux 环境（如需迁移）
- **Python**: 默认环境
- **计算资源**: CPU + 多进程
- **长时间训练**: 使用 `nohup` 后台运行

## 性能目标

基于 Tofino 框架的经验，预期性能指标：

- **R² Score**: > 0.85（优秀）
- **MAE**: < 5%（相对误差）
- **RMSE**: < 10%
- **过拟合控制**: 训练R² - 测试R² < 0.15

## 技术栈

- **Python 3.x**
- **scikit-learn**: 随机森林、梯度提升
- **pandas**: 数据处理
- **numpy**: 数值计算
- **matplotlib**: 可视化
- **joblib**: 模型序列化

## 下一步计划

1. ✅ 构建基础预测器（dpdk_predictor.py）
2. ✅ 构建 V2 改进版（dpdk_predictor_v2.py）
3. ⬜ 数据探索和特征分析
4. ⬜ 模型训练和评估
5. ⬜ 性能优化和调参
6. ⬜ 可视化分析

## 参考

本项目参考了 Tofino 框架的设计思路，针对 DPDK 平台进行了适配和优化。
