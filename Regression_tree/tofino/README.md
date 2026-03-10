# Dryad P4 资源预测系统 - V4 最终版 (Tiles 级精确预测)

## 项目概述

本项目是 Dryad 框架的核心组件，专门用于预测 P4 程序在 Tofino 硬件上的资源消耗。V4 版本实现了从“百分比预测”到“**Tiles 绝对值预测**”的重大飞跃，有效规避了传统百分比预测在硬件映射时的舍入误差，并集成了编译失败（超限）预警机制。

## 核心技术更新 (V4)

### 1. 预测目标演进：Tiles 级精度
- **旧版 (V1-V3)**: 预测 TCAM/SRAM 使用百分比。存在由于 Tiles 离散性导致的 1-2% 舍入偏离。
- **新版 (V4)**: 直接预测 **TCAM Tiles (0-288)** 和 **SRAM Tiles (0-960)** 绝对数量，确保与硬件物理分布完全一致。

### 2. 增强特征工程 (21 维可解释特征)
系统采用 21 个物理含义明确的特征，涵盖了 P4 表配置的各个维度：
- **基础类型 (8维)**: 8 个标志位的匹配方式编码 (Exact=0, Ternary=1, Range=2, LPM=3)。
- **硬件位宽 (8维)**: 对应字段的物理 Bit 位宽 (1, 8, 16)。
- **规模特征 (1维)**: 归一化后的 P4 Table Size (支持 0 - 20,000 条规则规模)。
- **分布统计 (4维)**: 配置中每种匹配方式出现的频率计数。

### 3. 数据集强化：正负样本闭环
系统不仅学习了 9,000+ 常规成功编译案例，还合并了 **87 个因资源超限而编译失败** 的极端样本 (`compile_results_p4_new_exceeded_theoretical.csv`)。这使得模型具备了识别“即使理论可行，硬件也无法布局”的边界能力。

## 性能评估报告

基于 V4 模型在测试集上的全量验证，各指标性能如下表所示（单位已换算为百分比以便于跨平台对比）：

| 指标 (Target) | MAE (平均绝对误差 %) | RMSE (均方根误差 %) |
| :--- | :--- | :--- |
| **TCAM (Tiles)** | **0.41%** | **1.11%** |
| **SRAM (Tiles)** | **0.01%** | **0.03%** |
| **Pipeline Stages** | **0.77%** | **1.66%** |

> **评估结论**：模型在 SRAM 预测上几乎达到完美精度；在极难预测的 TCAM 和 Stages 指标上，误差均被控制在 1% 左右，完全满足遗传算法在优化过程中的惩罚项评估需求。

## 系统架构与文件规范

### 文件夹结构
```bash
tofino/
├── data/                        # 核心数据集
│   ├── final_merged_resource_analysis.csv    # 主数据集 (Tiles 级)
│   └── compile_results_p4_new_exceeded_theoretical.csv # 失败样本集
├── scripts/                     # 训练与验证脚本
│   ├── dryad_predictor.py       # V4 核心训练器 (支持 Tiles 预测 + 样本合并)
│   └── model_evaluator.py       # V4 综合评估器 (Tiles 级真值对比)
├── models/                      # 持久化 V4 模型 (v2.pkl 为版本后辍)
│   ├── tcam_model_v2.pkl        # Tiles 级 TCAM 模型
│   ├── sram_model_v2.pkl        # Tiles 级 SRAM 模型
│   ├── stages_model_v2.pkl      # 整数级 Stages 模型
│   └── scaler_v2.pkl            # V4 专用特征标准化器
```

## 快速上手

### 1. 模型训练与预警分析
运行主训练脚本，系统会自动合并数据集、进行特征工程，并输出**失败边界风险分析**：
```bash
python scripts/dryad_predictor.py
```

### 2. 精确性能验证
使用评估脚本对比特定样本（如 `Sample_01968`）的预测值与真实 BFRT 编译结果：
```bash
python scripts/model_evaluator.py
```

### 3. 跨平台调用接口
在 Dryad 优化流程中，请统一通过 `Dryad/platform_predictor.py` 调用 `TofinoPlatformPredictor` 类。V4 版本接口已支持直接返回 `tcam_tiles` 和 `sram_tiles`。

## 技术亮点总结
1. **Tiles 级端到端预测**：规避百分比舍入误差。
2. **失败边界学习**：通过合并失败样本，模型学会了“说不”。
3. **21 维可解释特征**：舍弃了黑盒深度学习，采用物理含义明确的随机森林/梯度提升集成方案。
4. **过拟合极低**：通过正则化手段，训练与测试集误差差值保持在 0.1% 以内。
