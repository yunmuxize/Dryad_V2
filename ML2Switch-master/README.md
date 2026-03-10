# ML2Switch 框架指南

ML2Switch 是一个旨在将机器学习（Machine Learning）模型训练并高效部署到可编程交换机（P4 Switch）上的框架。它支持多种常见的机器学习算法，并能自动生成对应的 P4 代码和运行时规则。

## 核心功能

*   **多模型支持**：支持决策树 (DT)、随机森林 (RF, 包括 Planter 和 Netbeacon 方案) 以及 XGBoost (XGB)。
*   **端到端流程**：涵盖数据集加载、模型训练、一致性验证、P4 代码生成及规则导出。
*   **硬件兼容性**：生成的 P4 代码针对 Tofino 架构优化。

## 实验结果记录 (UNSW-NB15 数据集)

基于你提供的 `unsw_pkl` 数据集，以下是三个框架的运行结果汇总：

| 框架 | 关键超参数设置 | 测试集准确率 | 规则数量 (Rules) | 生成 P4 文件 |
| :--- | :--- | :--- | :--- | :--- |
| **DT** (决策树) | `max_depth=8` | **98.87%** | 177 | `DT/dt.p4` |
| **RF/Netbeacon** | `n_estimators=1, max_depth=5` | **97.65%** | 93 | `RF/Netbeacon/rf.p4` |
| **RF/Planter** | `n_estimators=3, max_depth=5` | **98.09%** | 124 | `RF/Planter/rf.p4` |

---

## 运行步骤与超参数说明

### 1. 运行步骤

所有的模型训练脚本均位于各自的算法文件夹内。运行前请确保已激活对应的 Python 环境。

*   **决策树 (DT)**:
    ```bash
    cd DT
    python train_dt.py --key unsw_pkl --max_depth 8
    ```
*   **随机森林 (Netbeacon)**:
    ```bash
    cd RF/Netbeacon
    python train_rf.py --key unsw_pkl --max_depth 5 --n_estimators 1 --num_classes 2
    ```
*   **随机森林 (Planter)**:
    ```bash
    cd RF/Planter
    python train_rf.py --key unsw_pkl --max_depth 5 --n_estimators 3 --class_num_bits 1
    ```

### 2. 关键超参数含义

| 参数 | 含义 | 说明 |
| :--- | :--- | :--- |
| `--key` | 数据集标识符 | 必须在 `utils/__init__.py` 中定义。例如 `unsw_pkl`。 |
| `--max_depth` | 决策树最大深度 | 控制模型复杂度。深度越大精度越高，但会消耗更多交换机阶段（Stages）。 |
| `--n_estimators`| 树的数量 (仅 RF) | 即随机森林中包含多少棵决策树。Planter 支持多棵树聚合，Netbeacon 建议树数量较少。 |
| `--num_classes` | 类别总数 | 数据集中的分类标签数量（如二分类为 2）。 |
| `--class_num_bits`| 类别占用位宽 | 存储分类结果需要的 bit 数。例如 2 分类占 1 bit，4 分类占 2 bit。 |

---

## 目录结构

*   `DT/`: 决策树相关代码及 P4 转换逻辑。
*   `RF/Netbeacon`: 传统的随机森林部署方案。
*   `RF/Planter`: 更加优化的随机森林部署方案（推荐用于多树场景）。
*   `utils/`: 数据加载接口 `load_data` 所在位置。

## 验证流程
1.  运行脚本后，检查当前目录下的 `.p4` 文件和 `_setup.py` 文件。
2.  对比 `xxx_sklearn.txt` 中的预测结果与交换机真实输出，确保一致性。

---
*本项目由团队协作开发，旨在推动可编程网络与人工智能的深度融合。*
