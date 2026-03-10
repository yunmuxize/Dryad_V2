# 技术报告：IIsy、NetBeacon 与 Planter 在可编程交换机上的模型转化架构与资源可解释性分析

**作者**: Antigravity  
**日期**: 2026-01-24  
**目标环境**: Intel Tofino (P4 Programmable Switch) / ISCX Dataset

---

## 1. 概述
本报告深度拆解了三种主流的将机器学习模型映射至 P4 可编程交换机的方案：**IIsy**、**NetBeacon** 和 **Planter**。报告结合源代码 (`dt2rules.py`, `rf2rules.py`) 分析了它们在处理 **ISCX (97维特征)** 复杂数据集时的具体转化机制，并结合实验数据（SRAM/TCAM 占用率）对不同模型配置下的硬件资源开销进行了可解释性分析。

**实验配置基准**:
*   **数据集**: ISCX (97 维特征，高维复杂流量)
*   **IIsy**: 单决策树 (DT)，深度 8。
*   **NetBeacon**: 随机森林 (RF)，4 棵树，深度 6。
*   **Planter**: 随机森林 (RF)，8 棵树，深度 4。

---

## 2. 核心转化机制深度拆解 (基于源码分析)

### 2.1 IIsy：单树特征离散化 (Single-Tree Discretization)
*   **源码定位**: `DT/dt2rules.py`
*   **P4 架构**: 
    1.  **特征表 (`tbl_fea_f*`)**: 使用 `Range Match`。每个特征一个表。
    2.  **模型表 (`tb_packet_cls`)**: 使用 `Ternary Match`。
*   **算法逻辑**:
    *   **离散化**: Python 脚本首先扫描整棵决策树，提取每个特征的所有分裂阈值 (Thresholds)。
    *   **编码**: `get_feature_table()` 函数将特征的数值区间映射为一个紧凑的 **Feature Code**。
    *   **匹配**: P4 流水线首先查特征表得到 Code，然后将所有特征的 Code 拼接 (`meta.codes_f0`, `meta.codes_f1`...) 作为 Key 去查模型表。
*   **资源特性**: 虽然也是“查表+匹配”结构，但因为只有 **一棵树**，每个特征只需要生成 **一套** Code。中间元数据 (Metadata) 极窄，模型表规模也受限于单树的叶子节点数，因此资源占用极低 (SRAM < 1%, TCAM < 5%)。

### 2.2 NetBeacon：基于 TCAM 的全路径展开 (TCAM-Based Path Expansion)
*   **源码定位**: `RF/Netbeacon/rf2rules.py`
*   **P4 架构**:
    1.  **特征表 (`tbl_fea_f*`)**: **关键差异点** —— 这里使用了 **`Ternary Match`** 而非 Range Match。
    2.  **模型表 (`tb_packet_cls`)**: 同样是 Ternary Match。
*   **算法逻辑**:
    *   **区间转三元组 (`range_to_tenary`)**: 这是 NetBeacon 的核心函数。它强制将一个数值区间 (e.g., `[10, 20]`) 拆解为多个掩码三元组 (e.g., `1xxxx`, `01xxx`) 以适配 TCAM。
    *   **TCAM 填充**: 由于 ISCX 特征未做强剪枝，每个特征的大量区间被转化为海量的 TCAM 条目。
*   **可解释性分析 (为什么 TCAM 71.2%?)**:
    *   **Range -> Ternary 代价**: Tofino 的 Range Match 单元是有限的，但 NetBeacon 选择用 TCAM 来模拟 Range。对于 ISCX 这样高维且阈值细碎的数据集，一个简单的区间可能分裂成 5-6 条 TCAM 规则。
    *   **规则累积**: 4 棵树 x 深度 6 的逻辑路径全展开后，规则基数本身就大，再乘以“区间转三元组”的膨胀系数，直接撑爆了 Stage 3-10 的 TCAM。

### 2.3 Planter：多树并行特征映射 (Multi-Tree Parallel Feature Mapping)
*   **源码定位**: `RF/Planter/rf2rules.py`
*   **P4 架构**:
    1.  **特征表 (`tbl_fea_f*`)**: 使用 `Range Match` (依赖 SRAM)。
    2.  **每棵树的子表 (`tbl_tree_*`)**: $N$ 个并行表，输出局部预测 (`pred_i`)。
    3.  **投票表 (`tb_packet_cls`)**: `Exact Match`，对 `pred_0`...`pred_7` 进行投票。
*   **算法逻辑**:
    *   **并行编码**: 在 `ac_fea_f*` 动作中，Planter 并不只输出一个 Code，而是为每一棵树输出一个独立的 Code (`meta.codes_0_f0`, `meta.codes_1_f0`, ...)。
    *   **源码证据**: `rf2rules.py` 中的 `split_codes` 函数会生成 `codes_%d_fea`。这意味着如果森林有 8 棵树，**每个特征表需要同时返回 8 个独立的编码**。
*   **可解释性分析 (为什么 SRAM 23.9% / Stage 7-8 满载?)**:
    *   **Action Data 膨胀**: 由于需要在一次特征查表中返回 8 棵树所需的编码，单条规则的 Action Data（动作数据）变得非常宽。
    *   **映射表副本**: 虽然物理上是一个特征表，但在逻辑上它承载了 8 棵树的视图。P4 编译器为了满足高吞吐，不得不复制这些巨大的映射表或将其分散。
    *   **局部拥塞**: 所有的 `tbl_tree_*` 和最终的 `vote_table` 都汇聚在流水线后段 (Stage 7/8)。大量的 Map RAM 被用于存储这些表的精确匹配项，导致局部 SRAM 瞬间 100%。

---

## 3. 实验数据深度对比总结

| 方案 | 机制 (源码级) | 资源瓶颈 | ISCX 表现 | 根本原因 |
| :--- | :--- | :--- | :--- | :--- |
| **IIsy** | Range Mapping -> Single Ternary Table | 无 | 极佳 | 单树逻辑简单，元数据窄。 |
| **NetBeacon** | **Range-to-Ternary Conversion** | **TCAM** | **严重溢出 (71%)** | `range_to_tenary` 算法导致规则数量呈倍数膨胀。 |
| **Planter** | **Parallel Encoding (1 Feature -> N Codes)** | **SRAM** | **局部拥塞 (100%)** | 8 棵树导致 Action Data 变宽，且所有树的查找表逻辑堆叠及并行查表压力过大。 |

## 4. 优化建议 (针对 Planter)
1.  **特征复用 (Feature Sharing)**: 修改 `rf2rules.py`，尝试让多棵树共用一套 Feature Code，而不是为每棵树生成独立 Code。这能将 Action Data 宽度减少 8 倍。
2.  **激进的特征选择**: ISCX 的 97 维特征是 SRAM 杀手。只保留 Top-16 特征，直接减少特征表的数量。
