# ML2Switch 框架完整技术分析

## 目录
1. [框架概述](#1-框架概述)
2. [数据集与特征配置](#2-数据集与特征配置)
3. [三种模型的训练逻辑](#3-三种模型的训练逻辑)
4. [P4表结构对比](#4-p4表结构对比)
5. [树模型到P4的转化逻辑](#5-树模型到p4的转化逻辑)
6. [参数如何影响资源消耗](#6-参数如何影响资源消耗)
7. [实验结果与分析](#7-实验结果与分析)
8. [总结与建议](#8-总结与建议)

---

## 1. 框架概述

ML2Switch 是一个将机器学习树模型（Decision Tree、Random Forest）部署到 Tofino 可编程交换机的框架。它包含三种部署方案：

| 方案 | 模型类型 | 原始论文 | P4表结构 | 核心特点 |
|------|---------|---------|---------|---------|
| **DT (IIsy)** | 决策树 | IIsy | 特征表 + 分类表 | 单树，最简洁 |
| **RF-NetBeacon** | 随机森林 | NetBeacon | 特征表(Ternary) + 模型表 | 多树合并，.dot文件解析 |
| **RF-Planter** | 随机森林 | Planter | 特征表 + 树表 + 投票表 | 多树并行，显式投票 |

---

## 2. 数据集与特征配置

### 2.1 支持的数据集

```python
# utils/__init__.py 中的 load_data 函数
def load_data(key="univ"):
    if key == "univ":      return load_univ_data()
    elif key == "unsw":    return load_unsw_data()
    elif key == "unsw_pkl":return load_unsw_pkl_data()
    elif key == "iscx":    return load_iscx_data()
```

### 2.2 特征配置对比

| 数据集 | 特征数量 | 特征列表 | 类别数 |
|--------|---------|---------|--------|
| UNSW-NB15 | 8 | (匿名数值特征) | 2 |
| UNIV | 10 | srcPort, dstPort, protocol, ip_ihl, ip_tos, ip_ttl, tcp_dataofs, tcp_window, udp_len, length | 多类 |
| ISCX | 10 | (从PKL直接加载) | 6 |

### 2.3 特征到P4字段的映射（人为定义）

**DT 和 Planter 使用相同映射：**
```python
header_names = {
    'f0': ['meta.srcPort', 16],       # 源端口
    'f1': ['meta.dstPort', 16],       # 目的端口
    'f2': ['hdr.ipv4.protocol', 8],   # IP协议
    'f3': ['hdr.ipv4.ihl', 4],        # IP头部长度
    'f4': ['hdr.ipv4.tos', 8],        # 服务类型
    'f5': ['hdr.ipv4.ttl', 8],        # TTL
    'f6': ['meta.dataOffset', 4],     # TCP数据偏移
    'f7': ['meta.window', 16],        # TCP窗口
    'f8': ['meta.udp_length', 16],    # UDP长度
    'f9': ['hdr.ipv4.totalLen', 16]   # IP总长度
}
```

**NetBeacon 使用不同映射：**
```python
header_names = {
    'f0': ['hdr.ipv4.protocol', 8],   # IP协议
    'f1': ['hdr.ipv4.ihl', 4],        # IP头部长度
    'f2': ['hdr.ipv4.tos', 8],        # 服务类型
    'f3': ['hdr.ipv4.flags', 3],      # IP标志位 ← 不同！
    'f4': ['hdr.ipv4.ttl', 8],        # TTL
    'f5': ['meta.dataOffset', 4],     # TCP数据偏移
    'f6': ['meta.flags', 8],          # TCP标志位 ← 不同！
    'f7': ['meta.window', 16],        # TCP窗口
    'f8': ['meta.udp_length', 16],    # UDP长度
    'f9': ['hdr.ipv4.totalLen', 16]   # IP总长度
}
```

**关键差异**：
- DT/Planter: f0/f1 是**端口信息**
- NetBeacon: f0/f1 是**协议信息**，f3/f6 包含**标志位**

---

## 3. 三种模型的训练逻辑

### 3.1 DT (Decision Tree) - IIsy方案

**训练脚本**：`DT/train_dt.py`

```python
def main(args):
    train_X, train_y, test_X, test_y = load_data(args.key)
    
    # 核心：sklearn 决策树训练
    clf = tree.DecisionTreeClassifier(
        max_depth=args.max_depth,  # 关键参数
        random_state=2023
    )
    clf.fit(train_X, train_y)
    
    # 转化为P4规则
    num_rules = export_dt(clf, len(test_X[0]))
```

**关键参数**：
- `--max_depth`：树的最大深度（默认8）
- `--key`：数据集标识符

### 3.2 RF-NetBeacon (Random Forest - NetBeacon方案)

**训练脚本**：`RF/Netbeacon/train_rf.py`

```python
def main(args):
    train_X, train_y, test_X, test_y = load_data(args.key)
    
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,  # 树的数量
        max_depth=args.max_depth,        # 树的深度
        random_state=2023
    )
    clf.fit(train_X, train_y)
    
    # 导出每棵树为 .dot 文件
    for i in range(len(clf.estimators_)):
        export_graphviz(clf.estimators_[i], 
                        out_file='output/rf_tree_{}.dot'.format(i))
    
    # 从.dot文件解析并生成P4规则
    num_rules = export_rf(len(test_X[0]), args.n_estimators)
```

**关键参数**：
- `--max_depth`：每棵树的最大深度（默认3）
- `--n_estimators`：树的数量（默认1）
- `--num_classes`：类别数量

**特殊流程**：先导出为 `.dot` 文件，再解析生成规则

### 3.3 RF-Planter (Random Forest - Planter方案)

**训练脚本**：`RF/Planter/train_rf.py`

```python
def main(args):
    train_X, train_y, test_X, test_y = load_data(args.key)
    
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=2023
    )
    clf.fit(train_X, train_y)
    
    # 直接从clf对象生成P4规则（带投票表）
    num_rules = export_rf(clf, len(test_X[0]), args.class_num_bits)
```

**关键参数**：
- `--max_depth`：每棵树的最大深度（默认3）
- `--n_estimators`：树的数量（默认2）
- `--class_num_bits`：存储类别的比特数（6类需要3bits）

---

## 4. P4表结构对比

### 4.1 DT (IIsy) - 两级表结构

```
┌─────────────────────────────────────────────────────────────┐
│                        P4 流水线                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌───────────────┐     ┌───────────────┐                   │
│   │ tbl_fea_f0    │ --> │ tbl_fea_f1    │ --> ... -->       │
│   │ (Range Match) │     │ (Range Match) │                   │
│   │ key: srcPort  │     │ key: dstPort  │                   │
│   │ action: code  │     │ action: code  │                   │
│   └───────────────┘     └───────────────┘                   │
│           ↓                     ↓                           │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   tb_packet_cls                      │   │
│   │                 (Ternary Match)                      │   │
│   │  key: codes_f0, codes_f1, ..., codes_fn : ternary   │   │
│   │  action: ac_packet_forward(port)                    │   │
│   │  规则数 = 叶子节点数                                   │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**表类型**：
1. **特征表 (tbl_fea_fx)**：每个使用的特征一个表，Range Match
2. **分类表 (tb_packet_cls)**：单表，Ternary Match

### 4.2 RF-NetBeacon - 两级表结构（但融合多树）

```
┌─────────────────────────────────────────────────────────────┐
│                        P4 流水线                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌───────────────┐     ┌───────────────┐                   │
│   │ tbl_fea_f0    │ --> │ tbl_fea_f2    │ --> ... -->       │
│   │ (Ternary)     │     │ (Ternary)     │                   │
│   │ 多树阈值合并   │     │ 多树阈值合并   │                   │
│   └───────────────┘     └───────────────┘                   │
│           ↓                     ↓                           │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   tb_packet_cls                      │   │
│   │                 (Ternary Match)                      │   │
│   │  key: codes_f0, codes_f2, ... : ternary             │   │
│   │  action: ac_packet_forward(port)                    │   │
│   │  规则数 = 所有树的叶子路径组合（去冲突后）              │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键差异**：
- 特征表使用 **Ternary Match**（而非Range）
- 多棵树的阈值被**合并**到同一套特征表中
- 模型表中的规则是**多树路径的笛卡尔积**（去冲突后）

### 4.3 RF-Planter - 三级表结构（显式投票）

```
┌─────────────────────────────────────────────────────────────┐
│                        P4 流水线                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1-N: 特征编码                                         │
│   ┌───────────────┐     ┌───────────────┐                   │
│   │ tbl_fea_f0    │ --> │ tbl_fea_f1    │ --> ...           │
│   │ (Range Match) │     │ (Range Match) │                   │
│   │ 输出: code0,  │     │ 输出: code0,  │                   │
│   │      code1... │     │      code1... │                   │
│   └───────────────┘     └───────────────┘                   │
│           ↓                     ↓                           │
│  Stage N+1 ~ N+T: 每棵树独立分类                             │
│   ┌───────────────┐     ┌───────────────┐                   │
│   │ tbl_tree_0    │     │ tbl_tree_1    │ ...               │
│   │ (Ternary)     │     │ (Ternary)     │                   │
│   │ 输出: pred_0  │     │ 输出: pred_1  │                   │
│   └───────────────┘     └───────────────┘                   │
│           ↓                     ↓                           │
│  Stage Final: 投票聚合                                       │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   tb_packet_cls                      │   │
│   │                  (Exact Match)                       │   │
│   │  key: pred_0, pred_1, ..., pred_n : exact           │   │
│   │  action: ac_packet_forward(voted_class)             │   │
│   │  规则数 = 所有可能的预测组合                           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键差异**：
- **三级结构**：特征表 → 树表 → 投票表
- 每棵树有**独立的树表** (tbl_tree_0, tbl_tree_1, ...)
- 投票表使用 **Exact Match**（而非Ternary）
- 投票逻辑：多数投票（majority voting）

---

## 5. 树模型到P4的转化逻辑

### 5.1 DT转化流程

#### Step 1: 遍历决策树获取所有分裂点
```python
def get_splits(dt, feature_names):
    """提取树中所有的分裂节点"""
    for i in range(n_nodes):
        threshold = clf.tree_.threshold[i]
        feature = features[i]
        if threshold != -2.0:  # -2.0表示叶子节点
            data.append([node_id, left_child_id, right_child_id, threshold, feature])
    return data
```

#### Step 2: 为每个特征构建阈值区间表
```python
def get_feature_table(splits_data, feature_name):
    """
    将特征的所有阈值转换为区间编码
    例如：阈值 [10, 50, 100] 
    转换为区间：[0,10), [10,50), [50,100), [100,∞)
    每个区间对应一个二进制编码
    """
    # 按阈值排序
    feature_data = feature_data.sort_values(by="Threshold")
    # 为每个区间生成编码
    code_table["code"] = binary_encoding(...)
    return code_table
```

#### Step 3: 遍历所有叶子节点获取路径编码
```python
def get_codes_and_masks(clf, feature_names):
    """
    为每个叶子节点生成：
    - Code: 路径上的分裂方向编码
    - Mask: 标记哪些特征参与了分裂
    """
    for branch in retrieve_branches(clf):
        # 遍历路径上每个节点
        for split in branch:
            if split in path:
                mask[index] = 1  # 这个特征参与了决策
        # 编码路径方向（左0右1）
        code = encode_path_direction(branch)
    return final_codes, masks
```

#### Step 4: 生成P4代码和规则
```python
def export_dt(clf, n_fea):
    # 1. 生成特征表P4代码
    for fea in used_features:
        fea_tbl += f"""
        table tbl_fea_{fea}{{
            key= {{{header_names[fea][0]} : range;}}
            actions = {{ac_fea_{fea};}}
            size={len(Ranges)};
        }}
        """
    
    # 2. 生成分类表P4代码
    tb_packet_cls = f"""
    table tb_packet_cls {{
        key = {{
            meta.codes_f0 : ternary;
            meta.codes_f1 : ternary;
            ...
        }}
        actions = {{ac_packet_forward;}}
        size = {num_leaves};
    }}
    """
    
    # 3. 生成BFRT规则
    for code, mask, class_id in zip(Final_Codes, Final_Masks, Classes):
        print(f"tb_packet_cls.add_with_ac_packet_forward("
              f"codes_f0={code[0]}, codes_f0_mask={mask[0]}, "
              f"..., port={class_id})")
```

### 5.2 NetBeacon转化流程

#### 核心差异：从.dot文件解析

```python
def get_rf_feature_thres(model_file, keys, tree_num):
    """从.dot文件提取所有阈值"""
    for i in range(tree_num):
        with open(model_file + f'_{i}.dot', 'r') as f:
            lines = f.readlines()
        for line in lines:
            # 解析: "0 [label="f0 <= 50\ngini = ..."
            m = re.search(r".*\[label=\"(.*?) <= (.*?)\\n.*", line)
            if m:
                feat_dict[m.group(1)].append(float(m.group(2)))
```

#### 多树路径合并（笛卡尔积 + 冲突检测）
```python
def get_rf_trees_table_entries(...):
    """
    生成模型表规则：
    - 遍历所有树的叶子节点组合
    - 检测路径冲突
    - 计算最终概率
    """
    for tup in product(*loop_val):  # 笛卡尔积
        # 检查是否有冲突的特征值
        for f in range(len(keys)):
            a = min(tree_leaves[i][f*2] for i in tup)
            b = min(tree_leaves[i][f*2+1] for i in tup)
            if a + b <= 0:  # 冲突！
                flag = 1
                break
        
        if flag == 0:  # 无冲突，添加规则
            leaf_sum = sum(leaf_info[i] for i in tup) / len(tup)
            tree_data.append([range_marks..., leaf_sum])
```

### 5.3 Planter转化流程

#### 核心差异：显式树表和投票表

```python
def export_rf(clf, n_fea, class_num_bits):
    """Planter的三级表结构"""
    
    # 1. 为每棵树生成独立的编码
    for tree_id in range(n_tree):
        Final_Codes, Final_Masks = get_codes_and_masks(
            clf.estimators_[tree_id], feature_names)
        Classe, Certain = get_classes(clf.estimators_[tree_id])
        model_dict[f"code{tree_id}"] = [Final_Codes, Final_Masks, Classe]
    
    # 2. 生成特征表（每个特征输出多棵树的编码）
    for fea in feature_names:
        fea_tbl += f"""
        action ac_fea_{fea}(bit<N> code0, bit<N> code1, ...){{
            meta.codes_0_{fea} = code0;
            meta.codes_1_{fea} = code1;
            ...
        }}
        table tbl_fea_{fea}{{
            key= {{{header_names[fea][0]} : range;}}
            actions = {{ac_fea_{fea};}}
        }}
        """
    
    # 3. 生成树表（每棵树独立分类）
    for i in range(n_tree):
        tree_tbl += f"""
        action ac_tree_{i}(bit<{class_num_bits}> cls){{
            meta.pred_{i} = cls;
        }}
        table tbl_tree_{i}{{
            key={{
                meta.codes_{i}_f0 : ternary;
                meta.codes_{i}_f1 : ternary;
                ...
            }}
            actions = {{ac_tree_{i};}}
            size = {num_leaves[i]};
        }}
        """
    
    # 4. 生成投票表（多数投票）
    for comb in comb_tree_preds([], forest_domain):
        voted_class = max(set(comb), key=comb.count)  # 多数投票
        print(f"tb_packet_cls.add_with_ac_packet_forward("
              f"pred_0={comb[0]}, pred_1={comb[1]}, ..., "
              f"port={voted_class})")
```

---

## 6. 参数如何影响资源消耗

### 6.1 max_depth 的影响

| max_depth | 最大叶子数 | 对资源的影响 |
|-----------|-----------|-------------|
| 3 | 8 | 叶子少，规则少，但模型能力弱 |
| 5 | 32 | 中等规模 |
| 8 | 256 | 叶子多，规则多，模型能力强 |
| 10 | 1024 | 可能超出硬件限制 |

**资源消耗公式**：
```
特征表规则数 ≈ Σ(每个特征的阈值数 + 1)
分类表规则数 = 叶子节点数 ≤ 2^max_depth
```

### 6.2 n_estimators 的影响（仅RF）

#### NetBeacon：
```
模型表规则数 ≈ Π(每棵树的叶子数) / 冲突过滤
```
- 多树路径组合呈**指数增长**
- 冲突检测会过滤部分规则

#### Planter：
```
树表规则数 = Σ(每棵树的叶子数)
投票表规则数 = Π(每棵树的类别数)
```
- 树表规则数**线性增长**
- 投票表规则数**指数增长**（但类别数通常较少）

### 6.3 实验数据验证（ISCX数据集）

| 模型 | depth | trees | 规则数 | 准确率 |
|------|-------|-------|--------|--------|
| DT | 5 | 1 | 64 | 93.65% |
| DT | 8 | 1 | 272 | 96.77% |
| RF-Planter | 5 | 2 | 155 | 93.23% |
| RF-Planter | 8 | 1 | 251 | 94.43% |
| RF-Planter | 8 | 3 | 913 | 96.22% |
| RF-NetBeacon | 5 | 2 | 323 | 93.23% |

**关键发现**：
1. **depth 是决定性因素**：depth 5→8 带来 3.12% 准确率提升
2. **trees 影响较小**：在 depth=8 时，trees 1→3 带来 1.79% 提升
3. **NetBeacon规则更多**：同配置下，NetBeacon规则数 > Planter规则数

---

## 7. 实验结果与分析

### 7.1 完整实验记录（ISCX数据集，6分类）

| 模型 | Depth | Trees | Train Acc | Test Acc | F1 (Macro) | Rules |
|------|-------|-------|-----------|----------|------------|-------|
| DT | 8 | 1 | 96.86% | **96.77%** | **0.8593** | 272 |
| RF-Planter | 8 | 3 | 96.26% | 96.22% | 0.8423 | 913 |
| RF-Planter | 8 | 1 | 94.50% | 94.43% | 0.8112 | 251 |
| DT | 5 | 1 | 93.71% | 93.65% | 0.7782 | 64 |
| RF-NetBeacon | 5 | 2 | 93.31% | 93.23% | 0.7716 | 323 |
| RF-Planter | 5 | 2 | 93.31% | 93.23% | 0.7716 | 155 |

### 7.2 关键发现

#### 1. DT (单树深度8) 优于 RF (多树深度5)
- **原因**：深度决定了模型的表达能力
- **公式**：表达能力 ∝ 2^depth，而非 trees × 2^depth

#### 2. 在相同深度下，DT 可能优于 RF
- **实验**：DT(d=8) = 96.77% > RF-Planter(d=8,t=3) = 96.22%
- **原因**：RF的投票机制在类别不平衡时可能稀释少数类信号

#### 3. 规则数量 ≠ 性能
- DT: 272 rules → 96.77%（每规则贡献 0.356%）
- RF: 913 rules → 96.22%（每规则贡献 0.105%）

### 7.3 类别不平的影响

| Class | 样本占比 | DT(d=8) F1 | RF(d=8,t=3) F1 |
|-------|---------|------------|----------------|
| 0 | 7% | 0.8348 | 0.8183 |
| **1** | **2%** | **0.4320** | **0.3667** |
| 2 | 68% | 0.9939 | 0.9900 |
| 3 | 7% | 0.9374 | 0.9138 |
| 4 | 11% | 0.9851 | 0.9868 |
| 5 | 5% | 0.9728 | 0.9783 |

**结论**：少数类(Class 1)在RF中表现更差，投票机制导致信号被稀释。

---

## 8. 总结与建议

### 8.1 三种方案对比

| 维度 | DT (IIsy) | RF-NetBeacon | RF-Planter |
|------|-----------|--------------|------------|
| **表结构** | 2级表 | 2级表 | 3级表 |
| **Match类型** | Range + Ternary | Ternary | Range + Ternary + Exact |
| **规则效率** | 高 | 中 | 中 |
| **Stage消耗** | 低 | 中 | 高 |
| **可扩展性** | 单树限制 | 多树合并复杂 | 多树并行清晰 |
| **适用场景** | 简单分类 | 需要融合策略 | 需要显式投票 |

### 8.2 参数配置建议

#### 对于高准确率需求：
```bash
# DT: 使用较深的树
python train_dt.py --key iscx --max_depth 10

# RF-Planter: 使用较深的树 + 适量树数
python train_rf.py --key iscx --max_depth 8 --n_estimators 3 --class_num_bits 3
```

#### 对于资源受限场景：
```bash
# DT: 浅层树，规则最少
python train_dt.py --key iscx --max_depth 5

# RF-Planter: 少量浅树
python train_rf.py --key iscx --max_depth 4 --n_estimators 2 --class_num_bits 3
```

### 8.3 框架选择建议

1. **首选 DT (IIsy)**：
   - 规则效率最高
   - Stage消耗最低
   - 在大多数场景下性能不输RF

2. **使用 RF-Planter 当**：
   - 需要模型可解释性（显式投票）
   - 硬件Stage充足
   - 需要集成多个分类器

3. **使用 RF-NetBeacon 当**：
   - 需要复杂的特征融合策略
   - 已有预训练的.dot模型文件
   - 需要与原论文保持兼容

---

## 附录：运行命令速查

### DT训练
```powershell
C:\Users\86177\anaconda3\envs\linc_env\python.exe "path\to\DT\train_dt.py" --key iscx --max_depth 8
```

### RF-NetBeacon训练
```powershell
C:\Users\86177\anaconda3\envs\linc_env\python.exe "path\to\RF\Netbeacon\train_rf.py" --key iscx --max_depth 5 --n_estimators 2 --num_classes 6
```

### RF-Planter训练
```powershell
C:\Users\86177\anaconda3\envs\linc_env\python.exe "path\to\RF\Planter\train_rf.py" --key iscx --max_depth 5 --n_estimators 2 --class_num_bits 3
```
