# Tofino P4 决策树编译器

本目录包含将机器学习决策树模型转换为 Tofino P4 规则的完整工具链。

## 目录结构

```
tofino/
├── data/                          # 模型数据目录
│   ├── iscx_depth_5_model.json    # 深度5决策树模型
│   ├── iscx_depth_7_model.json    # 深度7决策树模型
│   ├── iscx_depth_16_model.json   # 深度16决策树模型
│   ├── iscx_depth_26_model.json   # 深度26决策树模型
│   └── iscx_validation_samples_100.json  # 验证样本集
│
├── docs/                          # 技术文档
│   └── 标志位数据分析报告.md      # P4匹配方式量化分析报告
│
├── generated/                     # 生成的P4代码和规则
│   ├── iscx_depth_5/              # 深度5 (Range+Ternary) - 26条规则
│   ├── iscx_depth_7/              # 深度7 - 66条规则
│   ├── iscx_depth_16/             # 深度16 - 890条规则
│   ├── iscx_depth_26/             # 深度26 - 1542条规则
│   └── iscx_depth_5_all_ternary/  # 深度5 全Ternary - 5104条规则
│
├── scripts/                       # 脚本工具集
│   ├── generators/                # P4规则生成器
│   │   ├── generate_genetic_edt.py    # 主生成器 (Range+Ternary)
│   │   └── generate_all_ternary.py    # 全Ternary生成器
│   ├── validators/                # 验证脚本
│   │   ├── verify_rules.py            # 规则路径一致性验证
│   │   ├── verify_leaf_vs_rules.py    # 叶节点vs规则数对比
│   │   ├── verify_model_accuracy.py   # 单模型精度验证
│   │   └── verify_multi_depth_accuracy.py  # 多深度精度验证
│   ├── analysis/                  # 分析脚本
│   │   ├── calc_ternary_expansion.py  # Ternary展开计算器
│   │   ├── check_dead_paths.py        # 死路径检测
│   │   ├── prove_dead_path.py         # 死路径证明
│   │   └── search_optimal_no_range.py # 无Range最优搜索
│   ├── exporters/                 # 模型导出脚本
│   │   ├── export_model_and_samples.py    # 模型与样本导出
│   │   └── export_multi_depth_models.py   # 多深度模型批量导出
│   └── utils/                     # 工具脚本
│       └── json2pcap.py               # JSON样本转PCAP
│
└── README.md                      # 本文档
```

## 规则展开算法原理

### 1. 决策树到P4规则的转换流程

```
决策树 JSON 模型
       ↓
   DFS 遍历树
       ↓
  提取所有叶节点路径
       ↓
  聚合每条路径的特征约束
       ↓
  根据匹配类型展开规则
       ↓
  生成 BFRT 命令
```

### 2. 匹配类型与规则展开

P4 支持四种匹配类型，各有优劣：

| 匹配类型 | 硬件资源 | 适用场景 | 展开系数 |
|---------|---------|---------|---------|
| **exact** | SRAM | 精确值匹配 | 需枚举所有值 |
| **ternary** | TCAM | 任意位模式匹配 | log₂(范围) |
| **lpm** | TCAM | 前缀匹配 | 2×log₂(范围) |
| **range** | TCAM | 连续范围匹配 | 1 (无展开) |

### 3. Range-to-Ternary 分解算法

当不能使用 `range` 匹配时，需要将 `[start, end]` 范围分解为多条 ternary 规则：

```python
def range_to_ternary_decomposition(start, end, width):
    """
    将 [start, end] 范围分解为 (value, mask) 对列表
    
    算法思想：
    1. 如果是全范围 [0, 2^width-1]，返回单条通配规则 (0, 0)
    2. 如果是单值，返回精确匹配规则 (value, all_ones)
    3. 否则，递归分解：
       - 如果 start 是奇数，单独处理 start
       - 如果 end 是偶数，单独处理 end
       - 对中间部分右移1位递归处理
    """
```

**示例**：将 `[3, 12]` 分解为 ternary 规则 (4-bit):
```
[3, 12] → {3, 4-7, 8-11, 12}
        → (0011, 1111), (01**, ****), (10**, ****), (1100, 1111)
        → 4条ternary规则
```

### 4. 深度5模型配置对比

| 配置方案 | 匹配类型组合 | 规则数 | TCAM使用 | 适用场景 |
|---------|-------------|--------|---------|---------|
| **Range+Ternary** | 4×range + 4×ternary | **26** | ~5% | 默认推荐 |
| **全Ternary** | 8×ternary | **5104** | 高 | 禁用range时 |

### 5. 死路径过滤机制

决策树在 sklearn 中使用浮点阈值（如 `Syn > 1.0`），转换为整数后可能产生矛盾路径：

```
条件: TCP flags (Syn) > 1.0
硬件: TCP flags (Syn) >= 2
位宽: 1-bit (最大值为1)
结果: 区间 [2, 1] 无效 → 规则被自动过滤
```

| 深度 | 叶节点数 | 有效路径 | 死路径 |
|-----|---------|---------|--------|
| 5 | 26 | 26 | 0 |
| 7 | 71 | 66 | 5 |
| 16 | 1014 | 890 | 124 |
| 26 | 1795 | 1542 | 253 |

## 硬件资源约束

### Tofino 资源限制
- **TCAM**: 5% (40.55KB)
- **SRAM**: 1% (153.6KB)
- **Stages**: 1

### 推荐配置 (禁用Range时)
在无法使用 `range` 匹配的约束下，**全 Ternary** 是唯一可行的最优解：

```p4
table EDT {
    key = {
        hdr.ipv4.total_len   : ternary;
        hdr.ipv4.protocol    : ternary;
        hdr.ipv4.flags[1:1]  : ternary;
        hdr.ipv4.ttl         : ternary;
        meta.src_port        : ternary;
        meta.dst_port        : ternary;
        meta.tcp_flags[2:2]  : ternary;
        meta.tcp_flags[1:1]  : ternary;
    }
    actions = {SetClass;}
    size = 5104;  // 深度5模型展开后的实际规则数
    default_action = SetClass(1);
}
```

**理论计算**：
- 26条路径 × 平均196倍展开 ≈ 5104条规则
- 主要展开来源：`Src Port` (最大19条)、`Total length` (最大13条)、`Dst Port` (最大11条)

## 使用指南

### 1. 生成 P4 规则 (Range+Ternary)
```powershell
cd tofino/scripts/generators
python generate_genetic_edt.py
```

### 2. 生成全 Ternary 规则
```powershell
cd tofino/scripts/generators
python generate_all_ternary.py
```

### 3. 验证规则一致性
```powershell
cd tofino/scripts/validators
python verify_rules.py
```

### 4. 分析规则展开
```powershell
cd tofino/scripts/analysis
python calc_ternary_expansion.py
```

## 精度验证结果

| 模型深度 | 准确率 | Macro F1 | 规则数 |
|---------|--------|----------|--------|
| Depth 5 | 96%+ | 0.86+ | 26 |
| Depth 7 | 96% | 0.87 | 66 |
| Depth 16 | 98% | 0.90 | 890 |
| Depth 26 | 98% | 0.90 | 1542 |

## 技术文档

详细的匹配方式量化分析请参阅：[标志位数据分析报告](docs/标志位数据分析报告.md)
