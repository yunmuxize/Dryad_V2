# GPU加速训练指南

## 概述

LightGBM支持GPU加速训练，可以显著提升训练速度，特别是在处理大量参数组合时。

## 启用GPU训练

### 方法1: 命令行参数

```bash
# 标准训练 + GPU
python dryad_predictor_enhanced.py --gpu

# 扩展搜索 + GPU
python dryad_predictor_enhanced.py --extensive --gpu

# 优化训练脚本 + GPU
python scripts/train_optimized.py --gpu
```

### 方法2: 代码中启用

```python
from dryad_predictor_enhanced import DryadPredictorEnhanced

# 创建预测器时启用GPU
predictor = DryadPredictorEnhanced(use_gpu=True)

# 后续训练流程相同
predictor.load_and_process_data()
predictor.split_data()
predictor.train_models(extensive_search=True)
```

## GPU要求

1. **安装GPU版本的LightGBM**:
   ```bash
   # 如果已安装CPU版本，需要先卸载
   pip uninstall lightgbm
   
   # 安装GPU版本（需要CUDA支持）
   pip install lightgbm --install-option=--gpu
   ```

2. **CUDA环境**:
   - 需要安装NVIDIA CUDA Toolkit
   - 需要支持CUDA的GPU
   - 确保CUDA版本与LightGBM兼容

3. **验证GPU支持**:
   ```python
   import lightgbm as lgb
   print(lgb.__version__)
   # 尝试创建模型
   model = lgb.LGBMRegressor(device='gpu')
   ```

## 性能提升

根据参数组合数分析：

- **标准模式**: 2,268 组合 × 5折 = 11,340 次训练
- **扩展模式**: 138,240 组合 × 5折 = 691,200 次训练

GPU加速通常可以带来：
- **2-5倍**的训练速度提升（取决于GPU性能）
- 特别是在扩展模式下，GPU的优势更明显

## 注意事项

1. **GridSearchCV并行**: GPU模式下，`n_jobs=-1`会自动设置为`n_jobs=1`，因为GPU本身是并行计算的

2. **内存使用**: GPU训练会占用GPU显存，确保有足够的显存

3. **回退机制**: 如果GPU不可用，代码会自动回退到CPU模式

4. **混合使用**: 不建议同时使用CPU多进程和GPU，可能导致资源竞争

## 故障排除

### GPU不可用

如果遇到GPU相关错误，可以：
1. 检查CUDA是否正确安装
2. 检查GPU驱动是否最新
3. 回退到CPU模式：移除`--gpu`参数

### 显存不足

如果遇到显存不足：
1. 减少`n_estimators`参数范围
2. 减少`num_leaves`参数范围
3. 使用CPU模式

## 性能对比

| 模式 | 参数组合数 | CPU训练时间 | GPU训练时间（估算） | 加速比 |
|------|-----------|------------|-------------------|--------|
| 标准模式 | 2,268 | ~3-5分钟 | ~1-2分钟 | 2-3x |
| 扩展模式 | 138,240 | ~20-30小时 | ~5-10小时 | 3-4x |

*注：实际时间取决于硬件配置和数据集大小*

