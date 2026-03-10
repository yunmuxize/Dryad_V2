# 快速开始指南

## 🚀 三种训练方式

### 方式1: 标准训练（推荐首次使用）
```bash
python dryad_predictor_enhanced.py
```
- 训练时间：约5-10分钟
- 适合：快速测试和验证

### 方式2: 扩展搜索（推荐最终训练）
```bash
python dryad_predictor_enhanced.py --extensive
```
- 训练时间：约30-60分钟
- 适合：获得最佳性能

### 方式3: 后台训练（推荐长时间训练）
```bash
# Windows: 使用批处理文件
start_training.bat --extensive

# 或者直接使用start命令
start /MIN python train_background.py --extensive
```

## 📊 监控训练进度

在另一个终端窗口运行：
```bash
python monitor_training.py
```

## 📁 输出文件

训练完成后，检查以下目录：

1. **logs/** - 训练日志
   - `training_YYYYMMDD_HHMMSS.log` - 完整训练日志
   - `training_history_YYYYMMDD_HHMMSS.json` - 训练历史数据

2. **plots/** - 可视化图表
   - `prediction_scatter.png` - 预测vs真实值
   - `feature_importance.png` - 特征重要性
   - `residuals.png` - 残差分析
   - `learning_curves.png` - 学习曲线

3. **models/** - 训练好的模型
   - `unified_model.pkl` - 统一模型
   - `scaler.pkl` - 特征缩放器
   - `feature_names.pkl` - 特征名称

## 🎯 性能改进

### CPU预测优化
- ✅ 增强特征工程（添加交互特征）
- ✅ 自定义评分函数（CPU权重60%）
- ✅ 优化的超参数搜索
- ✅ RobustScaler（对异常值更鲁棒）

### Memory预测优化
- ✅ 保持现有优势
- ✅ 通过扩展搜索进一步提升

## ⚡ 性能对比

| 指标 | 原版本 | 增强版（预期） |
|------|--------|---------------|
| CPU R² | -0.0343 | > 0.2 |
| Memory R² | 0.8056 | > 0.85 |
| 特征数量 | 14 | 16 |
| 可视化 | ❌ | ✅ |
| 日志记录 | ❌ | ✅ |

## 🔍 查看结果

### 查看日志
```bash
# Windows
type logs\training_*.log | more

# 或直接打开日志文件查看
```

### 查看训练历史
```python
import json
with open('logs/training_history_*.json', 'r') as f:
    history = json.load(f)
    print(f"CPU R²: {history['cpu_test_r2'][-1]:.4f}")
    print(f"Memory R²: {history['memory_test_r2'][-1]:.4f}")
```

### 查看可视化
直接打开 `plots/` 目录中的PNG文件

## 💡 提示

1. **首次训练**：使用标准模式快速验证
2. **最终训练**：使用扩展模式获得最佳性能
3. **长时间训练**：使用后台模式，并用监控脚本观察进度
4. **性能分析**：查看可视化图表了解模型表现

## 🐛 故障排除

### 问题：训练时间过长
- 使用标准模式而非扩展模式
- 减少数据量（修改test_size）

### 问题：CPU预测仍然较差
- 查看特征重要性图
- 检查数据质量
- 考虑添加更多特征

### 问题：监控无输出
- 确认训练脚本正在运行
- 检查logs目录是否存在
- 查看是否有错误信息

