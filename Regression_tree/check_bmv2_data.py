import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('bmv2/data/processed_features.csv')

print("=" * 80)
print("  BMv2 数据集分析")
print("=" * 80)

print(f"\nMemory (KB) 统计:")
print(f"  最小值: {df['memory'].min():.2f} KB")
print(f"  最大值: {df['memory'].max():.2f} KB")
print(f"  平均值: {df['memory'].mean():.2f} KB")
print(f"  中位数: {df['memory'].median():.2f} KB")

print(f"\nSize 统计:")
print(f"  最小值: {df['size'].min()}")
print(f"  最大值: {df['size'].max()}")

print(f"\nSize=378 的样本:")
subset = df[df['size'] == 378]
if len(subset) > 0:
    print(subset[['size', 'memory']].to_string())
else:
    print("  没有 size=378 的样本")
    print(f"\n最接近 378 的样本:")
    df['diff'] = abs(df['size'] - 378)
    closest = df.nsmallest(5, 'diff')
    print(closest[['size', 'memory']].to_string())
