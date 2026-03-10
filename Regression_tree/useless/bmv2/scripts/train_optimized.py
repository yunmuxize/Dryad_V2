# -*- coding: utf-8 -*-
"""
优化的训练脚本 - 使用extensive_search模式，目标Memory R²达到90%
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dryad_predictor_enhanced import DryadPredictorEnhanced

def main(use_gpu=False):
    """优化的训练主函数"""
    print("=" * 70)
    print("优化训练 - 目标: Memory R² >= 0.90, CPU准确率提升")
    if use_gpu:
        print("GPU加速: 已启用")
    print("=" * 70)
    
    try:
        # 创建预测器
        predictor = DryadPredictorEnhanced(use_gpu=use_gpu)
        
        # 加载和处理数据
        print("\n[1/5] 加载和处理数据...")
        predictor.load_and_process_data()
        
        # 分割数据
        print("\n[2/5] 分割数据...")
        predictor.split_data()
        
        # 训练模型（使用extensive_search）
        print("\n[3/5] 开始训练模型（扩展超参数搜索）...")
        print("      - Memory模型：精细参数搜索，目标R² >= 0.90")
        print("      - CPU模型：优化分类准确率")
        print("      这可能需要较长时间，请耐心等待...")
        
        predictor.train_models(extensive_search=True)
        
        print("\n[4/5] 训练完成！")
        print("\n[5/5] 生成最终报告...")
        
        # 读取训练历史
        import json
        import glob
        history_files = glob.glob("logs/training_history_*.json")
        if history_files:
            latest_history = max(history_files, key=os.path.getmtime)
            with open(latest_history, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            print("\n" + "=" * 70)
            print("最终预测精度")
            print("=" * 70)
            
            if history.get('memory_test_r2') and len(history['memory_test_r2']) > 0:
                memory_r2 = history['memory_test_r2'][-1]
                memory_mae = history['memory_test_mae'][-1]
                print(f"\n[Memory回归模型]")
                print(f"  测试集 R²:  {memory_r2:.4f} {'✓ 达到目标!' if memory_r2 >= 0.90 else '⚠ 未达目标'}")
                print(f"  测试集 MAE: {memory_mae:.4f}")
            
            if history.get('cpu_test_accuracy') and len(history['cpu_test_accuracy']) > 0:
                cpu_acc = history['cpu_test_accuracy'][-1]
                cpu_f1 = history['cpu_test_f1'][-1]
                print(f"\n[CPU分类模型]")
                print(f"  测试集准确率: {cpu_acc:.4f}")
                print(f"  测试集 F1:    {cpu_f1:.4f}")
            
            print("\n" + "=" * 70)
        
        print("\n训练完成！所有结果已保存。")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    use_gpu = '--gpu' in sys.argv or '--use-gpu' in sys.argv
    success = main(use_gpu=use_gpu)
    sys.exit(0 if success else 1)

