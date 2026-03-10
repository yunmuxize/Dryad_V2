# -*- coding: utf-8 -*-
"""
标准模式 + GPU训练脚本
使用标准参数搜索（非扩展模式），启用GPU加速
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dryad_predictor_enhanced import DryadPredictorEnhanced

def main():
    """标准模式 + GPU训练主函数"""
    print("=" * 70)
    print("标准模式训练 + GPU加速")
    print("=" * 70)
    print("模式: 标准搜索（快速）")
    print("GPU: 已启用")
    print("预计训练时间: 1-3分钟（使用GPU）")
    print("=" * 70)
    
    try:
        # 创建预测器（启用GPU，不使用扩展搜索）
        predictor = DryadPredictorEnhanced(use_gpu=True)
        
        # 加载和处理数据
        print("\n[1/4] 加载和处理数据...")
        predictor.load_and_process_data()
        
        # 分割数据
        print("\n[2/4] 分割数据...")
        predictor.split_data()
        
        # 训练模型（标准模式，不使用extensive_search）
        print("\n[3/4] 开始训练模型（标准模式 + GPU加速）...")
        print("      - Memory模型：标准参数搜索")
        print("      - CPU模型：标准参数搜索")
        print("      - GPU加速：已启用")
        
        predictor.train_models(extensive_search=False)  # 标准模式
        
        print("\n[4/4] 训练完成！")
        
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
                print(f"  测试集 R²:  {memory_r2:.4f}")
                print(f"  测试集 MAE: {memory_mae:.4f}")
            
            if history.get('cpu_test_accuracy') and len(history['cpu_test_accuracy']) > 0:
                cpu_acc = history['cpu_test_accuracy'][-1]
                cpu_f1 = history['cpu_test_f1'][-1]
                print(f"\n[CPU分类模型]")
                print(f"  测试集准确率: {cpu_acc:.4f}")
                print(f"  测试集 F1:    {cpu_f1:.4f}")
            
            print("\n" + "=" * 70)
        
        print("\n训练完成！所有结果已保存。")
        print("- 模型文件: models/")
        print("- 日志文件: logs/")
        print("- 可视化图表: plots/")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




