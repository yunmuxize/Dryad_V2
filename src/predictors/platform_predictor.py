# -*- coding: utf-8 -*-
"""
平台资源预测器 - 封装 Regression_tree 的预测模型
用于预测决策树在不同平台上的资源消耗（TCAM, SRAM, Stages）
已适配 V3Final 21维特征模型
"""

import os
import numpy as np
from unified_predictor import UnifiedPredictor

class TofinoPlatformPredictor:
    """
    Tofino平台资源预测器
    
    输入：
        - feature_match_types: Dryad匹配类型编码 [RANGE, PREFIX, TERNARY, EXACT]
        - entries_count: 实际P4 entries数
        
    输出：
        - tcam_percent: TCAM使用率 (%)
        - sram_percent: SRAM使用率 (%)
        - stages: 使用的stage数 (机器学习模型预测)
    """
    
    # Dryad匹配类型编码 (已更新为：Exact=0, Ternary=1, Range=2, LPM=3)
    DRYAD_EXACT = 0
    DRYAD_TERNARY = 1
    DRYAD_RANGE = 2
    DRYAD_PREFIX = 3    # LPM
    
    # 预测器匹配类型编码 (0=Exact, 1=Ternary, 2=Range, 3=LPM)
    PREDICTOR_EXACT = 0
    PREDICTOR_TERNARY = 1
    PREDICTOR_RANGE = 2
    PREDICTOR_LPM = 3
    
    def __init__(self):
        """初始化预测器"""
        try:
            self.predictor = UnifiedPredictor()
            self.available = True
            print("[OK] Tofino V3Final 预测器加载成功")
        except Exception as e:
            print(f"[ERROR] 预测器加载失败: {e}")
            self.available = False
            raise e
            
    def _convert_match_types(self, dryad_match_types):
        """
        转换Dryad编码 → 预测器编码
        
        现在 Dryad 编码已与预测器编码统一为:
        Exact=0, Ternary=1, Range=2, LPM=3
        故直接返回即可。
        """
        return dryad_match_types
    
    def predict(self, feature_match_types, entries_count):
        """
        预测资源消耗（带非负约束）
        """
        if not self.available:
            raise RuntimeError("TofinoPredictor is not available.")
        
        # 转换匹配类型编码
        predictor_match_types = self._convert_match_types(feature_match_types)
        
        # 调用 unified_predictor
        # 内部会自动根据 21 维特征进行预测
        result = self.predictor.predict(
            match_types=predictor_match_types,
            size=entries_count
        )
        
        tofino_result = result.get('Tofino', {})
        
        # 应用非负约束，防止预测值为负
        tcam_percent = max(0.0, tofino_result.get('TCAM (%)', 0.0))
        sram_percent = max(0.0, tofino_result.get('SRAM (%)', 0.0))
        tcam_tiles = max(0.0, tofino_result.get('TCAM (Tiles)', 0.0))
        sram_tiles = max(0.0, tofino_result.get('SRAM (Tiles)', 0.0))
        stages = max(1, int(tofino_result.get('Stages', 1)))  # Stages最小为1
        
        return {
            'tcam_percent': tcam_percent,
            'sram_percent': sram_percent,
            'tcam_tiles': tcam_tiles,
            'sram_tiles': sram_tiles,
            'stages': stages,
            'is_valid': True
        }

class PlatformConstraints:
    """平台约束验证器"""
    
    @staticmethod
    def validate_tofino(feature_match_types, feature_list):
        """
        验证Tofino平台约束 (V4 标准)
        """
        # 1-bit特征列表 (标准化名称)
        one_bit_features = ['flags[1:1]', 'tcp_flags[2:2]', 'tcp_flags[1:1]']
        
        # 约束1: 1-bit特征不能用RANGE (编码 2)
        for i, feature in enumerate(feature_list):
            if feature in one_bit_features:
                if feature_match_types[i] == 2: # MatchType.RANGE
                    return False
        
        # 约束2: LPM (编码 3) 最多一次
        lpm_count = sum(1 for mt in feature_match_types if mt == 3)
        if lpm_count > 1:
            return False
        
        return True
    
    @staticmethod
    def get_platform_config(platform='tofino'):
        """获取平台资源配置"""
        configs = {
            'tofino': {
                'max_stages': 12,
                'tcam_limit': 80.0,
                'sram_limit': 80.0
            }
        }
        return configs.get(platform, configs['tofino'])

if __name__ == "__main__":
    # 测试代码 (V4 21维特征模型验证)
    predictor = TofinoPlatformPredictor()
    
    # 用户指定的两个样本进行验证
    # Sample 1: sample_01968 (Large Table, Multi-Range)
    # Range, Range, Ternary, Ternary, Range, Range, LPM, Exact
    # Encodings: [2, 2, 1, 1, 2, 2, 3, 0]
    cases = [
        {
            'name': 'Sample_01968 (Large Table)',
            'match': [2, 2, 1, 1, 2, 2, 3, 0],
            'size': 16474,
            'truth': {'TCAM': 324, 'SRAM': 14, 'Stages': 14}
        },
        {
            'name': 'Sample_01999 (Hybrid Mix)',
            'match': [0, 1, 1, 0, 0, 1, 3, 1],
            'size': 11125,
            'truth': {'TCAM': 44, 'SRAM': 2, 'Stages': 2}
        }
    ]
    
    print("\n" + "=" * 60)
    print("  Tofino V4 模型精度验证")
    print("=" * 60)
    
    for case in cases:
        print(f"\nCase: {case['name']} (Size={case['size']})")
        res = predictor.predict(case['match'], case['size'])
        
        t = case['truth']
        print(f"  预测结果 -> TCAM: {res['tcam_tiles']:.2f} Tiles, SRAM: {res['sram_tiles']:.2f} Tiles, Stages: {res['stages']}")
        print(f"  真实数据 -> TCAM: {t['TCAM']} Tiles, SRAM: {t['SRAM']} Tiles, Stages: {t['Stages']}")
        
        err_tcam = res['tcam_tiles'] - t['TCAM']
        err_sram = res['sram_tiles'] - t['SRAM']
        err_stages = res['stages'] - t['Stages']
        
        print(f"  预测误差 -> TCAM: {err_tcam:+.2f}, SRAM: {err_sram:+.2f}, Stages: {err_stages:+d}")
        print("-" * 40)
    
    print("\n[OK] 验证程序执行完毕")
