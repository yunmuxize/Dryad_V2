# -*- coding: utf-8 -*-
"""测试Tofino预测器"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm.tofino_predictor import TofinoPredictor

# 测试预测器
print("Testing TofinoPredictor...")
predictor = TofinoPredictor()

# 测试预测
match_config = {
    'Total length': 2,      # range
    'Protocol': 1,          # ternary
    'IPV4 Flags (DF)': 1,   # ternary
    'Time to live': 2,      # range
    'Src Port': 2,          # range
    'Dst Port': 2,          # range
    'TCP flags (Reset)': 1, # ternary
    'TCP flags (Syn)': 1    # ternary
}

print("\nPredictions for different rule counts:")
print(f"{'Rules':>8} | {'TCAM%':>8} | {'SRAM%':>8} | {'Stages':>6}")
print("-" * 40)

for rules in [100, 500, 1000, 2000, 5000, 10000]:
    result = predictor.predict(match_config, rules)
    tcam = result['tcam_pct']
    sram = result['sram_pct']
    stages = result['stages']
    print(f"{rules:>8} | {tcam:>8.2f} | {sram:>8.2f} | {stages:>6}")

print("\nTest complete!")
