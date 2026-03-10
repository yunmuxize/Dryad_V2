import pandas as pd
import numpy as np
import os
from unified_predictor import UnifiedPredictor

def validate_tofino_samples(predictor, n_samples=5):
    """从 Tofino 数据集随机抽取样本进行验证"""
    print("=" * 80)
    print("  Tofino 数据集验证")
    print("=" * 80)
    
    data_path = os.path.join('tofino', 'data', 'merged_resource_usage.csv')
    df = pd.read_csv(data_path)
    
    # 随机抽取样本
    samples = df.sample(n_samples, random_state=np.random.randint(0, 10000))
    
    match_cols = [
        'match_total_len', 'match_protocol', 'match_flags_1', 'match_ttl',
        'match_src_port', 'match_dst_port', 'match_tcp_flags_2', 'match_tcp_flags_1'
    ]
    
    tcam_errors = []
    sram_errors = []
    
    for idx, row in samples.iterrows():
        match_types = row[match_cols].values.astype(int).tolist()
        size = int(row['size'])
        actual_tcam = float(str(row['tcam_percent']).replace('%', ''))
        actual_sram = float(str(row['sram_percent']).replace('%', ''))
        
        # 预测
        pred = predictor.predict(match_types, size)
        pred_tcam = pred['Tofino']['TCAM (%)']
        pred_sram = pred['Tofino']['SRAM (%)']
        
        # 计算误差
        tcam_error = pred_tcam - actual_tcam
        sram_error = pred_sram - actual_sram
        tcam_errors.append(abs(tcam_error))
        sram_errors.append(abs(sram_error))
        
        # 打印结果
        print(f"\n【样本 {idx}】")
        print(f"  Match Types: {match_types}")
        print(f"  Size: {size}")
        print(f"  TCAM: 实际={actual_tcam:.4f}%, 预测={pred_tcam:.4f}%, 误差={tcam_error:+.4f}%")
        print(f"  SRAM: 实际={actual_sram:.4f}%, 预测={pred_sram:.4f}%, 误差={sram_error:+.4f}%")
        
        # 误差评级
        tcam_rating = "[优秀]" if abs(tcam_error) < 0.5 else "[一般]" if abs(tcam_error) < 2.0 else "[较大]"
        sram_rating = "[优秀]" if abs(sram_error) < 0.01 else "[一般]" if abs(sram_error) < 0.05 else "[较大]"
        print(f"  评级: TCAM {tcam_rating}, SRAM {sram_rating}")
    
    # 统计
    print(f"\n{'─' * 80}")
    print(f"Tofino 验证统计 (n={n_samples}):")
    print(f"  TCAM MAE: {np.mean(tcam_errors):.4f}%")
    print(f"  TCAM 最大误差: {np.max(tcam_errors):.4f}%")
    print(f"  SRAM MAE: {np.mean(sram_errors):.6f}%")
    print(f"  SRAM 最大误差: {np.max(sram_errors):.6f}%")
    
    return tcam_errors, sram_errors

def validate_bmv2_samples(predictor, n_samples=5):
    """从 BMv2 数据集随机抽取样本进行验证"""
    print("\n" + "=" * 80)
    print("  BMv2 数据集验证")
    print("=" * 80)
    
    data_path = os.path.join('bmv2', 'ml_features.csv')
    df = pd.read_csv(data_path)
    
    # 随机抽取样本
    samples = df.sample(n_samples, random_state=np.random.randint(0, 10000))
    
    match_cols = [
        'total_len', 'protocol', 'flags', 'ttl',
        'src_port', 'dst_port', 'tcp_flags_2', 'tcp_flags_1'
    ]
    
    rss_errors = []
    rss_percent_errors = []
    
    for idx, row in samples.iterrows():
        match_types = row[match_cols].values.astype(int).tolist()
        size = int(row['size'])
        actual_rss_bytes = float(row['memory'])  # 单位：字节
        actual_rss_mb = actual_rss_bytes / 1024 / 1024
        
        # 预测
        pred = predictor.predict(match_types, size)
        pred_rss_mb = pred['BMv2']['Total RSS (MB)']
        
        # 计算误差
        rss_error_mb = pred_rss_mb - actual_rss_mb
        rss_error_percent = (rss_error_mb / actual_rss_mb) * 100 if actual_rss_mb > 0 else 0
        rss_errors.append(abs(rss_error_mb))
        rss_percent_errors.append(abs(rss_error_percent))
        
        # 打印结果
        print(f"\n【样本 {idx}】")
        print(f"  Match Types: {match_types}")
        print(f"  Size: {size}")
        print(f"  RSS: 实际={actual_rss_mb:.2f} MB, 预测={pred_rss_mb:.2f} MB")
        print(f"  误差: {rss_error_mb:+.2f} MB ({rss_error_percent:+.2f}%)")
        
        # 误差评级
        rating = "[优秀]" if abs(rss_error_percent) < 5 else "[一般]" if abs(rss_error_percent) < 15 else "[较大]"
        print(f"  评级: {rating}")
    
    # 统计
    print(f"\n{'─' * 80}")
    print(f"BMv2 验证统计 (n={n_samples}):")
    print(f"  RSS MAE: {np.mean(rss_errors):.2f} MB")
    print(f"  RSS 最大误差: {np.max(rss_errors):.2f} MB")
    print(f"  RSS 百分比 MAE: {np.mean(rss_percent_errors):.2f}%")
    print(f"  RSS 百分比最大误差: {np.max(rss_percent_errors):.2f}%")
    
    return rss_errors, rss_percent_errors

def validate_ebpf_logic(predictor, n_tests=3):
    """验证 eBPF 预测逻辑"""
    print("\n" + "=" * 80)
    print("  eBPF 预测逻辑验证")
    print("=" * 80)
    
    test_cases = [
        {
            'name': 'Ternary 匹配',
            'match': [1, 0, 0, 0, 0, 0, 0, 0],
            'size': 1000,
            'expected_map_memory': 58.0 * 1000,
            'expected_instructions': 1077,
            'expected_jit_size': 2919
        },
        {
            'name': 'LPM 匹配',
            'match': [3, 0, 0, 0, 0, 0, 0, 0],
            'size': 2000,
            'expected_map_memory': 0.0,
            'expected_instructions': 994,
            'expected_jit_size': 2453
        },
        {
            'name': 'Exact 匹配',
            'match': [0, 0, 0, 0, 0, 0, 0, 0],
            'size': 1500,
            'expected_map_memory': 13.0 * 1500,
            'expected_instructions': 992,
            'expected_jit_size': 2454
        }
    ]
    
    all_correct = True
    
    for i, test in enumerate(test_cases[:n_tests], 1):
        pred = predictor.predict(test['match'], test['size'])
        ebpf_pred = pred['eBPF']
        
        print(f"\n【测试 {i}: {test['name']}】")
        print(f"  Match Types: {test['match']}")
        print(f"  Size: {test['size']}")
        
        # 验证 Map Memory
        map_correct = ebpf_pred['Map Memory (Bytes)'] == test['expected_map_memory']
        print(f"  Map Memory: {ebpf_pred['Map Memory (Bytes)']} Bytes (期望: {test['expected_map_memory']}) {'[OK]' if map_correct else '[FAIL]'}")
        
        # 验证 Instructions
        inst_correct = ebpf_pred['Estimated Instructions'] == test['expected_instructions']
        print(f"  Instructions: {ebpf_pred['Estimated Instructions']} (期望: {test['expected_instructions']}) {'[OK]' if inst_correct else '[FAIL]'}")
        
        # 验证 JIT Size
        jit_correct = ebpf_pred['JIT Size (Bytes)'] == test['expected_jit_size']
        print(f"  JIT Size: {ebpf_pred['JIT Size (Bytes)']} Bytes (期望: {test['expected_jit_size']}) {'[OK]' if jit_correct else '[FAIL]'}")
        
        test_passed = map_correct and inst_correct and jit_correct
        all_correct = all_correct and test_passed
        print(f"  结果: {'[通过]' if test_passed else '[失败]'}")
    
    print(f"\n{'─' * 80}")
    print(f"eBPF 逻辑验证: {'[全部通过]' if all_correct else '[存在错误]'}")
    
    return all_correct

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("  Dryad 统一预测器 - 交叉验证")
    print("=" * 80)
    print("\n正在加载预测模型...")
    
    predictor = UnifiedPredictor()
    print("模型加载完成！\n")
    
    # 1. Tofino 验证
    tofino_tcam_errors, tofino_sram_errors = validate_tofino_samples(predictor, n_samples=8)
    
    # 2. BMv2 验证
    bmv2_errors, bmv2_percent_errors = validate_bmv2_samples(predictor, n_samples=8)
    
    # 3. eBPF 逻辑验证
    ebpf_correct = validate_ebpf_logic(predictor, n_tests=3)
    
    # 总结
    print("\n" + "=" * 80)
    print("  验证总结")
    print("=" * 80)
    print(f"\n【Tofino】")
    print(f"  TCAM MAE: {np.mean(tofino_tcam_errors):.4f}%")
    print(f"  SRAM MAE: {np.mean(tofino_sram_errors):.6f}%")
    print(f"  评级: {'[优秀]' if np.mean(tofino_tcam_errors) < 1.0 else '[一般]' if np.mean(tofino_tcam_errors) < 2.0 else '[需改进]'}")
    
    print(f"\n【BMv2】")
    print(f"  RSS MAE: {np.mean(bmv2_errors):.2f} MB")
    print(f"  RSS 百分比 MAE: {np.mean(bmv2_percent_errors):.2f}%")
    print(f"  评级: {'[优秀]' if np.mean(bmv2_percent_errors) < 10 else '[一般]' if np.mean(bmv2_percent_errors) < 20 else '[需改进]'}")
    
    print(f"\n【eBPF】")
    print(f"  逻辑验证: {'[通过]' if ebpf_correct else '[失败]'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
