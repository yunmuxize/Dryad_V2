# -*- coding:utf-8 -*-
"""
LPM vs Ternary 算法原理对比与验证
"""

def range_to_ternary(start, end, width):
    """标准Ternary分解"""
    if start > end: return []
    if start == 0 and end == (1 << width) - 1: return [('*' * width,)]
    if start == end: return [(format(start, f'0{width}b'),)]
    
    result = []
    if start % 2 == 1:
        result.append((format(start, f'0{width}b'),))
        start += 1
    if start <= end and end % 2 == 0:
        result.append((format(end, f'0{width}b'),))
        end -= 1
    if start <= end:
        sub = range_to_ternary(start >> 1, end >> 1, width - 1)
        for (pattern,) in sub:
            # 高位保持，低位通配
            result.append((pattern + '*',))
    return result

def range_to_lpm(start, end, width):
    """LPM分解 - 只能匹配 prefix/len 形式"""
    if start > end: return []
    if start == 0 and end == (1 << width) - 1: 
        return [('*' * width, 0)]
    
    result = []
    current = start
    
    while current <= end:
        # 找最大对齐的2^k块
        best_k = 0
        for k in range(width + 1):
            block_size = 1 << k
            # current必须对齐到block_size
            if current % block_size == 0:
                # 块必须完全在范围内
                if current + block_size - 1 <= end:
                    best_k = k
        
        block_size = 1 << best_k
        prefix_len = width - best_k
        prefix = current >> best_k
        
        pattern = format(prefix, f'0{prefix_len}b') + '*' * best_k if prefix_len > 0 else '*' * width
        result.append((pattern, prefix_len))
        current += block_size
    
    return result

def compare_single_range(start, end, width, name=""):
    """对比单个范围的分解结果"""
    ternary = range_to_ternary(start, end, width)
    lpm = range_to_lpm(start, end, width)
    
    print(f"\n范围 [{start}, {end}] ({width}-bit) {name}")
    print(f"  Ternary: {len(ternary)} 条")
    for t in ternary[:5]:
        print(f"    {t[0]}")
    if len(ternary) > 5:
        print(f"    ... 共 {len(ternary)} 条")
    
    print(f"  LPM: {len(lpm)} 条")
    for l in lpm[:5]:
        print(f"    {l[0]} (prefix_len={l[1]})")
    if len(lpm) > 5:
        print(f"    ... 共 {len(lpm)} 条")
    
    return len(ternary), len(lpm)

def main():
    print("=" * 70)
    print("LPM vs Ternary 算法原理对比")
    print("=" * 70)
    
    print("\n【核心原理】")
    print("Ternary: 可以在任意位置放置通配符 (*)")
    print("LPM: 只能在低位连续放置通配符 (prefix/*)")
    print("\n理论上，LPM 是 Ternary 的子集。")
    print("但对于决策树的阈值分割，两者产生的规则数往往相同。")
    
    print("\n" + "=" * 70)
    print("典型范围测试")
    print("=" * 70)
    
    # 测试各种典型范围
    test_cases = [
        (0, 255, 8, "全范围"),
        (0, 100, 8, "从0开始"),
        (100, 255, 8, "到最大值"),
        (50, 150, 8, "中间范围"),
        (0, 1389, 16, "Total length典型值"),
        (256, 65535, 16, "Src Port典型值"),
        (0, 610, 16, "Dst Port典型值"),
    ]
    
    total_ternary = 0
    total_lpm = 0
    
    for start, end, width, name in test_cases:
        t, l = compare_single_range(start, end, width, name)
        total_ternary += t
        total_lpm += l
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"测试范围总计: Ternary {total_ternary} 条, LPM {total_lpm} 条")
    
    if total_ternary == total_lpm:
        print("\n✓ LPM 和 Ternary 产生相同数量的规则")
        print("\n原因分析:")
        print("1. 决策树的阈值分割产生的范围通常是 [0, threshold] 或 [threshold+1, max]")
        print("2. 这类范围的 Ternary 分解和 LPM 分解算法本质相同")
        print("3. 两者都是将范围分解为 2^k 对齐的块")
        print("4. 差异只在于表示方式:")
        print("   - Ternary: (value, mask)")
        print("   - LPM: (prefix, prefix_length)")
    else:
        print(f"\n△ LPM 和 Ternary 规则数不同: 差值 = {total_lpm - total_ternary}")

if __name__ == "__main__":
    main()
