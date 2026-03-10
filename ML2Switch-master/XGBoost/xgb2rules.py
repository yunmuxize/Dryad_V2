# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : xgb2rules.py
# Time       ：2023-05-06 15:32
# Author     ：Haolin Yan
# Description：
"""
import numpy as np
import re
from itertools import product
import math
import time
from tqdm import tqdm
import fileinput
from shutil import copyfile


def add_to_template(outname, placeholder, code):
    with fileinput.FileInput(outname, inplace=True) as file:
        for line in file:
            print(line.replace(placeholder, code), end='')


def split_codes(code, used_features, add=""):
    code = code[2:]  # 删除"0b"
    idx = 0
    content = ""
    for fea in used_features:
        fea_n = fea[0].lower()
        content += "codes_" + fea_n + add + "=" + "0b" + code[idx:idx + fea[1]] + ","
        idx = idx + fea[1]
    return content


def get_xgb_feature_thres(model_file, keys):
    with open(model_file, 'r') as f:
        lines = f.readlines()
    feat_dict = {}
    for key in keys:
        feat_dict[key] = []
    for line in lines:
        if ":[" in line:
            m = re.search(r".*\[(.*?)<(.*?)\].*", line.strip(), re.M | re.I)
            feat_dict[m.group(1)].append(float(m.group(2)))
    for key in feat_dict.keys():
        for i in range(len(feat_dict[key])):
            feat_dict[key][i] = math.ceil(feat_dict[key][i])  # 向上取整 xgboost节点是f<a
        feat_dict[key] = list(np.unique(np.array(feat_dict[key])))

    return feat_dict


# 得到模型表表项
# model_file 模型文件，feat_dict 各个特征的阈值，key_encode_bits range mark占的位数，pkts第几个包,可选
# 返回值为列表，每个元素代表一条表项，内容为各个特征的range mark和分类结果
def get_xgb_trees_table_entries(model_file, keys, feat_dict, key_encode_bits, pkts=None, num_classes=2):  # 遍历一次
    with open(model_file, 'r') as f:
        lines = f.readlines()
    tree_data = []
    tree_leaves = []  # 每行是一个叶子节点，记录该叶子节点路径上各个特征中从左往右(<a的类型)中a最小（最靠左），从右往左（>b，负数表示）中b最小（最靠右）
    trees = []
    leafs = []
    for line in lines:
        if "booster" in line:  # 新的树开始的行，即下面的行是新树叶子节点的信息
            trees.append(len(tree_leaves))
            nodes = {}
            nodes[str(0)] = [1000, 0] * len(keys)  # 假设不会超过1000个阈值
        if "yes" in line:
            m = re.search(r"(.*?):\[(.*?)<(.*?)\] yes=(.*?),no=(.*?),.*", line.strip(), re.M | re.I)
            feat = m.group(2)
            thre = math.ceil(float(m.group(3)))  # 向上取整
            nodes[m.group(4)] = nodes[m.group(1)].copy()
            nodes[m.group(4)][keys.index(feat) * 2] = min(nodes[m.group(4)][keys.index(feat) * 2],  # 路径上多个节点用一个特征，取最小集合
                                                          feat_dict[feat].index(thre) + 1)
            nodes[m.group(5)] = nodes[m.group(1)].copy()
            nodes[m.group(5)][keys.index(feat) * 2 + 1] = min(nodes[m.group(5)][keys.index(feat) * 2 + 1],
                                                              -feat_dict[feat].index(thre) - 1)
        if "leaf" in line:
            m = re.search(r"(.*?):leaf=(.*?)\n", line.strip('\t'), re.M | re.I)
            tree_leaves.append(nodes[m.group(1)])  # 叶子编号
            leafs.append(float(m.group(2)))  # 叶子节点信息

    trees.append(len(tree_leaves))
    print("tree_leaves: ", trees)
    print("judge leaf conflict ...")
    loop_val = []
    for i in range(len(trees))[:-1]:
        loop_val.append(range(trees[i], trees[i + 1]))
    print(loop_val, len(loop_val))
    for tup in tqdm(product(*loop_val)):
        flag = 0
        for f in range(len(keys)):  # 检查是否有特征取值冲突
            a = 1000;
            b = 1000;  # 假设不会超过1000个阈值
            for i in tup:
                a = min(tree_leaves[i][f * 2], a)
                b = min(tree_leaves[i][f * 2 + 1], b)
            if a + b <= 0:
                flag = 1
                break

        # 这里可以增加语义冲突的判断
        if flag == 0:
            if pkts is None:
                tree_data.append([])  #
            else:
                tree_data.append([pkts])  # 数据包个数
            for f in range(len(keys)):
                a = 1000;
                b = 1000;
                for i in tup:  # 合并多个路径信息
                    a = min(tree_leaves[i][f * 2], a)
                    b = min(tree_leaves[i][f * 2 + 1], b)
                key = keys[f]
                te = get_model_table_range_mark(key_encode_bits[key], a, b, len(feat_dict[key]))
                tree_data[-1].extend([int(get_value_mask(te, key_encode_bits[key])[0], 2),
                                      int(get_value_mask(te, key_encode_bits[key])[1], 2)])  # 每个特征的值和mask
            leaf_sum = 0.0
            for i in tup:
                leaf_sum += leafs[i]
            tree_data[-1].append(round(sigmoid(leaf_sum) * 100))  # 分类概率
    return tree_data


def find_next_split(minz, maxz):
    count = 0
    while (minz >> count) & 1 == 0 and (minz + (1 << count)) < maxz:
        count += 1
    if (minz + (1 << count)) > maxz:
        return 1 << (count - 1)
    return 1 << count


# 前缀方式Prefix
# 返回值 range_to_tenary(0,48) return ([0, 32], [32, 16]),表示从值0开始，长32，下一段从值32开始，长16
def range_to_tenary(minz, maxz):  # [minz,maxz)
    if maxz <= minz:
        return [[], []]
    start_num = []
    bcount = []
    while True:
        a = find_next_split(minz, maxz)
        start_num.append(minz)
        bcount.append(a)
        if minz + a == maxz:
            break
        minz += a
    return start_num, bcount


# 根据位数和长度，得到mask
# length 是特征位数，即mask位数，num是mask覆盖的数目,即range_to_tenary返回的bcount
def get_mask(length, num):
    a = int(np.log2(num))
    result = '0b'
    for i in range(length - a):
        result += '1'
    for i in range(a):  # mask位置为0
        result += '0'
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_value_mask(s, bits):  # 将1**00转变为10000和mask 10011,bits是位数
    value = s.copy()
    mask = ['1'] * bits
    for i in range(len(s)):
        if s[i] == '*':
            value[i] = '0'  # 把*替代，随便0，1都行
            mask[i] = '0'  # 数据面mask中 0代表任意
    return ''.join(value), ''.join(mask)


# 得到feature table中基区间的range mark
# length是range mark需要的位数，valid是当前特征阈值个数，num是基区间是第几个，
def get_feature_table_range_mark(length, num, valid):
    result = '0b'
    for i in range(num):
        result += '0'
    for i in range(valid - num):
        result += '1'
    for i in range(length - valid):  # 无效的位默认为0
        result += '0'
    return result


# 得到model table中相连区间的range mark
# length是range mark位数，valid是当前特征阈值个数，a,b分别代表相连区间的左端和右端，
def get_model_table_range_mark(length, a, b, valid):
    result = ['0'] * length  # 初始化为0
    for n in range(valid):
        if n < abs(b):
            result[n] = '0'
        elif n >= a - 1:
            result[n] = '1'
        else:
            result[n] = '*'
    return result


trans = lambda x: list(map(float, x.strip('[').strip(']').split(',')))


def list_to_proba(ll):  # 将rf叶子节点中的value 变成概率
    ll = trans(ll)
    re = []
    for i in ll:
        re.append(i / np.sum(ll))
    return re


# 得到feature table表项，包括key和action的参数
# feat_dict 各个特征的阈值，key_bits特征占的位数，key_encode_bits range mark占的位数，pkts第几个包,可选，
# 返回字典，key为特征，value为列表，列表中每一项是对应特征表中一条表项，包括：[优先级, value, mask, action参数,（包数）]
def get_feature_table_entries(feat_dict, key_bits, key_encode_bits, pkts=None):
    feat_table_datas = {}
    for key in feat_dict.keys():
        thres = feat_dict[key]
        if "avg" in key and pkts & (pkts - 1) != 0:  # pkts 不是2的幂次时
            print("avg in key, pkts ", pkts & (pkts - 1))
            thres = list(np.array(thres) * pkts)
        feat_table = []
        sum1 = 0
        priority = 1
        for i in range(len(thres)):
            best_start = 0
            best_end = 0
            min_entries = 100000
            end = thres[i]
            start_time = time.time()

            while (i != len(thres) - 1 and end < thres[i + 1]) or (i == len(thres) - 1 and end < (2 ** key_bits[key])):
                right_entries = len(range_to_tenary(thres[i], end)[0])
                start = thres[i - 1] if i > 0 else 0
                while start >= 0:
                    temp = range_to_tenary(start, end)
                    now_entries = len(temp[0]) + right_entries
                    if now_entries < min_entries:
                        min_entries = now_entries
                        best_start = start
                        best_end = end
                    if len(temp[0]) > 1 and temp[1][0] - temp[1][1] < 0:
                        start += temp[1][0] - temp[1][1]
                    else:
                        break
                # if time.time()-start_time>1: #控制时间如果有必要
                #     break
                if len(temp[0]) > 1 and temp[1][-1] - temp[1][-2] < 0:
                    end += temp[1][-2] - temp[1][-1]
                else:
                    break
            # print(min_entries,thres[i-1],thres[i],best_start,best_end)
            temp = range_to_tenary(thres[i], best_end)
            for j in range(len(temp[0])):
                feat_table.append([priority, temp[0][j], int(get_mask(key_bits[key], temp[1][j]), 2),
                                   int(get_feature_table_range_mark(key_encode_bits[key], i + 1, len(thres)), 2)])
                if pkts is not None:
                    feat_table[-1].append(pkts)
            priority += 1
            temp = range_to_tenary(best_start, best_end)
            for j in range(len(temp[0])):
                feat_table.append([priority, temp[0][j], int(get_mask(key_bits[key], temp[1][j]), 2),
                                   int(get_feature_table_range_mark(key_encode_bits[key], i, len(thres)), 2)])
                if pkts is not None:
                    feat_table[-1].append(pkts)
            priority += 1

            sum1 += min_entries
        print("The entries of {} is {}.".format(key, len(feat_table)))
        feat_table_datas[key] = feat_table
    return feat_table_datas


def get_bin_table(keys, bin_count_bits, QL=4):  # bin_count_bits是bin count变量位数
    bin_table_data = []
    mask_value = (((2 ** bin_count_bits) - 1) >> QL) << QL
    for key in keys:
        if "bin" in key:
            start_value = int(key[4:]) * 2 ** QL
            bin_table_data.append([start_value, mask_value])
    return bin_table_data


def export_xgb(model_name, n_fea):
    source_file = "tmpl.p4"
    destination_file = "xgb.p4"
    setup_file = "xgb_setup.py"
    copyfile(source_file, destination_file)

    num_rules = 0
    pkt_feat = ["f%d" % i for i in range(n_fea)]
    ############################ 自定义内容 #############################
    # 特征占用的bits
    pkt_feat_bits = [16, 16, 8, 4, 8, 8, 4, 16, 16, 16]
    # 参考tmpl.p4&../P4/headers.p4, 修改特征对应的PHV字段
    header_names = {'f0': ['meta.srcPort', 16],
                    'f1': ['meta.dstPort', 16],
                    'f2': ['hdr.ipv4.protocol', 8],
                    'f3': ['hdr.ipv4.ihl', 4],
                    'f4': ['hdr.ipv4.tos', 8],
                    'f5': ['hdr.ipv4.ttl', 8],
                    'f6': ['meta.dataOffset', 4],
                    'f7': ['meta.window', 16],
                    'f8': ['meta.udp_length', 16],
                    'f9': ['hdr.ipv4.totalLen', 16]}

    ##################################################################

    key_bits = {}
    for i in range(len(pkt_feat)):
        key_bits[pkt_feat[i]] = pkt_feat_bits[i]

    feat_dict_thres = get_xgb_feature_thres(model_name, pkt_feat)

    feat_dict = {}
    for key in pkt_feat:
        feat_dict[key] = list(set(feat_dict_thres[key]))
        feat_dict[key].sort()

    ignored_fea = []
    pkt_feat_mark_bit = []
    for k in feat_dict.keys():
        if len(feat_dict[k]) == 0:
            ignored_fea.append(k)
        pkt_feat_mark_bit.append(len(feat_dict[k]))

    range_mark_bits = {}
    for i in range(len(pkt_feat)):
        range_mark_bits[pkt_feat[i]] = pkt_feat_mark_bit[i]

    # 删除冗余特征
    for key in ignored_fea:
        pkt_feat.remove(key)
        del feat_dict[key]
        del range_mark_bits[key]
        del key_bits[key]
        print("Delete %s" % key)

    feat_table_datas = get_feature_table_entries(feat_dict, key_bits, range_mark_bits)
    tree_data_datas = get_xgb_trees_table_entries(model_name, pkt_feat, feat_dict, range_mark_bits)

    num_rules += len(tree_data_datas)
    for v in feat_table_datas.values():
        num_rules += len(v)

    # 生成P4
    fea_tbl = ""
    tbl_apply = ""
    tbl_model_key = ""
    meta_code = ""

    for fea in pkt_feat:
        meta_code += "bit<%d> codes_%s;\n\t" % (range_mark_bits[fea], fea)
        tbl_model_key += "meta.codes_%s : ternary;\n\t\t" % fea

        fea_tbl += "action ac_fea_%s(bit<%d> code){\n" % (fea, range_mark_bits[fea])
        fea_tbl += "\t\tmeta.codes_%s = code;\n" % fea
        fea_tbl += "\t}\n\n\t"

        fea_tbl += "table tbl_fea_%s{\n" % fea
        fea_tbl += "\t\tkey= {%s : ternary;}\n" % header_names[fea][0]
        fea_tbl += "\t\tactions = {ac_fea_%s;}\n" % fea
        fea_tbl += "\t\tsize=%d;\n" % len(feat_table_datas[fea])
        fea_tbl += "\t}\n\n\t"
        tbl_apply += "tbl_fea_%s.apply();\n\t\t" % fea

    key = "==model_size=="
    add_to_template(destination_file, key, str(len(tree_data_datas)))
    key = "==fea_tbl=="
    add_to_template(destination_file, key, fea_tbl)
    key = "==apply_tbl=="
    add_to_template(destination_file, key, tbl_apply)
    key = "==codes_ternary=="
    add_to_template(destination_file, key, tbl_model_key)
    key = "==codes=="
    add_to_template(destination_file, key, meta_code)
    
    # 生成 python
    # setup python file
    f = open(setup_file, "w")
    print("p4 = bfrt.xgb.pipe\n", file=f)
    clear_tables = """
def clear_all(verbose=True, batching=True):
    global p4
    global bfrt
    for table_types in (['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR'],
                        ['SELECTOR'],
                        ['ACTION_PROFILE']):
        for table in p4.info(return_info=True, print_info=False):
            if table['type'] in table_types:
                if verbose:
                    print("Clearing table {:<40} ... ".
                          format(table['full_name']), end='', flush=True)
                table['node'].clear(batch=batching)
                if verbose:
                    print('Done')
clear_all(verbose=True)\n
    """
    print(clear_tables, file=f)
    
    for fea in pkt_feat:
        tbl_name = "tbl_fea_%s" % fea
        fea_tbl_data = feat_table_datas[fea]
        fea_name = header_names[fea][0].split(".")[-1].lower()
        print(tbl_name + " = p4.Ingress."+tbl_name, file=f)
        print('', file=f)
        for rule in fea_tbl_data:
            print(tbl_name+".add_with_ac_fea_"+fea+"("+fea_name+"="+"%d, " % rule[1] + fea_name+"_mask="+"%d, " % rule[2] + "match_priority=%d, " % rule[0] + "code=%d" % rule[3] + ")", file=f)
        print('', file=f)
        
        
    code_tbl_name = "tb_packet_cls"
    print(code_tbl_name + " = p4.Ingress."+code_tbl_name, file=f)
    print('', file=f)
    
    for rule in tree_data_datas:
        cla = int(rule[-1] >= 50)
        code_str = ", ".join(["codes_%s=%d, codes_%s_mask=%d" % (pkt_feat[i], rule[i*2], pkt_feat[i], rule[i*2+1]) for i in range(len(pkt_feat))])
        print(code_tbl_name+".add_with_ac_packet_forward("+ code_str + ", port=%d" % cla + ")", file=f)
    print('', file=f)    
    print("bfrt.complete_operations()", file=f)
    f.close()
    
    return num_rules



