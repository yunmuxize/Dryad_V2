# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : __init__.py
# Time       ：2023-05-06 15:08
# Author     ：Haolin Yan
# Description：
"""
import pandas as pd
import pickle
import numpy as np
import os

# 获取项目根目录 (假设 utils 文件夹在项目根目录下)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_unsw_pkl_data():
    with open(os.path.join(ROOT_DIR, "datasets", "x_train.pkl"), "rb") as f:
        train_X = pickle.load(f)
    with open(os.path.join(ROOT_DIR, "datasets", "y_train.pkl"), "rb") as f:
        train_y = pickle.load(f)
    with open(os.path.join(ROOT_DIR, "datasets", "x_test.pkl"), "rb") as f:
        test_X = pickle.load(f)
    with open(os.path.join(ROOT_DIR, "datasets", "y_test.pkl"), "rb") as f:
        test_y = pickle.load(f)
    
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

def load_unsw_data():
    train_path = os.path.join(ROOT_DIR, "datasets", "unsw_train_b.csv")
    test_path = os.path.join(ROOT_DIR, "datasets", "unsw_test_b.csv")
    
    iot_feature_names = ["srcPort", "dstPort", "protocol", "ip_ihl", "ip_tos", "ip_ttl", "tcp_dataofs", "tcp_window",
                         "udp_len", "length", "label"]
    
    df_train = pd.read_csv(train_path)[iot_feature_names]
    train_X, train_y = df_train.values[:, :-1], df_train.values[:, -1]

    df_test = pd.read_csv(test_path)[iot_feature_names]
    test_X, test_y = df_test.values[:, :-1], df_test.values[:, -1]

    return train_X, train_y, test_X, test_y

def load_univ_data():
    # 注意：这里的路径逻辑根据原代码可能是相对于某个特定目录，这里统一使用 ROOT_DIR
    train_path = os.path.join(ROOT_DIR, "datasets", "univ", "univ1-1.csv")
    eval_path = os.path.join(ROOT_DIR, "datasets", "univ", "univ1-2.csv")

    iot_feature_names = ["srcPort", "dstPort", "protocol", "ip_ihl", "ip_tos", "ip_ttl", "tcp_dataofs",
                         "tcp_window", "udp_len", "length", 'flowSize']

    # 如果文件不存在，尝试原代码中的相对路径 (可能是为了兼容旧的调用方式)
    if not os.path.exists(train_path):
         train_path = "../../datasets/univ1-1.csv"
         eval_path = "../../datasets/univ1-2.csv"

    try:
        train_data = pd.read_csv(train_path)[iot_feature_names].values
        eval_data = pd.read_csv(eval_path)[iot_feature_names].values
        train_X, train_y = train_data[:, :-1], train_data[:, -1]
        test_X, test_y = eval_data[:, :-1], eval_data[:, -1]
        return train_X, train_y, test_X, test_y
    except Exception as e:
        print(f"Error loading univ data: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

def load_iscx_data():
    train_path = os.path.join(ROOT_DIR, "datasets", "iscx", "data_train_iscx_C.pkl")
    test_path = os.path.join(ROOT_DIR, "datasets", "iscx", "data_eval_iscx_C.pkl")
    
    with open(train_path, "rb") as f:
        data = pickle.load(f)
    train_X, train_y = data[:, :-1], data[:, -1]
    
    with open(test_path, "rb") as f:
        data = pickle.load(f)
    test_X, test_y = data[:, :-1], data[:, -1]

    return train_X, train_y, test_X, test_y

def load_data(key="univ"):
    if key == "univ":
        return load_univ_data()
    elif key == "unsw":
        return load_unsw_data()
    elif key == "unsw_pkl":
        return load_unsw_pkl_data()
    elif key == "iscx":
        return load_iscx_data()
    else:
        raise ValueError("Invalid dataset!")
    
def DecToBinary(data, num_bits=8):
    bit_str = bin(data)[2:].zfill(num_bits)
    return np.array([int(bit) for bit in bit_str])
