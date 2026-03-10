# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : generate_pcaps.py
# Time       ：2023-05-07 13:51
# Author     ：Haolin Yan
# Description：
"""
import sys
sys.path.append("../")
from scapy.all import *
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP, TCP, Raw
from utils import load_univ_data, load_iscx_data
from tqdm import tqdm
import numpy as np

def generate_univ(n_test=200):
    def generate_pkt(sample):
        ip = IP(ihl=sample[3],
                tos=sample[4],
                len=sample[9],
                proto=sample[2],
                ttl=sample[5])
        if sample[2] == 6:
            l4 = TCP(sport=sample[0],
                     dport=sample[1],
                     dataofs=sample[6],
                     window=sample[7])
        else:
            l4 = UDP(sport=sample[0],
                     dport=sample[1],
                     len=sample[8])

        payload = sample[8] - sample[3] * 4 - len(l4)
        pkt = Ether() / ip / l4 / Raw(load=b"\x01" * payload)
        return pkt
    train_X, train_y, test_X, test_y = load_univ_data()
    pcap = []
    for i in tqdm(range(n_test)):
        pcap.append(generate_pkt(test_X[i, :]))
    wrpcap('univ_test.pcap', pcap)

def generate_iscx(n_test=200):
    def generate_pkt(sample):
        ip = IP(ihl=sample[1],
                tos=sample[2],
                len=sample[9],
                flags=sample[3],
                proto=sample[0],
                ttl=sample[4])
        if sample[0] == 6:
            l4 = TCP(
                     dataofs=sample[5],
                     flags=sample[6],
                     window=sample[7])
        else:
            l4 = UDP(
                     len=sample[8])

        payload = sample[9] - sample[1] * 4 - len(l4)
        # print(payload)
        pkt = Ether() / ip / l4 / Raw(load=b"\x01" * payload)
        return pkt
    train_X, train_y, test_X, test_y = load_iscx_data()
    pcap = []
    for i in tqdm(range(n_test)):
        pcap.append(generate_pkt(test_X[i, :]))
    wrpcap('iscx_test.pcap', pcap)

if __name__ == '__main__':
    # generate_univ()
    generate_iscx()


