# -*- coding:utf-8 -*-
"""
Generate PCAP from JSON samples.
Modified from pkl2pcap.py to support JSON input.
"""

import os
import json
import sys
from io import StringIO
import numpy as np
from scapy.all import IP, TCP, UDP, Raw, wrpcap

def features_to_packet(features, feature_list):
    """
    Construct a network packet based on features.
    """
    feature_dict = {}
    for i, feat_name in enumerate(feature_list):
        feature_dict[feat_name] = features[i]
    
    total_length = int(feature_dict.get('Total length', 60))
    protocol = int(feature_dict.get('Protocol', 6))
    df_flag = int(feature_dict.get('IPV4 Flags (DF)', 0))
    ttl = int(feature_dict.get('Time to live', 64))
    src_port = int(feature_dict.get('Src Port', 12345))
    dst_port = int(feature_dict.get('Dst Port', 80))
    tcp_rst = int(feature_dict.get('TCP flags (Reset)', 0))
    tcp_syn = int(feature_dict.get('TCP flags (Syn)', 0))
    
    ip_flags = 0
    if df_flag:
        ip_flags = 0x02
    
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.200"
    
    if protocol == 6:  # TCP
        tcp_flags = 0
        if tcp_syn:
            tcp_flags |= 0x02
        if tcp_rst:
            tcp_flags |= 0x04
        if not tcp_syn and not tcp_rst:
            tcp_flags = 0x10
        
        tcp = TCP(sport=src_port, dport=dst_port, flags=tcp_flags, seq=1000)
        packet = IP(src=src_ip, dst=dst_ip, ttl=ttl, flags=ip_flags) / tcp
        
        ip_header_len = 20
        tcp_header_len = 20
        data_len = max(0, total_length - ip_header_len - tcp_header_len)
        
        if data_len > 0:
            packet = packet / Raw(b'X' * data_len)
    
    elif protocol == 17:  # UDP
        udp = UDP(sport=src_port, dport=dst_port)
        packet = IP(src=src_ip, dst=dst_ip, ttl=ttl, flags=ip_flags, proto=17) / udp
        
        ip_header_len = 20
        udp_header_len = 8
        data_len = max(0, total_length - ip_header_len - udp_header_len)
        
        if data_len > 0:
            packet = packet / Raw(b'X' * data_len)
    
    else:
        packet = IP(src=src_ip, dst=dst_ip, ttl=ttl, flags=ip_flags, proto=protocol)
        ip_header_len = 20
        data_len = max(0, total_length - ip_header_len)
        if data_len > 0:
            packet = packet / Raw(b'X' * data_len)
    
    return packet

def main():
    json_path = r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_validation_samples_100.json"
    output_pcap = r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_validation_100.pcap"
    
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]
    
    print(f"Loading JSON samples from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    packets = []
    print(f"Constructing {len(samples)} packets...")
    
    for i, sample in enumerate(samples):
        features = sample['features']
        pkt = features_to_packet(features, feature_list)
        packets.append(pkt)
    
    print(f"Writing PCAP to: {output_pcap}")
    wrpcap(output_pcap, packets)
    print("Success!")

if __name__ == '__main__':
    main()
