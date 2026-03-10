from scapy.all import *
import pickle
import time
send_timestamp = []
pcap_path = "/home/p4/infocom23_shared/P4/UNIV/iscx_test.pcap"
packets = rdpcap(pcap_path)
print("prepare to send packet.")
num_send = 0
for pkt in packets:
    sendp(pkt, iface="veth0")
    num_send += 1
    send_timestamp.append(time.time())
print("send:%d" % num_send)
with open("send_timestamp.pkl", "wb") as f:
    pickle.dump(send_timestamp, f)
