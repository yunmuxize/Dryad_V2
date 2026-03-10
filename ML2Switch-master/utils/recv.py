from scapy.all import *
import time
import pickle
recv_timestamp = []
MAX_NUM_RECV = 180
predicted_results = ""
def callback(packet):
    global recv_timestamp, predicted_results
    print("recv %d pkt" % len(recv_timestamp))
    predicted_results += "%d\n" % int(packet['IP'].ttl)

    recv_timestamp.append(time.time())
    if len(recv_timestamp) >= MAX_NUM_RECV:
        with open("recv_timestamp.pkl","wb") as f:
            pickle.dump(recv_timestamp, f)
        
        with open("predicted_result.txt", "w") as f:
            f.write(predicted_results)
        exit()

if __name__ == "__main__":
    sniff(prn=callback, iface='veth5', count=0)

