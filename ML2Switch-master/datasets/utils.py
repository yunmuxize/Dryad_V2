import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def load_dataset(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data[:, :-1], data[:, -1]
eval_X, eval_y = load_dataset("iscx/data_eval_iscx_C.pkl")
X_train, y_train = load_dataset("iscx/data_train_iscx_C.pkl")
X_test, X_dev, y_test, y_dev = train_test_split(eval_X, eval_y, test_size=0.1, shuffle=True, random_state=2023)
with open("iscx_C.pkl", "wb") as f:
        pickle.dump(dict(train=(X_train, y_train),
                         val=(X_dev, y_dev),
                            test=(X_test, y_test)), f)
        
eval_X, eval_y = load_dataset("iscx/data_eval_iscx.pkl")
X_train, y_train = load_dataset("iscx/data_train_iscx.pkl")
X_test, X_dev, y_test, y_dev = train_test_split(eval_X, eval_y, test_size=0.1, shuffle=True, random_state=2023)
with open("iscx.pkl", "wb") as f:
        pickle.dump(dict(train=(X_train, y_train),
                         val=(X_dev, y_dev),
                            test=(X_test, y_test)), f)
        




# def read_data(filename):
#     data = pd.read_csv(filename)
#     return data
# # read data from train.csv and test.csv
# train_data = read_data("univ1-1.csv")
# test_data = read_data("univ1-2.csv")
# print(train_data.shape, test_data.shape)

# # split test data as dev and test data
# test_data, dev_data = train_test_split(test_data, test_size=0.1, shuffle=True, random_state=2023)
# print(train_data.shape, dev_data.shape, test_data.shape)

# with open("univ.pkl", "wb") as f:
#     pickle.dump(dict(train=train_data,
#                          val=test_data,
#                             test=test_data), f)

# def DecToBinary(data, num_bits=8):
#     bit_str = bin(data)[2:].zfill(num_bits)
#     return [int(bit) for bit in bit_str]


# def toBinaryFormat(df, feature_names, feature_bits):
#     results = {}
#     for i, feature in enumerate(feature_names):
#         df_b = np.array([DecToBinary(v, feature_bits[i]) for v in df[feature].values])
#         for j in range(feature_bits[i]):
#             results[feature + "_" + str(j)] = df_b[:, j]
#     return pd.DataFrame(results)

# train_path = "univ1-1.csv"
# eval_path = "univ1-2.csv"

# iot_feature_names = ["srcPort", "dstPort", "protocol", "ip_ihl", "ip_tos", "ip_ttl", "tcp_dataofs",
#                          "tcp_window", "udp_len", "length", 'flowSize']

# feature_bits = [16, 16, 8, 4, 8, 8, 4, 16, 16, 16, 16]

# train_data = pd.read_csv(train_path)[iot_feature_names]
# eval_data = pd.read_csv(eval_path)[iot_feature_names]

# train_data_b = toBinaryFormat(train_data, iot_feature_names[:-1], feature_bits[:-1])
# eval_data_b = toBinaryFormat(eval_data, iot_feature_names[:-1], feature_bits[:-1])
# train_data_b["flowSize"] = train_data["flowSize"]
# eval_data_b["flowSize"] = eval_data["flowSize"]

# print(train_data.head(0))
# print(train_data_b.head(0))

# train_data.to_csv("univ_train_c.csv", index=False)
# eval_data.to_csv("univ_eval_c.csv", index=False)
# train_data_b.to_csv("univ_train_c_b.csv", index=False)
# eval_data_b.to_csv("univ_eval_c_b.csv", index=False)
    







# path = "iscx_c.pkl"
# try:
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
# except EOFError:
#     print("Pickle file is truncated.")

# # X_train, y_train = data["train"]
# # print(X_train.shape, y_train.shape)
