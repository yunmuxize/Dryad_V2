from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append("../")
from sklearn import tree
import wandb
import argparse
from dt2rules import export_dt
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
def load_iscx_c(path, key):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[key]

def sample_data(path, load_func, new_cls, old_cls):
    X_test, y_test = load_func(path, key="test")
    X_train, y_train = load_func(path, key="train")
    X_dev, y_dev = load_func(path, key="val")   

    def sample_old_data(X, y, cfg, key="train"):
        """sample old data"""
        X_old, y_old = [], []
        for c, r in cfg.items():
            mask = np.isin(y, [c])
            X_ = X[mask]
            y_ = y[mask]
            if key != "test":
                X_ = X_[:int(X_.shape[0] * r)]
                y_ = y_[:int(y_.shape[0] * r)]
            X_old.append(X_)
            y_old.append(y_)
        X_old = np.concatenate(X_old, axis=0)
        y_old = np.concatenate(y_old, axis=0)
        return X_old, y_old
    
    X_train_old, y_train_old = sample_old_data(X_train, y_train, old_cls)
    X_dev_old, y_dev_old = sample_old_data(X_dev, y_dev, old_cls)
    X_test_old, y_test_old = sample_old_data(X_test, y_test, old_cls, key="test")
    X_train_new, y_train_new = sample_old_data(X_train, y_train, new_cls)
    X_dev_new, y_dev_new = sample_old_data(X_dev, y_dev, new_cls)
    X_test_new, y_test_new = sample_old_data(X_test, y_test, new_cls, key="test")

    # return them in a dictionary
    return {
        "X_train_old": X_train_old,
        "y_train_old": y_train_old,
        "X_train_new": X_train_new,
        "y_train_new": y_train_new,
        "X_dev_old": X_dev_old,
        "y_dev_old": y_dev_old,
        "X_dev_new": X_dev_new,
        "y_dev_new": y_dev_new,
        "X_test_old": X_test_old,   
        "y_test_old": y_test_old,
        "X_test_new": X_test_new,
        "y_test_new": y_test_new,
    }

def main(args):
    # Streaming
    old_cfg = {0:1, 1:1, 2:1}
    new_cfg = {4:args.ratio}
    
    # P2P
    old_cfg = {0:1, 1:1, 2:1, 4:args.ratio}
    new_cfg = {3:args.ratio}

    #VoIP
    old_cfg = {0:1, 1:1, 2:1, 3:args.ratio, 4:args.ratio}
    new_cfg = {5:args.ratio}

    data = sample_data("../datasets/iscx_C.pkl", load_iscx_c, new_cfg, old_cfg)

    train_X, train_y = np.concatenate((data["X_train_old"], data["X_train_new"]), axis=0), np.concatenate((data["y_train_old"], data["y_train_new"]), axis=0)
    dev_X, dev_y = np.concatenate((data["X_dev_old"], data["X_dev_new"]), axis=0), np.concatenate((data["y_dev_old"], data["y_dev_new"]), axis=0)
    train_X, train_y = np.concatenate((train_X, dev_X), axis=0), np.concatenate((train_y, dev_y), axis=0)
    print("new train data size: ", len(data["X_train_new"]) + len(data["X_dev_new"]))
    print("new test data size: ", len(data["X_test_new"]))
    search_cfg = {"max_depth": [3,5,7,9,10,11]}
    print("Decision Tree, Ratio: {:.3f}".format(args.ratio))
    test_all_X = np.concatenate((data["X_test_old"], data["X_test_new"]), axis=0)
    test_all_y = np.concatenate((data["y_test_old"], data["y_test_new"]), axis=0)
    for max_depth in search_cfg["max_depth"]:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        clf.fit(train_X, train_y)
        train_acc = clf.score(train_X, train_y)
        old_test_acc = clf.score(data["X_test_old"], data["y_test_old"])
        new_test_acc = clf.score(data["X_test_new"], data["y_test_new"])
        all_test_acc = clf.score(test_all_X, test_all_y)
        num_rules = export_dt(clf, len(test_all_X[0]))
        print("Max depth: {}, Old test accuracy: {:.3f}, New test accuracy: {:.3f}, All test accuracy: {:.3f}, Num rules: {}".format(max_depth, old_test_acc, new_test_acc, all_test_acc, num_rules))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.0005)
    args = parser.parse_args()
    main(args)