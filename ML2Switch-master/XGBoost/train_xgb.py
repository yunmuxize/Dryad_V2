# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_xgb.py
# Time       ：2023-05-06 15:07
# Author     ：Haolin Yan
# Description：
"""
import sys
sys.path.append("../")
from utils import load_data
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import wandb
from xgb2rules import export_xgb
import argparse


def main(args):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    train_X, train_y, test_X, test_y = load_data(args.key)
    clf = XGBClassifier(n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        random_state=2023)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    test_acc = accuracy_score(test_y, preds)
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    print(classification_report(test_y, preds, digits=6))
    clf.get_booster().dump_model(args.out)

    num_rules = export_xgb(args.out, len(test_X[0]))

    if args.use_wandb:
        wandb.log(dict(train_acc=train_acc, test_acc=test_acc, rules=num_rules))
        wandb.finish()

    # 保存预测结果，用于在交换机上验证
    with open("xgb_sklearn.txt", "w") as f:
        f.write("\n".join(["%d" % i for i in preds]))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", default=True, type=bool)
    parser.add_argument("--wandb_project", default="ML2Switch", type=str)
    parser.add_argument("--key", default="univ", type=str)
    parser.add_argument("--n_estimators", default=3, type=int)
    parser.add_argument("--max_depth", default=10, type=int)
    parser.add_argument("--out", default="xgb.txt", type=str)
    args = parser.parse_args()
    main(args)









