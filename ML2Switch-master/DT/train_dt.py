from sklearn.metrics import classification_report, accuracy_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data
from sklearn import tree
import wandb
import argparse
from dt2rules import export_dt

def main(args):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    train_X, train_y, test_X, test_y = load_data(args.key)
    clf = tree.DecisionTreeClassifier(max_depth=args.max_depth, random_state=2023)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    test_acc = accuracy_score(test_y, preds)
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    print(classification_report(test_y, preds, digits=6))

    num_rules = export_dt(clf, len(test_X[0]))

    if args.use_wandb:
        wandb.log(dict(train_acc=train_acc, test_acc=test_acc, rules=num_rules))
        wandb.finish()
    print("Train acc: %.4f, Test acc: %.4f, Num rules: %d" % (train_acc, test_acc, num_rules))

    # 保存预测结果，用于在交换机上验证
    # 保存预测结果，用于在交换机上验证
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dt_sklearn.txt"), "w") as f:
        f.write("\n".join(["%d" % i for i in preds]))

if __name__ == '__main__':
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", default=False, type=bool)
    parser.add_argument("--wandb_project", default="ML2Switch", type=str)
    parser.add_argument("--key", default="univ", type=str)
    parser.add_argument("--max_depth", default=8, type=int)
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    print("Total execution time: %.4f seconds" % (end_time - start_time))








