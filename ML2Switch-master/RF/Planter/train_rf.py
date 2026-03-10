from sklearn.metrics import classification_report, accuracy_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_data
import wandb
import argparse
from sklearn.ensemble import RandomForestClassifier
from rf2rules import export_rf


def main(args):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    train_X, train_y, test_X, test_y = load_data(args.key)

    clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                 max_depth=args.max_depth,
                                 random_state=2023)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    test_acc = accuracy_score(test_y, preds)
    train_acc = accuracy_score(train_y, clf.predict(train_X))
    print(classification_report(test_y, preds, digits=6))

    num_rules = export_rf(clf, len(test_X[0]), args.class_num_bits)

    if args.use_wandb:
        wandb.log(dict(train_acc=train_acc, test_acc=test_acc, rules=num_rules))
        wandb.finish()

    # 保存预测结果，用于在交换机上验证
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_sklearn.txt"), "w") as f:
        f.write("\n".join(["%d" % i for i in preds]))

    print("Train acc: %.6f, Test acc: %.6f, Rules: %d" % (train_acc, test_acc, num_rules))

if __name__ == '__main__':
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", default=False, type=bool)
    parser.add_argument("--wandb_project", default="ML2Switch", type=str)
    parser.add_argument("--key", default="univ", type=str)
    parser.add_argument("--max_depth", default=3, type=int)
    parser.add_argument("--class_num_bits", default=1, type=int)
    parser.add_argument("--n_estimators", default=2, type=int)
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    print("Total execution time: %.4f seconds" % (end_time - start_time))








