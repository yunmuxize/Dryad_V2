from sklearn.metrics import classification_report, accuracy_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_data
import wandb
import argparse
from sklearn.ensemble import RandomForestClassifier
from rf2rules import export_rf
import os
from sklearn.tree import export_graphviz

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

    class_names = ["cls%d" % i for i in range(args.num_classes)]
    feature_names = ["f%d" % i for i in range(len(test_X[0]))]

    for i in range(len(clf.estimators_)):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        export_graphviz(clf.estimators_[i], out_file=os.path.join(output_dir, 'rf_tree_{}.dot'.format(i)),
                        feature_names=feature_names,
                        class_names=class_names,
                        rounded=True, proportion=False,
                        precision=4, filled=True)

    num_rules = export_rf(len(test_X[0]), args.n_estimators)

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
    parser.add_argument("--num_classes", default=6, type=int)
    parser.add_argument("--n_estimators", default=1, type=int)
    args = parser.parse_args()
    main(args)
    end_time = time.time()
    print("Total execution time: %.4f seconds" % (end_time - start_time))








