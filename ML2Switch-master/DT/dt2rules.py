"""
Dataloader for different task
"""
import pandas as pd
import numpy as np
import fileinput
import os
from shutil import copyfile
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_to_template(outname, placeholder, code):
    with fileinput.FileInput(outname, inplace=True) as file:
        for line in file:
            print(line.replace(placeholder, code), end='')


def split_codes(code, used_features, add=""):
    code = code[2:]  # 删除"0b"
    idx = 0
    content = ""
    for fea in used_features:
        fea_n = fea[0].lower()
        content += "codes_" + fea_n + add + "=" + "0b" + code[idx:idx + fea[1]] + ","
        idx = idx + fea[1]
    return content

def comb_tree_preds(comb, forest_domain):
    new_comb = []
    if len(forest_domain) == 1:
        current_tree = forest_domain.pop()
        if len(comb) == 0:
            for f in current_tree:
                new_comb.append([f])
        else:
            for f in current_tree:
                for c in comb:
                    new_comb.append([*c, f])
    else:
        current_tree = forest_domain.pop()
        if len(comb) == 0:
            for f in current_tree:
                new_comb.append([f])
        else:
            for f in current_tree:
                for c in comb:
                    new_comb.append([*c, f])
        new_comb = comb_tree_preds(new_comb, forest_domain)
    return new_comb


## get list of splits crossed to get to leaves
def retrieve_branches(estimator):
    number_nodes = estimator.tree_.node_count
    children_left_list = estimator.tree_.children_left
    children_right_list = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # Calculate if a node is a leaf
    is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
    # Store the branches paths
    paths = []
    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]
            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output
        else:
            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]
            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])
            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left_list[i]])
                paths.append([i, children_right_list[i]])


## get classes and certainties
def get_classes(clf):
    leaves = []
    classes = []
    certainties = []
    for branch in list(retrieve_branches(clf)):
        leaves.append(branch[-1])
    for leaf in leaves:
        if clf.tree_.n_outputs == 1:
            value = clf.tree_.value[leaf][0]
        else:
            value = clf.tree_.value[leaf].T[0]
        class_name = np.argmax(value)
        certainty = int(round(max(value) / sum(value), 2) * 100)
        classes.append(class_name)
        certainties.append(certainty)
    return classes, certainties


## get the codes corresponging to the branches followed
def get_leaf_paths(clf):
    depth = clf.max_depth
    branch_codes = []
    for branch in list(retrieve_branches(clf)):
        code = [0] * len(branch)
        for i in range(1, len(branch)):
            if (branch[i] == clf.tree_.children_left[branch[i - 1]]):
                code[i] = 0
            elif (branch[i] == clf.tree_.children_right[branch[i - 1]]):
                code[i] = 1
        branch_codes.append(list(code[1:]))
    return branch_codes


def get_splits_per_tree(clf, feature_names):
    data = []
    n_nodes = clf.tree_.node_count
    # set feature names
    features = [feature_names[i] for i in clf.tree_.feature]
    # generate dataframe with all thresholds and features
    for i in range(0, n_nodes):
        node_id = i
        left_child_id = clf.tree_.children_left[i]
        right_child_id = clf.tree_.children_right[i]
        threshold = clf.tree_.threshold[i]
        feature = features[i]
        if threshold != -2.0:
            data.append([node_id, left_child_id,
                         right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["NodeID", "LeftID", "RightID", "Threshold", "Feature"]
    return data


## get all splits from the tree
def get_splits(dt, feature_names):
    data = []
    # generate dataframe with all thresholds and features
    clf = dt
    n_nodes = clf.tree_.node_count
    features = [feature_names[i] for i in clf.tree_.feature]
    for i in range(0, n_nodes):
        node_id = i
        left_child_id = clf.tree_.children_left[i]
        right_child_id = clf.tree_.children_right[i]
        threshold = clf.tree_.threshold[i]
        feature = features[i]
        if threshold != -2.0:
            # data.append([t, node_id, left_child_id,
            #              right_child_id, threshold, feature])
            data.append([0, node_id, left_child_id,
                         right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["Tree", "NodeID", "LeftID", "RightID", "Threshold", "Feature"]
    return data


## gets the feature table of each feature from the splits
def get_feature_table(splits_data, feature_name):
    feature_data = splits_data[splits_data["Feature"] == feature_name]
    feature_data = feature_data.sort_values(by="Threshold")
    feature_data = feature_data.reset_index(drop=True)
    ##
    # feature_data["Threshold"] = (feature_data["Threshold"]).astype(int)
    feature_data["Threshold"] = feature_data["Threshold"].astype(int)
    ##
    code_table = pd.DataFrame()
    code_table["Threshold"] = feature_data["Threshold"]
    # print(feature_data)
    # create a column for each split in each tree
    for tree_id, node in zip(list(feature_data["Tree"]), list(feature_data["NodeID"])):
        colname = "s" + str(tree_id) + "_" + str(node)
        code_table[colname] = np.where((code_table["Threshold"] <=
                                        feature_data[(feature_data["NodeID"] == node) &
                                                     (feature_data["Tree"] == tree_id)]["Threshold"].values[0]), 0, 1)
    # add a row to represent the values above the largest threshold
    temp = [max(code_table["Threshold"]) + 1]
    temp.extend(list([1] * (len(code_table.columns) - 1)))
    code_table.loc[len(code_table)] = temp
    code_table = code_table.drop_duplicates(subset=['Threshold'])
    code_table = code_table.reset_index(drop=True)
    return code_table


## get feature tables with ranges and codes only
def get_feature_codes_with_ranges(feature_table, num_of_trees):
    Codes = pd.DataFrame()
    for tree_id in range(num_of_trees):
        colname = "code" + str(tree_id)
        Codes[colname] = feature_table[
            feature_table[[col for col in feature_table.columns if ('s' + str(tree_id) + '_') in col]].columns[
            0:]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
        Codes[colname] = ["0b" + x for x in Codes[colname]]
    feature_table["Range"] = [0] * len(feature_table)
    feature_table["Range"].loc[0] = "0," + str(feature_table["Threshold"].loc[0])
    for i in range(1, len(feature_table)):
        if (i == (len(feature_table)) - 1):
            feature_table["Range"].loc[i] = str(feature_table["Threshold"].loc[i]) + "," + str(
                feature_table["Threshold"].loc[i])
        else:
            feature_table["Range"].loc[i] = str(feature_table["Threshold"].loc[i - 1] + 1) + "," + str(
                feature_table["Threshold"].loc[i])
    Ranges = feature_table["Range"]
    return Ranges, Codes


## get the order of the splits to enable code generation
def get_order_of_splits(data, feature_names):
    splits_order = []
    for feature_name in feature_names:
        feature_data = data[data.iloc[:, 4] == feature_name]
        feature_data = feature_data.sort_values(by="Threshold")
        for node in list(feature_data.iloc[:, 0]):
            splits_order.append(node)
    return splits_order


def get_codes_and_masks(clf, feature_names):
    splits = get_order_of_splits(get_splits_per_tree(clf, feature_names), feature_names)
    depth = clf.max_depth
    codes = []
    masks = []
    for branch, coded in zip(list(retrieve_branches(clf)), get_leaf_paths(clf)):
        code = [0] * len(splits)
        mask = [0] * len(splits)
        for index, split in enumerate(splits):
            if split in branch:
                mask[index] = 1
        masks.append(mask)
        codes.append(code)
    masks = pd.DataFrame(masks)
    masks['Mask'] = masks[masks.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    masks = ["0b" + x for x in masks['Mask']]
    indices = range(0, len(splits))
    temp = pd.DataFrame(columns=["split", "index"], dtype=object)
    temp["split"] = splits
    temp["index"] = indices
    final_codes = []
    for branch, code, coded in zip(list(retrieve_branches(clf)), codes, get_leaf_paths(clf)):
        indices_to_use = temp[temp["split"].isin(branch)].sort_values(by="split")["index"]
        for i, j in zip(range(0, len(coded)), list(indices_to_use)):
            code[j] = coded[i]
        final_codes.append(code)
    final_codes = pd.DataFrame(final_codes)
    final_codes["Code"] = final_codes[final_codes.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)), axis=1)
    final_codes = ["0b" + x for x in final_codes["Code"]]
    return final_codes, masks

def export_dt(clf, n_fea):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(base_dir, "tmpl.p4")
    destination_file = os.path.join(base_dir, "dt.p4")
    setup_file = os.path.join(base_dir, "dt_setup.py")
    copyfile(source_file, destination_file)

    num_rules = 0
    feature_names = ["f%d" % i for i in range(n_fea)]
    ############################ 自定义内容 #############################
    # 参考tmpl.p4&../P4/headers.p4, 修改特征对应的PHV字段
    header_names = {'f0': ['meta.srcPort', 16],
                    'f1': ['meta.dstPort', 16],
                    'f2': ['hdr.ipv4.protocol', 8],
                    'f3': ['hdr.ipv4.ihl', 4],
                    'f4': ['hdr.ipv4.tos', 8],
                    'f5': ['hdr.ipv4.ttl', 8],
                    'f6': ['meta.dataOffset', 4],
                    'f7': ['meta.window', 16],
                    'f8': ['meta.udp_length', 16],
                    'f9': ['hdr.ipv4.totalLen', 16]}

    ##################################################################
    used_features = []

    Final_Codes, Final_Masks = get_codes_and_masks(clf, feature_names)
    Classe, Certain = get_classes(clf)
    num_rules += len(Classe)

    # 写入P4
    key = "==model_size=="
    add_to_template(destination_file, key, str(len(Classe)))

    # Find feature splits
    fea_tbl = ""
    tbl_apply = ""
    tbl_model_key = ""
    meta_code = ""
    data = get_splits(clf, feature_names)
    action_data_bits = []

    # setup python file
    f = open(setup_file, "w")
    print("p4 = bfrt.iisy.pipe\n", file=f)
    clear_tables = """
def clear_all(verbose=True, batching=True):
    global p4
    global bfrt
    for table_types in (['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR'],
                        ['SELECTOR'],
                        ['ACTION_PROFILE']):
        for table in p4.info(return_info=True, print_info=False):
            if table['type'] in table_types:
                if verbose:
                    print("Clearing table {:<40} ... ".
                          format(table['full_name']), end='', flush=True)
                table['node'].clear(batch=batching)
                if verbose:
                    print('Done')
clear_all(verbose=True)\n
    """
    print(clear_tables, file=f)

    used_features_ = data["Feature"].unique()
    for fea in feature_names:
        if fea not in used_features_:
            continue
        fea_space_split = get_feature_table(data, fea)
        # Get feature table
        Ranges, Codes = get_feature_codes_with_ranges(fea_space_split, 1)
        action_data_bit = len(Codes["code0"][0]) - 2
        action_data_bits.append(action_data_bit)
        used_features.append([fea, action_data_bit])
        num_rules += len(Ranges)

        # python
        tbl_name = "tbl_fea_%s" % fea
        print(tbl_name + " = p4.Ingress." + tbl_name, file=f)
        print('', file=f)

        ub = 2 ** header_names[fea][1] - 1

        for ran, code in zip(Ranges, Codes.iloc[:, 0]):
            fea_name = header_names[fea][0].split(".")[-1].lower()
            if (ran == Ranges[len(Ranges) - 1]):
                print(tbl_name + ".add_with_ac_fea_" + fea + "(" + fea_name + "_start=" + str(ran.split(",")[0]) + \
                      ", " + fea_name + "_end=" + str(ub) + ", code=" + str(code) + ")", file=f)
            else:
                print(tbl_name + ".add_with_ac_fea_" + fea + "(" + fea_name + "_start=" + str(ran.split(",")[0]) + \
                      ", " + fea_name + "_end=" + str(ran.split(",")[1]) + ", code=" + str(code) + ")", file=f)
        print('', file=f)

        # P4
        fea_tbl += "action ac_fea_%s(bit<%d> code){\n" % (fea, action_data_bit)
        fea_tbl += "\t\tmeta.codes_%s = code;\n" % fea
        fea_tbl += "\t}\n\n\t"

        fea_tbl += "table tbl_fea_%s{\n" % fea
        fea_tbl += "\t\tkey= {%s : range;}\n" % header_names[fea][0]
        fea_tbl += "\t\tactions = {ac_fea_%s;}\n" % fea
        fea_tbl += "\t\tsize=%d;\n" % len(Ranges)
        fea_tbl += "\t}\n\n\t"
        tbl_apply += "tbl_fea_%s.apply();\n\t\t" % fea

        meta_code += "bit<%d> codes_%s;\n\t" % (action_data_bit, fea)
        tbl_model_key += "meta.codes_%s : ternary;\n\t\t" % fea

    code_tbl_name = "tb_packet_cls"
    print(code_tbl_name + " = p4.Ingress." + code_tbl_name, file=f)
    print('', file=f)
    for cod, mas, cla, cer in zip(Final_Codes, Final_Masks, Classe, Certain):
        print(code_tbl_name + ".add_with_ac_packet_forward(" + split_codes(cod, used_features) +
              split_codes(mas, used_features, add="_mask") + "port=", cla, ")", file=f)
        print('', file=f)

    print("bfrt.complete_operations()", file=f)
    f.close()

    key = "==fea_tbl=="
    add_to_template(destination_file, key, fea_tbl)

    key = "==apply_tbl=="
    add_to_template(destination_file, key, tbl_apply)

    key = "==codes_ternary=="
    add_to_template(destination_file, key, tbl_model_key)

    key = "==codes=="
    add_to_template(destination_file, key, meta_code)

    return num_rules

