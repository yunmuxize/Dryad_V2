import json
import os

model_path = r"c:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_depth_5_model.json"

def count_leaves(node):
    if "children" not in node or not node["children"]:
        return 1
    
    count = 0
    for child in node["children"]:
        count += count_leaves(child)
    return count

def main():
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    with open(model_path, 'r') as f:
        model = json.load(f)
    
    tree = model['tree_structure']
    leaves = count_leaves(tree)
    
    print(f"Leaf count for {model_path}: {leaves}")

if __name__ == "__main__":
    main()
