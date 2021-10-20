#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 #
########################################################################################
# python -m awesome_autodl.bins.show_infos --root awesome_autodl/raw_data
########################################################################################
import argparse
from pathlib import Path
from awesome_autodl.utils import load_yaml
from awesome_autodl.utils import dump_yaml
from awesome_autodl.utils import check_and_sort_by_date
from awesome_autodl.utils import filter_ele_w_value


name2file = {
    "Automated Problem Formulation": "Automated_Problem_Formulation.yaml",
    "Automated Data Engineering": "Automated_Data_Engineering.yaml",
    "Neural Architecture Search": "Neural_Architecture_Search.yaml",
    "Hyperparameter Optimization": "Hyperparameter_Optimization.yaml",
    "Automated Deployment": "Automated_Deployment.yaml",
    "Automated Maintenance": "Automated_Maintenance.yaml",
}

# add abbreviation
name2file["ADE"] = name2file["Automated Data Engineering"]
name2file["NAS"] = name2file["Neural Architecture Search"]
name2file["HPO"] = name2file["Hyperparameter Optimization"]
name2file["AD"] = name2file["Automated Deployment"]
name2file["AM"] = name2file["Automated Maintenance"]


def scale(xlist, scale):
    for x in xlist:
        print(int(x * scale))


def show_nas(root):
    abbrv = load_yaml(root / "abbrv.yaml")
    topic_path = root / "papers" / name2file["NAS"]
    assert topic_path.exists(), f"Did not find {topic_path}"
    print(f"Process NAS topic from {topic_path}")
    data = load_yaml(topic_path)
    data = check_and_sort_by_date(data)
    print(f"Find {len(data)} papers for NAS")
    # Search Space Analysis
    nasnet_like_search_space_papers = filter_ele_w_value(data, "search_space", "NASNet")
    print(
        f"NASNet-like search space: {len(nasnet_like_search_space_papers)}/{len(data)} : {len(nasnet_like_search_space_papers)*100./len(data):.3f}"
    )
    mbconv_based_search_space_papers = filter_ele_w_value(
        data, "search_space", "MBConv"
    )
    print(
        f"MBConv-based search space: {len(mbconv_based_search_space_papers)}/{len(data)} : {len(mbconv_based_search_space_papers)*100./len(data):.3f}"
    )
    size_based_search_space_papers = filter_ele_w_value(data, "search_space", "size")
    print(
        f"Size-related search space: {len(size_based_search_space_papers)}/{len(data)} : {len(size_based_search_space_papers)*100./len(data):.3f}"
    )
    ratio_keys = ["NASNet", "MBConv", "size"]
    ratios = [len(filter_ele_w_value(data, "search_space", key)) for key in ratio_keys]
    ratios = [float(ratio) / len(data) for ratio in ratios]
    ratios.append(1 - sum(ratios))
    # Search Strategy
    ratio_keys = ["Differential", "RL", "Evolution"]
    ratios = [
        len(filter_ele_w_value(data, "search_strategy", key)) for key in ratio_keys
    ]
    ratios = [float(ratio) / len(data) for ratio in ratios]
    ratios.append(1 - sum(ratios))
    # scale(ratios, 500)

    # Efficient candidate evaluation
    weight_sharing_papers = (
        filter_ele_w_value(data, "eval_boost", "weight sharing")
        + filter_ele_w_value(data, "eval_boost", "HyperNet")
        + filter_ele_w_value(data, "eval_boost", "weight i")
        + filter_ele_w_value(data, "eval_boost", "Net2Net")
    )
    print(
        f"Weight sharing: {len(weight_sharing_papers)}/{len(data)} : {len(weight_sharing_papers)*100./len(data):.3f}"
    )


def show_all(root):

    name2data = dict()
    total = 0
    for name, file in name2file.items():
        if len(name) < 10:
            topic_path = root / "papers" / file
            data = load_yaml(topic_path)
            name2data[name] = data
            total += len(data)
    for name, data in name2data.items():
        print(f"{name:5s}: {len(data)}/{total} = {len(data)*1./total:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis the AutoDL papers.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="The path for the data directory.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    show_all(root)
    # show_nas(root)
