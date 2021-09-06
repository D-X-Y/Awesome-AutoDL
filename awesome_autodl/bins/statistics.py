#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 #
########################################################################################
# python -m awesome_autodl.bins.statistics --topic "ADE" --root awesome_autodl/raw_data
# python -m awesome_autodl.bins.statistics --topic "NAS" --root awesome_autodl/raw_data
# python -m awesome_autodl.bins.statistics --topic "HPO" --root awesome_autodl/raw_data
# python -m awesome_autodl.bins.statistics --topic "AD"  --root awesome_autodl/raw_data
# python -m awesome_autodl.bins.statistics --topic "AM"  --root awesome_autodl/raw_data
########################################################################################
import argparse
from pathlib import Path
from awesome_autodl.utils import load_yaml
from awesome_autodl.utils import dump_yaml
from awesome_autodl.utils import check_and_sort_by_date


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


def main(root, topic):
    abbrv = load_yaml(root / "abbrv.yaml")
    topic_path = root / "papers" / name2file[topic]
    assert topic_path.exists(), f"Did not find {topic_path}"
    print(f"Process {topic_path}")
    data = load_yaml(topic_path)
    data = check_and_sort_by_date(data)
    print(f"Find {len(data)} papers for {topic}")
    dump_yaml(data, path=topic_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis the AutoDL papers.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="The path for the data directory.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        choices=list(name2file.keys()),
        help="Choose the AutoDL sub-topic.",
    )
    args = parser.parse_args()
    main(Path(args.root), args.topic)
