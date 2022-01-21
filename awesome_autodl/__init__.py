#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.09 #
#####################################################################################
# Automated Deep Learning: Neural Architecture Search Is Not the End, arXiv 2021.12 #
#####################################################################################
# This package is used to analyze the AutoDL-related papers. More detailed reports  #
# can be found in the above paper.                                                  #
#####################################################################################
from pathlib import Path
from collections import OrderedDict


def version():
    versions = ["v0.1"]  # 2021.09.03
    versions = ["v0.2"]  # 2021.09.04
    versions = ["v0.3"]  # 2022.01.17
    versions = ["v1.0"]  # 2022.01.20
    versions = ["v1.1"]  # 2022.01.21
    return versions[-1]


def autodl_topic2file():
    topic2file = OrderedDict()
    topic2file["Automated Problem Formulation"] = "Automated_Problem_Formulation.yaml"
    topic2file["Automated Data Engineering"] = "Automated_Data_Engineering.yaml"
    topic2file["Neural Architecture Search"] = "Neural_Architecture_Search.yaml"
    topic2file["Hyperparameter Optimization"] = "Hyperparameter_Optimization.yaml"
    topic2file["Automated Deployment"] = "Automated_Deployment.yaml"
    topic2file["Automated Maintenance"] = "Automated_Maintenance.yaml"
    return topic2file


def root():
    return Path(__file__).parent


def get_data_dir():
    return root() / "raw_data"


def get_bib_abbrv_file():
    return get_data_dir() / "abbrv.bib"


def autodl_topic2path():
    topic2file = autodl_topic2file()
    topic2path = OrderedDict()
    xdir = get_data_dir() / "papers"
    for topic, file_name in topic2file.items():
        topic2path[topic] = xdir / file_name
        if not topic2path[topic].exists():
            ValueError(f"Can not find {topic} at {topic2path[topic]}")
    return topic2path


def autodl_topic2papers():
    from awesome_autodl.utils import load_yaml, dump_yaml
    from awesome_autodl.data_cls import AutoDLpaper

    topic2path = autodl_topic2path()
    topic2papers = OrderedDict()
    for topic, xpath in topic2path.items():
        if not xpath.exists():
            ValueError(f"Can not find {topic} at {xpath}.")
        papers = []
        raw_data = load_yaml(xpath)
        assert isinstance(
            raw_data, (list, tuple)
        ), f"invalid type of raw data: {type(raw_data)}"
        for per_data in raw_data:
            papers.append(AutoDLpaper(per_data))
        topic2papers[topic] = papers
        print(f"Load {topic} completed with {len(papers)} papers.")
    return topic2papers


def get_bib_abbrv_obj():
    from awesome_autodl.data_cls import BibAbbreviations

    xfile = str(get_bib_abbrv_file())
    return BibAbbreviations(xfile)
