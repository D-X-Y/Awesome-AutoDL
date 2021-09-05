#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.09 #
#####################################################
# An package to analyze paper statistics of AutoDL  #
# papers.                                           #
#####################################################
from pathlib import Path


def version():
    versions = ["0.0.1"]  # 2021.09.03
    versions = ["0.0.2"]  # 2021.09.04
    return versions[-1]


def root():
    return Path(__file__).parent
