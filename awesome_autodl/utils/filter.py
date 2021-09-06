#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 #
#####################################################
from copy import deepcopy


def filter_ele_w_value(paper_list, key, value):
    assert isinstance(paper_list, list)
    xlist = list()
    for paper in paper_list:
        if value in paper[key]:
            xlist.append(deepcopy(paper))
    return xlist
