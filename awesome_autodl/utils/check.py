#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 #
#####################################################
from copy import deepcopy


def check_paper_and_correct_format(paper):
    assert isinstance(paper, dict), f"Expect dict than {type(paper)}"
    paper = deepcopy(paper)
    necessary_keys = (
        "title",
        "search_space",
        "search_strategy",
        "eval_boost",
        "online_date",
        "venue",
        "application",
    )
    for key in necessary_keys:
        assert key in paper, f"Did not find {key} in {list(paper.keys())}"
        if key != "title" and isinstance(paper[key], str):
            paper[key] = ",".join(paper[key].split(", "))
    return paper


def check_and_sort_by_date(paper_list):
    assert isinstance(paper_list, list)
    xlist = list()
    for paper in paper_list:
        paper = check_paper_and_correct_format(paper)
        xlist.append(paper)
    return sorted(xlist, key=lambda x: x["online_date"])
