#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.01 #
#####################################################
# pytest ./tests/test_format.py -s                  #
#####################################################
from awesome_autodl import autodl_topic2papers
from awesome_autodl import AutoDLpaper


class TestFormat:
    """Test the format of the raw data."""

    def test_simple(self):
        topic2papers = autodl_topic2papers()
        for topic, papers in topic2papers.items():
            print(f'Collect {len(papers)} papers for "{topic}"')
