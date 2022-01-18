#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.01 #
#####################################################
# pytest ./tests/test_abbrv.py -s                   #
#####################################################
import os
from awesome_autodl import get_bib_abbrv_file
from awesome_autodl.data_cls import BibAbbreviations


class TestAbbrv:
    """Test the bib file for Abbreviations."""

    def test_init(self):
        xfile = str(get_bib_abbrv_file())
        assert os.path.isfile(xfile)
        obj = BibAbbreviations(xfile)
        assert len(obj) == 32
        print(obj)
