#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.01 #
#####################################################
import re
import os


class BibAbbreviations:
    """A class to maintain the paper venue abbreviation."""

    def __init__(self, xfile):
        xfile = str(xfile)
        assert os.path.isfile(xfile)
        with open(xfile) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]

        pattern = r'^@STRING\{([A-Z]|[a-z]|\s|_)*=(\s)*"([A-Z]|[a-z]|\s|\(|\)|\/|\,|\:|\-)*"\}$'
        self.prog = re.compile(pattern)
        self.abbrv2str = dict()
        for index, line in enumerate(lines):
            if line:
                if self.prog.match(line) is None:
                    raise ValueError(f"Incorrect line [{index}]: {line}")
                key, value = self.decode(line)
                if key in self.abbrv2str:
                    raise ValueError(f"Already defined {key}")
                self.abbrv2str[key] = value

    def __contains__(self, key):
        return key in self.abbrv2str

    def __getitem__(self, key):
        return self.abbrv2str[key]

    def __len__(self):
        return len(self.abbrv2str)

    def keys(self, sort=False):
        keys = list(self.abbrv2str.keys())
        if sort:
            keys = sorted(keys)
        return keys

    def decode(self, line):
        head = "@STRING{"
        assert len(line) > len(head)
        assert line[: len(head)] == head and line[-1] == "}"
        line = line[len(head) : -1]
        key, value = line.split("=")
        key, value = key.strip(" "), value.strip(" ")
        return key, value

    def __repr__(self):
        return f"{self.__class__.__name__}(" + f"{len(self)} abbrev pairs)"
