#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.08 #
#####################################################
import yaml
from pathlib import Path


def load_yaml(file_path):
    with open(str(file_path), "r") as cfile:
        data = yaml.safe_load(cfile)
    return data


def dump_yaml(data, indent=2, path=None):
    class NoAliasSafeDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    xstr = yaml.dump(
        data,
        None,
        allow_unicode=True,
        Dumper=NoAliasSafeDumper,
        indent=indent,
        sort_keys=False,
    )
    if path is not None:
        parent_dir = Path(path).resolve().parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w") as cfile:
            cfile.write(xstr)
    return xstr
