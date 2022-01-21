#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.01 #
#####################################################
from typing import Dict, Text, Any
from typing import Optional
import re


class AutoDLpaper:
    """A class to maintain the paper attribute."""

    # required fields, must include
    title: Text = None
    venue: Text = None
    venue_date: Text = None  # yyyy.mm
    online_date: Text = None  # yyyy.mm
    contacts: Optional[Dict[Text, Text]] = None  # a pair of name and email

    required_fields = ("title", "venue", "venue_date", "online_date", "contacts")

    # none-required fields
    search_space: Text = None
    search_strategy: Text = None
    candidate_evaluation: Text = None

    autodl_aspect_fields = ("search_space", "search_strategy", "candidate_evaluation")

    # non-required fileds
    links: Optional[Dict[Text, Text]] = None

    # misc
    discussed: bool = None
    misc: Text = None

    def __init__(self, data: Dict[Text, Any]):
        self.check_raw_data(data)
        self.reset_value(data)

    def reset_value(self, data):
        # set the basic value
        for field in self.required_fields[:-1]:
            if not isinstance(data[field], str):
                raise TypeError(
                    f"Expect {field} is str instead of {type(data[field])}."
                )
        self.title = data["title"]
        self.venue = data["venue"]
        date_pattern = re.compile("^([0-9]){4}.([0-9]){2}$")
        if date_pattern.match(data["venue_date"]) is None:
            raise ValueError(f"Invalid venue date ({data['venue_date']}) :: {data}")
        if date_pattern.match(data["online_date"]) is None:
            raise ValueError(f"Invalid online date ({data['online_date']}) :: {data}")
        self.venue_date = data["venue_date"]
        self.online_date = data["online_date"]

        # set the contact information
        if data["contacts"] is not None:
            assert isinstance(data["contacts"], dict)
            self.contacts = data["contacts"]
            # TODO(xuanyidong): check the email address
            # regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        # set the AutoDL aspect fields
        for field in self.autodl_aspect_fields:
            if data[field] is None:
                continue
            assert isinstance(data[field], str)
            setattr(self, field, data[field])

        # set the misc info
        if "discussed" in data:
            assert isinstance(
                data["discussed"], bool
            ), f"The discussed field must be a bool instead of a {type(data['discussed'])}"
            self.discussed = data["discussed"]
        if "misc" in data and data["misc"] is not None:
            assert isinstance(
                data["misc"], str
            ), f"The misc field must be a str instead of a {type(data['discussed'])}"
            self.misc = data["misc"]

    def check_raw_data(self, data):
        """Check whether the necessary field is included in `data`."""
        if not isinstance(data, dict):
            raise TypeError(f"Expect data to be a dict instead of {type(data)}")
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing {field} in {data}")
        for field in self.autodl_aspect_fields:
            if field not in data:
                raise ValueError(
                    f"Missing {field} in {data}"
                    + "Please leave this field as blank if you are not sure about it."
                )
        all_fields = (
            list(self.required_fields)
            + list(self.autodl_aspect_fields)
            + ["discussed", "misc", "links"]
        )
        for key in data.keys():
            if key not in all_fields:
                raise ValueError(f"Find unexpected field: {key} in {data}")

    @property
    def author_email(self):
        if self.contacts is None:
            return dict()
        else:
            return self.contacts

    def __repr__(self):
        return f"{self.__class__.__name__}({self.title})"
