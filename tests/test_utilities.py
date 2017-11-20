import os
import pytest

import partridge as ptg
import peartree as pt
from peartree.utilities import get_representative_feed, InvalidGTFS


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_extract_valid_feed():
    # Read in without name, or any
    # other optional arguments
    path = fixture('caltrain-2017-07-24.zip')
    feed = get_representative_feed(path)


def test_extract_empty_feed():
    path = fixture('empty.zip')
    with pytest.raises(InvalidGTFS):
        get_representative_feed(path, 'busiest', 'foobar')