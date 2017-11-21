import os
import pytest

import partridge as ptg

from peartree.paths import (generate_random_name,
							get_representative_feed,
							InvalidGTFS)


def fixture(filename):
    return os.path.join(os.path.dirname(__file__), 'fixtures', filename)


def test_generate_name():
	name = generate_random_name(10)
	assert len(name) == 10

	name = generate_random_name(12)
	assert len(name) == 12

	name = generate_random_name()
	assert isinstance(name, str)


def test_extract_valid_feed():
    # Read in without name, or any
    # other optional arguments
    path = fixture('caltrain-2017-07-24.zip')
    feed = get_representative_feed(path)
    assert isinstance(feed, ptg.gtfs.feed)
