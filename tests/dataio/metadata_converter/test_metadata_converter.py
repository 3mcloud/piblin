import pytest
from piblin.dataio.metadata_converter import MetadataConverter
"""

These tests are for the public API of the MetadataConverter class only. As this
class only takes string arguments, there is no need to create files on disk to
test its functionality.
"""

@pytest.fixture()
def default_metadata_converter():
    return MetadataConverter()


def test_to_string(default_metadata_converter):

    assert default_metadata_converter.\
               dict_to_string({}) == ""

    assert default_metadata_converter.\
               dict_to_string({"key": 1.0}) == "key=1.0"

    assert default_metadata_converter.\
               dict_to_string({"key": True}) == "key=True"

    assert default_metadata_converter.\
               dict_to_string({"key": "value"}) == "key=value"


def test_to_string_assignment_and_separation(default_metadata_converter):

    assert default_metadata_converter.\
               dict_to_string({"key-a": 1.0,
                               "key-b": 0.0}) == "key-a=1.0_key-b=0.0"


def test_parse_string():
    assert MetadataConverter.parse_string("value") == "value"
    assert MetadataConverter.parse_string("1.0") == 1.0
    assert MetadataConverter.parse_string("1") == 1


def test_to_dict_assignment(default_metadata_converter):

    assert default_metadata_converter.string_to_dict("") == {}

    assert default_metadata_converter.\
               string_to_dict("key=value") == {"key": "value"}

    assert default_metadata_converter.\
               string_to_dict("key=1.0") == {"key": 1.0}

    assert default_metadata_converter.\
               string_to_dict("key=1") == {"key": 1}


def test_to_dict_separator(default_metadata_converter):
    assert default_metadata_converter\
               .string_to_dict("key-a=1.0_key-b=0.0") == {"key-a": 1.0,
                                                          "key-b": 0.0}
