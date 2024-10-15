import pytest
import pathlib
from piblin.dataio.data_collection_reader import DataCollectionReader
from piblin.dataio.metadata_converter import MetadataConverter


@pytest.fixture()
def example_filepath(datadir):
    return pathlib.Path(datadir.join("example.csv"))


@pytest.fixture()
def external_metadata_filename(datadir):
    return datadir.join("filename_external/key=value.csv")


def test_external_metadata_filename(external_metadata_filename):
    reader = DataCollectionReader()
    measurements = reader.from_filepath(external_metadata_filename,
                                        parse_filename_metadata=True)
    assert measurements.details["key"] == "value"


@pytest.fixture()
def filename_with_support(datadir):
    return datadir.join("filename_file_support_external/key=value.csv")


def test_filename_with_support(filename_with_support):
    reader = DataCollectionReader()
    measurements = reader.from_filepath(filename_with_support,
                                        parse_filename_metadata=True)
    assert measurements.details["key"] == "value"
    assert measurements.details["support_key"] == "support_value"


@pytest.fixture()
def filename_with_support_duplicate(datadir):
    return datadir.join("duplicate_metadata/key=filename-value.csv")


def test_filename_with_support_duplicate(filename_with_support_duplicate):
    """Make sure file value is used when duplicate metadata in filename and supporting file."""
    reader = DataCollectionReader()
    measurements = reader.from_filepath(filename_with_support_duplicate,
                                        parse_filename_metadata=True)
    assert measurements.details["key"] == "json_value"


def test_supporting_json(example_filepath):
    reader = DataCollectionReader()
    measurements = reader.from_filepath(example_filepath)
    assert measurements.details["key"] == "value"


def test_supporting_json_directory(datadir):
    reader = DataCollectionReader()
    measurements = reader.from_directory(datadir)
    assert measurements.details["key"] == "value"


@pytest.fixture()
def directory_external_metadata(datadir):
    return datadir.join("directory_external_metadata")


def test_directory_external_metadata(directory_external_metadata):
    reader = DataCollectionReader()
    measurements = reader.from_directory(directory_external_metadata)
    assert measurements.details["shared_key"] == "value"


def custom_converter():
    return "TestResult"


@pytest.mark.parametrize("metadata", [None, MetadataConverter(), custom_converter])
def test_metadata_converters(metadata):
    reader = DataCollectionReader(metadata)
    metadata_converter = reader._DataCollectionReader__metadata_converter
    assert isinstance(metadata_converter, MetadataConverter) \
        or callable(metadata_converter)


def test_bad_metadata_converter():
    with pytest.raises(TypeError):
        reader = DataCollectionReader("d")
