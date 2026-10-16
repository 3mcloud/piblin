import pytest
import pathlib
import shutil
import piblin.dataio
from piblin.dataio.data_collection_reader import UnsupportedFileSubtypeError
from piblin.dataio.fileio.read import FileParsingException
from piblin.dataio.fileio.read.generic.generic_csv import CsvReader
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


class DatAsCsvReader(CsvReader):
    """Reads comma-separated .dat files; not selected by extension."""
    supported_extensions = set()
    calls = 0

    def _read_file_contents(self, filepath, **read_kwargs):
        DatAsCsvReader.calls += 1
        return super()._read_file_contents(filepath, **read_kwargs)


class FailingReader(CsvReader):
    supported_extensions = set()

    def _read_file_contents(self, filepath, **read_kwargs):
        raise FileParsingException("cannot parse")


@pytest.fixture()
def dat_filepath(example_filepath):
    dat = example_filepath.with_suffix(".dat")
    shutil.copy(example_filepath, dat)
    return dat


def test_unregistered_extension_needs_file_reader(dat_filepath):
    with pytest.raises(UnsupportedFileSubtypeError):
        DataCollectionReader().from_filepath(dat_filepath)


def test_file_reader_is_used(dat_filepath):
    DatAsCsvReader.calls = 0
    measurements = DataCollectionReader().from_filepath(dat_filepath, file_reader=DatAsCsvReader)
    assert DatAsCsvReader.calls == 1
    assert measurements.details["key"] == "value"


def test_file_reader_takes_precedence_over_extension(example_filepath):
    DatAsCsvReader.calls = 0
    DataCollectionReader().from_filepath(example_filepath, file_reader=DatAsCsvReader)
    assert DatAsCsvReader.calls == 1


def test_file_reader_list_is_tried_in_order(dat_filepath):
    DatAsCsvReader.calls = 0
    DataCollectionReader().from_filepath(dat_filepath, file_reader=[FailingReader, DatAsCsvReader])
    assert DatAsCsvReader.calls == 1


def test_file_reader_list_all_fail(dat_filepath):
    with pytest.raises(UnsupportedFileSubtypeError, match="FailingReader"):
        DataCollectionReader().from_filepath(dat_filepath, file_reader=[FailingReader])


@pytest.mark.parametrize("bad", [CsvReader(), "CsvReader", [], [CsvReader, int]])
def test_file_reader_type_is_checked(dat_filepath, bad):
    with pytest.raises(TypeError):
        DataCollectionReader().from_filepath(dat_filepath, file_reader=bad)


def test_read_functions_pass_file_reader(dat_filepath):
    DatAsCsvReader.calls = 0
    piblin.dataio.read_file(str(dat_filepath), file_reader=DatAsCsvReader)
    piblin.dataio.read_files([str(dat_filepath)], file_reader=DatAsCsvReader)
    piblin.dataio.read_directory(str(dat_filepath.parent), extension="dat",
                                 file_reader=DatAsCsvReader)
    piblin.dataio.read_directories([str(dat_filepath.parent)], extension="dat",
                                   file_reader=DatAsCsvReader)
    assert DatAsCsvReader.calls == 4
