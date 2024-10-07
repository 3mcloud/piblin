import pytest
import numpy as np
from piblin.dataio.fileio.read.generic.generic_csv import CsvReader


@pytest.fixture()
def minimal_measurement(datadir):
    filepath = datadir.join("minimal.csv")
    csv_reader = CsvReader()
    return csv_reader.data_from_filepath(filepath)


def test_tmpdir(minimal_measurement):
    assert len(minimal_measurement) == 1
    assert minimal_measurement[0].conditions == {}
    assert minimal_measurement[0].detail_names == {"source_disk_directory",
                                                   "source_file_name",
                                                   "source_file_name_head",
                                                   "source_file_path"}
    assert np.array_equal(minimal_measurement[0].datasets[0].x_values, [0.0, 1.0])
