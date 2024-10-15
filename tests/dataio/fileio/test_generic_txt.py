import pytest
import numpy as np
from piblin.dataio.fileio.read.generic.generic_txt import TxtReader


@pytest.fixture()
def minimal_measurement(datadir):
    filepath = datadir.join("minimal.txt")
    txt_reader = TxtReader()
    return txt_reader.data_from_filepath(filepath)[0]


def test_tmpdir(minimal_measurement):
    assert minimal_measurement.conditions == {}
    assert minimal_measurement.detail_names == {"source_disk_directory",
                                                "source_file_name",
                                                "source_file_name_head",
                                                "source_file_path"}
    assert np.array_equal(minimal_measurement.datasets[0].x_values, [0.0, 1.0])
