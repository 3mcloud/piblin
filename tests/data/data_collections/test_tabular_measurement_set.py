import pytest
from piblin.data.data_collections.tabular_measurement_set import TabularMeasurementSet
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset

@pytest.fixture()
def empty_tabular_dataset_no_metadata():
    return TabularMeasurementSet(data=[],
                                 n_metadata_columns=0,
                                 column_headers=[],
                                 dataset_types=[],
                                 dataset_end_indices=[])


@pytest.fixture()
def empty_tabular_dataset_one_metadata():
    return TabularMeasurementSet(data=[[0]],
                                 n_metadata_columns=1,
                                 column_headers=["metadata"],
                                 dataset_types=[],
                                 dataset_end_indices=[])


@pytest.fixture()
def single_scalar_tabular_dataset():
    return TabularMeasurementSet(data=[[0]],
                                 n_metadata_columns=0,
                                 column_headers=["scalar"],
                                 dataset_types=[ZeroDimensionalDataset],
                                 dataset_end_indices=[1])


@pytest.fixture()
def single_scalar_tabular_dataset_one_metadata_one_row():
    return TabularMeasurementSet(data=[[0, 0]],
                                 n_metadata_columns=1,
                                 column_headers=["metadata", "scalar"],
                                 dataset_types=[ZeroDimensionalDataset],
                                 dataset_end_indices=[1])


@pytest.fixture()
def single_scalar_tabular_dataset_one_metadata_two_rows():
    return TabularMeasurementSet(data=[[0, 0], [0, 0]],
                                 n_metadata_columns=1,
                                 column_headers=["metadata", "scalar"],
                                 dataset_types=[ZeroDimensionalDataset],
                                 dataset_end_indices=[1])


@pytest.fixture()
def two_scalar_tabular_dataset_one_row():
    return TabularMeasurementSet(data=[[0, 0]],
                                 n_metadata_columns=0,
                                 column_headers=["scalar_0", "scalar_1"],
                                 dataset_types=[ZeroDimensionalDataset, ZeroDimensionalDataset],
                                 dataset_end_indices=[1, 2])


@pytest.fixture()
def two_scalar_tabular_dataset_two_rows():
    return TabularMeasurementSet(data=[[0, 0], [0, 0]],
                                 n_metadata_columns=0,
                                 column_headers=["scalar_0", "scalar_1"],
                                 dataset_types=[ZeroDimensionalDataset, ZeroDimensionalDataset],
                                 dataset_end_indices=[1, 2])


def test_empty_tabular_dataset(two_scalar_tabular_dataset_two_rows):
    print(two_scalar_tabular_dataset_two_rows)
