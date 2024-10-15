import pytest
from piblin.data.datasets.abc.split_datasets.two_dimensional_dataset import TwoDimensionalDataset


@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_twod_dataset_single_point_default(twod_dataset_four_points, datadir):

    figure, axes = twod_dataset_four_points.visualize(include_text=False)

    return figure


@pytest.fixture()
def independent_variable_name():
    return "x"


@pytest.fixture()
def dependent_variable_value():
    return 1


@pytest.fixture()
def twod_dataset(dependent_variable_value):

    return TwoDimensionalDataset.create(x_values=[0],
                                        y_values=[0],
                                        z_values=[[dependent_variable_value]])


def test_remove_independent_variable_by_name_2d(independent_variable_name,
                                                dependent_variable_value,
                                                twod_dataset):

    dataset = twod_dataset.remove_independent_variable_by_name(independent_variable_name)

    assert dataset.number_of_independent_dimensions == 1
    assert dataset.y_values[0] == dependent_variable_value
