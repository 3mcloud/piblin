import numpy as np
from numpy import array  # NOT UNUSED
import pytest
from piblin.data.datasets.abc.split_datasets.one_dimensional_dataset import OneDimensionalDataset


def test_value_at_repeat_values(oned_dataset_repeated_values):
    assert oned_dataset_repeated_values.value_at(0.0) == 0.5
    assert oned_dataset_repeated_values.value_at(0.0, single_value=False) == [0.0, 1.0]


def test_value_at(oned_dataset_multiple_points_and_values):
    assert oned_dataset_multiple_points_and_values.value_at(0.0) == 0.0
    assert oned_dataset_multiple_points_and_values.value_at(1.0) == 1.0
    assert oned_dataset_multiple_points_and_values.value_at(0.5) == 0.5
    with pytest.raises(ValueError):
        oned_dataset_multiple_points_and_values.value_at(1.5)
    with pytest.raises(ValueError):
        oned_dataset_multiple_points_and_values.value_at(-0.5)


def test_x_value_setter_single_point(oned_dataset_single_point):
    oned_dataset_single_point.x_values = [1.0]
    assert oned_dataset_single_point.x_values == [1.0]


def test_x_value_setter_multi_point(oned_dataset_multiple_points):
    oned_dataset_multiple_points.y_values = [1.0, 1.0]
    assert np.array_equal(oned_dataset_multiple_points.y_values, [1.0, 1.0])


def test_y_value_setter_single_point(oned_dataset_single_point):
    oned_dataset_single_point.y_values = [1.0]
    assert oned_dataset_single_point.y_values == [1.0]


def test_y_value_setter_multi_point(oned_dataset_multiple_points):
    oned_dataset_multiple_points.x_values = [1.0, 2.0]
    assert np.array_equal(oned_dataset_multiple_points.x_values, [1.0, 2.0])


def test_as_ndarray(oned_dataset_single_point):
    assert np.array_equal(oned_dataset_single_point.as_ndarray(), [[0.0], [0.0]])


def test_one_line_str(oned_dataset_single_point,
                      oned_dataset_multiple_points):
    """Ensure the one line str is one line."""
    assert oned_dataset_single_point.one_line_str().count("\n") == 0
    assert oned_dataset_multiple_points.one_line_str().count("\n") == 0


def test_flatten_single_point(oned_dataset_single_point):
    column_labels, row = oned_dataset_single_point.flatten()

    assert len(column_labels) == 1
    assert column_labels[0] == "y(None)=f(x(None)=0.0)"
    assert row == [0.0]


def test_flatten_multi_point(oned_dataset_multiple_points):
    column_labels, row = oned_dataset_multiple_points.flatten()

    assert len(column_labels) == 2
    assert column_labels[0] == "y(None)=f(x(None)=0.0)"
    assert column_labels[1] == "y(None)=f(x(None)=1.0)"

    assert np.array_equal(row, [0.0, 0.0])


def test_unflatten_single_point(oned_dataset_single_point):
    unflattened_dataset = oned_dataset_single_point.unflatten(column_labels=["y(None)=f(x(None)=0.00)"],
                                                               values=np.array([0.0]))
    assert unflattened_dataset == oned_dataset_single_point


def test_flatten_round_trip_single_point(oned_dataset_single_point):
    flat_dataset = oned_dataset_single_point.flatten()
    assert oned_dataset_single_point.unflatten(flat_dataset[0], flat_dataset[1]) == oned_dataset_single_point


def test_unflatten_multi_point(oned_dataset_multiple_points):
    assert oned_dataset_multiple_points.unflatten(column_labels=["y(None)=f(x(None)=0.0)",
                                                                 "y(None)=f(x(None)=1.0)"],
                                                  values=np.array([0.0, 0.0])) == \
           oned_dataset_multiple_points


def test_flatten_round_trip_multi_point(oned_dataset_multiple_points):
    flat_dataset = oned_dataset_multiple_points.flatten()
    assert oned_dataset_multiple_points.unflatten(flat_dataset[0],
                                                  flat_dataset[1]) == oned_dataset_multiple_points


def test_repr(oned_dataset_single_point):
    """Ensure repr evals to equal object for single-value data."""
    assert eval(repr(oned_dataset_single_point)) == oned_dataset_single_point


def test_equals(oned_dataset_single_point):
    """Check the implementation of equality, including labelling."""
    # an instance is equal to itself
    assert oned_dataset_single_point == oned_dataset_single_point
    # an instance is not equal to another instance with different data
    assert OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]]) != \
           OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[1.0]])
    # an instance is equal to another instance with the same data
    assert OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]]) == \
           OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]])
    # two instances with the same data and different labels are not equal
    assert OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]],
                                 dependent_variable_names=["a"]) != \
           OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]],
                                 dependent_variable_names=["b"])
    # two instances with the same data/dependent variable name and different independent labels are not equal
    assert OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]],
                                 dependent_variable_names=["a"],
                                 independent_variable_names=["b"]) != \
           OneDimensionalDataset(dependent_variable_data=[0.0],
                                 independent_variable_data=[[0.0]],
                                 dependent_variable_names=["a"],
                                 independent_variable_names=["c"])


@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_oned_dataset_single_point_default(oned_dataset_single_point, datadir):

    figure, axes = oned_dataset_single_point.visualize(include_text=False)

    return figure


@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_oned_dataset_single_point_labelled_default(oned_dataset_single_point_labelled,
                                                              nondefault_xvalue_name,
                                                              nondefault_yvalue_name,
                                                              nondefault_unit,
                                                              datadir):

    figure, axes = oned_dataset_single_point_labelled.visualize(include_text=False)

    assert axes.get_xlabel() == f"{nondefault_xvalue_name}({nondefault_unit})"
    assert axes.get_ylabel() == f"{nondefault_yvalue_name}({nondefault_unit})"

    return figure


@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_oned_dataset_multiple_points(oned_dataset_multiple_points, datadir):

    figure, axes = oned_dataset_multiple_points.visualize(include_text=False)

    return figure


@pytest.fixture()
def independent_variable_name():
    return "x"


@pytest.fixture()
def dependent_variable_value():
    return 1


@pytest.fixture()
def oned_dataset(independent_variable_name,
                 dependent_variable_value):

    return OneDimensionalDataset.create(x_values=[0],
                                        x_name=independent_variable_name,
                                        y_values=[dependent_variable_value])


def test_remove_independent_variable_by_name(independent_variable_name,
                                             dependent_variable_value,
                                             oned_dataset):

    dataset = oned_dataset.remove_independent_variable_by_name(independent_variable_name)

    assert dataset.number_of_independent_dimensions == 0
    assert dataset.value == dependent_variable_value
