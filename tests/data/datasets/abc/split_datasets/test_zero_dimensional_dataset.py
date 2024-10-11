import numpy as np
from numpy import array  # NOT UNUSED
import matplotlib.axes
import matplotlib.figure
import pytest
from piblin.data.datasets.abc.split_datasets.zero_dimensional_dataset import ZeroDimensionalDataset
from piblin.data.datasets.abc.dataset import VisualizationError


def test_value_at_repeat_values(zero_float_dataset):
    assert zero_float_dataset.value == 0.0


def test_x_value_setter_single_point(zero_float_dataset):
    zero_float_dataset.value = 1.0
    assert zero_float_dataset.value == 1.0


def test_flatten_single_point(zero_float_dataset, nondefault_float_variable_name, nondefault_unit):
    column_labels, row = zero_float_dataset.flatten()

    assert len(column_labels) == 1
    assert column_labels[0] == f"{nondefault_float_variable_name}({nondefault_unit})"
    assert row == [0.0]


def test_unflatten_single_point(zero_float_dataset, nondefault_float_variable_name, nondefault_unit):
    unflattened_dataset = zero_float_dataset.unflatten(column_labels=[f"{nondefault_float_variable_name}({nondefault_unit})"],
                                                       values=np.array(0.0))
    assert unflattened_dataset == zero_float_dataset


def test_flatten_round_trip_single_point(zero_float_dataset):
    flat_dataset = zero_float_dataset.flatten()
    assert zero_float_dataset.unflatten(flat_dataset[0], flat_dataset[1]) == zero_float_dataset


def test_repr(zero_float_dataset):
    """Ensure repr evals to equal object for single-value data."""
    assert eval(repr(zero_float_dataset)) == zero_float_dataset


def test_equals(zero_float_dataset):
    """Check the implementation of equality, including labelling."""
    # an instance is equal to itself
    assert zero_float_dataset == zero_float_dataset
    # an instance is not equal to another instance with different data
    assert ZeroDimensionalDataset(dependent_variable_data=0.0) != \
           ZeroDimensionalDataset(dependent_variable_data=0.1)
    # an instance is equal to another instance with the same data
    assert ZeroDimensionalDataset(dependent_variable_data=0.0) == \
           ZeroDimensionalDataset(dependent_variable_data=0.0)
    # two instances with the same data and different labels are not equal
    assert ZeroDimensionalDataset(dependent_variable_data=0.0,
                                  dependent_variable_names=["a"]) != \
           ZeroDimensionalDataset(dependent_variable_data=0.0,
                                  dependent_variable_names=["b"])
    # two instances with the same data/dependent variable name and different independent labels are not equal
    assert ZeroDimensionalDataset(dependent_variable_data=0.0,
                                  dependent_variable_names=["a"]) == \
           ZeroDimensionalDataset(dependent_variable_data=0.0,
                                  dependent_variable_names=["a"])


def test_create():
    """Ensure that different parameter types result in the same scalar."""
    assert ZeroDimensionalDataset.create(0.0) == \
           ZeroDimensionalDataset.create([0.0]) == \
           ZeroDimensionalDataset.create(np.array(0.0)) == \
           ZeroDimensionalDataset.create(np.array([0.0]))


def test_create_default():
    """Test that the create method results in the expected object."""
    assert ZeroDimensionalDataset.create(value=0.0,
                                         label="d_0",
                                         unit=None) == ZeroDimensionalDataset(0.0)


def test_create_label():
    """Test that the create method with specified label results in the expected object."""
    assert ZeroDimensionalDataset.create(value=0.0,
                                         label="label") == ZeroDimensionalDataset(dependent_variable_data=0.0,
                                                                                  dependent_variable_names=["label"])


def test_access(zero_float_dataset, nondefault_float_variable_name):
    """Make sure the label and value properties are accessible."""
    assert zero_float_dataset.label == nondefault_float_variable_name
    assert zero_float_dataset.value == 0.0


def test_column_labels(zero_float_dataset, nondefault_float_variable_name, nondefault_unit):
    """Make sure the column labels are the correct type and value."""
    assert zero_float_dataset._encode_column_labels() == [f"{nondefault_float_variable_name}({nondefault_unit})"]


def test_flatten_dependent_variables(zero_float_dataset):
    """Make sure the dependent variables are the correct type and value."""
    assert zero_float_dataset.flatten_dependent_variables() == [0.0]


def test_flatten(zero_float_dataset, nondefault_float_variable_name, nondefault_unit):
    """Ensure the superclass flatten method works with overridden sub-methods."""
    column_labels, values = zero_float_dataset.flatten()
    assert column_labels == [f"{nondefault_float_variable_name}({nondefault_unit})"]
    assert values == [0.0]


def test_one_line_str(zero_float_dataset):
    """Ensure the one line str is one line."""
    assert zero_float_dataset.one_line_description.count("\n") == 0

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_zerod_dataset_false_nondefault_name_default_unit(nondefault_boolean_variable_name,
                                                                    zerod_dataset_false_nondefault_name_default_unit, datadir):
    """Ensure visualization output for single 0D false dataset does not change."""
    figure, axes = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False)

    assert isinstance(axes, matplotlib.axes.Axes)
    assert len(figure.axes) == 1
    assert axes.get_title() == nondefault_boolean_variable_name
    assert figure.texts == []
    assert (figure.get_size_inches() == zerod_dataset_false_nondefault_name_default_unit.DEFAULT_FIGURE_SIZE).all()

    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_zerod_dataset_false_nondefault_name_default_unit_custom_axes(zerod_dataset_false_nondefault_name_default_unit, nondefault_boolean_variable_name, empty_axes):
    """Ensure visualization with a custom axes works."""
    figure, axes = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False,
                                                                              axes=empty_axes)

    assert axes is empty_axes
    assert len(figure.axes) == 1
    assert figure.texts == []

    return figure


def test_list_of_axes_raises_error(zerod_dataset_false_nondefault_name_default_unit, empty_axes):
    """Ensure that providing a list of axes raises an error."""
    with pytest.raises(VisualizationError):
        _, _ = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False,
                                                                          axes=[empty_axes])

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_zerod_dataset_false_nondefault_name_default_unit_figure_title(zerod_dataset_false_nondefault_name_default_unit, nondefault_figure_title):
    """Ensure the figure title argument is respected."""
    figure, axes = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False,
                                                                              figure_title=nondefault_figure_title)

    assert figure.texts[0].get_text() == nondefault_figure_title

    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_zerod_dataset_false_nondefault_name_default_unit_axis_title(zerod_dataset_false_nondefault_name_default_unit, nondefault_axes_title):
    """Ensure the axis title argument is respected."""
    figure, axes = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False,
                                                                              axis_title=nondefault_axes_title)

    assert axes.get_title() == nondefault_axes_title

    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_zerod_dataset_false_nondefault_name_default_unit_both_title(zerod_dataset_false_nondefault_name_default_unit,
                                            nondefault_figure_title,
                                            nondefault_axes_title):
    """Ensure the figure and axis title arguments are respected."""
    figure, axes = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False,
                                                                              axis_title=nondefault_axes_title,
                                                                              figure_title=nondefault_figure_title)

    assert figure.texts[0].get_text() == nondefault_figure_title
    assert axes.get_title() == nondefault_axes_title

    return figure

@pytest.mark.skip
@pytest.mark.mpl_image_compare(hash_library="../../../../matplotlib_test_hashes.json")
def test_visualize_zerod_dataset_false_nondefault_name_default_unit_total_figsize(zerod_dataset_false_nondefault_name_default_unit, specific_figsize):
    figure, axes = zerod_dataset_false_nondefault_name_default_unit.visualize(include_text=False,
                                                                              total_figsize=specific_figsize)

    assert (figure.get_size_inches() == specific_figsize).all()

    return figure


# def test_unflatten(zero_float_dataset):
#     """Ensure the scalar value unflattens appropriately."""
#     assert zero_float_dataset.unflatten_dependent_variables([0.0]) == 0.0
#
#
# def test_decode_column_labels(zero_float_dataset, float_label, nondefault_unit):
#     """Ensure the labels and independent variables unflatten appropriately."""
#     result = zero_float_dataset.decode_column_labels([f"{float_label}({nondefault_unit})"])
#
#     assert result[0] is None
#     assert result[1] == "Float"
#     assert result[2] is None
